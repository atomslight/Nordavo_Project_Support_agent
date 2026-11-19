"""Model calls and final safety checks for one support decision."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nordavo.ai.language import LanguageAnalyzer
from nordavo.ai.prompts import DECISION_SYSTEM_PROMPT, DECISION_TICKET_PROMPT
from nordavo.ai.safety import delimit_ticket_text, redact_sensitive
from nordavo.models import Decision, DecisionResult, LanguageAssessment, Ticket

FAQ_IDS = frozenset(f"F{number:02d}" for number in range(1, 13))
_INTERNAL_FAQ_REFERENCE = re.compile(
    r"\s*\(?(?:FAQ\s*:?\s*)?\bF\d{2}\b\)?", re.IGNORECASE
)
logger = logging.getLogger(__name__)


class DecisionProviderError(RuntimeError):
    """Safe, customer-text-free failure from a decision provider."""

    def __init__(
        self,
        provider: str,
        error_type: str,
        *,
        status: int | None = None,
        code: str | int | None = None,
        raw_error: Exception | None = None,
    ) -> None:
        self.provider = provider
        self.error_type = error_type
        self.status = status
        self.code = code
        self.raw_error = raw_error
        details = [error_type]
        if status is not None:
            details.append(f"HTTP {status}")
        if code is not None:
            details.append(f"code {code}")
        super().__init__(f"{provider.title()} decision failed ({', '.join(details)})")


class _DecisionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)
    category: Literal["bug", "billing", "feature_request", "question", "other"]
    priority: Literal["P1", "P2", "P3", "P4"]
    action: Literal["reply", "handoff", "close"]
    faq_ids: list[str]
    draft_reply: str | None
    handoff_note: str | None
    confidence: float = Field(ge=0, le=1)


class GeminiDecisionClassifier:
    def __init__(self, client, model: str) -> None:
        self._client = client
        self._model = model

    def health_check(self) -> None:
        self.decide(_health_ticket(), "nl", "F01: Test FAQ")

    def decide(self, ticket: Ticket, language: str, faq_text: str) -> DecisionResult:
        system_prompt, ticket_prompt = _prompts(ticket, language, faq_text)
        try:
            response = self._client.interactions.create(
                model=self._model,
                input=ticket_prompt,
                system_instruction=system_prompt,
                generation_config={"temperature": 0},
                response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "roosterly_decision",
                    "strict": True,
                    "schema": _DecisionResponse.model_json_schema(),
                },
            },
            )
            return _result_from_json(
                response.output_text, "gemini", self._model, _usage(response)
            )
        except DecisionProviderError:
            raise
        except Exception as error:
            raise _provider_error("gemini", error) from None


class GroqDecisionClassifier:
    def __init__(self, client, model: str) -> None:
        self._client = client
        self._model = model

    def health_check(self) -> None:
        self.decide(_health_ticket(), "nl", "F01: Test FAQ")

    def decide(self, ticket: Ticket, language: str, faq_text: str) -> DecisionResult:
        system_prompt, ticket_prompt = _prompts(ticket, language, faq_text)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                reasoning_effort="none",
                reasoning_format="hidden",
                temperature=0.6,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ticket_prompt},
                ],
                response_format={"type": "json_object"},
            )
            return _result_from_json(
                response.choices[0].message.content,
                "groq",
                self._model,
                _usage(response),
            )
        except DecisionProviderError:
            raise
        except Exception as error:
            raise _provider_error("groq", error) from None


class DecisionAnalyzer:
    """Call the configured provider, then exactly one fallback."""

    def __init__(self, primary, fallback, audit=None) -> None:
        self._primary = primary
        self._fallback = fallback
        self._audit = audit

    def health_check(self) -> None:
        self._primary.health_check()
        self._fallback.health_check()
        self._event("health_check", stage="decision")

    #Entry point for decision making from the decision workflow. It first tries the primary provider, and if that fails, it falls back to the secondary provider. It also logs events for auditing purposes.
    def decide(self, ticket: Ticket, language: str, faq_text: str) -> DecisionResult:
        try:
            return replace(
                self._primary.decide(ticket, language, faq_text), fallback_used=False
            )
        except DecisionProviderError as error:
            _log_provider_event("fallback", error)
            self._event("fallback_used", provider=error.provider, error_type=error.error_type,
                        status=error.status, code=error.code)
            try:
                return replace(
                    self._fallback.decide(ticket, language, faq_text), fallback_used=True
                )
            except DecisionProviderError as error:
                _log_provider_event("failure", error)
                self._event("provider_failure", provider=error.provider, error_type=error.error_type,
                            status=error.status, code=error.code)
                raise error from None

    def _event(self, name: str, **details) -> None:
        if self._audit:
            self._audit.event(name, **details)


def validate_decision(
    result: DecisionResult,
    ticket: Ticket,
    language_result: LanguageAssessment,
    language_analyzer: LanguageAnalyzer,
) -> DecisionResult:
    """Convert only invalid or explicitly unsafe outgoing work to handoff."""
    decision = result.decision
    if decision.action == "reply" and decision.draft_reply:
        # FAQ IDs are internal grounding evidence, never customer-facing text.
        decision = replace(
            decision,
            draft_reply=_remove_internal_faq_references(decision.draft_reply),
        )
        result = replace(result, decision=decision)
    unsafe_topic = _always_human_topic(ticket)
    invalid = _invalid_decision(decision)
    if decision.action == "reply":
        reply_language = language_analyzer.analyze(decision.draft_reply or "").language
        invalid = invalid or reply_language in {"unknown"} or reply_language != language_result.language
    if decision.action == "handoff":
        note_language = language_analyzer.analyze(decision.handoff_note or "").language
        invalid = invalid or note_language != "en"
    if unsafe_topic and decision.action in {"reply", "close"}:
        invalid = True
    if invalid:
        # Do not invent a second model response after a failed safety check.
        # A fixed English handoff keeps the customer-facing action safe.
        reason = "Human review is required before this ticket can be answered safely."
        note = "Review this ticket manually because the proposed automated decision failed a safety check."
        decision = Decision(
            language=language_result.language,
            reason=reason,
            category=decision.category,
            priority=_priority_floor(decision.priority, ticket),
            action="handoff",
            faq_ids=(),
            draft_reply=None,
            handoff_note=note,
            confidence=decision.confidence,
        )
    else:
        decision = replace(decision, priority=_priority_floor(decision.priority, ticket))
    return replace(result, decision=decision)


def _remove_internal_faq_references(text: str) -> str:
    """Remove internal IDs without changing the meaning of the reply."""
    if _INTERNAL_FAQ_REFERENCE.search(text) is None:
        # Leave a valid reply untouched unless an internal reference was removed.
        return text
    cleaned = _INTERNAL_FAQ_REFERENCE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def _prompts(ticket: Ticket, language: str, faq_text: str) -> tuple[str, str]:
    return (
        DECISION_SYSTEM_PROMPT.format(ticket_language=language, faq_text=faq_text),
        DECISION_TICKET_PROMPT.format(ticket_text=delimit_ticket_text(ticket.subject, ticket.body)),
    )


def _result_from_json(
    output_text: str, provider: Literal["gemini", "groq"], model: str, usage: tuple[int | None, int | None]
) -> DecisionResult:
    try:
        response = _DecisionResponse.model_validate_json(output_text)
    except ValidationError as error:
        raise DecisionProviderError(provider, "ValidationError", raw_error=error) from None
    decision = Decision(
        language="unknown",  # The trusted Phase 1 result is attached by the workflow.
        reason=response.reason,
        category=response.category,
        priority=response.priority,
        action=response.action,
        faq_ids=tuple(response.faq_ids),
        draft_reply=response.draft_reply,
        handoff_note=response.handoff_note,
        confidence=response.confidence,
    )
    return DecisionResult(decision, provider, model, False, usage[0], usage[1])


def _invalid_decision(decision: Decision) -> bool:
    fields = (decision.reason, decision.draft_reply or "", decision.handoff_note or "")
    unsafe_text = any("<REDACTED" in value or redact_sensitive(value) != value for value in fields)
    if unsafe_text or any(faq_id not in FAQ_IDS for faq_id in decision.faq_ids):
        return True
    if decision.action == "reply":
        return not decision.faq_ids or not decision.draft_reply or decision.handoff_note is not None
    if decision.action == "handoff":
        return decision.draft_reply is not None or not decision.handoff_note
    return bool(decision.faq_ids or decision.draft_reply or decision.handoff_note)


def _always_human_topic(ticket: Ticket) -> bool:
    text = f"{ticket.subject}\n{ticket.body}".lower()
    terms = (
        "refund", "restitutie", "disputed charge", "chargeback", "betwiste",
        "cancel", "opzeg", "contract", "gdpr", "avg", "personal data", "persoonsgegevens",
        "security", "beveilig", "data breach", "datalek", "legal", "lawyer", "jurid",
        "leave", "switch provider", "opstappen", "verlaten",
    )
    return any(term in text for term in terms)


def _priority_floor(priority: str, ticket: Ticket) -> Literal["P1", "P2", "P3", "P4"]:
    text = f"{ticket.subject}\n{ticket.body}".lower()
    if any(term in text for term in ("security", "beveilig", "data breach", "datalek", "legal", "lawyer", "jurid")):
        return "P1"
    if any(term in text for term in ("refund", "restitutie", "charge", "contract", "cancel", "opzeg", "gdpr", "avg", "leave", "opstappen")):
        return "P2" if priority in {"P3", "P4"} else priority  # type: ignore[return-value]
    return priority  # type: ignore[return-value]


def _usage(response) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None
    return (
        getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None),
        getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None),
    )


def _provider_error(provider: Literal["gemini", "groq"], error: Exception) -> DecisionProviderError:
    return DecisionProviderError(
        provider,
        type(error).__name__,
        status=_safe_status(error),
        code=_safe_provider_code(provider, error),
        raw_error=error,
    )


def _safe_status(error: Exception) -> int | None:
    for name in ("status_code", "code"):
        value = getattr(error, name, None)
        if isinstance(value, int):
            return value
    return None


def _safe_provider_code(provider: Literal["gemini", "groq"], error: Exception) -> str | int | None:
    if provider == "gemini":
        value = getattr(error, "code", None)
        return value if isinstance(value, (str, int)) else None
    body = getattr(error, "body", None)
    api_error = body.get("error") if isinstance(body, dict) else None
    value = api_error.get("code") if isinstance(api_error, dict) else None
    return value if isinstance(value, (str, int)) else None


def _log_provider_event(event: str, error: DecisionProviderError) -> None:
    # Provider messages can echo prompt content, so logs use only known metadata.
    logger.warning(
        "decision_provider_%s provider=%s error_type=%s status=%s code=%s",
        event, error.provider, error.error_type, error.status, error.code,
    )


def _health_ticket() -> Ticket:
    return Ticket("health", "", "", None, "Vraag", "Hallo, ik heb een vraag over het rooster.")
