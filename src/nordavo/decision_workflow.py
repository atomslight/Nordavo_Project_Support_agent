"""Decision-only and explicit-send workflows layered on Phase 1."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from nordavo.ai.decision import DecisionAnalyzer, DecisionProviderError, validate_decision
from nordavo.ai.safety import redact_ticket
from nordavo.be.decision_state import DecisionStateStore
from nordavo.be.helpdesk import HelpdeskError
from nordavo.be.run_log import RunLog
from nordavo.be.state import StateStore
from nordavo.models import Decision, DecisionResult, LanguageAssessment, Ticket


_REPLY_ONLY_SUBJECT = re.compile(r"(?:re:\s*)+", re.IGNORECASE)
_EMPTY_TICKET_LANGUAGE = "nl"
_REUSABLE_DECISION_STATUSES = {"decided", "sending", "sent", "already_handled"}


@dataclass(frozen=True)
class DecisionSummary:
    fetched_tickets: int
    decided_tickets: int
    reused_decisions: int = 0
    sent_tickets: int = 0
    language_unresolved_tickets: int = 0
    unresolved_ticket_ids: tuple[str, ...] = ()


class TicketDecisionWorkflow:
    def __init__(self, helpdesk, state: StateStore, decisions: DecisionStateStore,
                 languages, analyzer: DecisionAnalyzer, faq_path: Path, audit=None) -> None:
        self._helpdesk = helpdesk
        self._state = state
        self._decisions = decisions
        self._languages = languages
        self._analyzer = analyzer
        self._faq_text = faq_path.read_text(encoding="utf-8")
        self._audit = audit

    def decide(self) -> DecisionSummary:
        fetched = decided = reused = 0
        for raw_ticket in self._helpdesk.fetch_open_tickets():
            fetched += 1
            # Only this short-lived value contains customer text before redaction.
            safe_ticket = redact_ticket(raw_ticket)
            self._state.upsert_fetched(safe_ticket)
            existing = self._decisions.get_current(safe_ticket)
            if existing is not None and existing["status"] in _REUSABLE_DECISION_STATUSES:
                # A matching saved decision is safe to reuse; do not spend
                # another model call just to recreate the same ticket action.
                reused += 1
                continue
            # Empty email metadata has no customer language worth sending to a model.
            if _is_empty_ticket(safe_ticket):
                self._save_decision(safe_ticket, _rule_close(
                    "Empty ticket with no customer request.", "empty-ticket-rule"
                ))
                decided += 1
                continue
            language = self._language_for(safe_ticket)
            if language.language == "unknown":
                self._save_decision(safe_ticket, _rule_close(
                    "No supported customer language was identified; ticket marked as spam.",
                    "unknown-language-rule",
                ))
                decided += 1
                continue
            # If the language is known, run the ./ai/decision.py model and save the result.
            result = self._analyzer.decide(safe_ticket, language.language, self._faq_text)
            result = replace(result, decision=replace(result.decision, language=language.language))
            result = validate_decision(result, safe_ticket, language, self._languages)
            self._save_decision(safe_ticket, result)
            decided += 1
        return DecisionSummary(fetched, decided, reused)

    def _save_decision(self, ticket: Ticket, result: DecisionResult) -> None:
        self._decisions.save_decided(ticket, result, result.decision.language)
        self._event("decision_saved", ticket_id=ticket.id, action=result.decision.action,
                    category=result.decision.category, priority=result.decision.priority,
                    provider=result.provider, model=result.model,
                    input_tokens=result.input_tokens, output_tokens=result.output_tokens)

    def _language_for(self, ticket: Ticket) -> LanguageAssessment:
        row = self._state.get(ticket.id)
        if row is not None and row["status"] in {"analyzed", "language_unresolved"} and row["language"]:
            return LanguageAssessment(row["language"], row["confidence"], row["language_source"], bool(row["fallback_used"]))
        result = self._languages.analyze(f"{ticket.subject}\n{ticket.body}")
        self._state.mark_analyzed(ticket.id, result)
        return result

    def send(self) -> int:
        sent = 0
        log = self._audit or RunLog("run")
        for row in self._decisions.pending():
            decision = self._decisions.read_decision(row)
            result = DecisionResult(decision, row["provider"], row["model"], bool(row["fallback_used"]), row["input_tokens"], row["output_tokens"])
            if row["status"] == "sending":
                try:
                    already_handled = self._helpdesk.ticket_is_handled(row["ticket_id"])
                except HelpdeskError as error:
                    if error.status == 404:
                        self._mark_send_failed(row["ticket_id"], result, log, error)
                        continue
                    raise
                if already_handled:
                    self._decisions.mark(row["ticket_id"], "already_handled")
                    log.write(row["ticket_id"], result, "already_handled")
                    continue
            # Commit before POST. A crash can then recover this exact decision
            # without asking a model to create a different customer response.
            self._decisions.mark(row["ticket_id"], "sending")
            self._event("send_started", ticket_id=row["ticket_id"], action=decision.action,
                        category=decision.category, priority=decision.priority)
            try:
                outcome = self._helpdesk.send_decision(row["ticket_id"], decision)
            except HelpdeskError as error:
                if _is_ticket_request_error(error):
                    self._mark_send_failed(row["ticket_id"], result, log, error)
                    continue
                # An uncertain failure stays `sending` so recovery checks the
                # Helpdesk before it ever retries the stored customer action.
                log.write(row["ticket_id"], result, "sending", type(error).__name__)
                raise
            status = "already_handled" if outcome == "already_handled" else "sent"
            self._decisions.mark(row["ticket_id"], status)
            log.write(row["ticket_id"], result, status)
            sent += int(status == "sent")
        return sent

    def _mark_send_failed(self, ticket_id: str, result: DecisionResult, log: RunLog, error: HelpdeskError) -> None:
        self._decisions.mark(ticket_id, "send_failed")
        log.write(
            ticket_id, result, "send_failed", f"HTTP {error.status}",
            code=error.code, problem_fields=error.problem_fields,
        )

    def _event(self, name: str, **details) -> None:
        if self._audit:
            self._audit.event(name, **details)


def _is_empty_ticket(ticket: Ticket) -> bool:
    subject = ticket.subject.strip()
    return not ticket.body.strip() and (
        not subject or _REPLY_ONLY_SUBJECT.fullmatch(subject) is not None
    )


def _is_ticket_request_error(error: HelpdeskError) -> bool:
    if error.status == 404 or error.status == 422:
        return True
    return error.status == 400 and error.code in {"invalid_json", "bad_request"}


def _rule_close(reason: str, model: str) -> DecisionResult:
    # A close still needs a supported language, even when the ticket has none.
    decision = Decision(
        language=_EMPTY_TICKET_LANGUAGE,
        reason=reason,
        category="other",
        priority="P4",
        action="close",
        faq_ids=(),
        draft_reply=None,
        handoff_note=None,
        confidence=1.0,
    )
    return DecisionResult(decision, "rule", model, False, None, None)
