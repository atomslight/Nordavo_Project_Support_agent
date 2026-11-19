"""Decision-only and explicit-send workflows layered on Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from nordavo.ai.decision import DecisionAnalyzer, DecisionProviderError, validate_decision
from nordavo.ai.safety import redact_ticket
from nordavo.be.decision_state import DecisionStateStore
from nordavo.be.helpdesk import HelpdeskError
from nordavo.be.run_log import RunLog
from nordavo.be.state import StateStore
from nordavo.models import DecisionResult, LanguageAssessment, Ticket


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
        unresolved_ticket_ids: list[str] = []
        for raw_ticket in self._helpdesk.fetch_open_tickets():
            fetched += 1
            # Only this short-lived value contains customer text before redaction.
            safe_ticket = redact_ticket(raw_ticket)
            self._state.upsert_fetched(safe_ticket)
            language = self._language_for(safe_ticket)
            if language.language == "unknown":
                self._state.mark_language_unresolved(safe_ticket.id)
                unresolved_ticket_ids.append(safe_ticket.id)
                self._event(
                    "language_unresolved", ticket_id=safe_ticket.id,
                    provider=language.source, confidence=language.confidence,
                    outcome="skipped",
                )
                continue
            existing = self._decisions.get_current(safe_ticket)
            if existing is not None and existing["status"] in {"decided", "sending", "sent", "already_handled"}:
                # A matching saved decision is safe to reuse; do not spend
                # another model call just to recreate the same ticket action.
                reused += 1
                continue
            result = self._analyzer.decide(safe_ticket, language.language, self._faq_text)
            result = replace(result, decision=replace(result.decision, language=language.language))
            result = validate_decision(result, safe_ticket, language, self._languages)
            self._decisions.save_decided(safe_ticket, result, language)
            self._event("decision_saved", ticket_id=safe_ticket.id, action=result.decision.action,
                        category=result.decision.category, priority=result.decision.priority,
                        provider=result.provider, model=result.model,
                        input_tokens=result.input_tokens, output_tokens=result.output_tokens)
            decided += 1
        return DecisionSummary(
            fetched, decided, reused,
            language_unresolved_tickets=len(unresolved_ticket_ids),
            unresolved_ticket_ids=tuple(unresolved_ticket_ids),
        )

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
            if row["status"] == "sending" and self._helpdesk.ticket_is_handled(row["ticket_id"]):
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
                # 400/422 are definite validation failures. Other failures may
                # have reached the server, so keep `sending` for GET recovery.
                if error.status in {400, 422}:
                    self._decisions.mark(row["ticket_id"], "send_failed")
                    log.write(row["ticket_id"], result, "send_failed", f"HTTP {error.status}")
                    continue
                log.write(row["ticket_id"], result, "sending", type(error).__name__)
                raise
            status = "already_handled" if outcome == "already_handled" else "sent"
            self._decisions.mark(row["ticket_id"], status)
            log.write(row["ticket_id"], result, status)
            sent += int(status == "sent")
        return sent

    def _event(self, name: str, **details) -> None:
        if self._audit:
            self._audit.event(name, **details)
