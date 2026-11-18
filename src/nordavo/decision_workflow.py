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
    sent_tickets: int = 0
    example_metrics: dict[str, dict[str, int]] | None = None


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
        self._languages.health_check()
        self._analyzer.health_check()
        example_metrics = self._evaluate_examples()
        self._event("examples_evaluated", stage="decision", outcome="completed")
        fetched = decided = 0
        for raw_ticket in self._helpdesk.fetch_open_tickets():
            fetched += 1
            # Only this short-lived value contains customer text before redaction.
            safe_ticket = redact_ticket(raw_ticket)
            self._state.upsert_fetched(safe_ticket)
            language = self._language_for(safe_ticket)
            if language.language == "unknown":
                raise DecisionProviderError("language", "UnknownTicketLanguage")
            existing = self._decisions.get_current(safe_ticket)
            if existing is not None and existing["status"] in {"decided", "sending", "sent", "already_handled"}:
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
        return DecisionSummary(fetched, decided, example_metrics=example_metrics)

    def _evaluate_examples(self) -> dict[str, dict[str, int]]:
        fields = ("language", "action", "category", "priority")
        matches = {field: 0 for field in fields}
        total = 0
        for example in self._helpdesk.fetch_examples():
            label = example.get("label", {})
            if not isinstance(label, dict):
                continue
            raw_ticket = Ticket.from_api(example)
            ticket = redact_ticket(raw_ticket)
            language = self._languages.analyze(f"{ticket.subject}\n{ticket.body}")
            result = self._analyzer.decide(ticket, language.language, self._faq_text)
            result = replace(result, decision=replace(result.decision, language=language.language))
            decision = validate_decision(result, ticket, language, self._languages).decision
            total += 1
            predicted = {
                "language": language.language, "action": decision.action,
                "category": decision.category, "priority": decision.priority,
            }
            for field in fields:
                matches[field] += int(predicted[field] == label.get(field))
        return {field: {"matches": matches[field], "total": total} for field in fields}

    def _language_for(self, ticket: Ticket) -> LanguageAssessment:
        row = self._state.get(ticket.id)
        if row is not None and row["status"] == "analyzed" and row["language"]:
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
