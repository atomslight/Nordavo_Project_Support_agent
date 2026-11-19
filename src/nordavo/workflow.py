from __future__ import annotations

import logging
from dataclasses import dataclass
from nordavo.ai.language import LanguageAnalyzer
from nordavo.ai.safety import redact_ticket
from nordavo.be.state import StateStore
from nordavo.models import LanguageAssessment, Ticket

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisSummary:
    labelled_examples_analyzed: int
    open_tickets_analyzed: int
    gemini_analyses: int
    groq_analyses: int
    fallback_analyses: int


class TicketAnalysisWorkflow:
    def __init__(
        self,
        helpdesk,
        state: StateStore,
        analyzer: LanguageAnalyzer,
        audit=None,
    ) -> None:
        self._helpdesk = helpdesk
        self._state = state
        self._analyzer = analyzer
        self._audit = audit

    def run(self) -> AnalysisSummary:
        example_results = self._analyze_examples()
        ticket_results = self._analyze_live_tickets()
        all_results = example_results + ticket_results

        return AnalysisSummary(
            labelled_examples_analyzed=len(example_results),
            open_tickets_analyzed=len(ticket_results),
            gemini_analyses=sum(result.source == "gemini" for result in all_results),
            groq_analyses=sum(result.source == "groq" for result in all_results),
            fallback_analyses=sum(result.fallback_used for result in all_results),
        )

    def _analyze_examples(self) -> list[LanguageAssessment]:
        results: list[LanguageAssessment] = []
        for payload in self._helpdesk.fetch_examples():
            # The API label is intentionally ignored: this command predicts,
            # but does not score or retain, example answers.
            raw_ticket = Ticket.from_api(payload)
            safe_ticket = redact_ticket(raw_ticket)
            result = self._analyzer.analyze(f"{safe_ticket.subject}\n{safe_ticket.body}")
            self._state.save_example_analysis(safe_ticket, result)
            self._event("ticket_processed", ticket_id=safe_ticket.id, provider=result.source, outcome="analyzed")
            results.append(result)
        return results

    def _analyze_live_tickets(self) -> list[LanguageAssessment]:
        results: list[LanguageAssessment] = []
        for raw_ticket in self._helpdesk.fetch_open_tickets():
            logger.debug("Analyzing ticket_id=%s", raw_ticket.id)
            # Redaction happens before every database and model boundary.
            safe_ticket = redact_ticket(raw_ticket)
            self._state.upsert_fetched(safe_ticket)
            language_result = self._analyzer.analyze(
                f"{safe_ticket.subject}\n{safe_ticket.body}"
            )
            self._state.mark_analyzed(safe_ticket.id, language_result)
            self._state.save_testing_analysis(safe_ticket, language_result)
            self._event("ticket_processed", ticket_id=safe_ticket.id, provider=language_result.source,
                        outcome="analyzed")
            results.append(language_result)
        return results

    def _event(self, name: str, **details) -> None:
        if self._audit:
            self._audit.event(name, **details)
