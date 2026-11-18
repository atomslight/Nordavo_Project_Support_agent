from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from nordavo.ai.language import LanguageAnalyzer
from nordavo.ai.safety import redact_sensitive, redact_ticket
from nordavo.be.report import LanguageCsvReport
from nordavo.be.state import StateStore
from nordavo.models import LanguageAssessment, Ticket

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisSummary:
    fetched_tickets: int
    gemini_analyses: int
    groq_analyses: int
    fallback_analyses: int
    example_total: int
    example_matches: int
    example_by_language: dict[str, dict[str, int]]


class TicketAnalysisWorkflow:
    def __init__(
        self,
        helpdesk,
        state: StateStore,
        analyzer: LanguageAnalyzer,
        report: LanguageCsvReport | None = None,
        audit=None,
    ) -> None:
        self._helpdesk = helpdesk
        self._state = state
        self._analyzer = analyzer
        self._report = report
        self._audit = audit
    #Entry point for the workflow.  This method is called by the CLI and tests.
    def run(self) -> AnalysisSummary:
        self._analyzer.health_check()
        example_total, example_matches, example_by_language = self._evaluate_examples()
        report_rows, gemini_count, groq_count, fallback_count = self._analyze_live_tickets()

        if self._report:
            self._report.write(report_rows)
        self._event("examples_evaluated", stage="language", outcome="completed")

        return AnalysisSummary(
            fetched_tickets=len(report_rows),
            gemini_analyses=gemini_count,
            groq_analyses=groq_count,
            fallback_analyses=fallback_count,
            example_total=example_total,
            example_matches=example_matches,
            example_by_language=example_by_language,
        )

    def _analyze_live_tickets(
        self,
    ) -> tuple[list[tuple[Ticket, LanguageAssessment]], int, int, int]:
        report_rows: list[tuple[Ticket, LanguageAssessment]] = []
        gemini_count = 0
        groq_count = 0
        fallback_count = 0

        for raw_ticket in self._helpdesk.fetch_open_tickets():
            logger.debug("Analyzing ticket_id=%s", raw_ticket.id)
            # Raw text is kept only for the explicitly requested local CSV report.
            # All persistence and Gemini work use this redacted copy instead.
            safe_ticket = redact_ticket(raw_ticket)
            self._state.upsert_fetched(safe_ticket)
            language_result = self._analyzer.analyze(
                f"{safe_ticket.subject}\n{safe_ticket.body}"
            )
            self._state.mark_analyzed(safe_ticket.id, language_result)
            self._event("ticket_processed", ticket_id=safe_ticket.id, provider=language_result.source,
                        outcome="analyzed")
            report_rows.append((raw_ticket, language_result))
            gemini_count += int(language_result.source == "gemini")
            groq_count += int(language_result.source == "groq")
            fallback_count += int(language_result.fallback_used)

        return report_rows, gemini_count, groq_count, fallback_count
    #Evaluate the language classification accuracy on labelled examples from the Helpdesk API.  
    #This is a separate evaluation from the live ticket analysis, which is not labelled and cannot be scored.
    def _evaluate_examples(self) -> tuple[int, int, dict[str, dict[str, int]]]:
        totals: Counter[str] = Counter()
        matches: Counter[str] = Counter()
        for example_number, example in enumerate(self._helpdesk.fetch_examples(), start=1):
            logger.debug("Evaluating labelled example=%s", example_number)
            label = example.get("label", {})
            expected = label.get("language")
            if expected not in {"nl", "en", "de"}:
                continue
            # Examples can contain the same customer data as live tickets.  They
            # are evaluation inputs, not an exception to the redaction boundary.
            text = "\n".join(
                (
                    redact_sensitive(str(example.get("subject", ""))),
                    redact_sensitive(str(example.get("body", ""))),
                )
            )
            language_result = self._analyzer.analyze(text)
            totals[expected] += 1
            if language_result.language == expected:
                matches[expected] += 1
        by_language = {
            language: {"total": totals[language], "matches": matches[language]}
            for language in sorted(totals)
        }
        return sum(totals.values()), sum(matches.values()), by_language

    def _event(self, name: str, **details) -> None:
        if self._audit:
            self._audit.event(name, **details)
