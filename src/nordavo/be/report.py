from __future__ import annotations

import csv
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from nordavo.models import LanguageAssessment, Ticket


class LanguageCsvReport:
    """Writes the explicitly requested raw-ticket review report."""

    _FIELDS = (
        "ticket_id",
        "received_at",
        "channel",
        "customer_plan",
        "original_subject",
        "original_body",
        "language",
        "confidence",
        "source",
        "fallback_used",
    )

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, rows: list[tuple[Ticket, LanguageAssessment]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=self._FIELDS)
                writer.writeheader()
                for ticket, assessment in rows:
                    writer.writerow(_report_row(ticket, assessment))
                stream.flush()
                os.fsync(stream.fileno())
            # Replacing the file at the end keeps the previous report if this run fails.
            temporary_path.replace(self._path)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()


def _report_row(ticket: Ticket, assessment: LanguageAssessment) -> dict[str, str | float]:
    return {
        "ticket_id": _safe_csv_cell(ticket.id),
        "received_at": _safe_csv_cell(ticket.received_at),
        "channel": _safe_csv_cell(ticket.channel),
        "customer_plan": _safe_csv_cell(ticket.customer_plan or ""),
        "original_subject": _safe_csv_cell(ticket.subject),
        "original_body": _safe_csv_cell(ticket.body),
        "language": assessment.language,
        "confidence": assessment.confidence,
        "source": assessment.source,
        "fallback_used": str(assessment.fallback_used).lower(),
    }


def _safe_csv_cell(value: str) -> str:
    """Prevent customer text from becoming a formula when opened in Excel."""
    # A leading apostrophe keeps formula-like customer text as plain text.
    return f"'{value}" if value.lstrip().startswith(("=", "+", "-", "@")) else value
