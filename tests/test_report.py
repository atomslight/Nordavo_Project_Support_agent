import csv
import tempfile
import unittest
from pathlib import Path

from nordavo.be.report import LanguageCsvReport
from nordavo.models import LanguageAssessment, Ticket


class LanguageCsvReportTests(unittest.TestCase):
    def test_writes_raw_ticket_text_and_neutralizes_spreadsheet_formulas(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output" / "language_results.csv"
            LanguageCsvReport(path).write(
                [
                    (
                        Ticket(
                            "T1",
                            "now",
                            "email",
                            None,
                            "  =important, subject",
                            "line one\nMy card is 4111 1111 1111 1111",
                        ),
                        LanguageAssessment("nl", 0.95, "gemini", True),
                    )
                ]
            )
            with path.open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
        self.assertEqual("'  =important, subject", row["original_subject"])
        self.assertEqual("line one\nMy card is 4111 1111 1111 1111", row["original_body"])
        self.assertEqual("nl", row["language"])
        self.assertEqual("true", row["fallback_used"])

