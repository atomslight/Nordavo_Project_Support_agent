import tempfile
import unittest
import sqlite3
from pathlib import Path

from nordavo.be.state import StateStore
from nordavo.models import LanguageAssessment, Ticket


class StateStoreTests(unittest.TestCase):
    def test_only_redacted_ticket_text_is_stored(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            ticket = Ticket("T1", "now", "email", None, "Subject", "<REDACTED_CARD>")
            store.upsert_fetched(ticket)
            store.mark_analyzed("T1", LanguageAssessment("nl", 0.95, "gemini", False))
            row = store.get("T1")
            store.close()
        self.assertEqual("<REDACTED_CARD>", row["body"])
        self.assertEqual("analyzed", row["status"])
        self.assertIsNone(row["action"])

    def test_same_ticket_can_be_refetched_without_losing_its_prior_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            ticket = Ticket("T1", "now", "email", None, "Subject", "redacted body")
            store.upsert_fetched(ticket)
            store.mark_analyzed("T1", LanguageAssessment("nl", 0.95, "gemini", False))
            store.upsert_fetched(ticket)
            row = store.get("T1")
            store.close()
        self.assertEqual("analyzed", row["status"])
        self.assertEqual("nl", row["language"])

    def test_phase_one_review_tables_store_the_requested_redacted_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = StateStore(path)
            example = Ticket("E1", "now", "email", "Plus", "Subject", "<REDACTED_CARD>")
            testing = Ticket("T1", "now", "form", None, "Vraag", "Body")
            result = LanguageAssessment("nl", 0.95, "gemini", False)
            store.save_example_analysis(example, result)
            store.save_testing_analysis(testing, result)
            store.close()
            connection = sqlite3.connect(path)
            example_row = connection.execute(
                "SELECT ticket_id, original_body, language FROM language_phase_1_examples"
            ).fetchone()
            testing_row = connection.execute(
                "SELECT ticket_id, channel, fallback_used FROM language_phase_1_testing"
            ).fetchone()
            example_columns = [
                row[1] for row in connection.execute("PRAGMA table_info(language_phase_1_examples)")
            ]
            testing_columns = [
                row[1] for row in connection.execute("PRAGMA table_info(language_phase_1_testing)")
            ]
            connection.close()
        self.assertEqual(("E1", "<REDACTED_CARD>", "nl"), example_row)
        self.assertEqual(("T1", "form", 0), testing_row)
        expected_columns = [
            "ticket_id", "received_at", "channel", "customer_plan", "original_subject",
            "original_body", "language", "confidence", "source", "fallback_used",
        ]
        self.assertEqual(expected_columns, example_columns)
        self.assertEqual(expected_columns, testing_columns)
