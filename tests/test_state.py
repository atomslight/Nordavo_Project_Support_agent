import tempfile
import unittest
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
