import sqlite3
import tempfile
import unittest
from pathlib import Path

from nordavo.ai.language import LanguageAnalyzer, LanguageProviderError
from nordavo.be.state import StateStore
from nordavo.models import LanguageAssessment, Ticket
from nordavo.workflow import TicketAnalysisWorkflow


class FakeHelpdesk:
    def fetch_examples(self):
        return [
            {
                "id": "E1",
                "subject": "Hallo",
                "body": "vraag NL91ABNA0417164300",
                "label": {"language": "nl", "private_note": "labels must stay unused"},
            },
            {"id": "E2", "subject": "Hello", "body": "question", "label": {"language": "en"}},
        ]

    def fetch_open_tickets(self):
        yield Ticket("T1", "now", "email", None, "Hallo", "vraag")
        yield Ticket("T2", "now", "email", None, "Hello", "question")


class GeminiClassifier:
    def health_check(self):
        pass

    def classify(self, text):
        language = "en" if "Hello" in text else "nl"
        return LanguageAssessment(language, 0.99, "gemini", False)


class GroqPrimaryClassifier:
    def health_check(self):
        pass

    def classify(self, text):
        return LanguageAssessment("nl", 0.99, "groq", False)


class NoFallback:
    def health_check(self):
        pass

    def classify(self, text):
        raise AssertionError("fallback should not be called")


class FailingHealthGeminiClassifier(GeminiClassifier):
    def health_check(self):
        raise AssertionError("workflow must not run health checks")


class CapturingGeminiClassifier(GeminiClassifier):
    def __init__(self):
        self.inputs = []

    def classify(self, text):
        self.inputs.append(text)
        return super().classify(text)


class CapturingGeminiClassifierForSensitiveTicket:
    def __init__(self):
        self.inputs = []

    def health_check(self):
        pass

    def classify(self, text):
        self.inputs.append(text)
        return LanguageAssessment("en", 0.90, "gemini", False)


class UnavailableGeminiClassifier:
    def health_check(self):
        pass

    def classify(self, text):
        raise LanguageProviderError("gemini", "unavailable")


class CapturingGroqClassifier:
    def __init__(self):
        self.inputs = []

    def health_check(self):
        pass

    def classify(self, text):
        self.inputs.append(text)
        return LanguageAssessment("en", 0.90, "groq", True)


class SensitiveTicketHelpdesk:
    def fetch_examples(self):
        return []

    def fetch_open_tickets(self):
        yield Ticket(
            "T-sensitive",
            "now",
            "email",
            None,
            "Payment question",
            "My card is 4111 1111 1111 1111",
        )


class TwoTicketHelpdesk:
    def fetch_examples(self):
        return []

    def fetch_open_tickets(self):
        yield Ticket("T-first", "now", "email", None, "First", "short text")
        yield Ticket("T-second", "now", "email", None, "Second", "short text")


class FailsOnSecondGeminiClassifier:
    def __init__(self):
        self.calls = 0

    def health_check(self):
        pass

    def classify(self, text):
        self.calls += 1
        if self.calls == 2:
            raise LanguageProviderError("gemini", "unavailable")
        return LanguageAssessment("en", 0.90, "gemini", False)


class UnavailableGroqClassifier:
    def health_check(self):
        pass

    def classify(self, text):
        raise LanguageProviderError("groq", "unavailable")


class WorkflowTests(unittest.TestCase):
    def test_analyzes_examples_and_tickets_without_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.sqlite3"
            store = StateStore(db_path)
            detector = CapturingGeminiClassifier()
            workflow = TicketAnalysisWorkflow(
                FakeHelpdesk(), store, LanguageAnalyzer(detector, NoFallback())
            )
            summary = workflow.run()
            first = store.get("T1")
            store.close()
            connection = sqlite3.connect(db_path)
            example = connection.execute(
                "SELECT original_body FROM language_phase_1_examples WHERE ticket_id='E1'"
            ).fetchone()
            live_ticket = connection.execute(
                "SELECT language FROM language_phase_1_testing WHERE ticket_id='T1'"
            ).fetchone()
            connection.close()
        self.assertEqual(2, summary.labelled_examples_analyzed)
        self.assertEqual(2, summary.open_tickets_analyzed)
        self.assertEqual("analyzed", first["status"])
        self.assertIsNone(first["action"])
        self.assertEqual("vraag <REDACTED_IBAN>", example[0])
        self.assertEqual("nl", live_ticket[0])
        self.assertNotIn("NL91ABNA0417164300", "\n".join(detector.inputs))
        self.assertNotIn("private_note", "\n".join(detector.inputs))

    def test_summary_counts_the_provider_that_actually_classified_tickets(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            workflow = TicketAnalysisWorkflow(
                FakeHelpdesk(),
                store,
                LanguageAnalyzer(GroqPrimaryClassifier(), NoFallback()),
            )
            summary = workflow.run()
            store.close()

        self.assertEqual(0, summary.gemini_analyses)
        self.assertEqual(4, summary.groq_analyses)
        self.assertEqual(0, summary.fallback_analyses)

    def test_analyze_does_not_run_a_provider_health_check(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            summary = TicketAnalysisWorkflow(
                FakeHelpdesk(),
                store,
                LanguageAnalyzer(FailingHealthGeminiClassifier(), NoFallback()),
            ).run()
            store.close()
        self.assertEqual(2, summary.open_tickets_analyzed)

    def test_sensitive_ticket_text_is_redacted_before_gemini(self):
        gemini = CapturingGeminiClassifierForSensitiveTicket()
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.sqlite3"
            store = StateStore(db_path)
            TicketAnalysisWorkflow(
                SensitiveTicketHelpdesk(),
                store,
                LanguageAnalyzer(gemini, NoFallback()),
            ).run()
            row = store.get("T-sensitive")
            store.close()
            connection = sqlite3.connect(db_path)
            review_row = connection.execute(
                "SELECT original_body FROM language_phase_1_testing WHERE ticket_id='T-sensitive'"
            ).fetchone()
            connection.close()
        self.assertNotIn("4111 1111 1111 1111", gemini.inputs[0])
        self.assertIn("<REDACTED_CARD>", gemini.inputs[0])
        self.assertNotIn("4111 1111 1111 1111", row["body"])
        self.assertEqual("My card is <REDACTED_CARD>", review_row[0])

    def test_sensitive_ticket_text_is_redacted_before_groq_fallback(self):
        groq = CapturingGroqClassifier()
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            TicketAnalysisWorkflow(
                SensitiveTicketHelpdesk(),
                store,
                LanguageAnalyzer(UnavailableGeminiClassifier(), groq),
            ).run()
            store.close()
        self.assertNotIn("4111 1111 1111 1111", groq.inputs[0])
        self.assertIn("<REDACTED_CARD>", groq.inputs[0])

    def test_prior_analysis_survives_when_both_providers_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            workflow = TicketAnalysisWorkflow(
                TwoTicketHelpdesk(),
                store,
                LanguageAnalyzer(FailsOnSecondGeminiClassifier(), UnavailableGroqClassifier()),
            )
            with self.assertRaises(LanguageProviderError):
                workflow.run()
            first = store.get("T-first")
            second = store.get("T-second")
            store.close()
        self.assertEqual("analyzed", first["status"])
        self.assertEqual("fetched", second["status"])

    def test_analysis_does_not_change_an_existing_csv_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "language_results.csv"
            path.write_text("previous report", encoding="utf-8")
            store = StateStore(Path(directory) / "state.sqlite3")
            workflow = TicketAnalysisWorkflow(
                FakeHelpdesk(), store, LanguageAnalyzer(GeminiClassifier(), NoFallback())
            )
            workflow.run()
            store.close()
            report = path.read_text(encoding="utf-8")
        self.assertEqual("previous report", report)
