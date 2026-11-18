import csv
import tempfile
import unittest
from pathlib import Path

from nordavo.ai.language import LanguageAnalyzer, LanguageProviderError
from nordavo.be.state import StateStore
from nordavo.be.report import LanguageCsvReport
from nordavo.models import LanguageAssessment, Ticket
from nordavo.workflow import TicketAnalysisWorkflow


class FakeHelpdesk:
    def fetch_examples(self):
        return [
            {
                "subject": "Hallo",
                "body": "vraag NL91ABNA0417164300",
                "label": {"language": "nl"},
            },
            {"subject": "Hello", "body": "question", "label": {"language": "en"}},
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
            store = StateStore(Path(directory) / "state.sqlite3")
            detector = CapturingGeminiClassifier()
            workflow = TicketAnalysisWorkflow(
                FakeHelpdesk(), store, LanguageAnalyzer(detector, NoFallback())
            )
            summary = workflow.run()
            first = store.get("T1")
            store.close()
        self.assertEqual(2, summary.fetched_tickets)
        self.assertEqual(2, summary.example_matches)
        self.assertEqual("analyzed", first["status"])
        self.assertIsNone(first["action"])
        self.assertNotIn("NL91ABNA0417164300", "\n".join(detector.inputs))

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
        self.assertEqual(2, summary.groq_analyses)
        self.assertEqual(0, summary.fallback_analyses)

    def test_sensitive_ticket_text_is_redacted_before_gemini(self):
        gemini = CapturingGeminiClassifierForSensitiveTicket()
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            report_path = Path(directory) / "language_results.csv"
            TicketAnalysisWorkflow(
                SensitiveTicketHelpdesk(),
                store,
                LanguageAnalyzer(gemini, NoFallback()),
                LanguageCsvReport(report_path),
            ).run()
            row = store.get("T-sensitive")
            with report_path.open(encoding="utf-8", newline="") as stream:
                report_row = next(csv.DictReader(stream))
            store.close()
        self.assertNotIn("4111 1111 1111 1111", gemini.inputs[0])
        self.assertIn("<REDACTED_CARD>", gemini.inputs[0])
        self.assertNotIn("4111 1111 1111 1111", row["body"])
        self.assertIn("4111 1111 1111 1111", report_row["original_body"])

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

    def test_failure_keeps_the_previous_csv_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "language_results.csv"
            path.write_text("previous report", encoding="utf-8")
            store = StateStore(Path(directory) / "state.sqlite3")
            workflow = TicketAnalysisWorkflow(
                TwoTicketHelpdesk(),
                store,
                LanguageAnalyzer(FailsOnSecondGeminiClassifier(), UnavailableGroqClassifier()),
                LanguageCsvReport(path),
            )
            with self.assertRaises(LanguageProviderError):
                workflow.run()
            store.close()
            report = path.read_text(encoding="utf-8")
        self.assertEqual("previous report", report)
