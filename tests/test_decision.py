import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nordavo.ai.decision import (
    DecisionAnalyzer, DecisionProviderError, GeminiDecisionClassifier,
    GroqDecisionClassifier, _prompts, validate_decision,
)
from nordavo.be.decision_state import DecisionStateStore
from nordavo.be.state import StateStore
from nordavo.decision_workflow import TicketDecisionWorkflow
from nordavo.models import Decision, DecisionResult, LanguageAssessment, Ticket


def ticket() -> Ticket:
    return Ticket("T1", "", "email", "Basis", "Vraag", "Hoe voeg ik iemand toe?")


def result(action="reply", faq_ids=("F01",), draft_reply="Dit is een antwoord.") -> DecisionResult:
    return DecisionResult(
        Decision("nl", "FAQ F01 applies.", "question", "P4", action, faq_ids,
                 draft_reply if action == "reply" else None,
                 "A human should review this." if action == "handoff" else None, 0.9),
        "gemini", "model", False, 12, 5,
    )


class FakeLanguage:
    def health_check(self):
        pass

    def analyze(self, text):
        return LanguageAssessment("nl", 0.9, "gemini", False)


class FakeDecision:
    def __init__(self, value):
        self.value = value

    def health_check(self):
        pass

    def decide(self, *args):
        return self.value


class FailingHealthDecision(FakeDecision):
    def health_check(self):
        raise AssertionError("workflow must not run health checks")


class DecisionTests(unittest.TestCase):
    def test_prompt_renders_literal_json_and_only_expected_placeholders(self):
        system_prompt, ticket_prompt = _prompts(ticket(), "nl", "F01: Add a person")
        self.assertIn('"reason": "One or two concise English sentences."', system_prompt)
        self.assertIn("The customer's detected language is: nl.", system_prompt)
        self.assertIn("F01: Add a person", system_prompt)
        self.assertIn("<untrusted_ticket_data>", ticket_prompt)

    def test_invalid_reply_becomes_deterministic_handoff(self):
        unsafe = result(faq_ids=("F99",))
        checked = validate_decision(unsafe, ticket(), FakeLanguage().analyze(""), FakeLanguage())
        self.assertEqual("handoff", checked.decision.action)
        self.assertIsNone(checked.decision.draft_reply)
        self.assertTrue(checked.decision.handoff_note)

    def test_non_english_handoff_note_becomes_the_safe_handoff(self):
        proposed = result(action="handoff", faq_ids=(), draft_reply=None)
        checked = validate_decision(proposed, ticket(), FakeLanguage().analyze(""), FakeLanguage())
        self.assertEqual("handoff", checked.decision.action)
        self.assertIn("Human review", checked.decision.reason)

    def test_decision_analyzer_uses_fallback_once(self):
        class Broken:
            def decide(self, *args):
                raise DecisionProviderError("gemini", "TimeoutError")

        analyzer = DecisionAnalyzer(Broken(), FakeDecision(result()))
        actual = analyzer.decide(ticket(), "nl", "FAQ")
        self.assertEqual("gemini", actual.provider)
        self.assertTrue(actual.fallback_used)

    def test_fallback_log_never_contains_customer_text(self):
        class Broken:
            def decide(self, *args):
                raise DecisionProviderError("gemini", "BadRequestError", status=400, code="bad_request")

        with self.assertLogs("nordavo.ai.decision", level="WARNING") as logs:
            DecisionAnalyzer(Broken(), FakeDecision(result())).decide(ticket(), "nl", "FAQ")
        self.assertIn("provider=gemini", logs.output[0])
        self.assertNotIn(ticket().body, "\n".join(logs.output))

    def test_provider_errors_do_not_expose_customer_text(self):
        secret = "customer says 4111 1111 1111 1111"

        class ApiError(Exception):
            status_code = 400

        gemini_client = SimpleNamespace(
            interactions=SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(ApiError(secret)))
        )
        groq_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(ApiError(secret)))
            )
        )
        for classifier in (GeminiDecisionClassifier(gemini_client, "model"), GroqDecisionClassifier(groq_client, "model")):
            with self.assertRaises(DecisionProviderError) as caught:
                classifier.decide(ticket(), "nl", "FAQ")
            self.assertNotIn(secret, str(caught.exception))

    def test_workflow_persists_a_decision_without_sending(self):
        class Helpdesk:
            def fetch_examples(self):
                raise AssertionError("decide must not fetch labelled examples")

            def fetch_open_tickets(self):
                yield ticket()

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.sqlite3"
            language_state = StateStore(db_path)
            decision_state = DecisionStateStore(db_path)
            try:
                workflow = TicketDecisionWorkflow(
                    Helpdesk(), language_state, decision_state, FakeLanguage(),
                    DecisionAnalyzer(FakeDecision(result()), FakeDecision(result())),
                    Path("data/knowledgebase/faq.md"),
                )
                summary = workflow.decide()
                stored = decision_state.get_current(ticket())
            finally:
                decision_state.close()
                language_state.close()
        self.assertEqual(1, summary.decided_tickets)
        self.assertEqual("decided", stored["status"])

    def test_decide_does_not_run_provider_health_checks(self):
        class Helpdesk:
            def fetch_open_tickets(self):
                yield ticket()

        class FailingHealthLanguage(FakeLanguage):
            def health_check(self):
                raise AssertionError("workflow must not run health checks")

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.sqlite3"
            language_state = StateStore(db_path)
            decision_state = DecisionStateStore(db_path)
            try:
                summary = TicketDecisionWorkflow(
                    Helpdesk(), language_state, decision_state, FailingHealthLanguage(),
                    DecisionAnalyzer(FailingHealthDecision(result()), FailingHealthDecision(result())),
                    Path("data/knowledgebase/faq.md"),
                ).decide()
            finally:
                decision_state.close()
                language_state.close()
        self.assertEqual(1, summary.decided_tickets)

    def test_unknown_language_is_saved_and_other_tickets_continue(self):
        second_ticket = Ticket("T2", "", "email", None, "Vraag", "Nog een vraag")

        class OneUnknownLanguage(FakeLanguage):
            def analyze(self, text):
                language = "unknown" if "Hoe voeg" in text else "nl"
                return LanguageAssessment(language, 0.1, "groq", False)

        class Helpdesk:
            def fetch_examples(self):
                return []

            def fetch_open_tickets(self):
                yield ticket()
                yield second_ticket

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.sqlite3"
            language_state = StateStore(db_path)
            decision_state = DecisionStateStore(db_path)
            try:
                workflow = TicketDecisionWorkflow(
                    Helpdesk(), language_state, decision_state, OneUnknownLanguage(),
                    DecisionAnalyzer(FakeDecision(result()), FakeDecision(result())),
                    Path("data/knowledgebase/faq.md"),
                )
                summary = workflow.decide()
                unresolved = language_state.get("T1")
                second_decision = decision_state.get_current(second_ticket)
            finally:
                decision_state.close()
                language_state.close()
        self.assertEqual(1, summary.language_unresolved_tickets)
        self.assertEqual(("T1",), summary.unresolved_ticket_ids)
        self.assertEqual("language_unresolved", unresolved["status"])
        self.assertEqual("decided", second_decision["status"])

    def test_sender_uses_the_saved_decision_once(self):
        class Helpdesk:
            def ticket_is_handled(self, ticket_id):
                return False

            def send_decision(self, ticket_id, decision):
                self.sent = (ticket_id, decision)
                return "sent"

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.sqlite3"
            decisions = DecisionStateStore(db_path)
            helpdesk = Helpdesk()
            try:
                language = LanguageAssessment("nl", 0.9, "gemini", False)
                decisions.save_decided(ticket(), result(), language)
                workflow = TicketDecisionWorkflow(helpdesk, None, decisions, None, None, Path("data/knowledgebase/faq.md"))
                sent = workflow.send()
                stored = decisions.get_current(ticket())
            finally:
                decisions.close()
        self.assertEqual(1, sent)
        self.assertEqual("T1", helpdesk.sent[0])
        self.assertEqual("sent", stored["status"])

    def test_sending_row_recovers_without_resending_when_ticket_is_handled(self):
        class Helpdesk:
            def ticket_is_handled(self, ticket_id):
                return True

            def send_decision(self, ticket_id, decision):
                raise AssertionError("a handled ticket must not be sent again")

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.sqlite3"
            decisions = DecisionStateStore(db_path)
            try:
                decisions.save_decided(ticket(), result(), LanguageAssessment("nl", 0.9, "gemini", False))
                decisions.mark("T1", "sending")
                workflow = TicketDecisionWorkflow(Helpdesk(), None, decisions, None, None, Path("data/knowledgebase/faq.md"))
                self.assertEqual(0, workflow.send())
                stored = decisions.get_current(ticket())
            finally:
                decisions.close()
        self.assertEqual("already_handled", stored["status"])
