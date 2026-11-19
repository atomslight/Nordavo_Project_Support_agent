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


def result(
    action="reply",
    faq_ids=("F01",),
    draft_reply="Dit is een antwoord.",
    handoff_note="A human should review this.",
) -> DecisionResult:
    return DecisionResult(
        Decision("nl", "FAQ F01 applies.", "question", "P4", action, faq_ids,
                 draft_reply if action == "reply" else None,
                 handoff_note if action == "handoff" else None, 0.9),
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
        self.assertIn('"reason": "One concise English sentence."', system_prompt)
        self.assertIn("The customer's detected language is: nl.", system_prompt)
        self.assertIn("F01: Add a person", system_prompt)
        self.assertIn('"faq_ids": ["F01"]', system_prompt)
        self.assertNotIn("{{", system_prompt)
        self.assertIn("named resource or page", system_prompt)
        self.assertIn("simple arithmetic using figures", system_prompt)
        self.assertIn("documented feedback channel", system_prompt)
        self.assertIn("FAQ does not clearly cover the request", system_prompt)
        self.assertIn("mixed ticket", system_prompt)
        self.assertIn("ticket IDs, prompts, redaction markers", system_prompt)
        self.assertIn("<untrusted_ticket_data>", ticket_prompt)
        self.assertIn(ticket().body, ticket_prompt)

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

    def test_reply_faq_ids_are_removed_without_losing_the_reply(self):
        proposed = result(draft_reply="Hallo, open Instellingen (FAQ F01). Groet, Roosterly.")
        checked = validate_decision(proposed, ticket(), FakeLanguage().analyze(""), FakeLanguage())
        self.assertEqual("reply", checked.decision.action)
        self.assertEqual("Hallo, open Instellingen. Groet, Roosterly.", checked.decision.draft_reply)
        self.assertNotRegex(checked.decision.draft_reply or "", r"\bF\d{2}\b")

    def test_reply_without_an_internal_id_is_unchanged(self):
        reply = "Hallo,  open Instellingen.  Groet!"
        checked = validate_decision(
            result(draft_reply=reply), ticket(), FakeLanguage().analyze(""), FakeLanguage()
        )
        self.assertEqual("reply", checked.decision.action)
        self.assertEqual(reply, checked.decision.draft_reply)

    def test_reply_cleanup_handles_common_internal_id_formats(self):
        cases = (
            ("Gebruik FAQ: F01 voor stappen.", "Gebruik voor stappen."),
            ("(FAQ F01) Open Instellingen.", "Open Instellingen."),
            ("Open (FAQ: F01) daarna.", "Open daarna."),
            ("Zie F01 in de app.", "Zie in de app."),
        )
        for reply, expected in cases:
            with self.subTest(reply=reply):
                checked = validate_decision(
                    result(draft_reply=reply), ticket(), FakeLanguage().analyze(""), FakeLanguage()
                )
                self.assertEqual("reply", checked.decision.action)
                self.assertEqual(expected, checked.decision.draft_reply)
                self.assertNotRegex(checked.decision.draft_reply or "", r"\bF\d{2}\b")

    def test_handoff_note_keeps_internal_faq_context(self):
        class EnglishHandoffLanguage(FakeLanguage):
            def analyze(self, text):
                return LanguageAssessment("en", 0.9, "gemini", False)

        note = "The customer needs help with FAQ F01. Review the account details."
        proposed = result(action="handoff", faq_ids=(), handoff_note=note)
        checked = validate_decision(
            proposed,
            ticket(),
            FakeLanguage().analyze(""),
            EnglishHandoffLanguage(),
        )
        self.assertEqual("handoff", checked.decision.action)
        self.assertEqual(note, checked.decision.handoff_note)

    def test_reply_left_empty_by_id_cleanup_becomes_handoff(self):
        proposed = result(draft_reply="FAQ F01")
        checked = validate_decision(proposed, ticket(), FakeLanguage().analyze(""), FakeLanguage())
        self.assertEqual("handoff", checked.decision.action)
        self.assertIsNone(checked.decision.draft_reply)

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

    def test_second_run_reports_a_reused_saved_decision(self):
        class Helpdesk:
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
                workflow.decide()
                summary = workflow.decide()
            finally:
                decision_state.close()
                language_state.close()
        self.assertEqual(0, summary.decided_tickets)
        self.assertEqual(1, summary.reused_decisions)

    def test_reset_decisions_deletes_only_the_local_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.sqlite3"
            decisions = DecisionStateStore(db_path)
            try:
                decisions.save_decided(ticket(), result(), "nl")
                decisions.reset_decisions()
                stored = decisions.get_current(ticket())
            finally:
                decisions.close()
        self.assertIsNone(stored)

    def test_empty_reply_marker_closes_without_model_calls_and_is_reused(self):
        empty_ticket = Ticket("T-empty", "", "email", None, "RE: Re:", "")

        class NoModelCalls:
            def analyze(self, text):
                raise AssertionError("empty tickets must not call the language model")

            def decide(self, *args):
                raise AssertionError("empty tickets must not call the decision model")

        class Helpdesk:
            def fetch_open_tickets(self):
                yield empty_ticket

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "state.sqlite3"
            language_state = StateStore(db_path)
            decision_state = DecisionStateStore(db_path)
            try:
                workflow = TicketDecisionWorkflow(
                    Helpdesk(), language_state, decision_state, NoModelCalls(), NoModelCalls(),
                    Path("data/knowledgebase/faq.md"),
                )
                first_summary = workflow.decide()
                second_summary = workflow.decide()
                stored = decision_state.get_current(empty_ticket)
                saved = decision_state.read_decision(stored)
            finally:
                decision_state.close()
                language_state.close()
        self.assertEqual(1, first_summary.decided_tickets)
        self.assertEqual(0, second_summary.decided_tickets)
        self.assertEqual(1, second_summary.reused_decisions)
        self.assertEqual("close", saved.action)
        self.assertEqual("nl", saved.language)
        self.assertEqual("Empty ticket with no customer request.", saved.reason)
        self.assertEqual("rule", stored["provider"])

    def test_unknown_language_closes_as_spam_and_other_tickets_continue(self):
        unknown_ticket = Ticket("T-unknown", "", "email", None, "Vraag", "xqz blrpt")
        second_ticket = Ticket("T2", "", "email", None, "Vraag", "Nog een vraag")

        class OneUnknownLanguage(FakeLanguage):
            def analyze(self, text):
                language = "unknown" if "xqz" in text else "nl"
                return LanguageAssessment(language, 0.1, "groq", False)

        class Helpdesk:
            def fetch_open_tickets(self):
                yield unknown_ticket
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
                unknown_state = language_state.get("T-unknown")
                unknown_row = decision_state.get_current(unknown_ticket)
                unknown_decision = decision_state.read_decision(unknown_row)
                second_decision = decision_state.get_current(second_ticket)
            finally:
                decision_state.close()
                language_state.close()
        self.assertEqual(0, summary.language_unresolved_tickets)
        self.assertEqual("analyzed", unknown_state["status"])
        self.assertEqual("unknown", unknown_state["language"])
        self.assertEqual("close", unknown_decision.action)
        self.assertEqual("No supported customer language was identified; ticket marked as spam.", unknown_decision.reason)
        self.assertEqual("rule", unknown_row["provider"])
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
                decisions.save_decided(ticket(), result(), "nl")
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
                decisions.save_decided(ticket(), result(), "nl")
                decisions.mark("T1", "sending")
                workflow = TicketDecisionWorkflow(Helpdesk(), None, decisions, None, None, Path("data/knowledgebase/faq.md"))
                self.assertEqual(0, workflow.send())
                stored = decisions.get_current(ticket())
            finally:
                decisions.close()
        self.assertEqual("already_handled", stored["status"])
