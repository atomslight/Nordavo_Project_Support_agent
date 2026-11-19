import io
import json
import unittest
from urllib.error import HTTPError

from nordavo.be.config import Settings
from nordavo.be.helpdesk import HelpdeskClient, HelpdeskError
from nordavo.models import Decision


class FakeResponse:
    status = 200
    headers = {}

    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def getcode(self):
        return self.status

    def close(self):
        pass


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self):
        return self.now

    def sleep(self, seconds: float):
        self.sleeps.append(seconds)
        self.now += seconds


class HelpdeskTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            helpdesk_base_url="https://example.test",
            helpdesk_api_key="key",
            candidate_email="me@example.test",
            gemini_api_key="gemini",
            gemini_model="gemini-model",
            groq_api_key="groq",
            groq_model="groq-model",
            database_path=__import__("pathlib").Path("ignored.db"),
        )

    def test_paginates_and_retries_transient_get_failures(self):
        responses = iter([
            HTTPError("https://example.test", 503, "down", {}, io.BytesIO()),
            FakeResponse({"tickets": [{"id": "T1"}], "next_cursor": "T1"}),
            FakeResponse({"tickets": [{"id": "T2"}], "next_cursor": None}),
        ])
        clock = FakeClock()

        def opener(*args, **kwargs):
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

        client = HelpdeskClient(
            self.settings, opener=opener, sleeper=clock.sleep, clock=clock.time
        )
        self.assertEqual(["T1", "T2"], [ticket.id for ticket in client.fetch_open_tickets()])
        self.assertEqual(2, len(clock.sleeps))
        self.assertEqual(1.0, clock.sleeps[0])
        self.assertAlmostEqual(0.2, clock.sleeps[1])

    def test_honors_retry_after_for_429_and_never_builds_a_post_request(self):
        responses = iter([
            HTTPError("https://example.test", 429, "busy", {"Retry-After": "3"}, io.BytesIO()),
            FakeResponse([]),
        ])
        requests = []
        clock = FakeClock()

        def opener(request, **kwargs):
            requests.append(request)
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

        client = HelpdeskClient(
            self.settings, opener=opener, sleeper=clock.sleep, clock=clock.time
        )
        self.assertEqual([], client.fetch_examples())
        self.assertEqual([3.0], clock.sleeps)
        self.assertTrue(all(request.get_method() == "GET" for request in requests))

    def test_401_stops_without_retry(self):
        clock = FakeClock()

        def opener(*args, **kwargs):
            raise HTTPError("https://example.test", 401, "unauthorized", {}, io.BytesIO())

        client = HelpdeskClient(
            self.settings, opener=opener, sleeper=clock.sleep, clock=clock.time
        )
        with self.assertRaisesRegex(Exception, "authentication failed"):
            client.fetch_examples()
        self.assertEqual([], clock.sleeps)

    def test_spaces_all_helpdesk_requests_to_five_per_second(self):
        clock = FakeClock()
        requests = []

        def opener(request, **kwargs):
            requests.append(request)
            return FakeResponse([])

        client = HelpdeskClient(
            self.settings, opener=opener, sleeper=clock.sleep, clock=clock.time
        )
        client.fetch_examples()
        client.fetch_examples()

        self.assertEqual(2, len(requests))
        self.assertEqual([0.2], clock.sleeps)

    def test_retry_attempts_use_the_same_rate_limit(self):
        clock = FakeClock()
        responses = iter([
            HTTPError("https://example.test", 429, "busy", {"Retry-After": "0"}, io.BytesIO()),
            FakeResponse([]),
        ])

        def opener(*args, **kwargs):
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

        client = HelpdeskClient(
            self.settings, opener=opener, sleeper=clock.sleep, clock=clock.time
        )
        self.assertEqual([], client.fetch_examples())
        self.assertEqual([0.0, 0.2], clock.sleeps)

    def test_caps_and_safely_parses_retry_after(self):
        for header, expected_delay in (("100", 30.0), ("not-a-number", 1.0)):
            with self.subTest(header=header):
                clock = FakeClock()
                responses = iter([
                    HTTPError(
                        "https://example.test", 429, "busy", {"Retry-After": header}, io.BytesIO()
                    ),
                    FakeResponse([]),
                ])

                def opener(*args, **kwargs):
                    response = next(responses)
                    if isinstance(response, Exception):
                        raise response
                    return response

                client = HelpdeskClient(
                    self.settings, opener=opener, sleeper=clock.sleep, clock=clock.time
                )
                client.fetch_examples()
                self.assertEqual([expected_delay], clock.sleeps)

    def test_non_retryable_errors_are_attempted_once(self):
        cases = (
            (400, {"error": "bad_request"}),
            (401, {"error": "unauthorized"}),
            (404, {"error": "not_found"}),
            (422, {"error": "validation_failed"}),
        )
        for status, payload in cases:
            with self.subTest(status=status):
                calls = []

                def opener(*args, **kwargs):
                    calls.append(1)
                    body = io.BytesIO(json.dumps(payload).encode())
                    raise HTTPError("https://example.test", status, "failure", {}, body)

                with self.assertRaises(HelpdeskError) as caught:
                    HelpdeskClient(self.settings, opener=opener).fetch_examples()
                self.assertEqual(status, caught.exception.status)
                self.assertEqual(1, len(calls))

    def test_422_keeps_only_safe_problem_field_names(self):
        payload = {
            "error": "validation_failed",
            "problems": {
                "draft_reply": "Customer card 4111 1111 1111 1111 is invalid",
                "priority": "P9 is invalid",
                "bad field with customer text": "do not retain this",
            },
        }

        def opener(*args, **kwargs):
            raise HTTPError(
                "https://example.test", 422, "failure", {}, io.BytesIO(json.dumps(payload).encode())
            )

        with self.assertRaises(HelpdeskError) as caught:
            HelpdeskClient(self.settings, opener=opener).fetch_examples()
        self.assertEqual("validation_failed", caught.exception.code)
        self.assertEqual(("draft_reply", "priority"), caught.exception.problem_fields)

    def test_422_accepts_field_entries_without_retaining_messages(self):
        payload = {
            "error": {"code": "validation_failed"},
            "problems": [
                {"field": "draft_reply", "message": "Customer says 4111 1111 1111 1111"},
                {"field": "priority", "message": "Invalid P9"},
            ],
        }

        def opener(*args, **kwargs):
            raise HTTPError(
                "https://example.test", 422, "failure", {}, io.BytesIO(json.dumps(payload).encode())
            )

        with self.assertRaises(HelpdeskError) as caught:
            HelpdeskClient(self.settings, opener=opener).fetch_examples()
        self.assertEqual(("draft_reply", "priority"), caught.exception.problem_fields)
        self.assertNotIn("4111", str(caught.exception))

    def test_409_returns_already_handled_without_retrying(self):
        calls = []

        def opener(*args, **kwargs):
            calls.append(1)
            raise HTTPError("https://example.test", 409, "handled", {}, io.BytesIO())

        decision = Decision("nl", "FAQ F01.", "question", "P4", "reply", ("F01",), "Antwoord", None, 0.9)
        outcome = HelpdeskClient(self.settings, opener=opener).send_decision("T1", decision)
        self.assertEqual("already_handled", outcome)
        self.assertEqual(1, len(calls))

    def test_posts_the_saved_reply_payload(self):
        requests = []

        def opener(request, **kwargs):
            requests.append(request)
            return FakeResponse({"ok": True})

        client = HelpdeskClient(self.settings, opener=opener)
        decision = Decision("nl", "FAQ F01.", "question", "P4", "reply", ("F01",), "Antwoord", None, 0.9)
        self.assertEqual("sent", client.send_decision("T1", decision))
        self.assertEqual("POST", requests[0].get_method())
        self.assertTrue(requests[0].full_url.endswith("/tickets/T1/reply"))
        self.assertIn(b'"draft_reply": "Antwoord"', requests[0].data)
        self.assertNotIn(b'"faq_ids"', requests[0].data)

    def test_posts_the_saved_close_payload_without_customer_text(self):
        requests = []

        def opener(request, **kwargs):
            requests.append(request)
            return FakeResponse({"ok": True})

        client = HelpdeskClient(self.settings, opener=opener)
        decision = Decision("nl", "Empty ticket with no customer request.", "other", "P4", "close", (), None, None, 1.0)
        self.assertEqual("sent", client.send_decision("T1", decision))
        self.assertTrue(requests[0].full_url.endswith("/tickets/T1/close"))
        self.assertIn(b'"language": "nl"', requests[0].data)
        self.assertNotIn(b'"draft_reply"', requests[0].data)
        self.assertNotIn(b'"handoff_note"', requests[0].data)

    def test_handled_field_is_the_only_recovery_signal(self):
        responses = iter([
            FakeResponse({"id": "T1", "handled": None}),
            FakeResponse({"id": "T1", "handled": {"action": "reply"}}),
        ])
        client = HelpdeskClient(self.settings, opener=lambda *args, **kwargs: next(responses))
        self.assertFalse(client.ticket_is_handled("T1"))
        self.assertTrue(client.ticket_is_handled("T1"))
