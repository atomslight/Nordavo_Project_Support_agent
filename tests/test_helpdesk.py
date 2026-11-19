import io
import json
import unittest
from urllib.error import HTTPError

from nordavo.be.config import Settings
from nordavo.be.helpdesk import HelpdeskClient
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
        sleeps = []

        def opener(*args, **kwargs):
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

        client = HelpdeskClient(self.settings, opener=opener, sleeper=sleeps.append)
        self.assertEqual(["T1", "T2"], [ticket.id for ticket in client.fetch_open_tickets()])
        self.assertEqual([1], sleeps)

    def test_honors_retry_after_for_429_and_never_builds_a_post_request(self):
        responses = iter([
            HTTPError("https://example.test", 429, "busy", {"Retry-After": "3"}, io.BytesIO()),
            FakeResponse([]),
        ])
        requests = []
        sleeps = []

        def opener(request, **kwargs):
            requests.append(request)
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

        client = HelpdeskClient(self.settings, opener=opener, sleeper=sleeps.append)
        self.assertEqual([], client.fetch_examples())
        self.assertEqual([3.0], sleeps)
        self.assertTrue(all(request.get_method() == "GET" for request in requests))

    def test_401_stops_without_retry(self):
        sleeps = []

        def opener(*args, **kwargs):
            raise HTTPError("https://example.test", 401, "unauthorized", {}, io.BytesIO())

        client = HelpdeskClient(self.settings, opener=opener, sleeper=sleeps.append)
        with self.assertRaisesRegex(Exception, "authentication failed"):
            client.fetch_examples()
        self.assertEqual([], sleeps)

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

    def test_handled_field_is_the_only_recovery_signal(self):
        responses = iter([
            FakeResponse({"id": "T1", "handled": None}),
            FakeResponse({"id": "T1", "handled": {"action": "reply"}}),
        ])
        client = HelpdeskClient(self.settings, opener=lambda *args, **kwargs: next(responses))
        self.assertFalse(client.ticket_is_handled("T1"))
        self.assertTrue(client.ticket_is_handled("T1"))
