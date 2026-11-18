from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from nordavo.be.config import Settings
from nordavo.models import Decision, Ticket


class HelpdeskError(RuntimeError):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


class HelpdeskClient:
    """Small Helpdesk client: GETs for discovery and explicit action POSTs."""

    def __init__(
        self,
        settings: Settings,
        *,
        opener: Callable[..., Any] = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        max_attempts: int = 3,
        audit=None,
    ) -> None:
        self._settings = settings
        self._opener = opener
        self._sleeper = sleeper
        self._max_attempts = max_attempts
        self._audit = audit

    def fetch_open_tickets(self) -> Iterator[Ticket]:
        cursor: str | None = None
        while True:
            query: dict[str, str | int] = {"limit": 10, "status": "open"}
            if cursor:
                query["cursor"] = cursor
            payload = self._get_json("/tickets", query)
            for ticket in payload.get("tickets", []):
                yield Ticket.from_api(ticket)
            cursor = payload.get("next_cursor")
            if cursor is None:
                return

    def fetch_examples(self) -> list[dict[str, Any]]:
        payload = self._get_json("/examples")
        return payload if isinstance(payload, list) else list(payload.get("examples", []))

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        payload = self._get_json(f"/tickets/{ticket_id}")
        return payload if isinstance(payload, dict) else {}

    def ticket_is_handled(self, ticket_id: str) -> bool:
        ticket = self.get_ticket(ticket_id)
        # The API returns `handled: null` for an open ticket. Do not guess from
        # a status field here: this check prevents a duplicate customer action.
        return ticket.get("handled") is not None

    def send_decision(self, ticket_id: str, decision: Decision) -> str:
        payload = {
            "language": decision.language,
            "category": decision.category,
            "priority": decision.priority,
            "reason": decision.reason,
            "confidence": decision.confidence,
        }
        if decision.action == "reply":
            payload["draft_reply"] = decision.draft_reply
        elif decision.action == "handoff":
            payload["handoff_note"] = decision.handoff_note
        return self._post_json(f"/tickets/{ticket_id}/{decision.action}", payload)

    def _get_json(self, path: str, query: dict[str, str | int] | None = None) -> Any:
        suffix = f"?{urlencode(query)}" if query else ""
        request_url = f"{self._settings.helpdesk_base_url}{path}{suffix}"
        request_headers = {
            "Authorization": f"Bearer {self._settings.helpdesk_api_key}",
            "X-Candidate-Email": self._settings.candidate_email,
        }
        if self._settings.candidate_name:
            request_headers["X-Candidate-Name"] = self._settings.candidate_name

        for attempt in range(self._max_attempts):
            request = Request(request_url, headers=request_headers, method="GET")
            try:
                response = self._opener(request, timeout=20)
                try:
                    raw_body = response.read()
                    response_status = getattr(response, "status", response.getcode())
                    response_headers = getattr(response, "headers", {})
                finally:
                    response.close()
                if 200 <= response_status < 300:
                    return json.loads(raw_body.decode("utf-8"))
                self._retry_or_raise(response_status, response_headers, attempt, "GET", path)
            except HTTPError as error:
                try:
                    self._retry_or_raise(error.code, error.headers or {}, attempt, "GET", path)
                finally:
                    error.close()
            except URLError as error:
                if attempt == self._max_attempts - 1:
                    raise HelpdeskError(None, f"Helpdesk connection failed: {error.reason}") from error
                self._sleeper(2**attempt)
            except json.JSONDecodeError as error:
                raise HelpdeskError(None, "Helpdesk returned invalid JSON") from error

        raise AssertionError("retry loop must return or raise")

    def _post_json(self, path: str, payload: dict[str, Any]) -> str:
        request_url = f"{self._settings.helpdesk_base_url}{path}"
        headers = self._headers() | {"Content-Type": "application/json"}
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(self._max_attempts):
            request = Request(request_url, data=body, headers=headers, method="POST")
            try:
                response = self._opener(request, timeout=20)
                try:
                    status = getattr(response, "status", response.getcode())
                    response_headers = getattr(response, "headers", {})
                finally:
                    response.close()
                if 200 <= status < 300:
                    return "sent"
                if status == 409:
                    return "already_handled"
                self._retry_or_raise(status, response_headers, attempt, "POST", path)
            except HTTPError as error:
                try:
                    if error.code == 409:
                        return "already_handled"
                    self._retry_or_raise(error.code, error.headers or {}, attempt, "POST", path)
                finally:
                    error.close()
            except URLError as error:
                if attempt == self._max_attempts - 1:
                    raise HelpdeskError(None, "Helpdesk connection failed") from None
                self._sleeper(2**attempt)
        raise AssertionError("retry loop must return or raise")

    def _retry_or_raise(
        self, status: int, headers: Any, attempt: int, method: str, path: str
    ) -> None:
        # Bad credentials cannot be fixed by waiting, so stop immediately.
        if status == 401:
            raise HelpdeskError(status, "Helpdesk authentication failed")
        # Only rate limits and temporary server failures are safe to retry.
        if status not in {429, 503} or attempt == self._max_attempts - 1:
            raise HelpdeskError(status, f"Helpdesk GET failed with HTTP {status}")
        retry_after = headers.get("Retry-After") if headers else None
        # The API's Retry-After instruction takes priority for a 429 response.
        retry_delay = float(retry_after) if status == 429 and retry_after else 2**attempt
        if self._audit:
            self._audit.event(
                "retry_wait", stage="helpdesk", status=status,
                delay_seconds=retry_delay, method=method,
                endpoint=_endpoint_template(path),
            )
        self._sleeper(retry_delay)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._settings.helpdesk_api_key}",
            "X-Candidate-Email": self._settings.candidate_email,
        }
        if self._settings.candidate_name:
            headers["X-Candidate-Name"] = self._settings.candidate_name
        return headers


def _endpoint_template(path: str) -> str:
    """Keep a useful endpoint label without recording a customer ticket ID."""
    parts = path.split("/")
    if len(parts) >= 3 and parts[1] == "tickets":
        return "/tickets/{id}" + (f"/{parts[3]}" if len(parts) > 3 else "")
    return path
