from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from nordavo.be.config import Settings
from nordavo.models import Decision, Ticket


_REQUEST_INTERVAL_SECONDS = 0.2
_MAX_RETRY_AFTER_SECONDS = 30.0
_KNOWN_ERROR_CODES = {
    "invalid_json", "bad_request", "missing_email", "unauthorized", "not_found",
    "already_handled", "validation_failed", "rate_limited", "service_unavailable",
}
_SAFE_FIELD_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")


class HelpdeskError(RuntimeError):
    def __init__(
        self,
        status: int | None,
        message: str,
        *,
        code: str | None = None,
        problem_fields: tuple[str, ...] = (),
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.problem_fields = problem_fields


class HelpdeskClient:
    """Small Helpdesk client: GETs for discovery and explicit action POSTs."""

    def __init__(
        self,
        settings: Settings,
        *,
        opener: Callable[..., Any] = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        max_attempts: int = 3,
        audit=None,
    ) -> None:
        self._settings = settings
        self._opener = opener
        self._sleeper = sleeper
        self._clock = clock
        self._max_attempts = max_attempts
        self._audit = audit
        self._last_request_at: float | None = None

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
                self._wait_for_request_slot()
                response = self._opener(request, timeout=20)
                try:
                    raw_body = response.read()
                    response_status = getattr(response, "status", response.getcode())
                    response_headers = getattr(response, "headers", {})
                finally:
                    response.close()
                if 200 <= response_status < 300:
                    return json.loads(raw_body.decode("utf-8"))
                self._retry_or_raise(
                    response_status, response_headers, attempt, "GET", path, raw_body
                )
            except HTTPError as error:
                try:
                    self._retry_or_raise(
                        error.code, error.headers or {}, attempt, "GET", path, error.read()
                    )
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
                self._wait_for_request_slot()
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
                self._retry_or_raise(status, response_headers, attempt, "POST", path, raw_body)
            except HTTPError as error:
                try:
                    if error.code == 409:
                        return "already_handled"
                    self._retry_or_raise(
                        error.code, error.headers or {}, attempt, "POST", path, error.read()
                    )
                finally:
                    error.close()
            except URLError as error:
                if attempt == self._max_attempts - 1:
                    raise HelpdeskError(None, "Helpdesk connection failed") from None
                self._sleeper(2**attempt)
        raise AssertionError("retry loop must return or raise")

    def _retry_or_raise(
        self,
        status: int,
        headers: Any,
        attempt: int,
        method: str,
        path: str,
        raw_body: bytes,
    ) -> None:
        # Bad credentials cannot be fixed by waiting, so stop immediately.
        if status == 401:
            raise _helpdesk_error(status, "Helpdesk authentication failed", raw_body)
        # Only rate limits and temporary server failures are safe to retry.
        if status not in {429, 503} or attempt == self._max_attempts - 1:
            raise _helpdesk_error(
                status, f"Helpdesk {method} failed with HTTP {status}", raw_body
            )
        retry_after = headers.get("Retry-After") if headers else None
        # The API's Retry-After instruction takes priority for a 429 response.
        retry_delay = _retry_delay(status, retry_after, attempt)
        if self._audit:
            self._audit.event(
                "retry_wait", stage="helpdesk", status=status,
                delay_seconds=retry_delay, method=method,
                endpoint=_endpoint_template(path),
            )
        self._sleeper(retry_delay)

    def _wait_for_request_slot(self) -> None:
        if self._last_request_at is not None:
            delay = self._last_request_at + _REQUEST_INTERVAL_SECONDS - self._clock()
            if delay > 0:
                self._sleeper(delay)
        # Every GET and POST shares the Helpdesk limit of five requests per second.
        self._last_request_at = self._clock()

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


def _retry_delay(status: int, retry_after: Any, attempt: int) -> float:
    if status != 429 or retry_after is None:
        return float(2**attempt)
    try:
        return min(max(float(retry_after), 0.0), _MAX_RETRY_AFTER_SECONDS)
    except (TypeError, ValueError):
        return float(2**attempt)


def _helpdesk_error(status: int, message: str, raw_body: bytes) -> HelpdeskError:
    code, problem_fields = _safe_error_metadata(raw_body)
    return HelpdeskError(status, message, code=code, problem_fields=problem_fields)


def _safe_error_metadata(raw_body: bytes) -> tuple[str | None, tuple[str, ...]]:
    """Keep only stable API metadata; error bodies can contain customer data."""
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, ()
    if not isinstance(payload, dict):
        return None, ()
    code = payload.get("error", payload.get("code"))
    if isinstance(code, dict):
        code = code.get("code")
    safe_code = code if isinstance(code, str) and code in _KNOWN_ERROR_CODES else None
    problems = payload.get("problems")
    if isinstance(problems, dict):
        field_names = list(problems)
    elif isinstance(problems, list):
        field_names = [item.get("field") for item in problems if isinstance(item, dict)]
    else:
        return safe_code, ()
    field_names = [name for name in field_names if isinstance(name, str) and _SAFE_FIELD_NAME.fullmatch(name)]
    return safe_code, tuple(sorted(field_names[:20]))
