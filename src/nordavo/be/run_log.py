"""Safe, local audit events for one CLI run."""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from nordavo.models import DecisionResult


_SAFE_FIELDS = {
    "action", "category", "code", "command", "delay_seconds", "error_type",
    "input_tokens", "model", "outcome", "output_tokens", "priority", "provider",
    "stage", "status", "ticket_id", "method", "endpoint",
}


class RunLog:
    """Write safe JSONL events and optionally show safe progress on the console."""

    def __init__(self, command: str, *, verbose: bool = False, directory: Path = Path("logs")) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{timestamp}-{uuid4().hex[:8]}"
        self._command = command
        self._verbose = verbose
        self._path = directory / f"{self.run_id}.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def event(self, name: str, **details: object) -> None:
        # Customer text can appear in exceptions and model output. Keep only
        # this fixed allow-list even if a future caller passes extra details.
        record = {"event": name, "run_id": self.run_id, "command": self._command}
        record.update({key: value for key, value in details.items() if key in _SAFE_FIELDS})
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            # Auditing must not stop the support workflow.
            print("Audit log could not be written", file=sys.stderr)
            return
        if self._verbose:
            values = " ".join(f"{key}={value}" for key, value in record.items() if key not in {"event", "command"})
            print(f"[{name}] {values}", file=sys.stderr)

    def failure(self, stage: str, error: Exception) -> None:
        self.event("run_failed", stage=stage, error_type=type(error).__name__, **_safe_error_details(error))
        if not self._verbose:
            return
        # Frame metadata explains where a failure occurred without printing an
        # exception message, which may contain customer-controlled content.
        for frame in traceback.extract_tb(error.__traceback__):
            print(
                f"[trace] {Path(frame.filename).name}:{frame.lineno} {frame.name} {type(error).__name__}",
                file=sys.stderr,
            )

    def write(self, ticket_id: str, result: DecisionResult, outcome: str, error: str | None = None) -> None:
        self.event(
            "send_finished", ticket_id=ticket_id, action=result.decision.action,
            category=result.decision.category, priority=result.decision.priority,
            provider=result.provider, model=result.model, input_tokens=result.input_tokens,
            output_tokens=result.output_tokens, outcome=outcome, error_type=error,
        )


def _safe_error_details(error: Exception) -> dict[str, object]:
    """Extract known metadata instead of using an exception's raw message."""
    details: dict[str, object] = {}
    for field in ("provider", "status", "code"):
        value = getattr(error, field, None)
        if isinstance(value, (str, int)):
            details[field] = value
    return details
