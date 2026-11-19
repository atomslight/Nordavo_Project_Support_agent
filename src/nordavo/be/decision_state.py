"""SQLite ledger for validated Helpdesk decisions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from nordavo.models import Decision, DecisionResult, Ticket


class DecisionStateStore:
    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ticket_decision (
                ticket_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL,
                language TEXT NOT NULL, decision_json TEXT NOT NULL,
                provider TEXT NOT NULL, model TEXT NOT NULL, fallback_used INTEGER NOT NULL,
                input_tokens INTEGER, output_tokens INTEGER, status TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def get_current(self, ticket: Ticket) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM ticket_decision WHERE ticket_id=? AND content_hash=?",
            (ticket.id, _content_hash(ticket)),
        ).fetchone()

    def save_decided(self, ticket: Ticket, result: DecisionResult, language: str) -> None:
        decision = result.decision
        payload = json.dumps(_decision_dict(decision), ensure_ascii=False, sort_keys=True)
        self._connection.execute(
            """
            INSERT INTO ticket_decision (
                ticket_id, content_hash, language, decision_json, provider, model,
                fallback_used, input_tokens, output_tokens, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'decided')
            ON CONFLICT(ticket_id) DO UPDATE SET
                content_hash=excluded.content_hash, language=excluded.language,
                decision_json=excluded.decision_json, provider=excluded.provider,
                model=excluded.model, fallback_used=excluded.fallback_used,
                input_tokens=excluded.input_tokens, output_tokens=excluded.output_tokens,
                status='decided', updated_at=CURRENT_TIMESTAMP
            """,
            (ticket.id, _content_hash(ticket), language, payload, result.provider,
             result.model, int(result.fallback_used), result.input_tokens, result.output_tokens),
        )
        self._connection.commit()

    def pending(self) -> list[sqlite3.Row]:
        return self._connection.execute(
            "SELECT * FROM ticket_decision WHERE status IN ('decided', 'sending') ORDER BY created_at"
        ).fetchall()

    def mark(self, ticket_id: str, status: str) -> None:
        self._connection.execute(
            "UPDATE ticket_decision SET status=?, updated_at=CURRENT_TIMESTAMP WHERE ticket_id=?",
            (status, ticket_id),
        )
        self._connection.commit()

    def reset_decisions(self) -> None:
        """Delete saved local decisions before a fresh decision-only run."""
        self._connection.execute("DELETE FROM ticket_decision")
        self._connection.commit()

    def read_decision(self, row: sqlite3.Row) -> Decision:
        payload = json.loads(row["decision_json"])
        return Decision(
            language=row["language"], reason=payload["reason"], category=payload["category"],
            priority=payload["priority"], action=payload["action"], faq_ids=tuple(payload["faq_ids"]),
            draft_reply=payload["draft_reply"], handoff_note=payload["handoff_note"],
            confidence=payload["confidence"],
        )


def _content_hash(ticket: Ticket) -> str:
    return hashlib.sha256(f"{ticket.subject}\n{ticket.body}".encode("utf-8")).hexdigest()


def _decision_dict(decision: Decision) -> dict[str, object]:
    return {
        "reason": decision.reason, "category": decision.category, "priority": decision.priority,
        "action": decision.action, "faq_ids": list(decision.faq_ids),
        "draft_reply": decision.draft_reply, "handoff_note": decision.handoff_note,
        "confidence": decision.confidence,
    }
