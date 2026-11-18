from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from nordavo.models import LanguageAssessment, Ticket


class StateStore:
    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ticket_state (
                ticket_id TEXT PRIMARY KEY,
                received_at TEXT NOT NULL,
                channel TEXT NOT NULL,
                customer_plan TEXT,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                language TEXT,
                confidence REAL,
                language_source TEXT,
                fallback_used INTEGER NOT NULL DEFAULT 0,
                action TEXT,
                status TEXT NOT NULL,
                fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                analyzed_at TEXT
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def upsert_fetched(self, ticket: Ticket) -> None:
        content_hash = _content_hash(ticket.subject, ticket.body)
        # The hash tells us whether a previously saved language result still
        # belongs to this exact redacted ticket content.
        ticket_values = _ticket_values(ticket, content_hash)
        self._connection.execute(
            """
            INSERT INTO ticket_state (
                ticket_id, received_at, channel, customer_plan, subject, body,
                content_hash, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'fetched')
            ON CONFLICT(ticket_id) DO UPDATE SET
                received_at=excluded.received_at,
                channel=excluded.channel,
                customer_plan=excluded.customer_plan,
                subject=excluded.subject,
                body=excluded.body,
                fetched_at=CURRENT_TIMESTAMP,
                language=CASE WHEN ticket_state.content_hash = excluded.content_hash THEN ticket_state.language ELSE NULL END,
                confidence=CASE WHEN ticket_state.content_hash = excluded.content_hash THEN ticket_state.confidence ELSE NULL END,
                language_source=CASE WHEN ticket_state.content_hash = excluded.content_hash THEN ticket_state.language_source ELSE NULL END,
                fallback_used=CASE WHEN ticket_state.content_hash = excluded.content_hash THEN ticket_state.fallback_used ELSE 0 END,
                status=CASE WHEN ticket_state.content_hash = excluded.content_hash THEN ticket_state.status ELSE 'fetched' END,
                content_hash=excluded.content_hash
            """,
            ticket_values,
        )
        self._connection.commit()

    def mark_analyzed(self, ticket_id: str, assessment: LanguageAssessment) -> None:
        self._connection.execute(
            """
            UPDATE ticket_state
            SET language=?, confidence=?, language_source=?, fallback_used=?,
                status='analyzed', analyzed_at=CURRENT_TIMESTAMP
            WHERE ticket_id=?
            """,
            (
                assessment.language,
                assessment.confidence,
                assessment.source,
                int(assessment.fallback_used),
                ticket_id,
            ),
        )
        self._connection.commit()

    def get(self, ticket_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM ticket_state WHERE ticket_id=?", (ticket_id,)
        ).fetchone()


def _content_hash(subject: str, body: str) -> str:
    return hashlib.sha256(f"{subject}\n{body}".encode("utf-8")).hexdigest()


def _ticket_values(ticket: Ticket, content_hash: str) -> tuple[str | None, ...]:
    """Build the values for one safe, redacted ticket row."""
    return (
        ticket.id,
        ticket.received_at,
        ticket.channel,
        ticket.customer_plan,
        ticket.subject,
        ticket.body,
        content_hash,
    )
