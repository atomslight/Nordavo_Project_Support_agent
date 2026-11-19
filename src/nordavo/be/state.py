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
        # ticket_state remains the cache that lets the decision workflow reuse
        # language results; the two tables below are simple Phase-1 review views.
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS language_phase_1_examples (
                ticket_id TEXT PRIMARY KEY,
                received_at TEXT NOT NULL,
                channel TEXT NOT NULL,
                customer_plan TEXT,
                original_subject TEXT NOT NULL,
                original_body TEXT NOT NULL,
                language TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                fallback_used INTEGER NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS language_phase_1_testing (
                ticket_id TEXT PRIMARY KEY,
                received_at TEXT NOT NULL,
                channel TEXT NOT NULL,
                customer_plan TEXT,
                original_subject TEXT NOT NULL,
                original_body TEXT NOT NULL,
                language TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                fallback_used INTEGER NOT NULL
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

    def mark_language_unresolved(self, ticket_id: str) -> None:
        """Keep the safe language result, but prevent repeat model calls on rerun."""
        self._connection.execute(
            "UPDATE ticket_state SET status='language_unresolved' WHERE ticket_id=?",
            (ticket_id,),
        )
        self._connection.commit()

    def reset_phase_one(self) -> None:
        """Delete local language state and review rows before a fresh analysis."""
        self._connection.execute("DELETE FROM ticket_state")
        self._connection.execute("DELETE FROM language_phase_1_examples")
        self._connection.execute("DELETE FROM language_phase_1_testing")
        self._connection.commit()

    def save_example_analysis(
        self, ticket: Ticket, assessment: LanguageAssessment
    ) -> None:
        self._save_phase_one_result("language_phase_1_examples", ticket, assessment)

    def save_testing_analysis(
        self, ticket: Ticket, assessment: LanguageAssessment
    ) -> None:
        self._save_phase_one_result("language_phase_1_testing", ticket, assessment)

    def get(self, ticket_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM ticket_state WHERE ticket_id=?", (ticket_id,)
        ).fetchone()

    def _save_phase_one_result(
        self, table: str, ticket: Ticket, assessment: LanguageAssessment
    ) -> None:
        # These review tables keep the requested column names, but callers pass
        # only the redacted Ticket that is safe to store locally.
        self._connection.execute(
            f"""
            INSERT INTO {table} (
                ticket_id, received_at, channel, customer_plan, original_subject,
                original_body, language, confidence, source, fallback_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticket_id) DO UPDATE SET
                received_at=excluded.received_at,
                channel=excluded.channel,
                customer_plan=excluded.customer_plan,
                original_subject=excluded.original_subject,
                original_body=excluded.original_body,
                language=excluded.language,
                confidence=excluded.confidence,
                source=excluded.source,
                fallback_used=excluded.fallback_used
            """,
            (
                ticket.id,
                ticket.received_at,
                ticket.channel,
                ticket.customer_plan,
                ticket.subject,
                ticket.body,
                assessment.language,
                assessment.confidence,
                assessment.source,
                int(assessment.fallback_used),
            ),
        )
        self._connection.commit()


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
