from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LanguageCode = Literal["nl", "en", "de", "unknown"]
LanguageSource = Literal["gemini", "groq"]
DecisionAction = Literal["reply", "handoff", "close"]
DecisionCategory = Literal["bug", "billing", "feature_request", "question", "other"]
Priority = Literal["P1", "P2", "P3", "P4"]


@dataclass(frozen=True)
class Ticket:
    """The ticket fields this feature reads from the Helpdesk API."""
    id: str
    received_at: str
    channel: str
    customer_plan: str | None
    subject: str
    body: str

    @classmethod
    def from_api(cls, payload: dict[str, object]) -> "Ticket":
        return cls(
            id=str(payload["id"]),
            received_at=str(payload.get("received_at", "")),
            channel=str(payload.get("channel", "")),
            customer_plan=(
                str(payload["customer_plan"])
                if payload.get("customer_plan") is not None
                else None
            ),
            subject=str(payload.get("subject", "")),
            body=str(payload.get("body", "")),
        )


@dataclass(frozen=True)
class LanguageAssessment:
    """One language result produced by Gemini or Groq."""
    language: LanguageCode
    confidence: float
    source: LanguageSource
    fallback_used: bool


@dataclass(frozen=True)
class Decision:
    """A validated decision ready for local persistence or Helpdesk delivery."""

    language: LanguageCode
    reason: str
    category: DecisionCategory
    priority: Priority
    action: DecisionAction
    faq_ids: tuple[str, ...]
    draft_reply: str | None
    handoff_note: str | None
    confidence: float


@dataclass(frozen=True)
class DecisionResult:
    """A decision plus only the provider metadata needed for safe run logs."""

    decision: Decision
    provider: LanguageSource
    model: str
    fallback_used: bool
    input_tokens: int | None
    output_tokens: int | None
