from __future__ import annotations

import json
import re

from nordavo.models import Ticket

# An IBAN is 15--34 alphanumeric characters.  Permit either its compact form
# or conventional four-character groups; do not permit arbitrary words between
# groups, otherwise a match can swallow the following customer text.
_IBAN = re.compile(
    r"(?<![A-Z0-9])(?:[A-Z]{2}\d{2}[A-Z0-9]{11,30}|"
    r"[A-Z]{2}\d{2}(?:[ -][A-Z0-9]{4}){2,7}(?:[ -][A-Z0-9]{1,4})?)(?![A-Z0-9])",
    re.I,
)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def redact_sensitive(text: str) -> str:
    text = _IBAN.sub("<REDACTED_IBAN>", text)
    return _CARD.sub("<REDACTED_CARD>", text)


def redact_ticket(ticket: Ticket) -> Ticket:
    """Create the only version of a ticket allowed in SQLite and Gemini prompts."""
    return Ticket(
        id=ticket.id,
        received_at=ticket.received_at,
        channel=ticket.channel,
        customer_plan=ticket.customer_plan,
        subject=redact_sensitive(ticket.subject),
        body=redact_sensitive(ticket.body),
    )


def delimit_ticket_text(subject: str, body: str) -> str:
    """Mark customer content as data, not instructions, in a Gemini prompt."""
    payload = json.dumps({"subject": subject, "body": body}, ensure_ascii=False)
    return f"<untrusted_ticket_data>{payload}</untrusted_ticket_data>"
