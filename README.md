# Nordavo — AI Support Ticket Agent

Nordavo is a Python command-line support agent for Roosterly. It reads helpdesk tickets, checks the supplied FAQ, and prepares a **reply, human handoff, or spam closure**. Decisions are reviewed and saved locally before any customer-facing action is explicitly authorized.

## How to run

Requires **Python 3.10+**, [uv](https://docs.astral.sh/uv/), Helpdesk access, and Gemini/Groq API keys. From PowerShell:

```powershell
git clone https://github.com/atomslight/Nordavo_Project_Support_agent.git
cd Nordavo_Project_Support_agent
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e .
Copy-Item .env.example .env
# Edit .env: set your own CANDIDATE_EMAIL and API keys.
python -m unittest discover -s tests -v
nordavo health-check
nordavo analyze
nordavo decide --verbose
```

`analyze` detects languages; `decide` saves proposed actions **without sending**. Only `nordavo run --send` submits actions to the Helpdesk—use it only when authorized. The FAQ must exist at `data/knowledgebase/faq.md` (not included in the reviewed uploads).

## Decisions

1. **Dutch-first, multilingual support:** detect Dutch, English, or German separately, then verify reply language; human handoff notes stay in English.
2. **FAQ-grounded decisions:** send the full supplied FAQ to the model; validate its structured output and treat customer text as untrusted to reduce unsupported answers and prompt injection.
3. **Safe delivery over speed:** persist decisions in SQLite and require an explicit send command. Gemini is primary, Groq fallback; saved decisions avoid unnecessary repeated calls.

I used **Codex in Plan mode** and **Ponytail** to guide small, beginner-readable changes and purposeful comments. Where the brief was unclear, I treated **P1 as urgency, not necessarily handoff** when the FAQ provides an approved outage-status page.

## Reliability

**503:** up to three attempts with backoff. **429:** obey `Retry-After`, with requests paced at five/second. **409:** stop; do not retry an already-handled ticket. Before sending, SQLite records `sending`; after a crash, a Helpdesk status check prevents blindly repeating an action. Logs keep allow-listed metadata; ticket instructions are untrusted and card/IBAN patterns are redacted before storage and model calls. **Limitations:** CVC/expiry redaction needs strengthening; optional raw-error output may expose customer data.

## Eval and fix

The uploaded test run reports **73 tests: 72 passed, 1 live-provider test skipped**. After changes to FAQ and feature-request prompts, **replies rose from 11 to 14 on 29 comparable tickets** (handoffs 17 → 14). This is **not an accuracy score**: the models differed across runs, and the required **10 labelled examples were analyzed but not scored before/after**.

**Illustrative cost per 100 tickets (Gemini only):** using approximately 2,000 input/162 output tokens from 18 logged Groq decisions as a size proxy, plus an *assumed* 100 input/20 output for language detection: `100 × [(2,100 × $0.30 + 182 × $2.50) / 1,000,000] = $0.1085 ≈ $0.11`. [Gemini prices](https://ai.google.dev/gemini-api/docs/pricing). **Not a measured bill**; excludes fallbacks/retries.

## What I would improve

**Next:** make Groq's decision output schema-constrained and regression-test the JSON-format failure observed during fallback. **Brief clarification:** explicitly distinguish a P1 outage requiring immediate escalation from one that permits a documented status-page reply.
