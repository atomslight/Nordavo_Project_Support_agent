# Nordavo support agent

The first feature records language analysis. Phase 2 adds a separate decision
ledger: review decisions with `decide`, then explicitly send them with `run --send`.

## Run

```powershell
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e .
$env:HELPDESK_BASE_URL = "https://nakywgkztvzkihdjnwbc.supabase.co/functions/v1/helpdesk"
$env:HELPDESK_API_KEY = "..."
$env:CANDIDATE_EMAIL = "you@example.com"
$env:GEMINI_API_KEY = "..."
$env:GROQ_API_KEY = "..."
$env:LANGUAGE_PRIMARY = "gemini"
$env:LANGUAGE_FALLBACK = "groq"
nordavo analyze
nordavo decide
nordavo run --send
```

## What happens when you run it

1. `cli.py` reads the `analyze` command and your environment settings.
2. `be/helpdesk.py` fetches every open ticket, following pagination cursors.
3. `workflow.py` redacts payment data, detects the language, saves safe state,
   and prepares the report rows.
4. `be/state.py` saves only redacted ticket text and language results in SQLite.
5. `be/report.py` replaces `output/language_results.csv` only after a complete,
   successful run.

The language code lives in `ai/`. `ai/language.py` contains the Gemini and
Groq requests, then chooses the configured provider order. `ai/safety.py`
redacts payment details and marks ticket text as untrusted data in prompts.

## Language providers

Gemini (`GEMINI_MODEL`, default `gemini-3.5-flash-lite`) is the default
primary provider. Groq (`GROQ_MODEL`, default `qwen/qwen3.8-27b`) is the
default fallback. Set `LANGUAGE_PRIMARY` and `LANGUAGE_FALLBACK` to reverse
the order; use only `gemini` or `groq`, and keep the values different.

Both providers are checked before ticket processing. Each receives only
redacted, untrusted ticket text. If both fail for a ticket, analysis stops
without replacing the previous CSV report.

## Data and privacy

The SQLite database stores only redacted ticket text. Set `NORDAVO_DB_PATH` to
put that database elsewhere.

`output/language_results.csv` intentionally contains the original subject and
body because it is for your local review. It may contain sensitive customer
data: it is ignored by Git and must not be committed, uploaded, or shared
outside the authorized project team.

## Decisions and sending

`nordavo decide` redacts a ticket, reuses its saved language result where
possible, gives the complete FAQ to the configured model pair, validates the
result, and stores the decision in SQLite. It never sends a Helpdesk action.

`nordavo run --send` first performs that same safe decision pass, then sends
only already-persisted decisions. The sender marks a row `sending` before a
POST; a restart checks that ticket before resending it. Temporary 503 and 429
responses retry, 409 is treated as already handled, and 400/422 become a
visible `send_failed` ledger row. Each attempt writes one safe JSON line to
`logs/` with no customer text or draft.

The model chooses reply, handoff, or close. A final code check turns malformed,
payment-leaking, wrong-language, or always-human reply/close decisions into a
handoff. Example metrics compare language, action, category, and priority;
they appear in the command summary. The decision prompt always receives the
full unchanged FAQ—there is no RAG or retrieval layer.
