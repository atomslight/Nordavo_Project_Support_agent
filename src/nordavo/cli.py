from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from nordavo.ai.language import (
    GeminiLanguageClassifier,
    GroqLanguageClassifier,
    LanguageAnalyzer,
    LanguageProviderError,
)
from nordavo.ai.decision import (
    DecisionAnalyzer,
    DecisionProviderError,
    GeminiDecisionClassifier,
    GroqDecisionClassifier,
)
from nordavo.be.config import (
    ConfigurationError,
    Settings,
    create_gemini_client,
    create_groq_client,
)
from nordavo.be.helpdesk import HelpdeskClient, HelpdeskError
from nordavo.be.decision_state import DecisionStateStore
from nordavo.be.run_log import RunLog
from nordavo.be.state import StateStore
from nordavo.workflow import TicketAnalysisWorkflow
from nordavo.decision_workflow import TicketDecisionWorkflow


def _build_analyzer(settings: Settings, audit=None) -> LanguageAnalyzer:
    gemini_classifier = GeminiLanguageClassifier(
        create_gemini_client(settings), settings.gemini_model
    )
    groq_classifier = GroqLanguageClassifier(
        create_groq_client(settings), settings.groq_model
    )
    if settings.language_primary == "gemini":
        return LanguageAnalyzer(gemini_classifier, groq_classifier, audit)
    return LanguageAnalyzer(groq_classifier, gemini_classifier, audit)


def _build_decision_analyzer(settings: Settings, audit=None) -> DecisionAnalyzer:
    gemini = GeminiDecisionClassifier(create_gemini_client(settings), settings.gemini_model)
    groq = GroqDecisionClassifier(create_groq_client(settings), settings.groq_model)
    return DecisionAnalyzer(gemini, groq, audit) if settings.language_primary == "gemini" else DecisionAnalyzer(groq, gemini, audit)


def _run_analysis(settings: Settings, audit: RunLog, *, reset: bool = False):
    state_store = StateStore(settings.database_path)
    try:
        if reset:
            state_store.reset_phase_one()
            audit.event("local_state_reset", stage="cli", outcome="completed")
        workflow = TicketAnalysisWorkflow(
            HelpdeskClient(settings, audit=audit),
            state_store,
            _build_analyzer(settings, audit),
            audit,
        )
        return workflow.run()
    finally:
        state_store.close()


def _run_decisions(
    settings: Settings, audit: RunLog, *, send: bool = False, reset: bool = False
):
    state_store = StateStore(settings.database_path)
    decision_store = DecisionStateStore(settings.database_path)
    try:
        if reset:
            state_store.reset_phase_one()
            decision_store.reset_decisions()
            audit.event("local_state_reset", stage="cli", outcome="completed")
        workflow = TicketDecisionWorkflow(
            HelpdeskClient(settings, audit=audit), state_store, decision_store, _build_analyzer(settings, audit),
            _build_decision_analyzer(settings, audit), Path("data/knowledgebase/faq.md"), audit,
        )
        summary = workflow.decide()
        return summary if not send else type(summary)(
            fetched_tickets=summary.fetched_tickets,
            decided_tickets=summary.decided_tickets,
            reused_decisions=summary.reused_decisions,
            sent_tickets=workflow.send(),
            language_unresolved_tickets=summary.language_unresolved_tickets,
            unresolved_ticket_ids=summary.unresolved_ticket_ids,
        )
    finally:
        decision_store.close()
        state_store.close()


def _run_health_check(settings: Settings, audit: RunLog) -> dict[str, str]:
    # This is intentionally manual so normal ticket commands do not spend four
    # provider calls on every run. It makes readiness easy to check on demand.
    _build_analyzer(settings, audit).health_check()
    _build_decision_analyzer(settings, audit).health_check()
    return {"decision": "healthy", "language": "healthy"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze, decide, send, or health-check Helpdesk tickets.")
    parser.add_argument("command", choices=["analyze", "decide", "run", "health-check"])
    parser.add_argument("--send", action="store_true", help="Allow `run` to send saved decisions.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete local records before analyze or decide; never available for sending.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show safe provider diagnostics without ticket text or keys.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show safe stage progress and sanitized traceback frames.",
    )
    parser.add_argument(
        "--show-raw-errors",
        action="store_true",
        help="Print raw provider errors locally; requires --verbose and cannot be used with run --send.",
    )
    args = parser.parse_args(argv)
    if args.command == "run" and not args.send:
        parser.error("`nordavo run` requires --send")
    if args.send and args.command != "run":
        parser.error("--send is only valid with `nordavo run`")
    if args.reset and args.command not in {"analyze", "decide"}:
        parser.error("--reset is only valid with `nordavo analyze` or `nordavo decide`")
    if args.show_raw_errors and not args.verbose:
        parser.error("--show-raw-errors requires --verbose")
    if args.show_raw_errors and args.command == "run":
        parser.error("--show-raw-errors is not allowed with `nordavo run --send`")
    if args.command not in {"analyze", "decide", "run", "health-check"}:
        return 2
    if args.debug or args.verbose:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
        logging.getLogger("nordavo").setLevel(logging.DEBUG)
    audit = RunLog(args.command, verbose=args.verbose, raw_errors=args.show_raw_errors)
    audit.event("run_started", stage="cli")
    try:
        settings = Settings.from_environment()
        if args.command == "health-check":
            summary = _run_health_check(settings, audit)
        elif args.command == "analyze":
            summary = _run_analysis(settings, audit, reset=args.reset)
        else:
            summary = _run_decisions(
                settings, audit, send=args.command == "run", reset=args.reset
            )
    except ConfigurationError as error:
        return _fail(audit, "configuration", error, 2)
    except HelpdeskError as error:
        return _fail(audit, "helpdesk", error, 3)
    except LanguageProviderError as error:
        return _fail(audit, "language", error, 4)
    except DecisionProviderError as error:
        return _fail(audit, "decision", error, 5)

    audit.event("run_finished", stage="cli", outcome="completed")
    print(json.dumps(summary if isinstance(summary, dict) else summary.__dict__, sort_keys=True))
    return 0


def _fail(audit: RunLog, stage: str, error: Exception, exit_code: int) -> int:
    audit.failure(stage, error)
    print(
        f"{_safe_error_message(error)} [stage={stage} run_id={audit.run_id} audit={audit.path}]",
        file=sys.stderr,
    )
    return exit_code


def _safe_error_message(error: Exception) -> str:
    if isinstance(error, (LanguageProviderError, DecisionProviderError, ConfigurationError)):
        return str(error)
    status = getattr(error, "status", None)
    return f"Helpdesk request failed (HTTP {status})" if status else "Helpdesk request failed"
