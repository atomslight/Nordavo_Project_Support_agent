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
from nordavo.be.report import LanguageCsvReport
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


def _run_analysis(settings: Settings, audit: RunLog):
    state_store = StateStore(settings.database_path)
    try:
        workflow = TicketAnalysisWorkflow(
            HelpdeskClient(settings, audit=audit),
            state_store,
            _build_analyzer(settings, audit),
            LanguageCsvReport(Path("output/language_results.csv")),
            audit,
        )
        return workflow.run()
    finally:
        state_store.close()


def _run_decisions(settings: Settings, audit: RunLog, *, send: bool = False):
    state_store = StateStore(settings.database_path)
    decision_store = DecisionStateStore(settings.database_path)
    try:
        workflow = TicketDecisionWorkflow(
            HelpdeskClient(settings, audit=audit), state_store, decision_store, _build_analyzer(settings, audit),
            _build_decision_analyzer(settings, audit), Path("data/knowledgebase/faq.md"), audit,
        )
        summary = workflow.decide()
        return summary if not send else type(summary)(
            summary.fetched_tickets,
            summary.decided_tickets,
            workflow.send(),
            summary.example_metrics,
        )
    finally:
        decision_store.close()
        state_store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze or decide Helpdesk tickets.")
    parser.add_argument("command", choices=["analyze", "decide", "run"])
    parser.add_argument("--send", action="store_true", help="Allow `run` to send saved decisions.")
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
    args = parser.parse_args(argv)
    if args.command == "run" and not args.send:
        parser.error("`nordavo run` requires --send")
    if args.send and args.command != "run":
        parser.error("--send is only valid with `nordavo run`")
    if args.command not in {"analyze", "decide", "run"}:
        return 2
    if args.debug or args.verbose:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
        logging.getLogger("nordavo").setLevel(logging.DEBUG)
    audit = RunLog(args.command, verbose=args.verbose)
    audit.event("run_started", stage="cli")
    try:
        settings = Settings.from_environment()
        summary = _run_analysis(settings, audit) if args.command == "analyze" else _run_decisions(settings, audit, send=args.command == "run")
    except ConfigurationError as error:
        return _fail(audit, "configuration", error, 2)
    except HelpdeskError as error:
        return _fail(audit, "helpdesk", error, 3)
    except LanguageProviderError as error:
        return _fail(audit, "language", error, 4)
    except DecisionProviderError as error:
        return _fail(audit, "decision", error, 5)

    audit.event("run_finished", stage="cli", outcome="completed")
    print(json.dumps(summary.__dict__, sort_keys=True))
    return 0


def _fail(audit: RunLog, stage: str, error: Exception, exit_code: int) -> int:
    audit.failure(stage, error)
    print(f"{_safe_error_message(error)} [run_id={audit.run_id}]", file=sys.stderr)
    return exit_code


def _safe_error_message(error: Exception) -> str:
    if isinstance(error, (LanguageProviderError, DecisionProviderError, ConfigurationError)):
        return str(error)
    status = getattr(error, "status", None)
    return f"Helpdesk request failed (HTTP {status})" if status else "Helpdesk request failed"
