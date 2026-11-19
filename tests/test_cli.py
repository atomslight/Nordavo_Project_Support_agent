import io
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

from nordavo import cli


class CliTests(unittest.TestCase):
    def test_builds_gemini_primary_and_groq_fallback(self):
        settings = SimpleNamespace(
            gemini_model="gemini", groq_model="groq", language_primary="gemini"
        )
        with (
            patch("nordavo.cli.create_gemini_client"),
            patch("nordavo.cli.create_groq_client"),
            patch("nordavo.cli.GeminiLanguageClassifier"),
            patch("nordavo.cli.GroqLanguageClassifier"),
        ):
            cli._build_analyzer(settings)

    def test_reverses_provider_order_from_settings(self):
        settings = SimpleNamespace(
            gemini_model="gemini", groq_model="groq", language_primary="groq"
        )
        gemini_classifier = object()
        groq_classifier = object()
        with (
            patch("nordavo.cli.create_gemini_client"),
            patch("nordavo.cli.create_groq_client"),
            patch("nordavo.cli.GeminiLanguageClassifier", return_value=gemini_classifier),
            patch("nordavo.cli.GroqLanguageClassifier", return_value=groq_classifier),
        ):
            analyzer = cli._build_analyzer(settings)

        self.assertIs(groq_classifier, analyzer._primary)
        self.assertIs(gemini_classifier, analyzer._fallback)

    def test_language_provider_failure_prints_only_the_safe_message(self):
        safe_error = cli.LanguageProviderError("gemini", "BadRequestError", status=400)
        stderr = io.StringIO()
        with (
            patch("nordavo.cli.Settings.from_environment", return_value=SimpleNamespace()),
            patch("nordavo.cli._run_analysis", side_effect=safe_error),
            redirect_stderr(stderr),
        ):
            exit_code = cli.main(["analyze"])

        self.assertEqual(4, exit_code)
        self.assertTrue(stderr.getvalue().startswith(str(safe_error) + " [stage=language run_id="))

    def test_raw_errors_are_rejected_for_sending(self):
        with self.assertRaises(SystemExit):
            cli.main(["run", "--send", "--verbose", "--show-raw-errors"])

    def test_manual_health_check_checks_both_provider_layers(self):
        language = SimpleNamespace(health_check=lambda: None)
        decision = SimpleNamespace(health_check=lambda: None)
        with (
            patch("nordavo.cli._build_analyzer", return_value=language) as build_language,
            patch("nordavo.cli._build_decision_analyzer", return_value=decision) as build_decision,
        ):
            result = cli._run_health_check(SimpleNamespace(), SimpleNamespace())

        self.assertEqual({"decision": "healthy", "language": "healthy"}, result)
        build_language.assert_called_once()
        build_decision.assert_called_once()
