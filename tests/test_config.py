import os
import tempfile
import unittest
from pathlib import Path

from nordavo.be.config import Settings, _load_project_env


class ConfigurationTests(unittest.TestCase):
    def test_loads_simple_project_env_values(self):
        previous = os.environ.get("GEMINI_MODEL")
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / ".env"
                path.write_text("# comment\nGEMINI_MODEL=gemini-3.5-flash-lite\n", encoding="utf-8")
                _load_project_env(path)
            self.assertEqual("gemini-3.5-flash-lite", os.environ["GEMINI_MODEL"])
        finally:
            if previous is None:
                os.environ.pop("GEMINI_MODEL", None)
            else:
                os.environ["GEMINI_MODEL"] = previous

    def test_provider_names_are_normalized_and_defaulted(self):
        settings = Settings(
            helpdesk_base_url="https://example.test",
            helpdesk_api_key="helpdesk",
            candidate_email="candidate@example.test",
            gemini_api_key="gemini",
            gemini_model="gemini-model",
            groq_api_key="groq",
            groq_model="groq-model",
            database_path=Path("state.sqlite3"),
            language_primary=" Gemini ",
        )

        self.assertEqual("gemini", settings.language_primary)
        self.assertEqual("groq", settings.language_fallback)

    def test_provider_names_must_differ(self):
        with self.assertRaises(ValueError):
            Settings(
                helpdesk_base_url="https://example.test",
                helpdesk_api_key="helpdesk",
                candidate_email="candidate@example.test",
                gemini_api_key="gemini",
                gemini_model="gemini-model",
                groq_api_key="groq",
                groq_model="groq-model",
                database_path=Path("state.sqlite3"),
                language_primary="groq",
                language_fallback="groq",
            )
