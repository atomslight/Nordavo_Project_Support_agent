from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator


class ConfigurationError(ValueError):
    """Raised when required runtime configuration is absent."""


def _load_project_env(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE form used by this project's local .env file."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if name:
            os.environ[name] = value.strip("\"'")


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    helpdesk_base_url: str
    helpdesk_api_key: str
    candidate_email: str
    gemini_api_key: str
    gemini_model: str
    groq_api_key: str
    groq_model: str
    database_path: Path
    candidate_name: str | None = None
    language_primary: Literal["gemini", "groq"] = "gemini"
    language_fallback: Literal["gemini", "groq"] = "groq"

    @field_validator("language_primary", "language_fallback", mode="before")
    @classmethod
    def normalize_provider_name(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def require_different_providers(self) -> "Settings":
        if self.language_primary == self.language_fallback:
            raise ValueError("LANGUAGE_PRIMARY and LANGUAGE_FALLBACK must be different")
        return self

    @classmethod
    def from_environment(cls) -> "Settings":
        # Local .env is a convenience for development; deployment variables still work.
        _load_project_env()
        required = {
            "HELPDESK_BASE_URL": os.getenv("HELPDESK_BASE_URL"),
            "HELPDESK_API_KEY": os.getenv("HELPDESK_API_KEY"),
            "CANDIDATE_EMAIL": os.getenv("CANDIDATE_EMAIL"),
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ConfigurationError("Missing environment variables: " + ", ".join(missing))
        try:
            return cls(
                helpdesk_base_url=required["HELPDESK_BASE_URL"].rstrip("/"),
                helpdesk_api_key=required["HELPDESK_API_KEY"],
                candidate_email=required["CANDIDATE_EMAIL"],
                gemini_api_key=required["GEMINI_API_KEY"],
                gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                groq_api_key=required["GROQ_API_KEY"],
                groq_model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
                database_path=Path(os.getenv("NORDAVO_DB_PATH", "data/nordavo.sqlite3")),
                candidate_name=os.getenv("CANDIDATE_NAME") or None,
                language_primary=os.getenv("LANGUAGE_PRIMARY", "gemini"),
                language_fallback=os.getenv("LANGUAGE_FALLBACK", "groq"),
            )
        except ValidationError as error:
            raise ConfigurationError("Invalid language provider configuration") from error


def create_gemini_client(settings: Settings):
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=30_000),
    )


def create_groq_client(settings: Settings):
    from groq import Groq

    return Groq(api_key=settings.groq_api_key, timeout=30.0)
