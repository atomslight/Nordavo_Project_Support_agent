from __future__ import annotations

import logging
from dataclasses import replace
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nordavo.ai.prompts import LANGUAGE_PROMPT
from nordavo.ai.safety import delimit_ticket_text
from nordavo.models import LanguageAssessment

logger = logging.getLogger(__name__)

class LanguageProviderError(RuntimeError):
    """A safe summary of a language-provider failure."""

    def __init__(
        self,
        provider: str,
        error_type: str,
        *,
        status: int | None = None,
        code: str | int | None = None,
        raw_error: Exception | None = None,
    ) -> None:
        self.provider = provider
        self.error_type = error_type
        self.status = status
        self.code = code
        self.raw_error = raw_error
        details = [error_type]
        if status is not None:
            details.append(f"HTTP {status}")
        if code is not None:
            details.append(f"code {code}")
        super().__init__(f"{provider.title()} language classification failed ({', '.join(details)})")


class _LanguageResponse(BaseModel):
    # Reject unexpected model fields instead of silently accepting a changed response.
    model_config = ConfigDict(extra="forbid")

    language: Literal["nl", "en", "de", "unknown"]
    confidence: float = Field(ge=0, le=1)


class LanguageClassifier(Protocol):
    def health_check(self) -> None: ...

    def classify(self, text: str) -> LanguageAssessment: ...


class GeminiLanguageClassifier:
    def __init__(self, client, model: str) -> None:
        self._client = client
        self._model = model

    def health_check(self) -> None:
        result = self.classify("Hallo, ik heb een vraag over het rooster.")
        if result.language != "nl":
            raise LanguageProviderError("gemini", "UnexpectedHealthCheckLanguage", code=result.language)

    def classify(self, text: str) -> LanguageAssessment:
        prompt = (
            LANGUAGE_PROMPT
            + "Return JSON with language (nl, en, de, or unknown) and "
            "confidence (a number from 0 to 1).\n"
            + delimit_ticket_text("", text)
        )
        try:
            response = self._client.interactions.create(
                model=self._model,
                input=prompt,
                generation_config={"temperature": 0},
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": _LanguageResponse.model_json_schema(),
                },
            )
        except Exception as error:
            raise _provider_error("gemini", error) from None
        return _assessment_from_json(response.output_text, "gemini")


class GroqLanguageClassifier:
    def __init__(self, client, model: str) -> None:
        self._client = client
        self._model = model

    def health_check(self) -> None:
        result = self.classify("Hallo, ik heb een vraag over het rooster.")
        if result.language != "nl":
            raise LanguageProviderError("groq", "UnexpectedHealthCheckLanguage", code=result.language)

    def classify(self, text: str) -> LanguageAssessment:
        prompt = (
            LANGUAGE_PROMPT
            + "Respond with JSON only, in this exact shape: "
            '{"language": "nl" | "en" | "de" | "unknown", '
            '"confidence": <number between 0 and 1>}.\n'
            + delimit_ticket_text("", text)
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                reasoning_effort="none",
                reasoning_format="hidden",
                temperature=0.6,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            output_text = response.choices[0].message.content
        except Exception as error:
            raise _provider_error("groq", error) from None
        return _assessment_from_json(output_text, "groq")


def _assessment_from_json(output_text: str, provider: Literal["gemini", "groq"]) -> LanguageAssessment:
    try:
        result = _LanguageResponse.model_validate_json(output_text)
    except ValidationError as error:
        # Pydantic may include the rejected model text, so retain only field paths and types.
        raise LanguageProviderError(
            provider,
            "ValidationError",
            code=_validation_summary(error),
            raw_error=error,
        ) from None
    return LanguageAssessment(result.language, result.confidence, provider, False)


def _validation_summary(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}:{item['type']}"
        for item in error.errors(include_input=False)
    )


def _provider_error(provider: Literal["gemini", "groq"], error: Exception) -> LanguageProviderError:
    """Keep provider diagnostics useful without retaining customer-controlled text."""
    return LanguageProviderError(
        provider,
        type(error).__name__,
        status=_safe_status(error),
        code=_safe_provider_code(provider, error),
        raw_error=error,
    )


def _safe_status(error: Exception) -> int | None:
    for name in ("status_code", "code"):
        value = getattr(error, name, None)
        if isinstance(value, int):
            return value
    return None


def _safe_provider_code(
    provider: Literal["gemini", "groq"], error: Exception
) -> str | int | None:
    if provider == "gemini":
        value = getattr(error, "code", None)
        return value if isinstance(value, (str, int)) else None

    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return None
    api_error = body.get("error")
    if not isinstance(api_error, dict):
        return None
    value = api_error.get("code")
    return value if isinstance(value, (str, int)) else None


def _log_provider_event(event: str, error: LanguageProviderError) -> None:
    logger.warning(
        "language_provider_%s provider=%s error_type=%s status=%s code=%s",
        event,
        error.provider,
        error.error_type,
        error.status,
        error.code,
    )


class LanguageAnalyzer:
    """Use the configured primary provider, then one fallback provider."""

    def __init__(self, primary: LanguageClassifier, fallback: LanguageClassifier, audit=None) -> None:
        self._primary = primary
        self._fallback = fallback
        self._audit = audit
    #Health check both providers before analyzing any tickets.  If the primary fails, the fallback is still checked.
    def health_check(self) -> None:
        failures: list[LanguageProviderError] = []
        for classifier in (self._primary, self._fallback):
            try:
                classifier.health_check()
            except LanguageProviderError as error:
                _log_provider_event("failure", error)
                failures.append(error)
        if failures:
            first = failures[0]
            raise LanguageProviderError(
                first.provider,
                first.error_type,
                status=first.status,
                code=first.code,
            ) from None
        self._event("health_check", stage="language")
    #Main entry point definition for language classification.  Try the primary provider first, then the fallback if it fails.
    def analyze(self, text: str) -> LanguageAssessment:
        try:
            return replace(self._primary.classify(text), fallback_used=False)
        except LanguageProviderError as primary_error:
            _log_provider_event("fallback", primary_error)
            self._event("fallback_used", provider=primary_error.provider, error_type=primary_error.error_type,
                        status=primary_error.status, code=primary_error.code)
        try:
            return replace(self._fallback.classify(text), fallback_used=True)
        except LanguageProviderError as fallback_error:
            _log_provider_event("failure", fallback_error)
            self._event("provider_failure", provider=fallback_error.provider, error_type=fallback_error.error_type,
                        status=fallback_error.status, code=fallback_error.code)
            raise LanguageProviderError(
                fallback_error.provider,
                fallback_error.error_type,
                status=fallback_error.status,
                code=fallback_error.code,
            ) from None

    def _event(self, name: str, **details) -> None:
        if self._audit:
            self._audit.event(name, **details)
