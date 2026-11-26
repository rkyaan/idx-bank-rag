"""Text generation through the Gemini API.

The generator owns everything between a finished prompt and a string of text: the
model call, the process-wide request budget, and the response cache. Keeping that
behind a small protocol lets the tests run without network access or quota.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from datetime import date
from typing import Protocol

from rag.config import Settings

REFUSAL_TEXT = (
    "Informasi itu tidak ada di dokumen yang tersedia. "
    "Saya hanya menjawab dari laporan tahunan yang sudah diindeks."
)

SYSTEM_INSTRUCTION = f"""Anda menjawab pertanyaan tentang laporan tahunan bank Indonesia.

Aturan:
1. Gunakan hanya informasi dari KUTIPAN di bawah ini.
2. Setiap kalimat yang memuat fakta atau angka harus diakhiri dengan sitasi dalam format
   [DOKUMEN, hlm. N], memakai label dokumen yang tertulis pada kutipan.
3. Jika kutipan tidak memuat jawabannya, balas persis dengan kalimat berikut tanpa tambahan
   apa pun: {REFUSAL_TEXT}
4. Jangan menebak, jangan memakai pengetahuan di luar kutipan, dan jangan memberi
   nasihat keuangan.
5. Jawab dalam bahasa Indonesia yang ringkas, paling banyak enam kalimat."""


class LLMUnavailable(RuntimeError):
    """The model could not produce an answer, for any reason the UI should hide."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class DailyCapReached(LLMUnavailable):
    """The process-wide request budget for today is spent."""


# Server-side hiccups worth one more attempt. A 429 is deliberately absent: retrying
# a rate limit spends quota and delays the answer instead of helping it.
RETRYABLE_CODES = frozenset({500, 502, 503, 504})


def with_retry(
    call: Callable[[], str],
    attempts: int,
    backoff_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Run a model call, retrying only the failures marked as transient."""
    last: LLMUnavailable | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except LLMUnavailable as exc:
            if not exc.retryable:
                raise
            last = exc
        if attempt < attempts:
            sleep(backoff_seconds * attempt)
    if last is None:
        raise LLMUnavailable("model call failed")
    raise last


class TextModel(Protocol):
    """Anything that turns a prompt into text."""

    def generate(self, prompt: str) -> str:
        """Return the model's answer for this prompt."""
        ...


class GeminiClient:
    """Thin wrapper over google-genai, so callers never touch SDK details."""

    def __init__(self, api_key: str, model: str, settings: Settings) -> None:
        from google import genai
        from google.genai import types

        self._genai = genai
        self._types = types
        self._model = model
        self._settings = settings
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(settings.llm_timeout_seconds * 1000)),
        )

    def generate(self, prompt: str) -> str:
        return with_retry(
            lambda: self._call(prompt),
            attempts=self._settings.llm_retry_attempts + 1,
            backoff_seconds=self._settings.llm_retry_backoff_seconds,
        )

    def _call(self, prompt: str) -> str:
        from google.genai import errors

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=self._settings.llm_temperature,
                    max_output_tokens=self._settings.llm_max_output_tokens,
                ),
            )
            text = (response.text or "").strip()
        except errors.APIError as exc:
            raise LLMUnavailable(
                f"model call failed with status {exc.status}",
                retryable=exc.code in RETRYABLE_CODES,
            ) from exc
        except Exception as exc:
            raise LLMUnavailable("model call failed", retryable=True) from exc

        if not text:
            raise LLMUnavailable("model returned an empty answer", retryable=True)
        return text


class Generator:
    """Adds a daily request budget and a response cache around a text model."""

    def __init__(
        self,
        model: TextModel,
        settings: Settings,
        today: Callable[[], date] = date.today,
    ) -> None:
        self.model = model
        self.settings = settings
        self._today = today
        self._lock = threading.Lock()
        self._cache: dict[str, str] = {}
        self._day: date | None = None
        self._used = 0

    @property
    def requests_used_today(self) -> int:
        with self._lock:
            return self._used

    @property
    def cached_answers(self) -> int:
        with self._lock:
            return len(self._cache)

    def generate(self, prompt: str) -> str:
        """Return a cached answer when possible, otherwise call the model once."""
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        self._consume_budget()
        text = self.model.generate(prompt)

        if self.settings.cache_size > 0:
            with self._lock:
                while len(self._cache) >= self.settings.cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = text
        return text

    def _consume_budget(self) -> None:
        cap = self.settings.daily_request_cap
        if cap <= 0:
            return
        today = self._today()
        with self._lock:
            if self._day != today:
                self._day = today
                self._used = 0
            if self._used >= cap:
                raise DailyCapReached(f"daily request cap of {cap} reached")
            self._used += 1
