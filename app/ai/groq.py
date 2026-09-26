"""Hosted LLM access through Groq's OpenAI-compatible API.

Implements the same `ChatModel` interface as the local Ollama model, so the
services do not know which one they use. Every request goes through a
`QuotaGuard` first: it is only sent if it fits the development safety limits,
and its token usage is recorded afterwards. A 429 from Groq is never retried;
it pauses all requests until Groq's `retry-after` has passed.

The API key is only ever put in the `Authorization` header; it is never
logged, stored or included in an error message.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager

import httpx

from app.ai.llm import LLMError, LLMUnavailable
from app.ai.quota import Block, QuotaExceeded, QuotaGuard, estimate_chat_tokens
from app.core.config import settings

CONNECT_TIMEOUT_SECONDS = 10.0
GENERATE_TIMEOUT_SECONDS = 120.0
PROVIDER = "groq"


class GroqChatModel:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        guard: QuotaGuard,
        *,
        base_url: str = "https://api.groq.com/openai/v1",
        reasoning_effort: str = "none",
        max_output_tokens: int = 700,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self.model_name = model_name
        self.guard = guard
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort.strip()
        self.max_output_tokens = max_output_tokens
        self._client = httpx.Client(
            timeout=httpx.Timeout(GENERATE_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS),
            transport=transport,
        )

    def __repr__(self) -> str:  # never show the key
        return f"GroqChatModel(model={self.model_name!r})"

    # ---------------------------------------------------------------- ChatModel

    def generate(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        max_tokens = self._output_cap(max_tokens)
        usage_id = self._reserve(system, prompt, max_tokens, streamed=False)
        payload = self._payload(system, prompt, max_tokens, temperature, stream=False)
        with self._recording(usage_id) as record:
            with self._translate_errors():
                response = self._client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
                )
            self._check_status(response)
            data = response.json()
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            record.update(data.get("usage") or {})
        content = content.strip()
        if not content:
            raise LLMError("The language model returned an empty response.")
        return content

    def stream(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Yield text as Groq writes it (Server-Sent Events).

        The quota check and reservation happen on the first `next()`, before the
        stream is opened; nothing is sent if the request does not fit. Usage is
        recorded once the stream ends, however it ends.
        """
        max_tokens = self._output_cap(max_tokens)
        usage_id = self._reserve(system, prompt, max_tokens, streamed=True)
        payload = self._payload(system, prompt, max_tokens, temperature, stream=True)
        produced = False
        with (
            self._recording(usage_id) as record,
            self._translate_errors(),
            self._client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
            ) as response,
        ):
            if response.status_code >= 400:
                response.read()
            self._check_status(response)
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise LLMError(f"The language model failed: {_error_text(chunk)}")
                usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage")
                if usage:
                    record.update(usage)
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        produced = True
                        yield piece
        if not produced:
            raise LLMError("The language model returned an empty response.")

    # ---------------------------------------------------------------- helpers

    def _output_cap(self, max_tokens: int) -> int:
        return max(1, min(max_tokens, self.max_output_tokens))

    def _reserve(self, system: str, prompt: str, max_tokens: int, streamed: bool) -> str:
        if not self._api_key:
            raise LLMUnavailable(
                "Groq is not configured: set GROQ_API_KEY in Backend/.env, or set "
                "LLM_PROVIDER=ollama to use the local model."
            )
        return self.guard.reserve(estimate_chat_tokens(system, prompt), max_tokens, streamed)

    @contextmanager
    def _recording(self, usage_id: str) -> Iterator[dict]:
        """Record the outcome of request `usage_id` exactly once, however it ends.

        The body fills the yielded dict with Groq's `usage` when it arrives.
        """
        usage: dict = {}
        try:
            yield usage
        except QuotaExceeded as error:  # Groq answered 429
            self.guard.complete(usage_id, error=str(error), rate_limited=True)
            raise
        except LLMError as error:
            self.guard.complete(usage_id, **_reported(usage), error=str(error))
            raise
        except ValueError as error:  # a reply that is not JSON
            self.guard.complete(usage_id, **_reported(usage), error="invalid response")
            raise LLMError("Groq returned an unexpected response.") from error
        except BaseException:
            # e.g. the browser disconnected mid-stream: keep what was reserved/reported.
            self.guard.complete(usage_id, **_reported(usage), error="stopped early")
            raise
        self.guard.complete(usage_id, **_reported(usage))

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}

    def _payload(
        self, system: str, prompt: str, max_tokens: int, temperature: float, stream: bool
    ) -> dict:
        payload: dict = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": stream,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
            if self.reasoning_effort != "none":
                # Thinking on: keep it out of the answer text.
                payload["reasoning_format"] = "hidden"
        return payload

    def _check_status(self, response: httpx.Response) -> None:
        self.guard.record_headers({k.lower(): v for k, v in response.headers.items()})
        status = response.status_code
        if status < 400:
            return
        if status == 429:
            retry_after = _retry_after(response)
            until = self.guard.pause(retry_after)
            snapshot = self.guard.snapshot()
            block = Block(
                "groq_rate_limited",
                "Groq rate limit reached. ResearchMind has paused requests to protect your "
                "development quota. Please wait before trying again.",
                until,
            )
            raise QuotaExceeded(block, snapshot)  # never retried
        if status == 401:
            raise LLMUnavailable("Groq rejected the API key. Check GROQ_API_KEY in Backend/.env.")
        if status == 404:
            raise LLMUnavailable(
                f"The Groq model {self.model_name} is not available. Check GROQ_MODEL."
            )
        detail = _error_text(_json_or_empty(response))
        suffix = f": {detail}" if detail else ""
        raise LLMError(f"Groq returned an error ({status}){suffix}")

    @contextmanager
    def _translate_errors(self) -> Iterator[None]:
        try:
            yield
        except httpx.ConnectError as error:
            raise LLMUnavailable(
                "Could not reach Groq. Check the internet connection and try again."
            ) from error
        except httpx.TimeoutException as error:
            raise LLMError("Groq took too long to respond.") from error
        except httpx.HTTPError as error:
            raise LLMError(f"Could not reach Groq ({type(error).__name__}).") from error


def _reported(usage: dict) -> dict:
    """Groq's reported token counts, as `QuotaGuard.complete` arguments."""
    return {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
    }


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _json_or_empty(response: httpx.Response) -> dict:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _error_text(data: dict) -> str:
    error = data.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")[:200]
    return str(error or "")[:200]


_guard: QuotaGuard | None = None


def get_quota_guard() -> QuotaGuard:
    """The guard for the configured Groq model (shared by requests and the usage report)."""
    global _guard
    if _guard is None or _guard.model != settings.groq_model:
        from app.db.database import get_session_factory

        _guard = QuotaGuard(get_session_factory, PROVIDER, settings.groq_model)
    return _guard


def set_quota_guard(guard: QuotaGuard | None) -> None:
    """Swap the guard (tests point it at their own database)."""
    global _guard
    _guard = guard


def groq_from_settings() -> GroqChatModel:
    return GroqChatModel(
        settings.groq_api_key,
        settings.groq_model,
        get_quota_guard(),
        base_url=settings.groq_base_url,
        reasoning_effort=settings.groq_reasoning_effort,
        max_output_tokens=settings.groq_max_output_tokens,
    )
