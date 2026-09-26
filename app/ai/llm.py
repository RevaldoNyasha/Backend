"""LLM access: the `ChatModel` interface, the local Ollama model, and `get_llm()`.

Services use the `ChatModel` interface and `get_llm()`, so tests can use a fake
and the provider can change without touching them. `LLM_PROVIDER` picks Groq
(`app/ai/groq.py`, the default) or Ollama (this module).
"""

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

import httpx

from app.core.config import settings

STATUS_TIMEOUT_SECONDS = 3.0
CONNECT_TIMEOUT_SECONDS = 5.0
# Generation on a small GPU can take minutes for long prompts.
GENERATE_TIMEOUT_SECONDS = 600.0
# Context window requested from Ollama (tokens). Fits qwen2.5:3b in 4 GB VRAM.
CONTEXT_TOKENS = 8192


class LLMError(Exception):
    """Generation failed; the message is safe to show to the user."""


class LLMUnavailable(LLMError):
    """Ollama is not running, or the model has not been pulled."""


class ChatModel(Protocol):
    model_name: str

    def generate(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str: ...

    def stream(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Yield the reply in pieces as the model writes it."""
        ...


class OllamaChatModel:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._client = httpx.Client(
            timeout=httpx.Timeout(GENERATE_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS),
            transport=transport,
        )

    def _payload(
        self, system: str, prompt: str, max_tokens: int, temperature: float, stream: bool
    ) -> dict:
        return {
            "model": self.model_name,
            "stream": stream,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": CONTEXT_TOKENS,
            },
        }

    def _check_status(self, response: httpx.Response) -> None:
        if response.status_code == 404:
            raise LLMUnavailable(
                f"The model {self.model_name} is not available. Run: ollama pull {self.model_name}"
            )
        if response.status_code >= 400:
            raise LLMError(f"The language model returned an error ({response.status_code}).")

    def generate(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        payload = self._payload(system, prompt, max_tokens, temperature, stream=False)
        with self._translate_errors():
            response = self._client.post(f"{self.base_url}/api/chat", json=payload)
        self._check_status(response)

        content = (response.json().get("message") or {}).get("content", "").strip()
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
        """Yield text pieces as Ollama produces them (one JSON object per line)."""
        payload = self._payload(system, prompt, max_tokens, temperature, stream=True)
        produced = False
        with (
            self._translate_errors(),
            self._client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response,
        ):
            if response.status_code >= 400:
                response.read()
                self._check_status(response)
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise LLMError(f"The language model failed: {chunk['error']}")
                piece = (chunk.get("message") or {}).get("content", "")
                if piece:
                    produced = True
                    yield piece
                if chunk.get("done"):
                    break
        if not produced:
            raise LLMError("The language model returned an empty response.")

    @contextmanager
    def _translate_errors(self) -> Iterator[None]:
        try:
            yield
        except httpx.ConnectError as error:
            raise LLMUnavailable(
                f"Ollama is not running at {self.base_url}. Start Ollama and try again."
            ) from error
        except httpx.TimeoutException as error:
            raise LLMError("The language model took too long to respond.") from error
        except httpx.HTTPError as error:
            raise LLMError(f"Could not reach the language model: {error}") from error


_llm: ChatModel | None = None
_llm_lock = threading.Lock()


def get_llm() -> ChatModel:
    """The configured model: Groq (default) or the local Ollama (`LLM_PROVIDER`)."""
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                if settings.llm_provider == "ollama":
                    _llm = OllamaChatModel(settings.ollama_base_url, settings.ollama_model)
                else:
                    from app.ai.groq import groq_from_settings

                    _llm = groq_from_settings()
    return _llm


def llm_is_metered() -> bool:
    """True when each request spends a hosted quota (Groq), so none should be wasted."""
    return settings.llm_provider == "groq"


def set_llm(llm: ChatModel | None) -> None:
    """Swap the model (tests use a fake)."""
    global _llm
    _llm = llm


def ollama_status() -> dict[str, object]:
    """Is the Ollama server reachable, and has the configured model been pulled?"""
    model = settings.ollama_model
    try:
        response = httpx.get(
            f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=STATUS_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        return {
            "baseUrl": settings.ollama_base_url,
            "model": model,
            "reachable": False,
            "modelAvailable": False,
            "error": f"{type(error).__name__}: {error}"[:300],
        }

    names = {entry.get("name", "") for entry in response.json().get("models", [])}
    # "qwen2.5:3b" matches exactly; a bare "qwen2.5" means the ":latest" tag.
    available = model in names or f"{model}:latest" in names
    return {
        "baseUrl": settings.ollama_base_url,
        "model": model,
        "reachable": True,
        "modelAvailable": available,
        "error": None if available else f"Model not pulled. Run: ollama pull {model}",
    }
