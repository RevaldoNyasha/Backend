import hashlib
import os
import re
from collections.abc import Iterator

import numpy as np
import pytest

os.environ["ENVIRONMENT"] = "testing"

from app.ai.embeddings import set_embedder
from app.ai.llm import set_llm
from app.ai.whisper import Segment, Transcript, TranscriptionError, set_transcriber
from app.core.config import settings

FAKE_DIMENSION = 64


class FakeEmbedder:
    """Deterministic bag-of-words vectors: texts sharing words point the same way.

    Good enough to test ranking and plumbing without loading BGE-M3.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), FAKE_DIMENSION), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % FAKE_DIMENSION
                vectors[row, slot] += 1.0
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class FakeLLM:
    """Records every prompt and answers with a canned (or scripted) reply."""

    def __init__(self, model_name: str = "fake-llm", replies: list[str] | None = None) -> None:
        self.model_name = model_name
        self.replies = list(replies or [])
        self.calls: list[dict[str, object]] = []

    def generate(
        self, system: str, prompt: str, *, max_tokens: int = 800, temperature: float = 0.2
    ) -> str:
        self.calls.append({"system": system, "prompt": prompt, "max_tokens": max_tokens})
        if self.replies:
            return self.replies.pop(0)
        return f"**Overview** Reply {len(self.calls)}.\n\n**Key points**\n- Point {len(self.calls)}"

    def stream(
        self, system: str, prompt: str, *, max_tokens: int = 800, temperature: float = 0.2
    ) -> Iterator[str]:
        """The same reply as `generate`, delivered word by word."""
        reply = self.generate(system, prompt, max_tokens=max_tokens, temperature=temperature)
        yield from re.findall(r"\S+\s*", reply)


@pytest.fixture(autouse=True)
def fake_llm() -> Iterator[FakeLLM]:
    """Every test gets a fake LLM; no test may call Ollama."""
    llm = FakeLLM()
    set_llm(llm)
    yield llm
    set_llm(None)


@pytest.fixture(autouse=True)
def fake_embedder() -> Iterator[FakeEmbedder]:
    """Every test gets the fake; no test may load the real model."""
    embedder = FakeEmbedder()
    set_embedder(embedder)
    yield embedder
    set_embedder(None)


class FakeTranscriber:
    """Stands in for Whisper: returns scripted segments, or raises.

    `segments` is what a recording "says"; `translation` is returned for
    `task="translate"`. Records every call.
    """

    model_name = "fake-whisper"

    def __init__(self) -> None:
        self.language = "en"
        self.segments = [
            Segment(0.0, 4.0, "P03 said transport costs to the market doubled this year."),
            Segment(4.0, 9.5, "The road floods in January, so trucks cannot reach the farms."),
        ]
        self.translation: list[Segment] | None = None
        self.error: str | None = None
        self.calls: list[dict[str, object]] = []

    def transcribe(self, audio, *, task="transcribe", language=None, on_progress=None):
        self.calls.append({"task": task, "language": language, "bytes": len(audio.read())})
        if self.error:
            raise TranscriptionError(self.error)
        segments = self.translation if task == "translate" else self.segments
        duration = segments[-1].end if segments else 0.0
        for segment in segments:
            if on_progress:
                on_progress(segment.end / duration)
        return Transcript(
            language="en" if task == "translate" else self.language,
            language_probability=0.97,
            duration=duration,
            segments=list(segments or []),
        )


@pytest.fixture(autouse=True)
def fake_transcriber() -> Iterator[FakeTranscriber]:
    """Every test gets the fake; no test may load Whisper."""
    transcriber = FakeTranscriber()
    set_transcriber(transcriber)
    yield transcriber
    set_transcriber(None)
