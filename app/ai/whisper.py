"""Speech-to-text with faster-whisper, running locally (no API, no quota).

Only this module talks to faster-whisper. Services use the `Transcriber`
interface and `get_transcriber()`, so tests use a fake and the model can change
without touching them.

The model loads on first use (and downloads on the very first run: `small` is
about 480 MB). With `WHISPER_DEVICE=auto` it tries the GPU and falls back to
the CPU when the CUDA libraries are missing (on Windows, faster-whisper needs
cuBLAS and cuDNN for CUDA 12 installed separately). PyAV decodes the audio, so
MP3, WAV, M4A, OGG, FLAC, WMA and the sound track of MP4/MOV/WEBM all work without
installing ffmpeg.
"""

import importlib.util
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

from app.core.config import settings

logger = logging.getLogger(__name__)

# Voice activity detection skips silence: faster, and fewer invented words.
VAD_FILTER = True
BEAM_SIZE = 5


class TranscriptionError(Exception):
    """The recording could not be transcribed; the message is safe to show."""


@dataclass(frozen=True)
class Segment:
    start: float  # seconds
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    language: str | None
    language_probability: float
    duration: float  # seconds of audio
    segments: list[Segment]

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments).strip()


ProgressCallback = Callable[[float], None]  # 0.0 .. 1.0 of the recording


class Transcriber(Protocol):
    model_name: str

    def transcribe(
        self,
        audio: BinaryIO,
        *,
        task: str = "transcribe",
        language: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> Transcript:
        """`task="translate"` gives English text instead of the spoken language."""
        ...


class WhisperTranscriber:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        compute_type: str = "",
    ) -> None:
        self.model_name = model_name
        self._device = device.strip().lower() or "auto"
        self._compute_type = compute_type.strip()
        self._model = None
        self._loaded_device: str | None = None
        self._lock = threading.Lock()
        self.status = "not_loaded"
        self.error: str | None = None

    @property
    def device(self) -> str | None:
        return self._loaded_device

    def transcribe(
        self,
        audio: BinaryIO,
        *,
        task: str = "transcribe",
        language: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> Transcript:
        with self._lock:  # one recording at a time: the model is large
            model = self._load()
            try:
                return self._run(model, audio, task, language, on_progress)
            except RuntimeError as error:
                # The GPU loaded but failed at the first real work (missing cuBLAS/cuDNN).
                if self._loaded_device != "cuda" or self._device != "auto":
                    raise TranscriptionError(f"Transcription failed: {error}") from error
                logger.warning("Whisper on the GPU failed (%s); using the CPU instead", error)
                self._model = None
                model = self._load(force_cpu=True)
                audio.seek(0)
                return self._run(model, audio, task, language, on_progress)

    def _run(self, model, audio, task, language, on_progress) -> Transcript:
        try:
            pieces, info = model.transcribe(
                audio,
                task=task,
                language=language or None,
                beam_size=BEAM_SIZE,
                vad_filter=VAD_FILTER,
            )
            duration = float(info.duration or 0.0)
            segments = []
            for piece in pieces:  # a generator: the work happens while iterating
                text = piece.text.strip()
                if text:
                    segments.append(Segment(float(piece.start), float(piece.end), text))
                if on_progress and duration > 0:
                    on_progress(min(1.0, float(piece.end) / duration))
        except RuntimeError:
            raise
        except Exception as error:  # PyAV: not audio, corrupt file, no sound track...
            logger.info("Could not decode audio: %s", error)
            raise TranscriptionError(
                "The recording could not be read. Check that the file plays and has sound."
            ) from error
        return Transcript(
            language=info.language,
            language_probability=float(info.language_probability or 0.0),
            duration=duration,
            segments=segments,
        )

    def _load(self, force_cpu: bool = False):
        if self._model is not None:
            return self._model
        if not force_cpu:
            _expose_nvidia_libraries()
        from faster_whisper import WhisperModel  # heavy import: only when needed

        self.status = "loading"
        if force_cpu:
            candidates = ["cpu"]
        else:
            candidates = ["cuda", "cpu"] if self._device == "auto" else [self._device]
        last_error: Exception | None = None
        for device in candidates:
            compute_type = self._compute_type or ("int8_float16" if device == "cuda" else "int8")
            try:
                self._model = WhisperModel(
                    self.model_name, device=device, compute_type=compute_type
                )
            # No GPU, missing CUDA libraries, a failed download: try the next device.
            except Exception as error:  # noqa: BLE001
                last_error = error
                logger.warning("Whisper could not load on %s: %s", device, error)
                continue
            self._loaded_device = device
            self.status = "ready"
            self.error = None
            logger.info("Whisper %s loaded on %s (%s)", self.model_name, device, compute_type)
            return self._model
        self.status = "error"
        self.error = f"{type(last_error).__name__}: {last_error}"[:300]
        raise TranscriptionError(
            f"The speech-to-text model ({self.model_name}) could not be loaded. "
            "Check the internet connection for its first download, then retry."
        ) from last_error


def _expose_nvidia_libraries() -> None:
    """Let CTranslate2 find cuBLAS/cuDNN installed from PyPI (Windows).

    `pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"` puts the DLLs in
    `site-packages/nvidia/*/bin`, which Windows does not search. Adding those
    folders (to the DLL search path and PATH) makes the GPU usable without
    copying files or editing the system PATH. Harmless when they are absent.
    """
    if os.name != "nt":
        return
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    folders = [
        str(folder)
        for root in spec.submodule_search_locations
        for folder in Path(root).glob("*/bin")
        if folder.is_dir()
    ]
    path = os.environ.get("PATH", "")
    for folder in folders:
        os.add_dll_directory(folder)
        if folder not in path:
            path = folder + os.pathsep + path
    os.environ["PATH"] = path


_transcriber: Transcriber | None = None
_transcriber_lock = threading.Lock()


def get_transcriber() -> Transcriber:
    global _transcriber
    if _transcriber is None:
        with _transcriber_lock:
            if _transcriber is None:
                _transcriber = WhisperTranscriber(
                    settings.whisper_model, settings.whisper_device, settings.whisper_compute_type
                )
    return _transcriber


def set_transcriber(transcriber: Transcriber | None) -> None:
    """Swap the transcriber (tests use a fake)."""
    global _transcriber
    _transcriber = transcriber


def transcriber_status() -> dict[str, object]:
    """For `/health/ai`: never loads the model, only reports."""
    transcriber = _transcriber
    return {
        "enabled": settings.transcription_enabled,
        "model": settings.whisper_model,
        "status": getattr(transcriber, "status", "not_loaded"),
        "device": getattr(transcriber, "device", None),
        "error": getattr(transcriber, "error", None),
    }
