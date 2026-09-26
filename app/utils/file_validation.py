from pathlib import Path

from app.core.config import settings
from app.models.document import (
    KIND_AUDIO,
    KIND_CSV,
    KIND_DOCX,
    KIND_NOTES,
    KIND_PDF,
    KIND_VIDEO,
    TYPE_INTERVIEW,
    TYPE_RESEARCH_NOTES,
)

KIND_AND_TYPE_BY_EXTENSION = {
    ".pdf": (KIND_PDF, TYPE_RESEARCH_NOTES),
    ".docx": (KIND_DOCX, TYPE_INTERVIEW),
    ".csv": (KIND_CSV, TYPE_RESEARCH_NOTES),
    ".txt": (KIND_NOTES, TYPE_RESEARCH_NOTES),
    # Recordings: transcribed with faster-whisper (PyAV decodes all of these).
    ".mp3": (KIND_AUDIO, TYPE_INTERVIEW),
    ".wav": (KIND_AUDIO, TYPE_INTERVIEW),
    ".m4a": (KIND_AUDIO, TYPE_INTERVIEW),
    ".ogg": (KIND_AUDIO, TYPE_INTERVIEW),
    ".flac": (KIND_AUDIO, TYPE_INTERVIEW),
    ".wma": (KIND_AUDIO, TYPE_INTERVIEW),  # Windows Media Audio (e.g. voice recorders)
    ".mp4": (KIND_VIDEO, TYPE_INTERVIEW),
    ".mov": (KIND_VIDEO, TYPE_INTERVIEW),
    ".webm": (KIND_VIDEO, TYPE_INTERVIEW),
}
ALLOWED_EXTENSIONS = frozenset(KIND_AND_TYPE_BY_EXTENSION)
DEFAULT_KIND = KIND_NOTES
DEFAULT_TYPE = TYPE_RESEARCH_NOTES

MAX_SIZE_MB_BY_KIND = {
    KIND_AUDIO: settings.upload_max_audio_size_mb,
    KIND_VIDEO: settings.upload_max_video_size_mb,
}

BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024


def resolve_kind_and_type(filename: str) -> tuple[str, str] | None:
    return KIND_AND_TYPE_BY_EXTENSION.get(Path(filename).suffix.lower())


def normalize_extension(filename: str) -> str:
    return Path(filename).suffix.lstrip(".").upper()


def max_size_mb_for_kind(kind: str) -> int:
    return MAX_SIZE_MB_BY_KIND.get(kind, settings.upload_max_document_size_mb)


def format_file_size(size_bytes: int) -> str:
    if size_bytes < BYTES_PER_KB:
        return f"{size_bytes} B"

    if size_bytes < BYTES_PER_MB:
        return _with_unit(size_bytes / BYTES_PER_KB, "KB")

    return _with_unit(size_bytes / BYTES_PER_MB, "MB")


def _with_unit(value: float, unit: str) -> str:
    if value < 10:
        return f"{value:.1f} {unit}"
    return f"{value:.0f} {unit}"
