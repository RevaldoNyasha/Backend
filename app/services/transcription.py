"""Turn uploaded audio and video into text: the recording's equivalent of extraction.

A recording waits while its status is ``processing``; the transcription worker
(`processing_worker.TranscriptionWorker`) calls `transcribe_document` for each:
1. Whisper transcribes it, updating `Document.processing_progress` as it goes.
2. If the owner turned on "Translate transcripts to English" and the recording
   is not in English, a second pass produces an English translation.
3. The transcript is saved (`transcription` result, with timed segments), the
   working text (the translation if any, else the transcript) becomes the
   document's extracted text, and it is cut into chunks that remember when each
   passage is spoken. The embedding step then makes them searchable.

From there a recording is like any other document: summaries, analysis,
reports and the assistant all read its text, and evidence from it carries a
timestamp such as "04:32".
"""

import io
import logging
import time

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.ai.whisper import Segment, Transcriber, TranscriptionError, get_transcriber
from app.core.config import settings
from app.models.document import (
    STATUS_FAILED,
    STATUS_PROCESSED,
    STATUS_PROCESSING,
    TRANSCRIBABLE_KINDS,
    Document,
)
from app.models.document_chunk import DocumentChunk
from app.models.project import Project
from app.models.result import (
    RESULT_EXTRACTED_TEXT,
    RESULT_TRANSCRIPTION,
    RESULT_TRANSLATION,
    Result,
)
from app.models.user import User
from app.services.auth import get_preferences
from app.services.project import mark_project_updated
from app.services.storage import get_storage
from app.utils.chunking import OVERLAP_CHARS, TARGET_CHARS

logger = logging.getLogger(__name__)

MAX_ERROR_LENGTH = 500
PROGRESS_EVERY_SECONDS = 3.0  # how often progress is written while transcribing

TRANSCRIPTION_OFF_ERROR = (
    "Audio transcription is turned off on this server (TRANSCRIPTION_ENABLED=false)."
)
AUTO_TRANSCRIBE_OFF_ERROR = (
    "Automatic transcription is off in Settings. Choose Retry to transcribe this recording."
)
NO_SPEECH_ERROR = "No speech was found in this recording."


def is_recording(document: Document) -> bool:
    return document.kind in TRANSCRIBABLE_KINDS


def hold_reason(db: Session, document: Document) -> str | None:
    """Why a new upload should not be transcribed automatically, or None to go ahead.

    Checked at upload time only: "Retry" always transcribes.
    """
    if not settings.transcription_enabled:
        return TRANSCRIPTION_OFF_ERROR
    owner = _owner(db, document)
    if owner is not None and not get_preferences(owner)["autoTranscribe"]:
        return AUTO_TRANSCRIBE_OFF_ERROR
    return None


def next_pending_transcription_id(db: Session) -> str | None:
    """Oldest recording still waiting to be transcribed."""
    return db.scalar(
        select(Document.id)
        .where(Document.status == STATUS_PROCESSING, Document.kind.in_(TRANSCRIBABLE_KINDS))
        .order_by(Document.uploaded_at)
        .limit(1)
    )


def pending_transcription_count(db: Session) -> int:
    return len(
        db.scalars(
            select(Document.id).where(
                Document.status == STATUS_PROCESSING, Document.kind.in_(TRANSCRIBABLE_KINDS)
            )
        ).all()
    )


def transcribe_document(
    db: Session, document_id: str, transcriber: Transcriber | None = None
) -> None:
    document = db.get(Document, document_id)
    if document is None or document.status != STATUS_PROCESSING or not is_recording(document):
        return
    model = transcriber or get_transcriber()
    translate = _wants_translation(db, document)

    try:
        content = get_storage().read(document.storage_path)
    except FileNotFoundError:
        _mark_failed(db, document_id, "The uploaded file is missing. Please upload it again.")
        return

    started = time.monotonic()
    try:
        _set_progress(db, document_id, "Transcribing: starting")
        transcript = model.transcribe(
            io.BytesIO(content),
            language=settings.whisper_language.strip() or None,
            on_progress=_progress_writer(db, document_id, "Transcribing"),
        )
        if not transcript.segments:
            _mark_failed(db, document_id, NO_SPEECH_ERROR)
            return
        translation = None
        if translate and transcript.language not in (None, "en"):
            _set_progress(db, document_id, "Translating to English: starting")
            translation = model.transcribe(
                io.BytesIO(content),
                task="translate",
                language=transcript.language,
                on_progress=_progress_writer(db, document_id, "Translating to English"),
            )
    except TranscriptionError as error:
        _mark_failed(db, document_id, str(error))
        return
    except Exception:
        logger.exception("Unexpected error transcribing document %s", document_id)
        _mark_failed(db, document_id, "Transcription failed unexpectedly. Try again.")
        return

    document = db.get(Document, document_id)
    if document is None or document.status != STATUS_PROCESSING:
        return  # deleted (or reset) while we worked

    working = translation if translation and translation.segments else transcript
    _replace_outputs(db, document)
    db.add(
        Result(
            project_id=document.project_id,
            document_id=document.id,
            result_type=RESULT_TRANSCRIPTION,
            content=transcript.text,
            model_name=f"faster-whisper {model.model_name}",
            details={
                "language": transcript.language,
                "languageProbability": round(transcript.language_probability, 3),
                "duration": round(transcript.duration, 2),
                "segments": _segments_json(transcript.segments),
            },
        )
    )
    if working is translation:
        db.add(
            Result(
                project_id=document.project_id,
                document_id=document.id,
                result_type=RESULT_TRANSLATION,
                content=translation.text,
                model_name=f"faster-whisper {model.model_name}",
                details={
                    "from": transcript.language,
                    "segments": _segments_json(translation.segments),
                },
            )
        )
    text, chunks = timed_chunks(working.segments)
    db.add(
        Result(
            project_id=document.project_id,
            document_id=document.id,
            result_type=RESULT_EXTRACTED_TEXT,
            content=text,
        )
    )
    for index, chunk in enumerate(chunks):
        db.add(
            DocumentChunk(
                project_id=document.project_id,
                document_id=document.id,
                chunk_index=index,
                page=None,
                **chunk,
            )
        )

    document.language = transcript.language
    document.status = STATUS_PROCESSED
    document.processing_error = None
    document.processing_progress = None
    _touch_project(db, document)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()  # deleted meanwhile: nothing to save
        return
    logger.info(
        "Transcribed %s (%.0fs of %s audio) into %d chunks in %.0fs",
        document_id,
        transcript.duration,
        transcript.language,
        len(chunks),
        time.monotonic() - started,
    )


def transcript_for(db: Session, document: Document) -> dict | None:
    """The saved transcript (and translation) of a recording, for the viewer."""
    results = {
        result.result_type: result
        for result in db.scalars(
            select(Result).where(
                Result.document_id == document.id,
                Result.result_type.in_((RESULT_TRANSCRIPTION, RESULT_TRANSLATION)),
            )
        )
    }
    transcript = results.get(RESULT_TRANSCRIPTION)
    if transcript is None:
        return None
    details = transcript.details or {}
    translation = results.get(RESULT_TRANSLATION)
    return {
        "document_id": document.id,
        "language": details.get("language"),
        "duration": details.get("duration"),
        "model": transcript.model_name,
        "segments": details.get("segments", []),
        "translation": (translation.details or {}).get("segments") if translation else None,
    }


# ---------------------------------------------------------------- chunks with timestamps


def timed_chunks(segments: list[Segment]) -> tuple[str, list[dict]]:
    """The full text, and chunks of whole segments (~1,000 characters) with times.

    Chunks end on segment boundaries so every chunk has an exact start and end
    time; a little of the previous chunk is repeated, as for documents.
    """
    pieces: list[tuple[int, int, Segment]] = []  # (start char, end char, segment)
    text = ""
    for segment in segments:
        if text:
            text += " "
        pieces.append((len(text), len(text) + len(segment.text), segment))
        text += segment.text

    chunks: list[dict] = []
    first = 0
    while first < len(pieces):
        last = first
        while last + 1 < len(pieces) and pieces[last + 1][1] - pieces[first][0] <= TARGET_CHARS:
            last += 1
        start_char, end_char = pieces[first][0], pieces[last][1]
        chunks.append(
            {
                "text": text[start_char:end_char],
                "char_start": start_char,
                "char_end": end_char,
                "start_seconds": round(pieces[first][2].start, 2),
                "end_seconds": round(pieces[last][2].end, 2),
            }
        )
        if last + 1 >= len(pieces):
            break
        # Next chunk starts with the last segments that fit in the overlap.
        next_first = last + 1
        while next_first - 1 > first and end_char - pieces[next_first - 1][0] <= OVERLAP_CHARS:
            next_first -= 1
        first = next_first
    return text, chunks


def format_timestamp(seconds: float) -> str:
    """`mm:ss`, or `h:mm:ss` from an hour on (the contract's Evidence.timestamp)."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------- helpers


def _segments_json(segments: list[Segment]) -> list[dict]:
    return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text} for s in segments]


def _owner(db: Session, document: Document) -> User | None:
    """The researcher whose project this is (their Settings apply)."""
    return db.scalar(
        select(User)
        .join(Project, Project.user_id == User.id)
        .where(Project.id == document.project_id)
    )


def _wants_translation(db: Session, document: Document) -> bool:
    owner = _owner(db, document)
    return owner is not None and get_preferences(owner)["autoTranslate"]


def _progress_writer(db: Session, document_id: str, label: str):
    last_written = 0.0

    def write(fraction: float) -> None:
        nonlocal last_written
        now = time.monotonic()
        if now - last_written >= PROGRESS_EVERY_SECONDS:
            last_written = now
            _set_progress(db, document_id, f"{label}: {round(fraction * 100)}%")

    return write


def _set_progress(db: Session, document_id: str, progress: str) -> None:
    document = db.scalar(
        select(Document).where(Document.id == document_id).execution_options(populate_existing=True)
    )
    if document is None or document.status != STATUS_PROCESSING:
        return
    document.processing_progress = progress[:120]
    try:
        db.commit()
    except IntegrityError:
        db.rollback()


def _replace_outputs(db: Session, document: Document) -> None:
    db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
    db.execute(
        delete(Result).where(
            Result.document_id == document.id,
            Result.result_type.in_(
                (RESULT_EXTRACTED_TEXT, RESULT_TRANSCRIPTION, RESULT_TRANSLATION)
            ),
        )
    )


def _mark_failed(db: Session, document_id: str, reason: str) -> None:
    db.rollback()
    document = db.get(Document, document_id)
    if document is None:
        return
    document.status = STATUS_FAILED
    document.processing_error = reason[:MAX_ERROR_LENGTH]
    document.processing_progress = None
    _touch_project(db, document)
    db.commit()
    logger.info("Recording %s was not transcribed: %s", document_id, reason)


def _touch_project(db: Session, document: Document) -> None:
    project = db.get(Project, document.project_id)
    if project is not None:
        mark_project_updated(project)
