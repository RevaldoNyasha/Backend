"""Background threads that do slow work outside HTTP requests.

- `ProcessingWorker`: extracts and chunks waiting documents, then embeds chunks.
- `TranscriptionWorker`: turns audio/video into text with Whisper (minutes per recording),
  in its own thread so a long recording never holds up other uploads.
- `GenerationWorker`: runs LLM jobs (summaries, analyses, reports), which can take minutes,
  so they never hold up processing of new uploads.

Good enough for development: no queue server, and nothing is lost on restart
because each work list is read from the database ("documents in processing",
"chunks without an embedding", "summaries in processing"). Requests call the
`notify_*` functions so work starts immediately instead of at the next poll.
"""

import logging
import threading
import time
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.embeddings import get_embedder
from app.core.config import settings
from app.models.result import (
    RESULT_ANALYSIS,
    RESULT_REPORT,
    RESULT_STATUS_PROCESSING,
    RESULT_SUMMARY,
    Result,
)
from app.services.embedding import embed_pending_chunks
from app.services.processing import next_pending_document_id, process_document

logger = logging.getLogger(__name__)

POLL_SECONDS = 5.0
# After the embedding model fails (e.g. offline download), wait this long before
# trying again instead of retrying — and logging — every poll.
EMBEDDING_RETRY_SECONDS = 60.0

SessionFactory = Callable[[], Session]


class BackgroundWorker:
    """A daemon thread that calls `run_pending()` when notified or every poll."""

    name = "background-worker"

    def __init__(self, session_factory: SessionFactory, poll_seconds: float = POLL_SECONDS):
        self._session_factory = session_factory
        self._poll_seconds = poll_seconds
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def notify(self) -> None:
        self._wake.set()

    def run_pending(self) -> int:
        raise NotImplementedError

    def _run(self) -> None:
        logger.info("%s started", self.name)
        while not self._stop.is_set():
            try:
                self.run_pending()
            except Exception:
                # Keep the worker alive (e.g. a locked database); retry at the next poll.
                logger.exception("%s error", self.name)
            self._wake.wait(self._poll_seconds)
            self._wake.clear()
        logger.info("%s stopped", self.name)


class ProcessingWorker(BackgroundWorker):
    name = "document-processing"

    def __init__(
        self,
        session_factory: SessionFactory,
        poll_seconds: float = POLL_SECONDS,
        embeddings_enabled: bool | None = None,
    ):
        super().__init__(session_factory, poll_seconds)
        self._embeddings_enabled = (
            settings.embeddings_enabled if embeddings_enabled is None else embeddings_enabled
        )
        self._embedding_retry_at = 0.0

    def run_pending(self) -> int:
        """Process every waiting document, then embed waiting chunks.

        Returns how many documents were processed.
        """
        processed = self.process_documents()
        if self._embeddings_enabled:
            self.embed_chunks()
        return processed

    def process_documents(self) -> int:
        handled = 0
        while not self._stop.is_set():
            with self._session_factory() as session:
                document_id = next_pending_document_id(session)
                if document_id is None:
                    return handled
                process_document(session, document_id)
            handled += 1
        return handled

    def embed_chunks(self) -> int:
        if time.monotonic() < self._embedding_retry_at:
            return 0
        embedded = 0
        try:
            while not self._stop.is_set():
                # Documents first: a new upload shouldn't wait behind a long backlog.
                if self.process_documents():
                    continue
                with self._session_factory() as session:
                    count = embed_pending_chunks(session, get_embedder())
                if count == 0:
                    break
                embedded += count
        except Exception:
            self._embedding_retry_at = time.monotonic() + EMBEDDING_RETRY_SECONDS
            logger.exception(
                "Embedding failed; chunks stay unembedded, retrying in %ds",
                EMBEDDING_RETRY_SECONDS,
            )
        return embedded


class TranscriptionWorker(BackgroundWorker):
    name = "transcription"

    def run_pending(self) -> int:
        """Transcribe every waiting recording, oldest first. Returns how many ran."""
        from app.services.transcription import next_pending_transcription_id, transcribe_document

        handled = 0
        while not self._stop.is_set():
            with self._session_factory() as session:
                document_id = next_pending_transcription_id(session)
                if document_id is None:
                    return handled
                transcribe_document(session, document_id)
            handled += 1
            notify_worker()  # the new chunks need embedding
        return handled


class GenerationWorker(BackgroundWorker):
    name = "llm-generation"

    def run_pending(self) -> int:
        """Run every waiting summary, analysis and report, oldest request first.

        Returns how many jobs ran.
        """
        # Imported here: these depend on the LLM module, which the processing
        # worker does not need.
        from app.services.analyzer import run_analysis
        from app.services.reports import run_report
        from app.services.summarizer import run_summary

        runners = {
            RESULT_SUMMARY: run_summary,
            RESULT_ANALYSIS: run_analysis,
            RESULT_REPORT: run_report,
        }
        handled = 0
        while not self._stop.is_set():
            with self._session_factory() as session:
                job = session.execute(
                    select(Result.id, Result.result_type)
                    .where(
                        Result.status == RESULT_STATUS_PROCESSING,
                        Result.result_type.in_(runners),
                    )
                    .order_by(Result.created_at)
                    .limit(1)
                ).first()
                if job is None:
                    return handled
                runners[job.result_type](session, job.id)
            handled += 1
        return handled


_processing_worker: ProcessingWorker | None = None
_transcription_worker: TranscriptionWorker | None = None
_generation_worker: GenerationWorker | None = None


def start_workers(session_factory: SessionFactory) -> None:
    global _processing_worker, _transcription_worker, _generation_worker
    _processing_worker = ProcessingWorker(session_factory)
    _processing_worker.start()
    if settings.transcription_enabled:
        _transcription_worker = TranscriptionWorker(session_factory)
        _transcription_worker.start()
    _generation_worker = GenerationWorker(session_factory)
    _generation_worker.start()


def stop_workers() -> None:
    global _processing_worker, _transcription_worker, _generation_worker
    for worker in (_processing_worker, _transcription_worker, _generation_worker):
        if worker is not None:
            worker.stop()
    _processing_worker = None
    _transcription_worker = None
    _generation_worker = None


def notify_worker() -> None:
    """New documents to process (no-op when workers are not running, e.g. in tests).

    Wakes the transcription worker too: an upload or retry may be a recording.
    """
    if _processing_worker is not None:
        _processing_worker.notify()
    if _transcription_worker is not None:
        _transcription_worker.notify()


def notify_generation_worker() -> None:
    """New LLM jobs to run (no-op when workers are not running)."""
    if _generation_worker is not None:
        _generation_worker.notify()
