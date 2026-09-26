from fastapi import APIRouter

from app.ai.embeddings import embedder_status
from app.ai.llm import ollama_status
from app.ai.whisper import transcriber_status
from app.core.config import settings
from app.core.dependencies import DbSession
from app.services.embedding import pending_embedding_count
from app.services.transcription import pending_transcription_count

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ai")
def ai_health(db: DbSession) -> dict[str, object]:
    """Readiness of the local AI stack. Never loads a model; it only reports.

    `embeddings.status` is `not_loaded` until the first chunk is embedded (the
    model loads on first use), then `loading`, `ready` or `error`.
    """
    if settings.llm_provider == "ollama":
        llm = {"provider": "ollama", **ollama_status()}
    else:
        # Never calls Groq: that would spend the quota just to report status.
        configured = bool(settings.groq_api_key.strip())
        llm = {
            "provider": "groq",
            "model": settings.groq_model,
            "configured": configured,
            "modelAvailable": configured,
            "error": None if configured else "GROQ_API_KEY is not set in Backend/.env",
        }
    embeddings = {
        **embedder_status(),
        "enabled": settings.embeddings_enabled,
        "pendingChunks": pending_embedding_count(db, settings.embedding_model),
    }
    transcription = {
        **transcriber_status(),
        "pendingRecordings": pending_transcription_count(db),
    }
    ready = (
        bool(llm["modelAvailable"])
        and embeddings["status"] != "error"
        and transcription["status"] != "error"
    )
    return {
        "status": "ok" if ready else "degraded",
        "llm": llm,
        "embeddings": embeddings,
        "transcription": transcription,
    }
