from fastapi import APIRouter

from app.core.config import settings
from app.core.dependencies import CurrentUser

router = APIRouter(tags=["usage"])


def llm_usage_report() -> dict | None:
    """The Groq development usage report, or None when the LLM is not metered (Ollama)."""
    if settings.llm_provider != "groq":
        return None
    from app.ai.groq import get_quota_guard

    return {"metered": True, **get_quota_guard().snapshot().to_dict()}


@router.get("/usage/groq")
def groq_usage(_: CurrentUser) -> dict:
    """How much of the development Groq allowance is used (our safety limits, not Groq's).

    Counts come from our own records of every request; nothing is sent to Groq to
    produce this. The API key is never included.
    """
    report = llm_usage_report()
    if report is None:
        return {"provider": settings.llm_provider, "metered": False, "model": settings.ollama_model}
    return report
