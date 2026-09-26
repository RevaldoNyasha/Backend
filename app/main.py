import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analysis import router as analysis_router
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.projects import router as projects_router
from app.api.reports import router as reports_router
from app.api.search import router as search_router
from app.api.summaries import router as summaries_router
from app.api.usage import router as usage_router
from app.api.users import router as users_router
from app.core.config import settings
from app.db.database import get_session_factory
from app.db.migrations import upgrade_database
from app.services.processing_worker import start_workers, stop_workers

logger = logging.getLogger(__name__)

MIN_SECRET_LENGTH = 32


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    environment = settings.environment.lower()
    if environment != "testing" and len(settings.jwt_secret_key) < MIN_SECRET_LENGTH:
        logger.warning(
            "JWT_SECRET_KEY is only %d characters: anyone who guesses it can sign in as any "
            "user. Set a long random value before exposing the API (see DEPLOYMENT.md).",
            len(settings.jwt_secret_key),
        )
    # Development applies migrations automatically; elsewhere run `alembic upgrade head`.
    if environment == "development" and settings.database_url:
        upgrade_database(settings.database_url)

    # Tests drive processing directly, so the thread only runs outside them.
    run_worker = (
        settings.processing_worker_enabled and environment != "testing" and settings.database_url
    )
    if run_worker:
        start_workers(get_session_factory())
    try:
        yield
    finally:
        if run_worker:
            stop_workers()


def allowed_origins() -> list[str]:
    """`CORS_ORIGINS`, plus the local and hosted frontends (so they always work)."""
    frontends = [settings.frontend_url, settings.hosted_frontend_url]
    origins = [*settings.cors_origins, *frontends]
    return [origin.strip().rstrip("/") for origin in dict.fromkeys(origins) if origin.strip()]


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
    # With the API public (Tailscale Funnel), the docs can be switched off.
    docs_url="/docs" if settings.api_docs_enabled else None,
    redoc_url="/redoc" if settings.api_docs_enabled else None,
    openapi_url="/openapi.json" if settings.api_docs_enabled else None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)  # unprefixed: /health, /health/ai
app.include_router(auth_router, prefix=settings.api_prefix)
app.include_router(projects_router, prefix=settings.api_prefix)
app.include_router(documents_router, prefix=settings.api_prefix)
app.include_router(search_router, prefix=settings.api_prefix)
app.include_router(summaries_router, prefix=settings.api_prefix)
app.include_router(conversations_router, prefix=settings.api_prefix)
app.include_router(chat_router, prefix=settings.api_prefix)
app.include_router(analysis_router, prefix=settings.api_prefix)
app.include_router(reports_router, prefix=settings.api_prefix)
app.include_router(usage_router, prefix=settings.api_prefix)
app.include_router(users_router, prefix=settings.api_prefix)
