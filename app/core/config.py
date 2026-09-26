from pathlib import Path

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
UPLOADS_ROOT = BACKEND_ROOT / "uploads"
BYTES_PER_MB = 1024 * 1024

DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Research Assistant API"
    api_prefix: str = "/api/v1"
    environment: str = "development"
    debug: bool = False

    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    database_url: str = ""

    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15

    # Where the frontend is served. Google / GitHub send the user back to
    # `{frontend_url}/auth/callback/{provider}`, which must be registered with them.
    frontend_url: str = "http://localhost:5173"
    # The deployed frontend (e.g. https://researchmind.pages.dev), used alongside the
    # local one: allowed by CORS and as a sign-in return address. Blank = local only.
    hosted_frontend_url: str = ""
    # Sign in with Google: an OAuth client from Google Cloud Console (blank = off).
    google_client_id: str = ""
    google_client_secret: str = ""
    # Sign in with GitHub: an OAuth App from GitHub Developer settings (blank = off).
    github_client_id: str = ""
    github_client_secret: str = ""
    # A second GitHub OAuth App for the hosted frontend (GitHub allows one callback
    # URL per app). Google needs no second client: add the redirect URI instead.
    github_hosted_client_id: str = ""
    github_hosted_client_secret: str = ""

    # Serve the interactive API docs at /docs. Turn off when the API is public.
    api_docs_enabled: bool = True

    storage_endpoint: str = ""
    storage_bucket: str = ""
    storage_access_key: str = ""
    storage_secret_key: str = ""

    # Which LLM answers questions and writes summaries: "groq" (hosted) or "ollama" (local).
    llm_provider: str = "groq"

    groq_api_key: str = ""
    groq_model: str = DEFAULT_GROQ_MODEL
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Qwen 3.8 thinks before answering unless told not to; thinking tokens count
    # against the quota. "none" turns it off; blank sends nothing.
    groq_reasoning_effort: str = "none"
    # Our own limits, deliberately below Groq's free tier (30 RPM, 1,000 RPD,
    # 8,000 TPM, 200,000 TPD) so development never uses the last of the quota.
    groq_minute_request_limit: int = 25
    groq_minute_token_limit: int = 7000
    groq_daily_request_limit: int = 900
    groq_daily_token_limit: int = 180000
    groq_minute_warning_percent: int = 80
    groq_daily_warning_percent: int = 80
    # Upper bound on generated tokens for any Groq call (answers, summaries, analysis).
    groq_max_output_tokens: int = 700

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = DEFAULT_OLLAMA_MODEL

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    # A 30-60 minute interview recording is roughly 30-60 MB.
    upload_max_audio_size_mb: int = 100
    upload_max_video_size_mb: int = 100

    # Audio/video transcription with faster-whisper (runs locally, no quota).
    transcription_enabled: bool = True
    # tiny | base | small | medium | large-v3; downloaded on first use (small ~ 480 MB).
    whisper_model: str = "small"
    # auto: the GPU when its CUDA libraries load, otherwise the CPU.
    whisper_device: str = "auto"
    # Blank: int8 on the CPU, int8_float16 on the GPU.
    whisper_compute_type: str = ""
    # Blank: detect the spoken language. A code (e.g. "en", "sn") skips detection.
    whisper_language: str = ""
    upload_max_document_size_mb: int = 5

    # Background thread that extracts, chunks and embeds uploaded documents.
    processing_worker_enabled: bool = True
    # Embed chunks with `embedding_model` (downloads ~2.3 GB on first use).
    embeddings_enabled: bool = True
    # Search results scoring below this cosine similarity are dropped. Calibrated
    # on BGE-M3 (2026-09-26): on-topic questions scored 0.44-0.69 (short, specific
    # questions against a whole passage score lowest), off-topic 0.24-0.38.
    # 0.45 wrongly rejected "Why did P03 plant less?" (0.443). Revisit as real
    # research data arrives.
    retrieval_min_score: float = 0.40

    project_name_max_length: int = 200
    project_description_max_length: int = 2000

    @field_validator("ollama_model", "groq_model", "embedding_model", mode="after")
    @classmethod
    def _blank_means_default(cls, value: str, info: ValidationInfo) -> str:
        # `.env` files often carry `OLLAMA_MODEL=` with no value.
        if value.strip():
            return value.strip()
        return {
            "ollama_model": DEFAULT_OLLAMA_MODEL,
            "groq_model": DEFAULT_GROQ_MODEL,
        }.get(info.field_name, DEFAULT_EMBEDDING_MODEL)

    @field_validator("llm_provider", mode="after")
    @classmethod
    def _known_provider(cls, value: str) -> str:
        provider = value.strip().lower() or "groq"
        if provider not in ("groq", "ollama"):
            raise ValueError("LLM_PROVIDER must be 'groq' or 'ollama'")
        return provider


settings = Settings()
