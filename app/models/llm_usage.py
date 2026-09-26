from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.utils.ids import new_id

USAGE_PENDING = "pending"  # sent (or about to be); tokens are the reservation
USAGE_SUCCESS = "success"
USAGE_FAILED = "failed"
USAGE_RATE_LIMITED = "rate_limited"  # Groq answered 429


class LLMUsage(Base):
    """One request to a metered LLM provider (Groq), for our own rate limiting.

    A row is written *before* the request, reserving its estimated tokens (input
    estimate + the maximum output), so concurrent requests see each other. When
    the reply arrives `total_tokens` becomes the provider's reported usage.
    """

    __tablename__ = "llm_usage"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        index=True,
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    provider: Mapped[str] = mapped_column(String(20), nullable=False)
    model: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=USAGE_PENDING)
    streamed: Mapped[bool] = mapped_column(nullable=False, default=False)
    estimated_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    max_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    # Reported by the provider when available.
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # What this request counts against the limits: the reservation until the
    # reply arrives, then the reported total (or the input estimate on failure).
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    error: Mapped[str | None] = mapped_column(String(300), nullable=True)
