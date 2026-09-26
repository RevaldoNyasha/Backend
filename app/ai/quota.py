"""Local protection for a metered LLM quota (Groq's free tier).

Every request is checked here *before* it is sent. If it could take usage past
one of our safety limits (set below Groq's real limits in `.env`), it is not
sent at all and `QuotaExceeded` explains which limit and roughly how long to
wait. Nothing relies on Groq answering 429.

How usage is counted
- Each request is a row in `llm_usage`, written before the request and
  reserving `estimated input tokens + maximum output tokens`. Once the reply
  arrives the reservation is replaced by the tokens Groq reports.
- Windows are rolling: the last 60 seconds and the last 24 hours (UTC). Groq
  does not publish when its daily limit resets, so a rolling day is the safe
  assumption; the reported `retryAt` is when *our* window frees enough room.
- After a 429 from Groq, all requests pause until its `retry-after` has
  passed (in memory; a restart clears it).

Token counts before a request are estimates (see `estimate_tokens`): no Qwen
tokenizer is available locally, so the estimate deliberately errs high.
"""

import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.ai.llm import LLMError
from app.core.config import settings
from app.models.llm_usage import (
    USAGE_FAILED,
    USAGE_PENDING,
    USAGE_RATE_LIMITED,
    USAGE_SUCCESS,
    LLMUsage,
)

MINUTE = timedelta(minutes=1)
DAY = timedelta(days=1)
DEFAULT_PAUSE_SECONDS = 60  # after a 429 without a retry-after header
# Background jobs (summaries, analysis) may wait this long for a minute window to free up.
BACKGROUND_WAIT_SECONDS = 65
DAILY_WINDOW_NOTE = (
    "Rolling 24-hour window tracked by ResearchMind. Groq does not publish its "
    "exact daily reset time."
)

# ---------------------------------------------------------------- token estimates

CHARS_PER_TOKEN = 3  # English averages ~4 characters per token; 3 over-counts on purpose
MESSAGE_OVERHEAD_TOKENS = 8  # role markers and separators per chat message
REQUEST_OVERHEAD_TOKENS = 16


def estimate_tokens(text: str) -> int:
    """A conservative token count for `text` (an estimate, not a tokenizer).

    ASCII text counts one token per 3 characters (real tokenizers average about
    4); every other character (accents, non-Latin scripts) counts as a whole
    token, since tokenizers split those finely.
    """
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    return math.ceil(ascii_chars / CHARS_PER_TOKEN) + (len(text) - ascii_chars)


def estimate_chat_tokens(system: str, prompt: str) -> int:
    """Input tokens for one chat request.

    `prompt` is the whole user message the application builds: conversation
    history, retrieved passages and the question all live inside it.
    """
    return (
        estimate_tokens(system)
        + estimate_tokens(prompt)
        + 2 * MESSAGE_OVERHEAD_TOKENS
        + REQUEST_OVERHEAD_TOKENS
    )


# ---------------------------------------------------------------- limits and snapshots


@dataclass(frozen=True)
class QuotaLimits:
    minute_requests: int
    minute_tokens: int
    daily_requests: int
    daily_tokens: int
    minute_warning_percent: int = 80
    daily_warning_percent: int = 80

    @classmethod
    def from_settings(cls) -> "QuotaLimits":
        return cls(
            minute_requests=settings.groq_minute_request_limit,
            minute_tokens=settings.groq_minute_token_limit,
            daily_requests=settings.groq_daily_request_limit,
            daily_tokens=settings.groq_daily_token_limit,
            minute_warning_percent=settings.groq_minute_warning_percent,
            daily_warning_percent=settings.groq_daily_warning_percent,
        )


@dataclass(frozen=True)
class Block:
    reason: str
    message: str
    retry_at: datetime | None

    def retry_after_seconds(self, now: datetime) -> int | None:
        if self.retry_at is None:
            return None
        return max(1, math.ceil((self.retry_at - now).total_seconds()))


@dataclass(frozen=True)
class UsageSnapshot:
    provider: str
    model: str
    limits: QuotaLimits
    minute_requests: int
    minute_tokens: int
    day_requests: int
    day_tokens: int
    now: datetime
    block: Block | None = None
    groq_reported: dict = field(default_factory=dict)

    def _percent(self, used: int, limit: int) -> float:
        return 100.0 * used / limit if limit > 0 else 100.0

    @property
    def daily_percent(self) -> float:
        return max(
            self._percent(self.day_requests, self.limits.daily_requests),
            self._percent(self.day_tokens, self.limits.daily_tokens),
        )

    @property
    def minute_percent(self) -> float:
        return max(
            self._percent(self.minute_requests, self.limits.minute_requests),
            self._percent(self.minute_tokens, self.limits.minute_tokens),
        )

    @property
    def warning_message(self) -> str | None:
        if self.daily_percent >= self.limits.daily_warning_percent:
            return (
                "Groq usage is getting close to the daily development limit. You have used "
                f"approximately {round(self.daily_percent)}% of the configured daily allowance."
            )
        if self.minute_percent >= self.limits.minute_warning_percent:
            return (
                "Groq usage is close to the per-minute development limit "
                f"(about {round(self.minute_percent)}%). Requests may pause briefly."
            )
        return None

    def to_dict(self) -> dict:
        """The development usage report (camelCase, safe for the frontend)."""
        warning = self.warning_message
        block = self.block
        return {
            "provider": self.provider,
            "model": self.model,
            "requests": {
                "minute": self.minute_requests,
                "minuteLimit": self.limits.minute_requests,
                "today": self.day_requests,
                "dailyLimit": self.limits.daily_requests,
            },
            "tokens": {
                "minute": self.minute_tokens,
                "minuteLimit": self.limits.minute_tokens,
                "today": self.day_tokens,
                "dailyLimit": self.limits.daily_tokens,
            },
            "dailyPercent": round(self.daily_percent, 1),
            "minutePercent": round(self.minute_percent, 1),
            "warning": warning is not None,
            "warningMessage": warning,
            "blocked": block is not None,
            "reason": block.reason if block else None,
            "message": block.message if block else None,
            "retryAfter": block.retry_after_seconds(self.now) if block else None,
            "retryAt": _iso(block.retry_at) if block and block.retry_at else None,
            "window": {"minute": "rolling 60 seconds", "daily": DAILY_WINDOW_NOTE},
            "groqReported": self.groq_reported,
        }


class QuotaExceeded(LLMError):
    """A request was not sent: it would pass a safety limit, or Groq said 429."""

    def __init__(self, block: Block, snapshot: UsageSnapshot) -> None:
        super().__init__(block.message)
        self.block = block
        self.snapshot = snapshot

    @property
    def reason(self) -> str:
        return self.block.reason

    def retry_after_seconds(self) -> int | None:
        return self.block.retry_after_seconds(self.snapshot.now)

    def to_dict(self) -> dict:
        return {
            "blocked": True,
            "reason": self.block.reason,
            "message": self.block.message,
            "retryAfter": self.retry_after_seconds(),
            "retryAt": _iso(self.block.retry_at) if self.block.retry_at else None,
            "usage": self.snapshot.to_dict(),
        }


# ---------------------------------------------------------------- waiting (background jobs)

_wait_budget: ContextVar[float] = ContextVar("quota_wait_budget", default=0.0)


@contextmanager
def allow_quota_wait(max_seconds_per_request: float) -> Iterator[None]:
    """Let requests in this block wait out a *per-minute* limit instead of failing.

    For background jobs (summaries, analysis) that make several requests in a
    row. Daily limits never wait; a request that is simply too large never waits.
    """
    token = _wait_budget.set(max_seconds_per_request)
    try:
        yield
    finally:
        _wait_budget.reset(token)


# ---------------------------------------------------------------- the guard

MINUTE_REASONS = ("minute_request_limit", "minute_token_limit", "groq_rate_limited")


class QuotaGuard:
    """Checks, reserves and records requests to one metered provider/model."""

    def __init__(
        self,
        session_factory: Callable[[], sessionmaker[Session]],
        provider: str,
        model: str,
        limits: QuotaLimits | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._session_factory = session_factory
        self.provider = provider
        self.model = model
        self._limits = limits
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._paused_until: datetime | None = None
        self._groq_reported: dict = {}

    @property
    def limits(self) -> QuotaLimits:
        return self._limits or QuotaLimits.from_settings()

    # -- reading

    def snapshot(self) -> UsageSnapshot:
        with self._session() as db:
            return self._snapshot(db, self._clock(), extra_tokens=0)

    # -- before a request

    def reserve(self, estimated_input_tokens: int, max_output_tokens: int, streamed: bool) -> str:
        """Record the request as pending and return its id, or raise `QuotaExceeded`."""
        needed = estimated_input_tokens + max_output_tokens
        while True:
            with self._lock, self._session() as db:
                now = self._clock()
                snapshot = self._snapshot(db, now, extra_tokens=needed)
                if snapshot.block is None:
                    row = LLMUsage(
                        created_at=now,
                        provider=self.provider,
                        model=self.model,
                        status=USAGE_PENDING,
                        streamed=streamed,
                        estimated_input_tokens=estimated_input_tokens,
                        max_output_tokens=max_output_tokens,
                        total_tokens=needed,
                    )
                    db.add(row)
                    db.commit()
                    return row.id
            block = snapshot.block
            wait = block.retry_after_seconds(now) if block.reason in MINUTE_REASONS else None
            budget = _wait_budget.get()
            if wait is None or wait > budget:
                raise QuotaExceeded(block, snapshot)
            self._sleep(wait)

    # -- after a request

    def complete(
        self,
        usage_id: str,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error: str | None = None,
        rate_limited: bool = False,
    ) -> None:
        with self._session() as db:
            row = db.get(LLMUsage, usage_id)
            if row is None:
                return
            row.completed_at = self._clock()
            row.input_tokens = input_tokens
            row.output_tokens = output_tokens
            if rate_limited:
                row.status = USAGE_RATE_LIMITED
                row.total_tokens = 0  # Groq refused it before doing any work
            elif error is not None:
                row.status = USAGE_FAILED
                reported = (input_tokens or 0) + (output_tokens or 0)
                row.total_tokens = reported or row.estimated_input_tokens
            else:
                row.status = USAGE_SUCCESS
                if input_tokens is not None or output_tokens is not None:
                    row.total_tokens = (input_tokens or 0) + (output_tokens or 0)
                else:
                    row.total_tokens = row.estimated_input_tokens + row.max_output_tokens
            row.error = error[:300] if error else None
            db.commit()

    def pause(self, seconds: float | None) -> datetime:
        """Stop all requests for a while (Groq answered 429). Returns when it ends."""
        until = self._clock() + timedelta(seconds=seconds or DEFAULT_PAUSE_SECONDS)
        with self._lock:
            if self._paused_until is None or until > self._paused_until:
                self._paused_until = until
            return self._paused_until

    def record_headers(self, headers: dict[str, str]) -> None:
        """Keep Groq's own view of the limits (RPD and TPM remaining) for the report."""
        names = {
            "x-ratelimit-limit-requests": "dailyRequestLimit",
            "x-ratelimit-remaining-requests": "dailyRequestsRemaining",
            "x-ratelimit-reset-requests": "dailyRequestsReset",
            "x-ratelimit-limit-tokens": "minuteTokenLimit",
            "x-ratelimit-remaining-tokens": "minuteTokensRemaining",
            "x-ratelimit-reset-tokens": "minuteTokensReset",
        }
        reported = {key: headers[name] for name, key in names.items() if name in headers}
        if reported:
            self._groq_reported = {**reported, "at": _iso(self._clock())}

    # -- internals

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with self._session_factory()() as db:
            yield db

    def _snapshot(self, db: Session, now: datetime, extra_tokens: int) -> UsageSnapshot:
        limits = self.limits
        rows = [
            (_aware(created), tokens)
            for created, tokens in db.execute(
                select(LLMUsage.created_at, LLMUsage.total_tokens)
                .where(
                    LLMUsage.provider == self.provider,
                    LLMUsage.model == self.model,
                    LLMUsage.created_at > now - DAY,
                )
                .order_by(LLMUsage.created_at)
            ).all()
        ]
        minute_rows = [(created, tokens) for created, tokens in rows if created > now - MINUTE]
        snapshot = UsageSnapshot(
            provider=self.provider,
            model=self.model,
            limits=limits,
            minute_requests=len(minute_rows),
            minute_tokens=sum(tokens for _, tokens in minute_rows),
            day_requests=len(rows),
            day_tokens=sum(tokens for _, tokens in rows),
            now=now,
            groq_reported=dict(self._groq_reported),
        )
        block = self._block_for(snapshot, rows, minute_rows, extra_tokens)
        if block is None:
            return snapshot
        return UsageSnapshot(**{**snapshot.__dict__, "block": block})

    def _block_for(
        self,
        s: UsageSnapshot,
        day_rows: list[tuple[datetime, int]],
        minute_rows: list[tuple[datetime, int]],
        extra_tokens: int,
    ) -> Block | None:
        limits, now = s.limits, s.now
        # A plain status check (`extra_tokens=0`) asks whether a minimal request fits.
        tokens = max(extra_tokens, 1)

        if tokens > limits.minute_tokens or tokens > limits.daily_tokens:
            return Block(
                "request_too_large",
                f"This request needs about {tokens:,} tokens, more than the development "
                f"limit of {min(limits.minute_tokens, limits.daily_tokens):,} tokens per "
                "window, so it was not sent to Groq. Try a shorter question or document.",
                None,
            )

        # Daily limits first: they are the ones that cannot be waited out quickly.
        if s.day_requests + 1 > limits.daily_requests:
            return Block(
                "daily_request_limit",
                "Development Groq limit reached: the daily request allowance "
                f"({limits.daily_requests:,}) is used up. Please wait until the daily "
                "quota refreshes.",
                _free_at([(t, 1) for t, _ in day_rows], 1, limits.daily_requests, DAY),
            )
        if s.day_tokens + tokens > limits.daily_tokens:
            return Block(
                "daily_token_limit",
                "Development Groq limit reached: the daily token allowance "
                f"({limits.daily_tokens:,}) would be exceeded. Please wait until the daily "
                "quota refreshes.",
                _free_at(day_rows, tokens, limits.daily_tokens, DAY),
            )

        paused = self._paused_until
        if paused is not None and paused > now:
            return Block(
                "groq_rate_limited",
                "Groq rate limit reached. ResearchMind has paused requests to protect your "
                "development quota. Please wait before trying again.",
                paused,
            )

        if s.minute_requests + 1 > limits.minute_requests:
            return Block(
                "minute_request_limit",
                "Groq usage limit reached for development: too many requests this minute. "
                "Please wait until the minute window resets before trying again.",
                _free_at([(t, 1) for t, _ in minute_rows], 1, limits.minute_requests, MINUTE),
            )
        if s.minute_tokens + tokens > limits.minute_tokens:
            return Block(
                "minute_token_limit",
                "Groq usage limit reached for development: the per-minute token allowance "
                "would be exceeded. Please wait until the minute window resets before trying "
                "again.",
                _free_at(minute_rows, tokens, limits.minute_tokens, MINUTE),
            )
        return None


def _free_at(
    rows: list[tuple[datetime, int]], needed: int, limit: int, window: timedelta
) -> datetime | None:
    """When enough of the oldest usage leaves the rolling window for `needed` more."""
    excess = sum(amount for _, amount in rows) + needed - limit
    freed = 0
    for created, amount in rows:  # oldest first
        freed += amount
        if freed >= excess:
            return created + window
    return None


def _aware(value: datetime) -> datetime:
    # SQLite returns naive datetimes; everything here is UTC.
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
