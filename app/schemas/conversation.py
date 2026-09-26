from pydantic import ConfigDict, Field

from app.schemas.base import Schema, UtcDateTime


class ConversationCreate(Schema):
    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    # Optional: an untitled conversation is named after its first question.
    title: str | None = Field(default=None, max_length=200)


class ConversationUpdate(Schema):
    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    title: str = Field(min_length=1, max_length=200)


class ChatRequest(Schema):
    """Contract §7.4. `projectId` is accepted for the frontend's sake; the path wins."""

    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    question: str = Field(min_length=1, max_length=2000)
    # Continue this conversation; omit to start a new one.
    conversation_id: str | None = None
    project_id: str | None = None


class ConversationResponse(Schema):
    """Contract §7.2 `Conversation`, plus `createdAt`."""

    id: str
    project_id: str
    title: str
    preview: str
    message_count: int
    created_at: UtcDateTime
    updated_at: UtcDateTime


class MessageResponse(Schema):
    """Contract §7.1 `ChatMessage`, plus `conversationId`."""

    id: str
    role: str
    content: str
    created_at: UtcDateTime
    project_id: str
    conversation_id: str
    # Evidence[] for assistant answers; null for questions.
    evidence: list[dict] | None = None


class ChatResponse(MessageResponse):
    """The assistant's answer, plus the Groq development usage (null with Ollama)."""

    usage: dict | None = None
