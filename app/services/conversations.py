"""A project's conversations with the assistant and their messages.

Every function takes the project (already checked against the user by the
`CurrentProject` dependency) and filters by it, so conversations never cross
projects.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.conversation import ROLE_USER, Conversation, Message
from app.models.project import Project

DEFAULT_TITLE = "New chat"
TITLE_MAX_LENGTH = 80
PREVIEW_MAX_LENGTH = 160


@dataclass(frozen=True)
class ConversationSummary:
    conversation: Conversation
    preview: str
    message_count: int


def list_conversations(db: Session, project: Project) -> list[ConversationSummary]:
    """Newest activity first, with each conversation's first question as a preview."""
    conversations = list(
        db.scalars(
            select(Conversation)
            .where(Conversation.project_id == project.id)
            .order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
        )
    )
    if not conversations:
        return []
    ids = [conversation.id for conversation in conversations]

    counts = dict(
        db.execute(
            select(Message.conversation_id, func.count(Message.id))
            .where(Message.conversation_id.in_(ids))
            .group_by(Message.conversation_id)
        ).all()
    )
    first_question_at = (
        select(Message.conversation_id, func.min(Message.created_at).label("first_at"))
        .where(Message.conversation_id.in_(ids), Message.role == ROLE_USER)
        .group_by(Message.conversation_id)
        .subquery()
    )
    previews = dict(
        db.execute(
            select(Message.conversation_id, Message.content).join(
                first_question_at,
                (Message.conversation_id == first_question_at.c.conversation_id)
                & (Message.created_at == first_question_at.c.first_at),
            )
        ).all()
    )
    return [
        ConversationSummary(
            conversation=conversation,
            preview=_shorten(previews.get(conversation.id, ""), PREVIEW_MAX_LENGTH),
            message_count=counts.get(conversation.id, 0),
        )
        for conversation in conversations
    ]


def summarize(db: Session, conversation: Conversation) -> ConversationSummary:
    first_question = db.scalar(
        select(Message.content)
        .where(Message.conversation_id == conversation.id, Message.role == ROLE_USER)
        .order_by(Message.created_at)
        .limit(1)
    )
    count = db.scalar(
        select(func.count()).select_from(Message).where(Message.conversation_id == conversation.id)
    )
    return ConversationSummary(
        conversation=conversation,
        preview=_shorten(first_question or "", PREVIEW_MAX_LENGTH),
        message_count=count or 0,
    )


def create_conversation(db: Session, project: Project, title: str | None = None) -> Conversation:
    conversation = Conversation(
        project_id=project.id,
        title=_shorten(title.strip(), TITLE_MAX_LENGTH) if title and title.strip() else DEFAULT_TITLE,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def rename_conversation(db: Session, conversation: Conversation, title: str) -> Conversation:
    conversation.title = _shorten(title.strip(), TITLE_MAX_LENGTH) or DEFAULT_TITLE
    db.commit()
    db.refresh(conversation)
    return conversation


def delete_conversation(db: Session, conversation: Conversation) -> None:
    db.delete(conversation)  # messages go with it (ON DELETE CASCADE)
    db.commit()


def list_messages(db: Session, conversation: Conversation) -> list[Message]:
    """Oldest first, the order the chat window shows them in."""
    return list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at, Message.id)
        )
    )


def latest_conversation(db: Session, project: Project) -> Conversation | None:
    return db.scalar(
        select(Conversation)
        .where(Conversation.project_id == project.id)
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
        .limit(1)
    )


def add_message(
    db: Session,
    conversation: Conversation,
    role: str,
    content: str,
    evidence: list[dict] | None = None,
) -> Message:
    """Append a message, bump the conversation, and title it after its first question."""
    message = Message(
        conversation_id=conversation.id,
        project_id=conversation.project_id,
        role=role,
        content=content,
        evidence=evidence,
    )
    db.add(message)
    if role == ROLE_USER and conversation.title == DEFAULT_TITLE:
        conversation.title = _shorten(" ".join(content.split()), TITLE_MAX_LENGTH)
    conversation.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(message)
    return message


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
