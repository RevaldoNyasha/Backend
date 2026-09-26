from fastapi import APIRouter, status

from app.core.dependencies import CurrentConversation, CurrentProject, DbSession
from app.models.conversation import Message
from app.schemas.conversation import (
    ConversationCreate,
    ConversationResponse,
    ConversationUpdate,
    MessageResponse,
)
from app.services.conversations import (
    ConversationSummary,
    create_conversation,
    delete_conversation,
    latest_conversation,
    list_conversations,
    list_messages,
    rename_conversation,
    summarize,
)

router = APIRouter(tags=["conversations"])

CONVERSATIONS = "/projects/{project_id}/conversations"
CONVERSATION = CONVERSATIONS + "/{conversation_id}"


def to_response(summary: ConversationSummary) -> ConversationResponse:
    conversation = summary.conversation
    return ConversationResponse(
        id=conversation.id,
        project_id=conversation.project_id,
        title=conversation.title,
        preview=summary.preview,
        message_count=summary.message_count,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def to_message(message: Message) -> MessageResponse:
    return MessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        project_id=message.project_id,
        conversation_id=message.conversation_id,
        evidence=message.evidence,
    )


@router.get(CONVERSATIONS, response_model=list[ConversationResponse])
def index(project_id: str, current_project: CurrentProject, db: DbSession):
    """This project's conversations, most recently active first."""
    return [to_response(summary) for summary in list_conversations(db, current_project)]


@router.post(CONVERSATIONS, response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create(
    project_id: str,
    current_project: CurrentProject,
    db: DbSession,
    payload: ConversationCreate | None = None,
):
    conversation = create_conversation(db, current_project, payload.title if payload else None)
    return to_response(summarize(db, conversation))


@router.patch(CONVERSATION, response_model=ConversationResponse)
def rename(
    project_id: str,
    conversation_id: str,
    payload: ConversationUpdate,
    current_conversation: CurrentConversation,
    db: DbSession,
):
    conversation = rename_conversation(db, current_conversation, payload.title)
    return to_response(summarize(db, conversation))


@router.delete(CONVERSATION, status_code=status.HTTP_204_NO_CONTENT)
def destroy(
    project_id: str,
    conversation_id: str,
    current_conversation: CurrentConversation,
    db: DbSession,
) -> None:
    delete_conversation(db, current_conversation)


@router.get(CONVERSATION + "/messages", response_model=list[MessageResponse])
def messages(
    project_id: str,
    conversation_id: str,
    current_conversation: CurrentConversation,
    db: DbSession,
):
    """The conversation's messages, oldest first."""
    return [to_message(message) for message in list_messages(db, current_conversation)]


@router.get("/projects/{project_id}/messages", response_model=list[MessageResponse])
def project_messages(project_id: str, current_project: CurrentProject, db: DbSession):
    """Contract §7.3: the messages of the project's most recent conversation."""
    conversation = latest_conversation(db, current_project)
    if conversation is None:
        return []
    return [to_message(message) for message in list_messages(db, conversation)]
