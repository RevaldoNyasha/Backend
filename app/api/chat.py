import json
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.ai.embeddings import get_embedder
from app.ai.llm import LLMError, LLMUnavailable, get_llm
from app.ai.quota import QuotaExceeded
from app.api.conversations import to_message
from app.api.usage import llm_usage_report
from app.core.dependencies import CurrentProject, DbSession
from app.schemas.conversation import ChatRequest, ChatResponse
from app.services.assistant import (
    ConversationNotFound,
    PreparedAnswer,
    answer_question,
    finish_answer,
    prepare_answer,
    stream_answer,
)
from app.services.retrieval import SearchUnavailable

router = APIRouter(tags=["assistant"])


@router.post("/projects/{project_id}/chat", response_model=ChatResponse)
def chat(
    project_id: str,
    payload: ChatRequest,
    current_project: CurrentProject,
    db: DbSession,
) -> ChatResponse:
    """Answer a question from this project's documents (contract §7.4).

    Returns the assistant's message once it is complete; its `conversationId` is
    the conversation the exchange was saved in (a new one when none was given).
    See `/chat/stream` for the word-by-word version. `usage` is the Groq
    development usage after this answer (null with Ollama). When a development
    limit would be passed, nothing is sent to Groq: `429` with the details.
    """
    try:
        reply = answer_question(
            db,
            current_project,
            payload.question,
            llm=get_llm(),
            embedder=get_embedder(),
            conversation_id=payload.conversation_id,
        )
    except (ConversationNotFound, SearchUnavailable, LLMError) as error:
        raise _http_error(error) from None
    return ChatResponse(**to_message(reply.answer).model_dump(), usage=llm_usage_report())


@router.post("/projects/{project_id}/chat/stream")
def chat_stream(
    project_id: str,
    payload: ChatRequest,
    current_project: CurrentProject,
    db: DbSession,
) -> StreamingResponse:
    """Stream the answer as Server-Sent Events while the model writes it.

    Events (each `data:` is JSON):
    - `token` `{"text": "…"}`: the next piece of the answer.
    - `done` `{"message": ChatMessage, "usage": …}`: the saved answer, with
      `evidence` and `conversationId`, and the Groq usage. Only now is the
      exchange stored.
    - `error` `{"detail": "…"}`: generation failed midway; nothing was saved.

    Problems found before streaming starts (unknown conversation, search or model
    unavailable, a Groq development limit) are ordinary HTTP errors, as for
    `/chat`: the quota is checked before the Groq stream is opened.
    """
    llm = get_llm()
    try:
        prepared = prepare_answer(
            db, current_project, payload.question, get_embedder(), payload.conversation_id
        )
        pieces = stream_answer(llm, prepared)
        # Wait for the first piece here, so a stopped Ollama is a clean HTTP error.
        first = next(pieces)
    except (ConversationNotFound, SearchUnavailable, LLMError) as error:
        raise _http_error(error) from None
    except StopIteration:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The language model returned an empty response.",
        ) from None

    # The request's session may be closed while the response streams, so the
    # answer is saved in a fresh session on the same database.
    bind = db.get_bind()
    return StreamingResponse(
        _events(prepared, first, pieces, bind),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _events(prepared: PreparedAnswer, first: str, pieces: Iterator[str], bind) -> Iterator[str]:
    parts = [first]
    yield _sse("token", {"text": first})
    try:
        for piece in pieces:
            parts.append(piece)
            yield _sse("token", {"text": piece})
    except QuotaExceeded as error:  # Groq answered 429 midway
        yield _sse("error", {"detail": str(error), **error.to_dict()})
        return
    except LLMError as error:
        yield _sse("error", {"detail": str(error)})
        return

    with Session(bind=bind, expire_on_commit=False) as session:
        try:
            reply = finish_answer(session, prepared, "".join(parts))
        except ConversationNotFound:
            yield _sse("error", {"detail": "The conversation was deleted while answering."})
            return
        message = to_message(reply.answer).model_dump(mode="json", by_alias=True)
    yield _sse("done", {"message": message, "usage": llm_usage_report()})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, QuotaExceeded):
        retry_after = error.retry_after_seconds()
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=error.to_dict(),
            headers={"Retry-After": str(retry_after)} if retry_after else None,
        )
    if isinstance(error, ConversationNotFound):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    if isinstance(error, SearchUnavailable):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The assistant is unavailable: the embedding model could not be loaded",
        )
    if isinstance(error, LLMUnavailable):
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error))
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error))
