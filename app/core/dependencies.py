from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.database import get_db
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.project import Project
from app.models.user import User
from app.services.auth import is_token_revoked

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AccessToken:
    """The validated claims of the bearer token on the current request."""

    jti: str
    user_id: str
    expires_at: datetime


def get_access_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> AccessToken:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_access_token(credentials.credentials)
        token = AccessToken(
            jti=str(payload["jti"]),
            user_id=str(payload["sub"]),
            expires_at=datetime.fromtimestamp(int(payload["exp"]), UTC),
        )
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    if is_token_revoked(db, token.jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token


def get_current_user(
    token: Annotated[AccessToken, Depends(get_access_token)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    user = db.get(User, token.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


DbSession = Annotated[Session, Depends(get_db)]
CurrentToken = Annotated[AccessToken, Depends(get_access_token)]
CurrentUser = Annotated[User, Depends(get_current_user)]


def get_current_project(
    project_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> Project:
    project = db.scalar(
        select(Project).where(
            Project.id == project_id,
            Project.user_id == current_user.id,
        )
    )
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    return project


def get_current_document(
    project_id: str,
    document_id: str,
    db: DbSession,
    current_user: CurrentUser,
) -> Document:
    document = db.scalar(
        select(Document)
        .join(Project, Project.id == Document.project_id)
        .where(
            Document.id == document_id,
            Project.id == project_id,
            Project.user_id == current_user.id,
        )
    )
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return document


CurrentProject = Annotated[Project, Depends(get_current_project)]
CurrentDocument = Annotated[Document, Depends(get_current_document)]


def get_current_conversation(
    conversation_id: str,
    db: DbSession,
    current_project: CurrentProject,
) -> Conversation:
    """The conversation, only if it belongs to the path's project (itself owner-checked)."""
    conversation = db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.project_id == current_project.id,
        )
    )
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return conversation


CurrentConversation = Annotated[Conversation, Depends(get_current_conversation)]


__all__ = [
    "AccessToken",
    "CurrentConversation",
    "CurrentDocument",
    "CurrentProject",
    "CurrentToken",
    "CurrentUser",
    "DbSession",
    "get_access_token",
    "get_current_user",
    "get_db",
]
