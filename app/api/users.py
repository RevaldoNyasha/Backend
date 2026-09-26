"""The signed-in researcher's own account: profile, preferences, sign-in methods."""

from fastapi import APIRouter
from sqlalchemy import select

from app.core.dependencies import CurrentUser, DbSession
from app.models.oauth_account import OAuthAccount
from app.models.user import User
from app.schemas.auth import (
    LinkedProvider,
    Preferences,
    PreferencesUpdate,
    ProfileUpdate,
    SecurityResponse,
    UserResponse,
)
from app.services.auth import get_preferences, update_preferences, update_profile

router = APIRouter(tags=["users"])


@router.patch("/users/me", response_model=UserResponse)
def edit_profile(payload: ProfileUpdate, current_user: CurrentUser, db: DbSession) -> User:
    """Change name, title or organization (contract §10.4). Email is not editable here."""
    return update_profile(db, current_user, payload.model_dump(exclude_unset=True))


@router.get("/users/me/preferences", response_model=Preferences)
def read_preferences(current_user: CurrentUser) -> dict:
    return get_preferences(current_user)


@router.put("/users/me/preferences", response_model=Preferences)
def save_preferences(payload: PreferencesUpdate, current_user: CurrentUser, db: DbSession) -> dict:
    """Save the settings toggles; only the ones sent change."""
    changes = payload.model_dump(exclude_unset=True, exclude_none=True, by_alias=True)
    return update_preferences(db, current_user, changes)


@router.get("/users/me/security", response_model=SecurityResponse)
def read_security(current_user: CurrentUser, db: DbSession) -> SecurityResponse:
    """How this account can sign in. Never includes the password hash."""
    linked = db.scalars(
        select(OAuthAccount)
        .where(OAuthAccount.user_id == current_user.id)
        .order_by(OAuthAccount.created_at)
    ).all()
    return SecurityResponse(
        has_password=bool(current_user.password_hash),
        providers=[
            LinkedProvider(
                provider=account.provider, email=account.email, linked_at=account.created_at
            )
            for account in linked
        ],
    )
