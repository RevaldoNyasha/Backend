from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.security import hash_password, verify_password
from app.models.revoked_token import RevokedToken
from app.models.user import User

# Keys and defaults of `User.preferences` (camelCase, as the frontend sends them).
DEFAULT_PREFERENCES = {"autoTranscribe": True, "autoTranslate": False, "showEvidence": True}


def normalize_email(email: str) -> str:
    return email.strip().lower()


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email))


def create_user(
    db: Session,
    name: str,
    email: str,
    password: str,
    organization: str | None = None,
) -> User:
    user = User(
        name=name,
        email=email,
        password_hash=hash_password(password),
        organization=organization.strip() if organization else None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = get_user_by_email(db, email)
    if user is None or not verify_password(password, user.password_hash):
        return None
    return user


def is_token_revoked(db: Session, jti: str) -> bool:
    return db.get(RevokedToken, jti) is not None


def revoke_token(db: Session, jti: str, user_id: str, expires_at: datetime) -> None:
    """Deny-list one access token until it expires, and purge rows that already have."""
    db.execute(delete(RevokedToken).where(RevokedToken.expires_at < datetime.now(UTC)))
    if db.get(RevokedToken, jti) is None:
        db.add(RevokedToken(jti=jti, user_id=user_id, expires_at=expires_at))
    db.commit()


def update_profile(db: Session, user: User, changes: dict) -> User:
    """Apply the profile fields that were sent; blank title/organization clear them."""
    if changes.get("name"):
        user.name = changes["name"]
    for field in ("title", "organization"):
        if field in changes:
            setattr(user, field, changes[field] or None)
    db.commit()
    db.refresh(user)
    return user


def get_preferences(user: User) -> dict:
    return {**DEFAULT_PREFERENCES, **(user.preferences or {})}


def update_preferences(db: Session, user: User, changes: dict) -> dict:
    user.preferences = {
        **get_preferences(user),
        **{key: value for key, value in changes.items() if key in DEFAULT_PREFERENCES},
    }
    db.commit()
    return get_preferences(user)


class WrongPassword(Exception):
    pass


def change_password(
    db: Session, user: User, current_password: str | None, new_password: str
) -> None:
    """Set a new password. Accounts that have one must confirm it first; accounts
    created with Google/GitHub (no password yet) can add one."""
    if user.password_hash and not verify_password(current_password or "", user.password_hash):
        raise WrongPassword
    user.password_hash = hash_password(new_password)
    db.commit()
