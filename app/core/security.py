from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from pwdlib import PasswordHash
from pwdlib.exceptions import PwdlibError
from pwdlib.hashers.argon2 import Argon2Hasher

from app.core.config import settings
from app.utils.ids import new_id

_password_hasher = PasswordHash((Argon2Hasher(),))


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    try:
        return _password_hasher.verify(password, hashed_password)
    except (PwdlibError, ValueError):
        return False


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    if not settings.jwt_secret_key:
        raise RuntimeError("JWT_SECRET_KEY must be configured before creating access tokens")

    issued_at = datetime.now(UTC)
    token_lifetime = (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload: dict[str, Any] = {
        "sub": str(subject),
        # Unique token id, so a single token can be revoked on logout.
        "jti": new_id(),
        "iat": issued_at,
        "exp": issued_at + token_lifetime,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    if not settings.jwt_secret_key:
        raise RuntimeError("JWT_SECRET_KEY must be configured before decoding access tokens")

    return jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
        options={"require": ["exp", "sub", "jti"]},
    )
