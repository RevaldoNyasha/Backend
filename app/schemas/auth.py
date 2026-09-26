from pydantic import ConfigDict, Field

from app.schemas.base import Schema, UtcDateTime


class UserCreate(Schema):
    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    name: str = Field(min_length=1, max_length=255)
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)
    organization: str | None = Field(default=None, max_length=255)


class UserLogin(Schema):
    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=128)


class UserResponse(Schema):
    id: str
    name: str
    email: str
    role: str
    title: str | None = None
    organization: str | None = None
    avatar_url: str | None = None
    created_at: UtcDateTime


class OAuthStartResponse(Schema):
    authorization_url: str
    # Echoed back by the provider; the frontend keeps it to check the callback.
    state: str


class TokenResponse(Schema):
    user: UserResponse
    access_token: str
    expires_in: int


class ProfileUpdate(Schema):
    """`PATCH /users/me` (contract §10.4). Only the fields sent are changed."""

    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    name: str | None = Field(default=None, min_length=1, max_length=255)
    # Empty clears the field.
    title: str | None = Field(default=None, max_length=120)
    organization: str | None = Field(default=None, max_length=255)


class Preferences(Schema):
    """Settings-page preferences (contract §10.4). The theme stays per device."""

    auto_transcribe: bool = True
    auto_translate: bool = False
    show_evidence: bool = True


class PreferencesUpdate(Schema):
    auto_transcribe: bool | None = None
    auto_translate: bool | None = None
    show_evidence: bool | None = None


class PasswordChange(Schema):
    """`PATCH /auth/password`. `currentPassword` is required once the account has one."""

    model_config = ConfigDict(populate_by_name=True)

    current_password: str | None = Field(default=None, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class LinkedProvider(Schema):
    provider: str
    email: str | None = None
    linked_at: UtcDateTime


class SecurityResponse(Schema):
    """How this account signs in: a password, and/or linked Google / GitHub accounts."""

    has_password: bool
    providers: list[LinkedProvider]
