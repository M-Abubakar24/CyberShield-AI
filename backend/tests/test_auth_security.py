
from datetime import timedelta

import jwt

from app.auth.security import (
    get_password_hash,
    verify_password,
    create_access_token,
)
from app.config import settings


def test_password_hashing():
    password = "StrongPassword123!"

    hashed = get_password_hash(password)

    assert hashed != password
    assert verify_password(password, hashed)
    assert not verify_password("WrongPassword123!", hashed)


def test_access_token_creation():
    token = create_access_token(
        {"sub": "1"},
    )

    payload = jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
    )

    assert payload["sub"] == "1"
    assert "exp" in payload


def test_access_token_custom_expiry():
    token = create_access_token(
        {"sub": "2"},
        expires_delta=timedelta(minutes=10),
    )

    payload = jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
    )

    assert payload["sub"] == "2"
    assert "exp" in payload
