
from datetime import timedelta

import jwt
import pytest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database.connection import Base, get_db
from app.auth.dependencies import get_current_user
from app.auth.security import create_access_token
from app.config import settings
from app.models.user import User


# Create an isolated in-memory SQLite database
TEST_DATABASE_URL = "sqlite://"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


@pytest.fixture
def client():
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def create_test_user():
    db = TestingSessionLocal()

    user = User(
        full_name="Test User",
        email="test@example.com",
        hashed_password="unused",
        is_active=True,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    user_id = user.id
    db.close()

    return user_id


def test_profile_without_token_returns_401(client):
    response = client.get("/api/auth/me")

    assert response.status_code == 401


def test_profile_with_invalid_token_returns_401(client):
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert response.status_code == 401


def test_profile_with_valid_token_returns_200(client):
    user_id = create_test_user()

    token = create_access_token({"sub": str(user_id)})

    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["email"] == "test@example.com"
    assert response.json()["full_name"] == "Test User"


def test_profile_with_expired_token_returns_401(client):
    user_id = create_test_user()

    token = create_access_token(
        {"sub": str(user_id)},
        expires_delta=timedelta(seconds=-1),
    )

    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication token has expired."


def test_profile_with_nonexistent_user_returns_401(client):
    token = create_access_token({"sub": "99999"})

    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
