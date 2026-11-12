
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.auth.dependencies import get_current_user


client = TestClient(app)

TEST_USER_ID = 101

TEST_USER = SimpleNamespace(
    id=TEST_USER_ID,
    email="test@example.com",
    full_name="Test User",
    is_active=True,
)


@pytest.fixture(autouse=True)
def authenticated_user():
    """Simulate an authenticated user for chat API tests."""
    app.dependency_overrides[get_current_user] = lambda: TEST_USER

    yield

    app.dependency_overrides.pop(get_current_user, None)


def test_chat_rejects_unauthenticated_request():
    """Chat endpoint should reject requests without authentication."""
    app.dependency_overrides.pop(get_current_user, None)

    response = client.post(
        "/api/chat/ask",
        json={"question": "Explain SQL Injection."},
    )

    assert response.status_code == 401


def test_chat_success():
    """Test successful AI chat with user-specific retrieval."""
    mock_result = {
        "answer": "SQL Injection is a vulnerability caused by unsafe SQL queries.",
        "sources": [
            {
                "source_id": "S1",
                "filename": "owasp.txt",
                "page_number": 1,
                "distance": 0.25,
                "content_preview": "SQL Injection prevention techniques.",
            }
        ],
        "retrieval_count": 1,
    }

    with patch(
        "app.api.chat.ask_cybersecurity_assistant",
        return_value=mock_result,
    ) as mock_assistant:

        response = client.post(
            "/api/chat/ask",
            json={
                "question": "Explain SQL Injection.",
                "top_k": 5,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["question"] == "Explain SQL Injection."
    assert data["data"]["answer"] == mock_result["answer"]
    assert data["data"]["retrieval_count"] == 1
    assert len(data["data"]["sources"]) == 1

    mock_assistant.assert_called_once_with(
        question="Explain SQL Injection.",
        user_id=TEST_USER_ID,
        top_k=5,
    )


def test_chat_without_relevant_documents():
    """Test chat response when no relevant documents are found."""
    mock_result = {
        "answer": "I couldn't find relevant information in your cybersecurity knowledge base.",
        "sources": [],
        "retrieval_count": 0,
    }

    with patch(
        "app.api.chat.ask_cybersecurity_assistant",
        return_value=mock_result,
    ):
        response = client.post(
            "/api/chat/ask",
            json={"question": "Explain SQL Injection."},
        )

    assert response.status_code == 200

    data = response.json()["data"]

    assert data["sources"] == []
    assert data["retrieval_count"] == 0


def test_chat_rejects_short_question():
    """Questions shorter than five characters should be rejected."""
    response = client.post(
        "/api/chat/ask",
        json={"question": "Hi"},
    )

    assert response.status_code == 422


def test_chat_rejects_invalid_top_k():
    """Top-k must be between 1 and 10."""
    response = client.post(
        "/api/chat/ask",
        json={
            "question": "Explain SQL Injection.",
            "top_k": 20,
        },
    )

    assert response.status_code == 422


def test_chat_handles_value_error():
    """Value errors should return HTTP 400."""
    with patch(
        "app.api.chat.ask_cybersecurity_assistant",
        side_effect=ValueError("Invalid question."),
    ):
        response = client.post(
            "/api/chat/ask",
            json={"question": "Explain SQL Injection."},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid question."


def test_chat_handles_generation_error():
    """Unexpected AI errors should return HTTP 502."""
    with patch(
        "app.api.chat.ask_cybersecurity_assistant",
        side_effect=RuntimeError("Gemini service unavailable."),
    ):
        response = client.post(
            "/api/chat/ask",
            json={"question": "Explain SQL Injection."},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "AI response generation failed."
