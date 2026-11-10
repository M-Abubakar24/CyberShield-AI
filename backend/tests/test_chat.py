
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)


def test_chat_success():
    """Test a successful cybersecurity question."""

    mock_result = {
        "answer": "SQL Injection occurs when untrusted input is included in SQL queries.",
        "sources": [
            {
                "source_id": "S1",
                "filename": "sample_owasp.txt",
                "page_number": 1,
                "distance": 0.35,
                "content_preview": "SQL Injection prevention guidance.",
            }
        ],
        "retrieval_count": 1,
    }

    with patch(
        "app.api.chat.ask_cybersecurity_assistant",
        return_value=mock_result,
    ) as mock_chat:

        response = client.post(
            "/api/chat/ask",
            json={
                "question": "What is SQL Injection?",
                "top_k": 5,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["question"] == "What is SQL Injection?"
    assert data["data"]["answer"] == mock_result["answer"]
    assert len(data["data"]["sources"]) == 1

    mock_chat.assert_called_once_with(
        question="What is SQL Injection?",
        top_k=5,
    )


def test_chat_rejects_short_question():
    """Questions shorter than 5 characters should be rejected."""

    response = client.post(
        "/api/chat/ask",
        json={"question": "Hi"},
    )

    assert response.status_code == 422


def test_chat_rejects_invalid_top_k():
    """top_k must be between 1 and 10."""

    response = client.post(
        "/api/chat/ask",
        json={
            "question": "Explain SQL Injection",
            "top_k": 15,
        },
    )

    assert response.status_code == 422


def test_chat_handles_value_error():
    """A ValueError should return HTTP 400."""

    with patch(
        "app.api.chat.ask_cybersecurity_assistant",
        side_effect=ValueError("No relevant cybersecurity information found."),
    ):

        response = client.post(
            "/api/chat/ask",
            json={"question": "Explain an unsupported topic"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "No relevant cybersecurity information found."
    )


def test_chat_handles_generation_error():
    """Unexpected AI errors should return HTTP 502."""

    with patch(
        "app.api.chat.ask_cybersecurity_assistant",
        side_effect=Exception("Simulated Gemini service failure"),
    ):

        response = client.post(
            "/api/chat/ask",
            json={"question": "Explain SQL Injection"},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "AI response generation failed."


def test_chat_returns_unsupported_topic_message():
    """Test the response when no relevant knowledge is retrieved."""

    mock_result = {
        "answer": "I couldn't find relevant information in the cybersecurity knowledge base.",
        "sources": [],
        "retrieval_count": 0,
    }

    with patch(
        "app.api.chat.ask_cybersecurity_assistant",
        return_value=mock_result,
    ):

        response = client.post(
            "/api/chat/ask",
            json={"question": "Explain quantum computing"},
        )

    assert response.status_code == 200
    assert response.json()["data"]["sources"] == []
    assert response.json()["data"]["retrieval_count"] == 0
    assert "couldn't find relevant information" in (
        response.json()["data"]["answer"]
    )
