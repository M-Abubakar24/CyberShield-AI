
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
    """Simulate an authenticated user for document API tests."""
    app.dependency_overrides[get_current_user] = lambda: TEST_USER

    yield

    app.dependency_overrides.pop(get_current_user, None)


def test_documents_reject_unauthenticated_requests():
    """Protected document endpoints should reject missing authentication."""
    app.dependency_overrides.pop(get_current_user, None)

    response = client.get("/api/documents/")

    assert response.status_code == 401


def test_document_stats():
    """Test knowledge base statistics for the current user."""
    with (
        patch(
            "app.api.documents.get_document_count",
            return_value=2,
        ) as mock_documents,
        patch(
            "app.api.documents.get_collection_count",
            return_value=5,
        ) as mock_chunks,
    ):
        response = client.get("/api/documents/stats")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["total_documents"] == 2
    assert data["total_chunks"] == 5
    assert data["collection"] == "cybersecurity_knowledge"

    mock_documents.assert_called_once_with(user_id=TEST_USER_ID)
    mock_chunks.assert_called_once_with(user_id=TEST_USER_ID)


def test_list_documents():
    """Test listing only the current user's documents."""
    mock_documents = [
        {
            "document_id": "doc-1",
            "filename": "sample_owasp.txt",
            "chunks": 2,
        },
        {
            "document_id": "doc-2",
            "filename": "sample_idor.txt",
            "chunks": 3,
        },
    ]

    with patch(
        "app.api.documents.list_documents",
        return_value=mock_documents,
    ) as mock_list:
        response = client.get("/api/documents/")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["total_documents"] == 2
    assert len(data["documents"]) == 2

    mock_list.assert_called_once_with(user_id=TEST_USER_ID)


def test_upload_document_success():
    """Test successful upload with the current user's ID."""
    mock_result = {
        "success": True,
        "document_id": "doc-test-1",
        "filename": "test.txt",
        "chunks_created": 2,
    }

    with patch(
        "app.api.documents.ingest_document",
        return_value=mock_result,
    ) as mock_ingest:
        response = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "test.txt",
                    b"Sample cybersecurity document content.",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-test-1"

    mock_ingest.assert_called_once()

    call_data = mock_ingest.call_args.kwargs

    assert call_data["filename"] == "test.txt"
    assert call_data["user_id"] == TEST_USER_ID


def test_upload_empty_document():
    """Empty files should be rejected."""
    response = client.post(
        "/api/documents/upload",
        files={
            "file": ("empty.txt", b"", "text/plain")
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_upload_rejects_unsupported_document():
    """Invalid document formats should return HTTP 400."""
    with patch(
        "app.api.documents.ingest_document",
        side_effect=ValueError("Unsupported file format."),
    ):
        response = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "test.exe",
                    b"sample content",
                    "application/octet-stream",
                )
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file format."


def test_search_when_no_documents_exist():
    """Search should return empty results for an empty user knowledge base."""
    with patch(
        "app.api.documents.get_collection_count",
        return_value=0,
    ) as mock_count:
        response = client.post(
            "/api/documents/search",
            json={"query": "SQL Injection"},
        )

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["message"] == "No documents have been indexed yet."

    mock_count.assert_called_once_with(user_id=TEST_USER_ID)


def test_search_documents_success():
    """Test semantic search scoped to the current user."""
    mock_results = {
        "documents": [
            ["SQL Injection is a web security vulnerability."]
        ],
        "metadatas": [
            [
                {
                    "filename": "sample_owasp.txt",
                    "page_number": 1,
                    "user_id": str(TEST_USER_ID),
                }
            ]
        ],
        "distances": [[0.35]],
    }

    with (
        patch(
            "app.api.documents.get_collection_count",
            return_value=2,
        ),
        patch(
            "app.api.documents.generate_query_embedding",
            return_value=[0.1, 0.2, 0.3],
        ),
        patch(
            "app.api.documents.search_documents",
            return_value=mock_results,
        ) as mock_search,
    ):
        response = client.post(
            "/api/documents/search",
            json={
                "query": "What is SQL Injection?",
                "top_k": 5,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert len(data["results"]) == 1
    assert data["results"][0]["metadata"]["filename"] == "sample_owasp.txt"
    assert data["results"][0]["distance"] == 0.35

    mock_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        user_id=TEST_USER_ID,
        top_k=5,
    )


def test_search_rejects_short_query():
    """Search queries shorter than three characters should be rejected."""
    response = client.post(
        "/api/documents/search",
        json={"query": "Hi"},
    )

    assert response.status_code == 422


def test_delete_document_success():
    """Test deletion scoped to the current user."""
    with patch(
        "app.api.documents.delete_document",
        return_value=3,
    ) as mock_delete:
        response = client.delete("/api/documents/doc-test-1")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["document_id"] == "doc-test-1"
    assert data["deleted_chunks"] == 3

    mock_delete.assert_called_once_with(
        document_id="doc-test-1",
        user_id=TEST_USER_ID,
    )


def test_delete_nonexistent_document():
    """Deleting a missing or unowned document should return HTTP 404."""
    with patch(
        "app.api.documents.delete_document",
        return_value=0,
    ) as mock_delete:
        response = client.delete("/api/documents/unknown-doc")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."

    mock_delete.assert_called_once_with(
        document_id="unknown-doc",
        user_id=TEST_USER_ID,
    )


def test_upload_file_exceeds_maximum_size():
    """Oversized uploads should return HTTP 413."""
    with patch(
        "app.api.documents.settings.MAX_FILE_SIZE_MB",
        1,
    ):
        response = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "large.txt",
                    b"A" * (1024 * 1024 + 1),
                    "text/plain",
                )
            },
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "File exceeds maximum size."


def test_upload_handles_unexpected_processing_error():
    """Unexpected ingestion errors should return HTTP 500."""
    with patch(
        "app.api.documents.ingest_document",
        side_effect=RuntimeError("Unexpected processing error."),
    ):
        response = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "test.txt",
                    b"Valid cybersecurity content.",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Document processing failed."


def test_search_handles_unexpected_error():
    """Unexpected search errors should return HTTP 500."""
    with (
        patch(
            "app.api.documents.get_collection_count",
            return_value=2,
        ),
        patch(
            "app.api.documents.generate_query_embedding",
            side_effect=RuntimeError("Embedding service failed."),
        ),
    ):
        response = client.post(
            "/api/documents/search",
            json={"query": "Explain SQL Injection."},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Document search failed."


def test_upload_rejects_missing_file():
    """Upload endpoint should reject requests without a file."""
    response = client.post("/api/documents/upload")

    assert response.status_code == 422
