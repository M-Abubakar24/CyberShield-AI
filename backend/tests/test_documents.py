
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_document_stats():
    """Test knowledge base statistics."""

    with (
        patch("app.api.documents.get_document_count", return_value=2),
        patch("app.api.documents.get_collection_count", return_value=5),
    ):
        response = client.get("/api/documents/stats")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["total_documents"] == 2
    assert data["total_chunks"] == 5
    assert data["collection"] == "cybersecurity_knowledge"


def test_list_documents():
    """Test listing indexed documents."""

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
    ):
        response = client.get("/api/documents/")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["total_documents"] == 2
    assert len(data["documents"]) == 2


def test_upload_document_success():
    """Test successful document upload."""

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
    """Search should return an empty result when the knowledge base is empty."""

    with patch(
        "app.api.documents.get_collection_count",
        return_value=0,
    ):
        response = client.post(
            "/api/documents/search",
            json={"query": "SQL Injection"},
        )

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["message"] == "No documents have been indexed yet."


def test_search_documents_success():
    """Test successful semantic document search."""

    mock_results = {
        "documents": [[
            "SQL Injection is a web security vulnerability."
        ]],
        "metadatas": [[
            {
                "filename": "sample_owasp.txt",
                "page_number": 1,
            }
        ]],
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
        ),
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


def test_search_rejects_short_query():
    """Search queries shorter than three characters should be rejected."""

    response = client.post(
        "/api/documents/search",
        json={"query": "Hi"},
    )

    assert response.status_code == 422


def test_delete_document_success():
    """Test successful document deletion."""

    with patch(
        "app.api.documents.delete_document",
        return_value=3,
    ):
        response = client.delete("/api/documents/doc-test-1")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["document_id"] == "doc-test-1"
    assert data["deleted_chunks"] == 3


def test_delete_nonexistent_document():
    """Deleting a missing document should return HTTP 404."""

    with patch(
        "app.api.documents.delete_document",
        return_value=0,
    ):
        response = client.delete("/api/documents/unknown-doc")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."

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
