
import uuid

import chromadb
import pytest

from app.rag import vector_store


USER_A = "user-101"
USER_B = "user-202"


@pytest.fixture
def real_collection(tmp_path, monkeypatch):
    """Create an isolated, real ChromaDB collection for integration tests."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection_name = f"test_{uuid.uuid4().hex[:12]}"

    collection = client.create_collection(name=collection_name)

    monkeypatch.setattr(vector_store, "collection", collection)

    yield collection

    client.delete_collection(name=collection_name)


def add_test_document(user_id, document_id, filename, text, embedding):
    """Insert one document chunk into the real ChromaDB collection."""
    return vector_store.add_documents(
        chunks=[text],
        embeddings=[embedding],
        metadatas=[
            {
                "document_id": document_id,
                "filename": filename,
                "page_number": 1,
            }
        ],
        document_id=document_id,
        user_id=user_id,
    )


def test_real_chromadb_search_isolates_users(real_collection):
    """Each user should retrieve only their own documents."""

    add_test_document(
        USER_A,
        "doc-a",
        "user_a.txt",
        "User A private cybersecurity document.",
        [0.1, 0.2, 0.3],
    )

    add_test_document(
        USER_B,
        "doc-b",
        "user_b.txt",
        "User B private network security document.",
        [0.2, 0.3, 0.4],
    )

    results_a = vector_store.search_documents(
        query_embedding=[0.1, 0.2, 0.3],
        user_id=USER_A,
        top_k=5,
    )

    results_b = vector_store.search_documents(
        query_embedding=[0.2, 0.3, 0.4],
        user_id=USER_B,
        top_k=5,
    )

    assert len(results_a["documents"][0]) == 1
    assert len(results_b["documents"][0]) == 1

    assert "User A" in results_a["documents"][0][0]
    assert "User B" not in results_a["documents"][0][0]

    assert "User B" in results_b["documents"][0][0]
    assert "User A" not in results_b["documents"][0][0]


def test_real_chromadb_document_lists_are_isolated(real_collection):
    """Each user should see only their own uploaded documents."""

    add_test_document(
        USER_A,
        "doc-a",
        "private_a.pdf",
        "Document belonging to user A.",
        [0.1, 0.2, 0.3],
    )

    add_test_document(
        USER_B,
        "doc-b",
        "private_b.pdf",
        "Document belonging to user B.",
        [0.2, 0.3, 0.4],
    )

    documents_a = vector_store.list_documents(USER_A)
    documents_b = vector_store.list_documents(USER_B)

    assert len(documents_a) == 1
    assert len(documents_b) == 1

    assert documents_a[0]["filename"] == "private_a.pdf"
    assert documents_b[0]["filename"] == "private_b.pdf"


def test_real_chromadb_user_counts_are_separate(real_collection):
    """Collection counts should be scoped to each user."""

    add_test_document(
        USER_A,
        "doc-a1",
        "a1.txt",
        "First document for A.",
        [0.1, 0.2, 0.3],
    )

    add_test_document(
        USER_A,
        "doc-a2",
        "a2.txt",
        "Second document for A.",
        [0.2, 0.3, 0.4],
    )

    add_test_document(
        USER_B,
        "doc-b1",
        "b1.txt",
        "First document for B.",
        [0.3, 0.4, 0.5],
    )

    assert vector_store.get_collection_count(USER_A) == 2
    assert vector_store.get_collection_count(USER_B) == 1

    assert vector_store.get_document_count(USER_A) == 2
    assert vector_store.get_document_count(USER_B) == 1


def test_user_cannot_delete_another_users_document(real_collection):
    """A user must not be able to delete another user's document."""

    add_test_document(
        USER_B,
        "private-doc-b",
        "user_b.txt",
        "This document belongs to user B.",
        [0.1, 0.2, 0.3],
    )

    deleted_count = vector_store.delete_document(
        "private-doc-b",
        USER_A,
    )

    assert deleted_count == 0

    # User B's document must still exist.
    documents_b = vector_store.list_documents(USER_B)

    assert len(documents_b) == 1
    assert documents_b[0]["document_id"] == "private-doc-b"


def test_user_can_delete_only_their_own_document(real_collection):
    """Deleting an owned document must not affect another user."""

    add_test_document(
        USER_A,
        "doc-a",
        "user_a.txt",
        "Document belonging to A.",
        [0.1, 0.2, 0.3],
    )

    add_test_document(
        USER_B,
        "doc-b",
        "user_b.txt",
        "Document belonging to B.",
        [0.2, 0.3, 0.4],
    )

    deleted_count = vector_store.delete_document("doc-a", USER_A)

    assert deleted_count == 1

    assert vector_store.list_documents(USER_A) == []

    documents_b = vector_store.list_documents(USER_B)

    assert len(documents_b) == 1
    assert documents_b[0]["document_id"] == "doc-b"


def test_legacy_documents_without_user_id_are_not_visible(real_collection):
    """Legacy chunks without ownership metadata must remain inaccessible."""

    # Insert a legacy record directly, without user_id.
    real_collection.add(
        ids=["legacy-document_0"],
        documents=["Legacy document without ownership metadata."],
        embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[
            {
                "document_id": "legacy-document",
                "filename": "legacy.txt",
            }
        ],
    )

    assert vector_store.get_collection_count(USER_A) == 0
    assert vector_store.get_collection_count(USER_B) == 0

    assert vector_store.list_documents(USER_A) == []
    assert vector_store.list_documents(USER_B) == []

    results = vector_store.search_documents(
        query_embedding=[0.1, 0.2, 0.3],
        user_id=USER_A,
    )

    assert results["documents"] == [[]]
