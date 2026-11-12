
from unittest.mock import MagicMock

import pytest

from app.rag import vector_store


USER_A = "user-101"
USER_B = "user-202"


@pytest.fixture
def mock_collection(monkeypatch):
    """Replace the real ChromaDB collection with a mock."""
    mock = MagicMock()
    monkeypatch.setattr(vector_store, "collection", mock)
    return mock


# ADD DOCUMENTS

def test_add_documents_success(mock_collection):
    ids = vector_store.add_documents(
        chunks=["First chunk", "Second chunk"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[
            {"document_id": "doc-1", "filename": "test.txt"},
            {"document_id": "doc-1", "filename": "test.txt"},
        ],
        document_id="doc-1",
        user_id=USER_A,
    )

    assert ids == ["doc-1_0", "doc-1_1"]

    metadata = mock_collection.add.call_args.kwargs["metadatas"]
    assert all(item["user_id"] == USER_A for item in metadata)


def test_add_documents_empty_chunks(mock_collection):
    result = vector_store.add_documents(
        chunks=[],
        embeddings=[],
        metadatas=[],
        document_id="doc-1",
        user_id=USER_A,
    )

    assert result == []
    mock_collection.add.assert_not_called()


def test_add_documents_rejects_mismatched_lengths(mock_collection):
    with pytest.raises(ValueError, match="must have equal lengths"):
        vector_store.add_documents(
            chunks=["First chunk", "Second chunk"],
            embeddings=[[0.1, 0.2]],
            metadatas=[
                {"document_id": "doc-1"},
                {"document_id": "doc-1"},
            ],
            document_id="doc-1",
            user_id=USER_A,
        )

    mock_collection.add.assert_not_called()


def test_add_documents_rejects_missing_user_id(mock_collection):
    with pytest.raises(ValueError, match="user_id is required"):
        vector_store.add_documents(
            chunks=["First chunk"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"document_id": "doc-1"}],
            document_id="doc-1",
            user_id="",
        )

    mock_collection.add.assert_not_called()


# SEARCH DOCUMENTS

def test_search_documents_empty_collection(mock_collection):
    mock_collection.get.return_value = {
        "ids": [],
        "metadatas": [],
    }

    result = vector_store.search_documents(
        query_embedding=[0.1, 0.2],
        user_id=USER_A,
    )

    assert result == {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    mock_collection.query.assert_not_called()


def test_search_documents_success(mock_collection):
    mock_collection.get.return_value = {
        "ids": ["chunk-1", "chunk-2", "chunk-3"],
        "metadatas": [],
    }

    mock_collection.query.return_value = {
        "documents": [["First relevant chunk", "Second relevant chunk"]],
        "metadatas": [[
            {"filename": "first.txt", "user_id": USER_A},
            {"filename": "second.txt", "user_id": USER_A},
        ]],
        "distances": [[0.20, 0.35]],
    }

    result = vector_store.search_documents(
        query_embedding=[0.1, 0.2],
        user_id=USER_A,
        top_k=5,
    )

    assert len(result["documents"][0]) == 2
    assert result["documents"][0][0] == "First relevant chunk"
    assert result["distances"][0] == [0.20, 0.35]

    mock_collection.query.assert_called_once()

    assert mock_collection.query.call_args.kwargs["where"] == {
        "user_id": USER_A
    }


def test_search_documents_applies_user_and_document_filters(mock_collection):
    mock_collection.get.return_value = {
        "ids": [f"chunk-{i}" for i in range(5)],
        "metadatas": [],
    }

    mock_collection.query.return_value = {
        "documents": [["Document chunk"]],
        "metadatas": [[{"document_id": "doc-123", "user_id": USER_A}]],
        "distances": [[0.25]],
    }

    vector_store.search_documents(
        query_embedding=[0.1, 0.2],
        user_id=USER_A,
        document_id="doc-123",
    )

    call_kwargs = mock_collection.query.call_args.kwargs

    assert call_kwargs["where"] == {
        "$and": [
            {"user_id": USER_A},
            {"document_id": "doc-123"},
        ]
    }


def test_search_documents_never_uses_global_filter(mock_collection):
    mock_collection.get.return_value = {
        "ids": [f"chunk-{i}" for i in range(5)],
        "metadatas": [],
    }

    mock_collection.query.return_value = {
        "documents": [["Chunk"]],
        "metadatas": [[{"filename": "test.txt", "user_id": USER_A}]],
        "distances": [[0.2]],
    }

    vector_store.search_documents(
        query_embedding=[0.1, 0.2],
        user_id=USER_A,
    )

    call_kwargs = mock_collection.query.call_args.kwargs

    assert call_kwargs["where"] is not None
    assert call_kwargs["where"] == {"user_id": USER_A}


def test_search_documents_filters_irrelevant_results(mock_collection):
    mock_collection.get.return_value = {
        "ids": ["chunk-1", "chunk-2", "chunk-3"],
        "metadatas": [],
    }

    mock_collection.query.return_value = {
        "documents": [["Relevant chunk", "Irrelevant chunk"]],
        "metadatas": [[
            {"filename": "relevant.txt", "user_id": USER_A},
            {"filename": "irrelevant.txt", "user_id": USER_A},
        ]],
        "distances": [[0.25, 0.85]],
    }

    result = vector_store.search_documents(
        query_embedding=[0.1, 0.2],
        user_id=USER_A,
        distance_threshold=0.60,
    )

    assert result["documents"][0] == ["Relevant chunk"]
    assert result["distances"][0] == [0.25]


def test_search_documents_sorts_and_limits_results(mock_collection):
    mock_collection.get.return_value = {
        "ids": [f"chunk-{i}" for i in range(10)],
        "metadatas": [],
    }

    mock_collection.query.return_value = {
        "documents": [["Third chunk", "First chunk", "Second chunk"]],
        "metadatas": [[
            {"filename": "third.txt", "user_id": USER_A},
            {"filename": "first.txt", "user_id": USER_A},
            {"filename": "second.txt", "user_id": USER_A},
        ]],
        "distances": [[0.30, 0.10, 0.20]],
    }

    result = vector_store.search_documents(
        query_embedding=[0.1, 0.2],
        user_id=USER_A,
        top_k=2,
    )

    assert result["documents"][0] == ["First chunk", "Second chunk"]
    assert result["distances"][0] == [0.10, 0.20]

    assert mock_collection.query.call_args.kwargs["n_results"] == 6


# DELETE DOCUMENT

def test_delete_document_success(mock_collection):
    mock_collection.get.return_value = {
        "ids": ["doc-1_0", "doc-1_1"],
        "metadatas": [{}, {}],
    }

    result = vector_store.delete_document("doc-1", USER_A)

    assert result == 2

    assert mock_collection.get.call_args.kwargs["where"] == {
        "$and": [
            {"document_id": "doc-1"},
            {"user_id": USER_A},
        ]
    }

    mock_collection.delete.assert_called_once_with(
        ids=["doc-1_0", "doc-1_1"]
    )


def test_delete_document_not_found(mock_collection):
    mock_collection.get.return_value = {
        "ids": [],
        "metadatas": [],
    }

    result = vector_store.delete_document("missing-doc", USER_A)

    assert result == 0
    mock_collection.delete.assert_not_called()


# COLLECTION COUNT

def test_get_collection_count(mock_collection):
    mock_collection.get.return_value = {
        "ids": [f"chunk-{i}" for i in range(15)],
        "metadatas": [],
    }

    result = vector_store.get_collection_count(USER_A)

    assert result == 15

    mock_collection.get.assert_called_once_with(
        where={"user_id": USER_A},
        include=["metadatas"],
    )


# RENAME DOCUMENT

def test_rename_document_success(mock_collection):
    mock_collection.get.return_value = {
        "ids": ["doc-1_0", "doc-1_1"],
        "metadatas": [
            {
                "document_id": "doc-1",
                "user_id": USER_A,
                "filename": "old.txt",
                "page_number": 1,
            },
            {
                "document_id": "doc-1",
                "user_id": USER_A,
                "filename": "old.txt",
                "page_number": 2,
            },
        ],
    }

    result = vector_store.rename_document(
        "doc-1",
        "new.txt",
        USER_A,
    )

    assert result == 2

    update_kwargs = mock_collection.update.call_args.kwargs

    assert update_kwargs["ids"] == ["doc-1_0", "doc-1_1"]

    assert all(
        metadata["filename"] == "new.txt"
        for metadata in update_kwargs["metadatas"]
    )

    assert all(
        metadata["user_id"] == USER_A
        for metadata in update_kwargs["metadatas"]
    )


def test_rename_document_not_found(mock_collection):
    mock_collection.get.return_value = {
        "ids": [],
        "metadatas": [],
    }

    result = vector_store.rename_document(
        "missing-doc",
        "new.txt",
        USER_A,
    )

    assert result == 0
    mock_collection.update.assert_not_called()


# LIST DOCUMENTS

def test_list_documents_groups_chunks_correctly(mock_collection):
    mock_collection.get.return_value = {
        "metadatas": [
            {
                "document_id": "doc-1",
                "user_id": USER_A,
                "filename": "security.pdf",
                "page_number": 1,
            },
            {
                "document_id": "doc-1",
                "user_id": USER_A,
                "filename": "security.pdf",
                "page_number": 1,
            },
            {
                "document_id": "doc-1",
                "user_id": USER_A,
                "filename": "security.pdf",
                "page_number": 2,
            },
            {
                "document_id": "doc-2",
                "user_id": USER_A,
                "filename": "network.txt",
                "page_number": 1,
            },
        ]
    }

    result = vector_store.list_documents(USER_A)

    assert len(result) == 2

    first_document = next(
        item for item in result
        if item["document_id"] == "doc-1"
    )

    assert first_document["filename"] == "security.pdf"
    assert first_document["chunks"] == 3
    assert first_document["pages"] == 2

    mock_collection.get.assert_called_once_with(
        where={"user_id": USER_A},
        include=["metadatas"],
    )


def test_list_documents_skips_missing_document_ids(mock_collection):
    mock_collection.get.return_value = {
        "metadatas": [
            {"filename": "unknown.txt"},
            {
                "document_id": "doc-1",
                "user_id": USER_A,
                "filename": "security.txt",
                "page_number": 1,
            },
        ]
    }

    result = vector_store.list_documents(USER_A)

    assert len(result) == 1
    assert result[0]["document_id"] == "doc-1"


def test_list_documents_handles_missing_metadata_fields(mock_collection):
    mock_collection.get.return_value = {
        "metadatas": [
            {"document_id": "doc-1"},
        ]
    }

    result = vector_store.list_documents(USER_A)

    assert result == [
        {
            "document_id": "doc-1",
            "filename": "unknown",
            "pages": 0,
            "chunks": 1,
        }
    ]


# DOCUMENT COUNT

def test_get_document_count(mock_collection):
    mock_collection.get.return_value = {
        "metadatas": [
            {"document_id": "doc-1", "filename": "one.txt"},
            {"document_id": "doc-1", "filename": "one.txt"},
            {"document_id": "doc-2", "filename": "two.txt"},
        ]
    }

    result = vector_store.get_document_count(USER_A)

    assert result == 2


# USER ISOLATION

def test_user_a_and_user_b_have_different_filters(mock_collection):
    mock_collection.get.return_value = {
        "ids": [f"chunk-{i}" for i in range(5)],
        "metadatas": [],
    }

    mock_collection.query.return_value = {
        "documents": [["Private chunk"]],
        "metadatas": [[{"user_id": USER_A}]],
        "distances": [[0.2]],
    }

    vector_store.search_documents(
        query_embedding=[0.1, 0.2],
        user_id=USER_A,
    )

    filter_a = mock_collection.query.call_args.kwargs["where"]

    vector_store.search_documents(
        query_embedding=[0.1, 0.2],
        user_id=USER_B,
    )

    filter_b = mock_collection.query.call_args.kwargs["where"]

    assert filter_a == {"user_id": USER_A}
    assert filter_b == {"user_id": USER_B}
    assert filter_a != filter_b
