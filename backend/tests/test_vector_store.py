
from unittest.mock import MagicMock

import pytest

from app.rag import vector_store


@pytest.fixture
def mock_collection(monkeypatch):
    """Replace the real ChromaDB collection with a mock."""
    mock = MagicMock()
    monkeypatch.setattr(vector_store, "collection", mock)
    return mock


# --------------------------------------------------
# ADD DOCUMENTS
# --------------------------------------------------

def test_add_documents_success(mock_collection):
    ids = vector_store.add_documents(
        chunks=["First chunk", "Second chunk"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[
            {"document_id": "doc-1", "filename": "test.txt"},
            {"document_id": "doc-1", "filename": "test.txt"},
        ],
        document_id="doc-1",
    )

    assert ids == ["doc-1_0", "doc-1_1"]
    mock_collection.add.assert_called_once()


def test_add_documents_empty_chunks(mock_collection):
    result = vector_store.add_documents(
        chunks=[],
        embeddings=[],
        metadatas=[],
        document_id="doc-1",
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
        )

    mock_collection.add.assert_not_called()


# --------------------------------------------------
# SEARCH DOCUMENTS
# --------------------------------------------------

def test_search_documents_empty_collection(mock_collection):
    mock_collection.count.return_value = 0

    result = vector_store.search_documents(
        query_embedding=[0.1, 0.2]
    )

    assert result == {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    mock_collection.query.assert_not_called()


def test_search_documents_success(mock_collection):
    mock_collection.count.return_value = 3

    mock_collection.query.return_value = {
        "documents": [[
            "First relevant chunk",
            "Second relevant chunk",
        ]],
        "metadatas": [[
            {"filename": "first.txt"},
            {"filename": "second.txt"},
        ]],
        "distances": [[0.20, 0.35]],
    }

    result = vector_store.search_documents(
        query_embedding=[0.1, 0.2],
        top_k=5,
    )

    assert len(result["documents"][0]) == 2
    assert result["documents"][0][0] == "First relevant chunk"
    assert result["distances"][0] == [0.20, 0.35]

    mock_collection.query.assert_called_once()


def test_search_documents_applies_document_filter(mock_collection):
    mock_collection.count.return_value = 5
    mock_collection.query.return_value = {
        "documents": [["Document chunk"]],
        "metadatas": [[{"document_id": "doc-123"}]],
        "distances": [[0.25]],
    }

    vector_store.search_documents(
        query_embedding=[0.1, 0.2],
        document_id="doc-123",
    )

    call_kwargs = mock_collection.query.call_args.kwargs

    assert call_kwargs["where"] == {
        "document_id": "doc-123"
    }


def test_search_documents_without_document_filter(mock_collection):
    mock_collection.count.return_value = 5
    mock_collection.query.return_value = {
        "documents": [["Chunk"]],
        "metadatas": [[{"filename": "test.txt"}]],
        "distances": [[0.2]],
    }

    vector_store.search_documents(
        query_embedding=[0.1, 0.2]
    )

    call_kwargs = mock_collection.query.call_args.kwargs

    assert call_kwargs["where"] is None


def test_search_documents_filters_irrelevant_results(mock_collection):
    mock_collection.count.return_value = 3
    mock_collection.query.return_value = {
        "documents": [[
            "Relevant chunk",
            "Irrelevant chunk",
        ]],
        "metadatas": [[
            {"filename": "relevant.txt"},
            {"filename": "irrelevant.txt"},
        ]],
        "distances": [[0.25, 0.85]],
    }

    result = vector_store.search_documents(
        query_embedding=[0.1, 0.2],
        distance_threshold=0.60,
    )

    assert result["documents"][0] == ["Relevant chunk"]
    assert result["distances"][0] == [0.25]


def test_search_documents_sorts_and_limits_results(mock_collection):
    mock_collection.count.return_value = 10
    mock_collection.query.return_value = {
        "documents": [[
            "Third chunk",
            "First chunk",
            "Second chunk",
        ]],
        "metadatas": [[
            {"filename": "third.txt"},
            {"filename": "first.txt"},
            {"filename": "second.txt"},
        ]],
        "distances": [[0.30, 0.10, 0.20]],
    }

    result = vector_store.search_documents(
        query_embedding=[0.1, 0.2],
        top_k=2,
    )

    assert result["documents"][0] == [
        "First chunk",
        "Second chunk",
    ]

    assert result["distances"][0] == [0.10, 0.20]

    call_kwargs = mock_collection.query.call_args.kwargs
    assert call_kwargs["n_results"] == 6


# --------------------------------------------------
# DELETE DOCUMENT
# --------------------------------------------------

def test_delete_document_success(mock_collection):
    mock_collection.get.return_value = {
        "ids": ["doc-1_0", "doc-1_1"],
        "metadatas": [{}, {}],
    }

    result = vector_store.delete_document("doc-1")

    assert result == 2

    mock_collection.delete.assert_called_once_with(
        ids=["doc-1_0", "doc-1_1"]
    )


def test_delete_document_not_found(mock_collection):
    mock_collection.get.return_value = {
        "ids": [],
        "metadatas": [],
    }

    result = vector_store.delete_document("missing-doc")

    assert result == 0
    mock_collection.delete.assert_not_called()


# --------------------------------------------------
# COLLECTION COUNT
# --------------------------------------------------

def test_get_collection_count(mock_collection):
    mock_collection.count.return_value = 15

    result = vector_store.get_collection_count()

    assert result == 15
    mock_collection.count.assert_called_once()


# --------------------------------------------------
# RENAME DOCUMENT
# --------------------------------------------------

def test_rename_document_success(mock_collection):
    mock_collection.get.return_value = {
        "ids": ["doc-1_0", "doc-1_1"],
        "metadatas": [
            {
                "document_id": "doc-1",
                "filename": "old.txt",
                "page_number": 1,
            },
            {
                "document_id": "doc-1",
                "filename": "old.txt",
                "page_number": 2,
            },
        ],
    }

    result = vector_store.rename_document(
        "doc-1",
        "new.txt",
    )

    assert result == 2

    update_kwargs = mock_collection.update.call_args.kwargs

    assert update_kwargs["ids"] == ["doc-1_0", "doc-1_1"]
    assert all(
        metadata["filename"] == "new.txt"
        for metadata in update_kwargs["metadatas"]
    )

    # Original metadata should not be modified.
    assert mock_collection.get.return_value["metadatas"][0]["filename"] == "old.txt"


def test_rename_document_not_found(mock_collection):
    mock_collection.get.return_value = {
        "ids": [],
        "metadatas": [],
    }

    result = vector_store.rename_document(
        "missing-doc",
        "new.txt",
    )

    assert result == 0
    mock_collection.update.assert_not_called()


# --------------------------------------------------
# LIST DOCUMENTS
# --------------------------------------------------

def test_list_documents_groups_chunks_correctly(mock_collection):
    mock_collection.get.return_value = {
        "metadatas": [
            {
                "document_id": "doc-1",
                "filename": "security.pdf",
                "page_number": 1,
            },
            {
                "document_id": "doc-1",
                "filename": "security.pdf",
                "page_number": 1,
            },
            {
                "document_id": "doc-1",
                "filename": "security.pdf",
                "page_number": 2,
            },
            {
                "document_id": "doc-2",
                "filename": "network.txt",
                "page_number": 1,
            },
        ]
    }

    result = vector_store.list_documents()

    assert len(result) == 2

    first_document = next(
        item for item in result
        if item["document_id"] == "doc-1"
    )

    assert first_document["filename"] == "security.pdf"
    assert first_document["chunks"] == 3
    assert first_document["pages"] == 2


def test_list_documents_skips_missing_document_ids(mock_collection):
    mock_collection.get.return_value = {
        "metadatas": [
            {"filename": "unknown.txt"},
            {
                "document_id": "doc-1",
                "filename": "security.txt",
                "page_number": 1,
            },
        ]
    }

    result = vector_store.list_documents()

    assert len(result) == 1
    assert result[0]["document_id"] == "doc-1"


def test_list_documents_handles_missing_metadata_fields(mock_collection):
    mock_collection.get.return_value = {
        "metadatas": [
            {"document_id": "doc-1"},
        ]
    }

    result = vector_store.list_documents()

    assert result == [
        {
            "document_id": "doc-1",
            "filename": "unknown",
            "pages": 0,
            "chunks": 1,
        }
    ]


# --------------------------------------------------
# DOCUMENT COUNT
# --------------------------------------------------

def test_get_document_count(mock_collection):
    mock_collection.get.return_value = {
        "metadatas": [
            {"document_id": "doc-1", "filename": "one.txt"},
            {"document_id": "doc-1", "filename": "one.txt"},
            {"document_id": "doc-2", "filename": "two.txt"},
        ]
    }

    result = vector_store.get_document_count()

    assert result == 2
