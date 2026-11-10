
from unittest.mock import MagicMock, patch

import pytest

from app.rag.embeddings import (
    get_embedding_model,
    generate_embeddings,
    generate_query_embedding,
)


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Clear the cached model before and after each test."""
    get_embedding_model.cache_clear()
    yield
    get_embedding_model.cache_clear()


def test_get_embedding_model_loads_model():
    """Test that the configured embedding model is loaded."""

    mock_model = MagicMock()

    with patch(
        "app.rag.embeddings.SentenceTransformer",
        return_value=mock_model,
    ) as mock_transformer:

        result = get_embedding_model()

    mock_transformer.assert_called_once()
    assert result is mock_model


def test_get_embedding_model_uses_cache():
    """The model should only load once because of lru_cache."""

    mock_model = MagicMock()

    with patch(
        "app.rag.embeddings.SentenceTransformer",
        return_value=mock_model,
    ) as mock_transformer:

        first_result = get_embedding_model()
        second_result = get_embedding_model()

    assert first_result is second_result
    mock_transformer.assert_called_once()


def test_generate_embeddings_with_valid_text():
    """Test generation of normalized document embeddings."""

    mock_model = MagicMock()

    mock_model.encode.return_value.tolist.return_value = [
        [0.6, 0.8],
        [0.8, 0.6],
    ]

    with patch(
        "app.rag.embeddings.get_embedding_model",
        return_value=mock_model,
    ):
        result = generate_embeddings([
            "SQL Injection",
            "Cross-Site Scripting",
        ])

    assert result == [
        [0.6, 0.8],
        [0.8, 0.6],
    ]

    mock_model.encode.assert_called_once_with(
        ["SQL Injection", "Cross-Site Scripting"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def test_generate_embeddings_with_empty_list():
    """Empty input should return an empty list without loading a model."""

    with patch(
        "app.rag.embeddings.get_embedding_model"
    ) as mock_get_model:

        result = generate_embeddings([])

    assert result == []
    mock_get_model.assert_not_called()


def test_generate_query_embedding():
    """Test that query embedding returns a single vector."""

    mock_model = MagicMock()

    mock_model.encode.return_value.tolist.return_value = [
        [0.6, 0.8]
    ]

    with patch(
        "app.rag.embeddings.get_embedding_model",
        return_value=mock_model,
    ):
        result = generate_query_embedding(
            "How can SQL Injection be prevented?"
        )

    assert result == [0.6, 0.8]


def test_embedding_model_loading_error():
    """Model loading errors should propagate to the caller."""

    with patch(
        "app.rag.embeddings.SentenceTransformer",
        side_effect=RuntimeError("Model loading failed"),
    ):

        with pytest.raises(
            RuntimeError,
            match="Model loading failed",
        ):
            get_embedding_model()


def test_embedding_generation_error():
    """Embedding generation errors should propagate to the caller."""

    mock_model = MagicMock()

    mock_model.encode.side_effect = RuntimeError(
        "Embedding generation failed"
    )

    with patch(
        "app.rag.embeddings.get_embedding_model",
        return_value=mock_model,
    ):

        with pytest.raises(
            RuntimeError,
            match="Embedding generation failed",
        ):
            generate_embeddings(["SQL Injection"])
