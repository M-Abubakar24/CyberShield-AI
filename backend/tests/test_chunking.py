
import pytest

from app.rag.chunking import chunk_text


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []


def test_whitespace_only_text_returns_empty_list():
    assert chunk_text("   \n\t   ") == []


def test_short_text_returns_single_chunk():
    text = "SQL Injection is a web security vulnerability."

    result = chunk_text(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert result == [text]


def test_text_is_normalized():
    text = "SQL   Injection\n\nis a   vulnerability."

    result = chunk_text(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert result == ["SQL Injection is a vulnerability."]


def test_long_text_creates_multiple_chunks():
    text = " ".join(["cybersecurity"] * 100)

    result = chunk_text(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(result) > 1
    assert all(len(chunk) <= 100 for chunk in result)


def test_chunks_have_overlap():
    text = " ".join([f"word{i}" for i in range(50)])

    result = chunk_text(
        text,
        chunk_size=50,
        chunk_overlap=15,
    )

    assert len(result) > 1

    # Consecutive chunks should share some text.
    for previous, current in zip(result, result[1:]):
        previous_words = set(previous.split())
        current_words = set(current.split())

        assert previous_words.intersection(current_words)


def test_chunk_size_zero_raises_error():
    with pytest.raises(
        ValueError,
        match="chunk_size must be greater than zero",
    ):
        chunk_text("Test text", chunk_size=0)


def test_negative_chunk_size_raises_error():
    with pytest.raises(ValueError):
        chunk_text("Test text", chunk_size=-10)


def test_negative_overlap_raises_error():
    with pytest.raises(ValueError):
        chunk_text(
            "Test text",
            chunk_size=100,
            chunk_overlap=-1,
        )


def test_overlap_equal_to_chunk_size_raises_error():
    with pytest.raises(ValueError):
        chunk_text(
            "Test text",
            chunk_size=100,
            chunk_overlap=100,
        )


def test_overlap_greater_than_chunk_size_raises_error():
    with pytest.raises(ValueError):
        chunk_text(
            "Test text",
            chunk_size=100,
            chunk_overlap=120,
        )


def test_chunks_preserve_all_words():
    text = " ".join([f"security{i}" for i in range(100)])

    result = chunk_text(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    reconstructed_words = set(" ".join(result).split())
    original_words = set(text.split())

    assert original_words.issubset(reconstructed_words)
