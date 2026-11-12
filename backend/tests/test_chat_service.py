
from unittest.mock import patch

from app.services.chat_service import ask_cybersecurity_assistant


USER_ID = 101


@patch("app.services.chat_service.generate_answer")
@patch("app.services.chat_service.search_documents")
@patch("app.services.chat_service.generate_query_embedding")
def test_chat_service_success(
    mock_embedding,
    mock_search,
    mock_generate,
):
    mock_embedding.return_value = [0.1, 0.2, 0.3]

    mock_search.return_value = {
        "documents": [[
            "SQL Injection occurs when untrusted input is used in SQL queries.",
            "Use parameterized queries to prevent SQL Injection.",
        ]],
        "metadatas": [[
            {"filename": "sql_security.pdf", "page_number": 1},
            {"filename": "secure_coding.pdf", "page_number": 3},
        ]],
        "distances": [[0.20, 0.30]],
    }

    mock_generate.return_value = (
        "SQL Injection can be prevented using parameterized queries."
    )

    result = ask_cybersecurity_assistant(
        question="How can I prevent SQL Injection?",
        user_id=USER_ID,
        top_k=5,
    )

    assert result["answer"] == (
        "SQL Injection can be prevented using parameterized queries."
    )
    assert result["retrieval_count"] == 2
    assert len(result["sources"]) == 2

    assert result["sources"][0]["source_id"] == "S1"
    assert result["sources"][0]["filename"] == "sql_security.pdf"
    assert result["sources"][0]["page_number"] == 1
    assert result["sources"][0]["distance"] == 0.20

    assert result["sources"][1]["source_id"] == "S2"
    assert result["sources"][1]["filename"] == "secure_coding.pdf"

    mock_embedding.assert_called_once_with(
        "How can I prevent SQL Injection?"
    )

    mock_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        user_id=USER_ID,
        top_k=5,
    )

    mock_generate.assert_called_once()


@patch("app.services.chat_service.generate_answer")
@patch("app.services.chat_service.search_documents")
@patch("app.services.chat_service.generate_query_embedding")
def test_chat_service_no_relevant_documents(
    mock_embedding,
    mock_search,
    mock_generate,
):
    mock_embedding.return_value = [0.1, 0.2, 0.3]

    mock_search.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    result = ask_cybersecurity_assistant(
        question="Explain quantum computing.",
        user_id=USER_ID,
    )

    assert result["answer"] == (
    "I couldn't find relevant information "
    "in your cybersecurity knowledge base."
)
    assert result["sources"] == []
    assert result["retrieval_count"] == 0

    mock_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        user_id=USER_ID,
        top_k=5,
    )

    mock_generate.assert_not_called()


@patch("app.services.chat_service.generate_answer")
@patch("app.services.chat_service.search_documents")
@patch("app.services.chat_service.generate_query_embedding")
def test_chat_service_uses_default_top_k(
    mock_embedding,
    mock_search,
    mock_generate,
):
    mock_embedding.return_value = [0.1, 0.2, 0.3]

    mock_search.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    ask_cybersecurity_assistant(
        question="What is cybersecurity?",
        user_id=USER_ID,
    )

    mock_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        user_id=USER_ID,
        top_k=5,
    )


@patch("app.services.chat_service.generate_answer")
@patch("app.services.chat_service.search_documents")
@patch("app.services.chat_service.generate_query_embedding")
def test_chat_service_uses_custom_top_k(
    mock_embedding,
    mock_search,
    mock_generate,
):
    mock_embedding.return_value = [0.1, 0.2, 0.3]

    mock_search.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    ask_cybersecurity_assistant(
        question="What is cybersecurity?",
        user_id=USER_ID,
        top_k=3,
    )

    mock_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        user_id=USER_ID,
        top_k=3,
    )


@patch("app.services.chat_service.generate_answer")
@patch("app.services.chat_service.search_documents")
@patch("app.services.chat_service.generate_query_embedding")
def test_chat_service_handles_missing_metadata_fields(
    mock_embedding,
    mock_search,
    mock_generate,
):
    mock_embedding.return_value = [0.1, 0.2, 0.3]

    mock_search.return_value = {
        "documents": [["Security information."]],
        "metadatas": [[{}]],
        "distances": [[0.25]],
    }

    mock_generate.return_value = "Security information response."

    result = ask_cybersecurity_assistant(
        question="Explain this security information.",
        user_id=USER_ID,
    )

    source = result["sources"][0]

    assert source["filename"] == "Unknown document"
    assert source["page_number"] == "Unknown"
    assert source["source_id"] == "S1"

    mock_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        user_id=USER_ID,
        top_k=5,
    )


@patch("app.services.chat_service.generate_answer")
@patch("app.services.chat_service.search_documents")
@patch("app.services.chat_service.generate_query_embedding")
def test_chat_service_passes_context_to_generator(
    mock_embedding,
    mock_search,
    mock_generate,
):
    mock_embedding.return_value = [0.1, 0.2, 0.3]

    mock_search.return_value = {
        "documents": [["Use parameterized queries."]],
        "metadatas": [[{
            "filename": "owasp.pdf",
            "page_number": 5,
        }]],
        "distances": [[0.15]],
    }

    mock_generate.return_value = "Use parameterized queries."

    ask_cybersecurity_assistant(
        question="How do I prevent SQL Injection?",
        user_id=USER_ID,
    )

    kwargs = mock_generate.call_args.kwargs

    assert kwargs["question"] == "How do I prevent SQL Injection?"
    assert "SOURCE S1" in kwargs["context"]
    assert "Filename: owasp.pdf" in kwargs["context"]
    assert "Page: 5" in kwargs["context"]
    assert "Use parameterized queries." in kwargs["context"]

    mock_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        user_id=USER_ID,
        top_k=5,
    )


@patch("app.services.chat_service.generate_answer")
@patch("app.services.chat_service.search_documents")
@patch("app.services.chat_service.generate_query_embedding")
def test_chat_service_limits_content_preview_to_250_characters(
    mock_embedding,
    mock_search,
    mock_generate,
):
    mock_embedding.return_value = [0.1, 0.2, 0.3]

    long_document = "A" * 400

    mock_search.return_value = {
        "documents": [[long_document]],
        "metadatas": [[{
            "filename": "long_document.txt",
            "page_number": 1,
        }]],
        "distances": [[0.20]],
    }

    mock_generate.return_value = "Test answer."

    result = ask_cybersecurity_assistant(
        question="Explain this security document.",
        user_id=USER_ID,
    )

    preview = result["sources"][0]["content_preview"]

    assert len(preview) == 250
    assert preview == long_document[:250]

    mock_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        user_id=USER_ID,
        top_k=5,
    )
