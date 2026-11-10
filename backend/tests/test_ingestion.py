
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from docx import Document

from app.rag.ingestion import (
    extract_document_text,
    ingest_document,
)


def create_docx():
    """Create a small DOCX file in memory."""
    document = Document()
    document.add_paragraph("Cybersecurity document.")
    document.add_paragraph("SQL Injection prevention.")
    
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_extract_txt_success():
    content = b"SQL Injection prevention."

    result = extract_document_text(content, ".txt")

    assert len(result) == 1
    assert result[0]["page_number"] == 1
    assert "SQL Injection prevention." in result[0]["text"]


def test_extract_txt_with_utf8_bom():
    content = b"\xef\xbb\xbfCybersecurity knowledge."

    result = extract_document_text(content, ".txt")

    assert result[0]["text"] == "Cybersecurity knowledge."


def test_extract_docx_success():
    content = create_docx()

    result = extract_document_text(content, ".docx")

    assert len(result) == 1
    assert result[0]["page_number"] == 1
    assert "Cybersecurity document." in result[0]["text"]
    assert "SQL Injection prevention." in result[0]["text"]


def test_extract_docx_ignores_empty_paragraphs():
    document = Document()
    document.add_paragraph("")
    document.add_paragraph("Useful security information.")

    buffer = BytesIO()
    document.save(buffer)

    result = extract_document_text(buffer.getvalue(), ".docx")

    assert len(result) == 1
    assert result[0]["text"] == "Useful security information."


def test_extract_pdf_success():
    mock_reader = SimpleNamespace(
        is_encrypted=False,
        pages=[
            SimpleNamespace(
                extract_text=lambda: "First page security content."
            ),
            SimpleNamespace(
                extract_text=lambda: "Second page security content."
            ),
        ],
    )

    with patch(
        "app.rag.ingestion.PdfReader",
        return_value=mock_reader,
    ):
        result = extract_document_text(b"fake pdf content", ".pdf")

    assert len(result) == 2
    assert result[0]["page_number"] == 1
    assert result[1]["page_number"] == 2


def test_extract_pdf_skips_empty_pages():
    mock_reader = SimpleNamespace(
        is_encrypted=False,
        pages=[
            SimpleNamespace(extract_text=lambda: ""),
            SimpleNamespace(extract_text=lambda: "Readable content."),
        ],
    )

    with patch(
        "app.rag.ingestion.PdfReader",
        return_value=mock_reader,
    ):
        result = extract_document_text(b"fake pdf content", ".pdf")

    assert len(result) == 1
    assert result[0]["page_number"] == 2


def test_extract_encrypted_pdf_raises_error():
    mock_reader = SimpleNamespace(is_encrypted=True, pages=[])

    with patch(
        "app.rag.ingestion.PdfReader",
        return_value=mock_reader,
    ):
        with pytest.raises(
            ValueError,
            match="Encrypted PDF files are not supported.",
        ):
            extract_document_text(b"fake pdf content", ".pdf")


def test_extract_unsupported_extension_raises_error():
    with pytest.raises(
        ValueError,
        match="Unsupported file type.",
    ):
        extract_document_text(b"content", ".exe")


def test_extract_empty_txt_raises_error():
    with pytest.raises(
        ValueError,
        match="No readable text found in document.",
    ):
        extract_document_text(b"   \n  ", ".txt")


def test_extract_empty_docx_raises_error():
    document = Document()
    buffer = BytesIO()
    document.save(buffer)

    with pytest.raises(
        ValueError,
        match="No readable text found in document.",
    ):
        extract_document_text(buffer.getvalue(), ".docx")


def test_ingest_rejects_unsupported_file():
    with pytest.raises(
        ValueError,
        match="Only PDF, TXT, and DOCX files are supported.",
    ):
        ingest_document(b"content", "malware.exe")


def test_ingest_txt_success():
    content = b"SQL Injection is a web security vulnerability."

    mock_embeddings = [[0.1, 0.2, 0.3]]

    with (
        patch(
            "app.rag.ingestion.generate_embeddings",
            return_value=mock_embeddings,
        ) as mock_generate,
        patch(
            "app.rag.ingestion.add_documents",
        ) as mock_add,
    ):
        result = ingest_document(content, "sample.txt")

    assert result["filename"] == "sample.txt"
    assert result["pages_processed"] == 1
    assert result["chunks_created"] == 1
    assert result["status"] == "indexed"
    assert result["document_id"]

    mock_generate.assert_called_once()
    mock_add.assert_called_once()

    stored_data = mock_add.call_args.kwargs

    assert stored_data["document_id"] == result["document_id"]
    assert stored_data["embeddings"] == mock_embeddings
    assert stored_data["metadatas"][0]["filename"] == "sample.txt"
    assert stored_data["metadatas"][0]["page_number"] == 1


def test_ingest_docx_success():
    content = create_docx()

    with (
        patch(
            "app.rag.ingestion.generate_embeddings",
            return_value=[[0.1, 0.2]],
        ),
        patch(
            "app.rag.ingestion.add_documents",
        ) as mock_add,
    ):
        result = ingest_document(content, "security.docx")

    assert result["filename"] == "security.docx"
    assert result["pages_processed"] == 1
    assert result["chunks_created"] >= 1
    assert result["status"] == "indexed"

    mock_add.assert_called_once()


def test_ingest_pdf_success():
    mock_reader = SimpleNamespace(
        is_encrypted=False,
        pages=[
            SimpleNamespace(
                extract_text=lambda: "PDF cybersecurity content."
            )
        ],
    )

    with (
        patch(
            "app.rag.ingestion.PdfReader",
            return_value=mock_reader,
        ),
        patch(
            "app.rag.ingestion.generate_embeddings",
            return_value=[[0.1, 0.2]],
        ),
        patch(
            "app.rag.ingestion.add_documents",
        ) as mock_add,
    ):
        result = ingest_document(b"fake pdf content", "security.pdf")

    assert result["filename"] == "security.pdf"
    assert result["pages_processed"] == 1
    assert result["chunks_created"] == 1

    mock_add.assert_called_once()
