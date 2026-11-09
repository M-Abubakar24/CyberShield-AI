
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from docx import Document
from pypdf import PdfReader

from app.rag.chunking import chunk_text
from app.rag.embeddings import generate_embeddings
from app.rag.vector_store import add_documents


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def extract_document_text(content: bytes, extension: str):
    pages = []

    if extension == ".pdf":
        reader = PdfReader(BytesIO(content))

        if reader.is_encrypted:
            raise ValueError("Encrypted PDF files are not supported.")

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                pages.append({
                    "page_number": page_number,
                    "text": text,
                })

    elif extension == ".docx":
        document = Document(BytesIO(content))

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

        if text.strip():
            pages.append({
                "page_number": 1,
                "text": text,
            })

    elif extension == ".txt":
        text = content.decode("utf-8-sig")

        if text.strip():
            pages.append({
                "page_number": 1,
                "text": text,
            })

    else:
        raise ValueError("Unsupported file type.")

    if not pages:
        raise ValueError("No readable text found in document.")

    return pages


def ingest_document(content: bytes, filename: str):
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Only PDF, TXT, and DOCX files are supported."
        )

    pages = extract_document_text(content, extension)

    document_id = str(uuid4())

    all_chunks = []
    all_metadatas = []

    for page in pages:
        chunks = chunk_text(page["text"])

        for chunk in chunks:
            all_chunks.append(chunk)

            all_metadatas.append({
                "document_id": document_id,
                "filename": Path(filename).name,
                "page_number": page["page_number"],
            })

    embeddings = generate_embeddings(all_chunks)

    add_documents(
        chunks=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
        document_id=document_id,
    )

    return {
        "document_id": document_id,
        "filename": Path(filename).name,
        "pages_processed": len(pages),
        "chunks_created": len(all_chunks),
        "status": "indexed",
    }
