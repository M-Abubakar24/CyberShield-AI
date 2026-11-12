
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.auth.dependencies import get_current_user
from app.config import settings
from app.models.user import User
from app.rag.ingestion import ingest_document
from app.rag.embeddings import generate_query_embedding
from app.rag.vector_store import (
    search_documents,
    get_collection_count,
    list_documents,
    get_document_count,
    delete_document,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    content = await file.read(
        settings.MAX_FILE_SIZE_MB * 1024 * 1024 + 1
    )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File exceeds maximum size.",
        )

    try:
        return ingest_document(
            content=content,
            filename=file.filename or "unknown",
            user_id=current_user.id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        logger.exception("Document upload failed")

        raise HTTPException(
            status_code=500,
            detail="Document processing failed.",
        ) from exc

    finally:
        await file.close()


@router.post("/search")
def search_document(
    request: SearchRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        user_id = current_user.id

        if get_collection_count(user_id=user_id) == 0:
            return {
                "results": [],
                "message": "No documents have been indexed yet.",
            }

        embedding = generate_query_embedding(request.query)

        results = search_documents(
            query_embedding=embedding,
            user_id=user_id,
            top_k=request.top_k,
        )

        return {
            "results": [
                {
                    "content": document,
                    "metadata": metadata,
                    "distance": distance,
                }
                for document, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]
        }

    except Exception as exc:
        logger.exception("Document search failed")

        raise HTTPException(
            status_code=500,
            detail="Document search failed.",
        ) from exc


@router.get("/stats")
def document_stats(
    current_user: User = Depends(get_current_user),
):
    """Return statistics for the current user's knowledge base."""

    user_id = current_user.id

    return {
        "success": True,
        "total_documents": get_document_count(user_id=user_id),
        "total_chunks": get_collection_count(user_id=user_id),
        "collection": "cybersecurity_knowledge",
    }


@router.get("/")
def get_documents(
    current_user: User = Depends(get_current_user),
):
    """List only documents belonging to the current user."""

    documents = list_documents(user_id=current_user.id)

    return {
        "success": True,
        "total_documents": len(documents),
        "documents": documents,
    }


@router.delete("/{document_id}")
def remove_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
):
    """Delete only a document belonging to the current user."""

    deleted_chunks = delete_document(
        document_id=document_id,
        user_id=current_user.id,
    )

    if deleted_chunks == 0:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {
        "success": True,
        "message": "Document deleted successfully.",
        "document_id": document_id,
        "deleted_chunks": deleted_chunks,
    }
