
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.rag.ingestion import ingest_document
from app.rag.embeddings import generate_query_embedding
from app.rag.vector_store import search_documents, get_collection_count

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["Documents"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read(settings.MAX_FILE_SIZE_MB * 1024 * 1024 + 1)

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds maximum size.")

    try:
        return ingest_document(content, file.filename or "unknown")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Document upload failed")
        raise HTTPException(status_code=500, detail="Document processing failed.") from exc


@router.post("/search")
def search_document(request: SearchRequest):
    if get_collection_count() == 0:
        return {"results": [], "message": "No documents have been indexed yet."}

    try:
        embedding = generate_query_embedding(request.query)
        results = search_documents(embedding, top_k=request.top_k)

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
        raise HTTPException(status_code=500, detail="Document search failed.") from exc


@router.get("/stats")
def document_stats():
    return {
        "total_chunks": get_collection_count(),
        "collection": "cybersecurity_knowledge",
    }
