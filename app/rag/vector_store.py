
import chromadb

from app.config import CHROMA_DIR


client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    name="cybersecurity_knowledge",
    metadata={"hnsw:space": "cosine"},
)


def add_documents(chunks, embeddings, metadatas, document_id):
    if not chunks:
        return []

    ids = [
        f"{document_id}_{index}"
        for index in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return ids


def search_documents(query_embedding, top_k=5, document_id=None):
    count = collection.count()

    if count == 0:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    filters = None

    if document_id:
        filters = {"document_id": document_id}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        where=filters,
        include=["documents", "metadatas", "distances"],
    )

    return results


def delete_document(document_id):
    collection.delete(
        where={"document_id": document_id}
    )


def get_collection_count():
    return collection.count()
