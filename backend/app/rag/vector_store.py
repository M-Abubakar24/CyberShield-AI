
import chromadb

from app.config import CHROMA_DIR


# Initialize persistent ChromaDB client
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    name="cybersecurity_knowledge",
    metadata={"hnsw:space": "cosine"},
)


def add_documents(chunks, embeddings, metadatas, document_id):
    """Store document chunks, embeddings, and metadata."""

    if not chunks:
        return []

    if len(chunks) != len(embeddings) or len(chunks) != len(metadatas):
        raise ValueError(
            "Chunks, embeddings, and metadata must have equal lengths."
        )

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



def search_documents(
    query_embedding,
    top_k=5,
    document_id=None,
    distance_threshold=0.60,
):
    """
    Retrieve the most relevant document chunks.

    Lower cosine distance indicates better semantic similarity.
    Retrieve extra candidates before applying the relevance threshold.
    """

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

    candidate_count = min(top_k * 3, count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        where=filters,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    filtered_results = [
        (document, metadata, distance)
        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        )
        if distance <= distance_threshold
    ]

    # Keep the best matching chunks first.
    filtered_results.sort(key=lambda item: item[2])
    filtered_results = filtered_results[:top_k]

    return {
        "documents": [[item[0] for item in filtered_results]],
        "metadatas": [[item[1] for item in filtered_results]],
        "distances": [[item[2] for item in filtered_results]],
    }


def delete_document(document_id):
    """Delete all chunks belonging to a document and return the count."""

    results = collection.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )

    document_ids = results["ids"]

    if not document_ids:
        return 0

    collection.delete(ids=document_ids)

    return len(document_ids)



def get_collection_count():
    """Return the total number of indexed chunks."""

    return collection.count()
def rename_document(document_id, new_filename):
    results = collection.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )

    if not results["ids"]:
        return 0

    updated_metadatas = []

    for metadata in results["metadatas"]:
        updated_metadata = metadata.copy()
        updated_metadata["filename"] = new_filename
        updated_metadatas.append(updated_metadata)

    collection.update(
        ids=results["ids"],
        metadatas=updated_metadatas,
    )

    return len(results["ids"])
    
def list_documents():
    """Return unique uploaded documents and their chunk counts."""

    results = collection.get(include=["metadatas"])

    documents = {}

    for metadata in results["metadatas"]:
        document_id = metadata.get("document_id")

        if not document_id:
            continue

        if document_id not in documents:
            documents[document_id] = {
                "document_id": document_id,
                "filename": metadata.get("filename", "unknown"),
                "pages": set(),
                "chunks": 0,
            }

        documents[document_id]["chunks"] += 1

        page_number = metadata.get("page_number")
        if page_number is not None:
            documents[document_id]["pages"].add(page_number)

    output = []

    for document in documents.values():
        document["pages"] = len(document["pages"])
        output.append(document)

    return output


def get_document_count():
    """Return the number of unique uploaded documents."""

    return len(list_documents())
