
import chromadb

from app.config import CHROMA_DIR


# Initialize persistent ChromaDB client
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    name="cybersecurity_knowledge",
    metadata={"hnsw:space": "cosine"},
)


def _validate_user_id(user_id):
    """Require a valid user ID for every document operation."""

    if user_id is None or not str(user_id).strip():
        raise ValueError("A valid user_id is required.")

    return str(user_id)


def add_documents(chunks, embeddings, metadatas, document_id, user_id):
    """Store document chunks with their owner's user ID."""

    user_id = _validate_user_id(user_id)

    if not chunks:
        return []

    if len(chunks) != len(embeddings) or len(chunks) != len(metadatas):
        raise ValueError(
            "Chunks, embeddings, and metadata must have equal lengths."
        )

    # Enforce ownership on every stored chunk.
    secured_metadatas = []

    for metadata in metadatas:
        secured_metadata = metadata.copy()
        secured_metadata["user_id"] = user_id
        secured_metadatas.append(secured_metadata)

    ids = [
        f"{document_id}_{index}"
        for index in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=secured_metadatas,
    )

    return ids



def search_documents(
    query_embedding,
    user_id,
    top_k=5,
    document_id=None,
    distance_threshold=0.60,
):
    """Search only documents belonging to the authenticated user."""

    user_id = _validate_user_id(user_id)

    # Count this user's records using a filtered get operation.
    user_records = collection.get(
        where={"user_id": user_id},
        include=["metadatas"],
    )

    user_count = len(user_records.get("ids", []))

    empty_result = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    if user_count == 0:
        return empty_result

    # Always apply the user ownership filter.
    filters = {"user_id": user_id}

    # If a document is specified, apply both filters.
    if document_id:
        filters = {
            "$and": [
                {"user_id": user_id},
                {"document_id": document_id},
            ]
        }

    candidate_count = min(top_k * 3, user_count)

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

    filtered_results.sort(key=lambda item: item[2])
    filtered_results = filtered_results[:top_k]

    return {
        "documents": [[item[0] for item in filtered_results]],
        "metadatas": [[item[1] for item in filtered_results]],
        "distances": [[item[2] for item in filtered_results]],
    }

def delete_document(document_id, user_id):
    """Delete a document only if it belongs to the specified user."""

    user_id = _validate_user_id(user_id)

    results = collection.get(
        where={
            "$and": [
                {"document_id": document_id},
                {"user_id": user_id},
            ]
        },
        include=["metadatas"],
    )

    document_ids = results.get("ids", [])

    if not document_ids:
        return 0

    collection.delete(ids=document_ids)

    return len(document_ids)


def get_collection_count(user_id):
    """Return the total indexed chunks belonging to one user."""

    user_id = _validate_user_id(user_id)

    results = collection.get(
        where={"user_id": user_id},
        include=["metadatas"],
    )

    return len(results.get("ids", []))


def rename_document(document_id, new_filename, user_id):
    """Rename a document only if it belongs to the specified user."""

    user_id = _validate_user_id(user_id)

    results = collection.get(
        where={
            "$and": [
                {"document_id": document_id},
                {"user_id": user_id},
            ]
        },
        include=["metadatas"],
    )

    if not results.get("ids"):
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


def list_documents(user_id):
    """List only documents belonging to the specified user."""

    user_id = _validate_user_id(user_id)

    results = collection.get(
        where={"user_id": user_id},
        include=["metadatas"],
    )

    documents = {}

    for metadata in results.get("metadatas", []):
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


def get_document_count(user_id):
    """Return the number of unique documents belonging to one user."""

    return len(list_documents(user_id))
