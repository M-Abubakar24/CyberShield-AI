
from app.rag.embeddings import generate_query_embedding
from app.rag.vector_store import search_documents
from app.rag.generation import generate_answer


def ask_cybersecurity_assistant(
    question: str,
    user_id: int,
    top_k: int = 5,
):
    # 1. Convert question into an embedding
    query_embedding = generate_query_embedding(question)

    # 2. Retrieve only the current user's document chunks
    results = search_documents(
        query_embedding=query_embedding,
        user_id=user_id,
        top_k=top_k,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return {
            "answer": (
                "I couldn't find relevant information "
                "in your cybersecurity knowledge base."
            ),
            "sources": [],
            "retrieval_count": 0,
        }

    # 3. Prepare context with source labels
    context_parts = []
    sources = []

    for index, document in enumerate(documents):
        metadata = metadatas[index]

        filename = metadata.get(
            "filename",
            "Unknown document",
        )

        page_number = metadata.get(
            "page_number",
            "Unknown",
        )

        source_label = f"S{index + 1}"

        context_parts.append(
            f"""
SOURCE {source_label}
Filename: {filename}
Page: {page_number}

Content:
{document}
"""
        )

        sources.append({
            "source_id": source_label,
            "filename": filename,
            "page_number": page_number,
            "distance": distances[index],
            "content_preview": document[:250],
        })

    context = "\n\n".join(context_parts)

    # 4. Generate AI response using only retrieved user content
    answer = generate_answer(
        question=question,
        context=context,
    )

    # 5. Return answer and references
    return {
        "answer": answer,
        "sources": sources,
        "retrieval_count": len(documents),
    }
