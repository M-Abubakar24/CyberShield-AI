
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_embedding_model():
    print("Loading cybersecurity embedding model...")

    model = SentenceTransformer(settings.EMBEDDING_MODEL)

    print("Embedding model loaded successfully.")

    return model


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    embeddings = get_embedding_model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def generate_query_embedding(query: str) -> list[float]:
    return generate_embeddings([query])[0]
