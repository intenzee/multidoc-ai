"""
backend/storage/vector_store.py

ChromaDB wrapper. Two collections, because the two embedding spaces are different:

  text_chunks : 384-d MiniLM vectors for text, OCR text, tables and image captions
  images      : 512-d CLIP vectors for images

Each collection records which embedding model built it. If you later change the
model in .env, the app refuses to mix old and new vectors and tells you to rebuild
(mixing vectors from different models gives meaningless similarity scores).
"""

import logging
from functools import lru_cache
from typing import Any

import chromadb

from backend.config import get_settings

logger = logging.getLogger(__name__)

TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "images"


class IndexMismatchError(RuntimeError):
    pass


@lru_cache
def _client() -> chromadb.ClientAPI:
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )


def _collection(name: str, model_name: str):
    collection = _client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "embedding_model": model_name},
    )
    built_with = (collection.metadata or {}).get("embedding_model")
    if built_with and built_with != model_name:
        raise IndexMismatchError(
            f"The '{name}' index was built with {built_with}, but the configured model is "
            f"{model_name}. Delete the data/chroma folder and reprocess your documents."
        )
    return collection


def text_collection():
    return _collection(TEXT_COLLECTION, get_settings().text_embedding_model)


def image_collection():
    return _collection(IMAGE_COLLECTION, get_settings().image_embedding_model)


def _clean_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool (no None)."""
    return {k: v for k, v in meta.items() if v is not None}


def add_items(collection, ids: list[str], embeddings: list[list[float]],
              documents: list[str] | None, metadatas: list[dict]) -> None:
    if not ids:
        return
    batch = 500
    for start in range(0, len(ids), batch):
        end = start + batch
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end] if documents else None,
            metadatas=[_clean_metadata(m) for m in metadatas[start:end]],
        )


def query(collection, embedding: list[float], k: int,
          document_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Nearest neighbours. Returns [{id, similarity, metadata}] best first."""
    if k <= 0 or collection.count() == 0:
        return []
    where = {"document_id": {"$in": document_ids}} if document_ids else None
    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(k, collection.count()),
        where=where,
        include=["metadatas", "distances"],
    )
    hits = []
    for item_id, meta, distance in zip(result["ids"][0], result["metadatas"][0],
                                       result["distances"][0]):
        # Chroma returns cosine DISTANCE (0 = identical); similarity = 1 - distance.
        hits.append({"id": item_id, "similarity": 1.0 - float(distance), "metadata": meta})
    return hits


def delete_document_vectors(document_id: str) -> None:
    for collection in (text_collection(), image_collection()):
        collection.delete(where={"document_id": document_id})


def reset_cache() -> None:
    """Forget the cached client (used by tests and evaluation runs with other data dirs)."""
    _client.cache_clear()
