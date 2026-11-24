"""
backend/embeddings/models.py

The three local models MultiDoc AI uses. All run on your laptop; none needs an API key.

  1. Text embedder   all-MiniLM-L6-v2 -> 384-number vector per text chunk / query.
                     Similar meaning => vectors point in a similar direction.
  2. Image embedder  CLIP ViT-B/32 -> 512-number vector. CLIP was trained on
                     image-caption pairs so that an IMAGE and a TEXT describing it
                     land close together. That is what makes "find the chart about
                     revenue" work: we embed the question with CLIP's TEXT encoder
                     and compare it to CLIP IMAGE vectors.
  3. Reranker        cross-encoder ms-marco-MiniLM-L-6-v2: reads (question, passage)
                     TOGETHER and outputs a relevance score. More accurate than
                     comparing two separately made vectors, but slower, so it only
                     re-orders the top few candidates.

The 384-d (MiniLM) and 512-d (CLIP) spaces are NOT comparable; they are stored in
separate ChromaDB collections and combined later by rank, never by raw score.

Models run on the GPU when one is available (Apple Silicon "mps" or NVIDIA "cuda"),
otherwise on the CPU; see MODEL_DEVICE in config.py. They are only USED here
(inference), never trained: all three are pre-trained models.

Models load lazily (on first use) and only once. The first load downloads the
weights from Hugging Face into ~/.cache/huggingface (about 90 MB, 600 MB and 90 MB).
"""

import logging
import os
import threading
from functools import lru_cache

# The newer "xet" download backend stalled on first download on this setup;
# plain HTTPS is reliable. Also hide per-model progress bars in the server log.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from PIL import Image

from backend.config import get_settings

logger = logging.getLogger(__name__)

# One lock per model: PyTorch inference from several threads at once is not
# guaranteed safe, and uploads are processed in background threads.
_text_lock = threading.Lock()
_image_lock = threading.Lock()
_rerank_lock = threading.Lock()
_load_lock = threading.Lock()


@lru_cache
def device() -> str:
    """The torch device the models run on: 'mps' (Apple GPU), 'cuda' (NVIDIA) or 'cpu'."""
    import torch

    wanted = get_settings().model_device
    available = {"cpu": True, "mps": torch.backends.mps.is_available(),
                 "cuda": torch.cuda.is_available()}
    if wanted == "auto":
        chosen = "cuda" if available["cuda"] else "mps" if available["mps"] else "cpu"
    elif available[wanted]:
        chosen = wanted
    else:
        logger.warning("MODEL_DEVICE=%s is not available on this machine; using CPU.", wanted)
        chosen = "cpu"
    logger.info("Local models will run on: %s", chosen)
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:
        pass
    return chosen


@lru_cache
def _text_model():
    from sentence_transformers import SentenceTransformer

    with _load_lock:
        name = get_settings().text_embedding_model
        logger.info("Loading text embedding model %s", name)
        return SentenceTransformer(name, device=device())


@lru_cache
def _image_model():
    from sentence_transformers import SentenceTransformer

    with _load_lock:
        name = get_settings().image_embedding_model
        logger.info("Loading image embedding model %s", name)
        return SentenceTransformer(name, device=device())


@lru_cache
def _reranker():
    from sentence_transformers import CrossEncoder

    with _load_lock:
        name = get_settings().reranker_model
        logger.info("Loading reranker %s", name)
        return CrossEncoder(name, device=device())


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Text chunks -> unit-length 384-d vectors (cosine similarity = dot product)."""
    if not texts:
        return []
    with _text_lock:
        vectors = _text_model().encode(texts, batch_size=32, normalize_embeddings=True,
                                       show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_images(images: list[Image.Image]) -> list[list[float]]:
    """Images -> unit-length 512-d CLIP vectors."""
    if not images:
        return []
    with _image_lock:
        vectors = _image_model().encode(images, batch_size=8, normalize_embeddings=True,
                                        show_progress_bar=False)
    return vectors.tolist()


def embed_text_for_image_search(text: str) -> list[float]:
    """A question -> CLIP text vector, comparable with CLIP image vectors."""
    with _image_lock:
        # CLIP's text encoder reads at most 77 tokens; long questions are truncated.
        vector = _image_model().encode([text[:300]], normalize_embeddings=True,
                                       show_progress_bar=False)
    return vector[0].tolist()


def rerank_scores(query: str, passages: list[str]) -> list[float]:
    """Higher = more relevant. Scores are unbounded logits (roughly -11 to +11)."""
    if not passages:
        return []
    with _rerank_lock:
        scores = _reranker().predict([(query, p) for p in passages], show_progress_bar=False)
    return [float(s) for s in scores]


def warm_up() -> None:
    """Load all models now (used by scripts so timing measurements exclude loading)."""
    embed_texts(["warm up"])
    embed_text_for_image_search("warm up")
    rerank_scores("warm up", ["warm up"])
