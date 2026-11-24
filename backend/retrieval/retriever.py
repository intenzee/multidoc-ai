"""
backend/retrieval/retriever.py

Find the evidence for a question. No LLM is used here (free, fast, testable).

    question
      ├─ dense   : MiniLM vector  -> Chroma "text_chunks"   (meaning)
      ├─ sparse  : BM25 keywords  -> SQLite chunks          (exact words, numbers, names)
      │     └─ fused with Reciprocal Rank Fusion (RRF)
      ├─ rerank  : cross-encoder scores (question, chunk) pairs together (precision)
      └─ images  : CLIP text vector -> Chroma "images"      (charts, figures, scans)
                   + the figure behind any retrieved image caption

Every item keeps all of its scores so the UI and the evaluation can show WHY it
was retrieved. Retrieval modes and options can be changed per request, which is
what the experiments (dense vs hybrid, with/without reranking...) use.
"""

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

from backend.config import get_settings
from backend.embeddings import models
from backend.storage import database as db
from backend.storage import vector_store

logger = logging.getLogger(__name__)

RRF_K = 60  # standard constant from Cormack et al. (2009)

STOPWORDS = set("""a an the of to in on for and or is are was were be been by with as at
from that this these those it its what which who whom how when where why do does did
can could should would will shall may might there their them they he she his her you your
we our i me my not no yes than then so if into about over under per""".split())


@dataclass
class RetrievalOptions:
    mode: Literal["dense", "hybrid"] = "hybrid"
    use_reranker: bool = True
    multimodal: bool = True
    top_k_text: int = 8
    top_k_images: int = 3
    final_top_k: int = 6

    @classmethod
    def from_settings(cls, **overrides) -> "RetrievalOptions":
        s = get_settings()
        values = dict(mode=s.retrieval_mode, use_reranker=s.use_reranker, multimodal=s.multimodal,
                      top_k_text=s.top_k_text, top_k_images=s.top_k_images,
                      final_top_k=s.final_top_k)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


@dataclass
class EvidenceItem:
    id: str  # chunk id or image id
    kind: Literal["text", "image"]
    content_type: str  # text | ocr | table | image_caption | image
    document_id: str
    document_name: str
    page_number: int
    text: str  # chunk text, or the caption for an image ("" if none)
    image_id: str | None = None
    context: str | None = None  # "document title, page N" (searched, not displayed)
    scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "content_type": self.content_type,
            "document_id": self.document_id, "document_name": self.document_name,
            "page_number": self.page_number, "text": self.text, "image_id": self.image_id,
            "image_url": f"/api/images/{self.image_id}" if self.image_id else None,
            "context": self.context,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


@dataclass
class RetrievalResult:
    query: str
    text_items: list[EvidenceItem]
    image_items: list[EvidenceItem]
    weak_evidence: bool  # True when even the best text match is poor
    options: RetrievalOptions
    timings_ms: dict

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "text_items": [i.to_dict() for i in self.text_items],
            "image_items": [i.to_dict() for i in self.image_items],
            "weak_evidence": self.weak_evidence,
            "options": self.options.__dict__,
            "timings_ms": self.timings_ms,
        }


# --- BM25 (keyword) index ------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class _BM25Cache:
    """BM25 index per set of documents; rebuilt only when documents change."""

    def __init__(self):
        self._lock = threading.Lock()
        self._version = 0
        self._cache: dict[tuple, tuple[int, object, list[dict]]] = {}

    def invalidate(self):
        with self._lock:
            self._version += 1
            self._cache.clear()

    def get(self, document_ids: list[str] | None):
        from rank_bm25 import BM25Okapi

        key = tuple(sorted(document_ids)) if document_ids else ("*",)
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached[0] == self._version:
                return cached[1], cached[2]
            version = self._version
        chunks = db.get_chunks(document_ids)
        corpus = [tokenize(f"{c.get('context') or ''} {c['text']}") for c in chunks]
        index = BM25Okapi(corpus) if any(corpus) else None
        with self._lock:
            if version == self._version:
                self._cache[key] = (version, index, chunks)
        return index, chunks


bm25_cache = _BM25Cache()


def bm25_search(query: str, document_ids: list[str] | None, k: int) -> list[tuple[str, float]]:
    index, chunks = bm25_cache.get(document_ids)
    tokens = tokenize(query)
    if index is None or not tokens:
        return []
    scores = index.get_scores(tokens)
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunks[i]["id"], float(scores[i])) for i in ranked if scores[i] > 0]


# --- Fusion ----------------------------------------------------------------------------


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """score(d) = sum over rankings of 1 / (k + rank). Robust to different score scales."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return fused


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


# --- Main entry ------------------------------------------------------------------------------


def retrieve(query: str, document_ids: list[str] | None = None,
             options: RetrievalOptions | None = None) -> RetrievalResult:
    settings = get_settings()
    options = options or RetrievalOptions.from_settings()
    timings: dict[str, float] = {}
    document_ids = document_ids or None
    candidates_k = max(options.top_k_text * 2, options.final_top_k)

    # 1. Dense search
    t0 = time.perf_counter()
    dense_hits = vector_store.query(vector_store.text_collection(), models.embed_query(query),
                                    candidates_k, document_ids)
    dense_scores = {h["id"]: h["similarity"] for h in dense_hits}
    timings["dense"] = (time.perf_counter() - t0) * 1000

    # 2. Keyword search + fusion
    bm25_scores: dict[str, float] = {}
    if options.mode == "hybrid":
        t0 = time.perf_counter()
        bm25_scores = dict(bm25_search(query, document_ids, candidates_k))
        timings["bm25"] = (time.perf_counter() - t0) * 1000
        fused = reciprocal_rank_fusion([list(dense_scores), list(bm25_scores)])
    else:
        fused = reciprocal_rank_fusion([list(dense_scores)])
    ordered_ids = sorted(fused, key=fused.get, reverse=True)

    # 3. Load chunk text from SQLite (ignores stale vectors), drop duplicate texts
    rows = db.get_chunks_by_ids(ordered_ids)
    items: list[EvidenceItem] = []
    seen_texts: set[str] = set()
    for chunk_id in ordered_ids:
        row = rows.get(chunk_id)
        if row is None:
            continue
        norm = _normalise(row["text"])
        if norm in seen_texts:
            continue
        seen_texts.add(norm)
        scores = {"rrf": fused[chunk_id]}
        if chunk_id in dense_scores:
            scores["dense"] = dense_scores[chunk_id]
        if chunk_id in bm25_scores:
            scores["bm25"] = bm25_scores[chunk_id]
        items.append(EvidenceItem(
            id=chunk_id, kind="text", content_type=row["content_type"],
            document_id=row["document_id"], document_name=row["document_name"],
            page_number=row["page_number"], text=row["text"], image_id=row["image_id"],
            context=row.get("context"), scores=scores))
    items = items[:candidates_k]

    # 4. Rerank (cross-encoder reads question + chunk together)
    weak = not items
    if options.use_reranker and items:
        t0 = time.perf_counter()
        rerank = models.rerank_scores(
            query, [f"{i.context}\n{i.text}" if i.context else i.text for i in items])
        for item, score in zip(items, rerank):
            item.scores["rerank"] = score
        items.sort(key=lambda i: i.scores["rerank"], reverse=True)
        timings["rerank"] = (time.perf_counter() - t0) * 1000
        weak = items[0].scores["rerank"] < settings.rerank_min_score
    elif items:
        weak = items[0].scores.get("dense", 0.0) < 0.2
    text_items = items[:options.final_top_k]

    # 5. Images: CLIP search + figures whose caption was retrieved
    image_items: list[EvidenceItem] = []
    if options.multimodal:
        t0 = time.perf_counter()
        clip_hits = vector_store.query(vector_store.image_collection(),
                                       models.embed_text_for_image_search(query),
                                       options.top_k_images, document_ids)
        candidates = {h["id"]: h["similarity"] for h in clip_hits
                      if h["similarity"] >= settings.image_min_similarity}
        for item in text_items:  # a RELEVANT caption brings its figure along
            relevant = (item.scores.get("rerank", 0.0) >= settings.rerank_min_score
                        if "rerank" in item.scores else not weak)
            if item.image_id and item.image_id not in candidates and relevant:
                candidates[item.image_id] = -1.0  # -1 = "found via caption"
        image_rows = db.get_images_by_ids(list(candidates))
        for image_id, similarity in sorted(candidates.items(), key=lambda kv: kv[1],
                                           reverse=True):
            row = image_rows.get(image_id)
            if row is None:
                continue
            scores = {"clip": similarity} if similarity >= 0 else {"via_caption": 1.0}
            image_items.append(EvidenceItem(
                id=image_id, kind="image", content_type="image",
                document_id=row["document_id"], document_name=row["document_name"],
                page_number=row["page_number"], text=row["caption"] or "",
                image_id=image_id, scores=scores))
        image_items = image_items[:settings.max_images_to_llm]
        timings["images"] = (time.perf_counter() - t0) * 1000

    timings = {k: round(v, 1) for k, v in timings.items()}
    timings["total"] = round(sum(timings.values()), 1)
    return RetrievalResult(query, text_items, image_items, weak and not image_items,
                           options, timings)
