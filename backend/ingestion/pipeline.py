"""
backend/ingestion/pipeline.py

Turn an uploaded PDF into searchable evidence:

    extract (text, OCR, tables, images)
      -> chunk text and tables
      -> caption images (LLM, optional, limited)
      -> embed chunks (MiniLM) and images (CLIP)
      -> store facts in SQLite and vectors in ChromaDB
      -> status = ready

Runs in a background thread after upload, one document at a time (a lock), so
two big PDFs don't fight for memory. Every outcome is recorded on the document:
`ready` with statistics, or `failed` with a readable error. Nothing fails silently.
"""

import logging
import shutil
import threading
import time
from pathlib import Path

from PIL import Image

from backend.config import get_settings
from backend.embeddings import models
from backend.generation import llm_service
from backend.ingestion import converters
from backend.ingestion.chunking import chunk_text, split_table
from backend.ingestion.pdf_extractor import PDFOpenError, extract_pdf
from backend.retrieval.retriever import bm25_cache
from backend.storage import database as db
from backend.storage import vector_store

logger = logging.getLogger(__name__)

_processing_lock = threading.Lock()


def document_pdf_path(document_id: str) -> Path:
    return get_settings().upload_dir / f"{document_id}.pdf"


def document_image_dir(document_id: str) -> Path:
    return get_settings().image_dir / document_id


def document_title(filename: str, metadata_title: str) -> str:
    """PDF title if present, otherwise the file name made readable."""
    if metadata_title:
        return metadata_title
    stem = filename.rsplit(".", 1)[0]
    return " ".join(stem.replace("_", " ").replace("-", " ").split())


def search_text(chunk: dict) -> str:
    """What gets embedded: a short context header + the chunk. Two reports with the same
    sentences (e.g. 2024 and 2025) are then still distinguishable by their title."""
    return f"{chunk['context']}\n{chunk['text']}" if chunk.get("context") else chunk["text"]


def build_chunks(document_id: str, pages) -> tuple[list[dict], dict]:
    settings = get_settings()
    chunks: list[dict] = []
    counts = {"duplicate_pages_not_indexed": 0}
    for page in pages:
        if page.kind == "duplicate":
            counts["duplicate_pages_not_indexed"] += 1
            continue
        content_type = "ocr" if page.ocr_used else "text"
        pieces = [(content_type, t) for t in chunk_text(page.text, settings.chunk_size,
                                                         settings.chunk_overlap)]
        for table in page.tables:
            for part in split_table(table, settings.chunk_size * 2):
                pieces.append(("table", f"Table on page {page.page_number}:\n{part}"))
        for index, (kind, text) in enumerate(pieces):
            chunks.append({
                "id": f"{document_id}:p{page.page_number}:c{index}",
                "document_id": document_id,
                "page_number": page.page_number,
                "chunk_index": index,
                "content_type": kind,
                "text": text,
                "image_id": None,
            })
    return chunks, counts


def process_document(document_id: str) -> None:
    """Entry point for the background task. Never raises."""
    with _processing_lock:
        try:
            _process(document_id)
        except Exception as exc:  # last line of defence: record, don't crash the server
            logger.exception("Processing failed for %s", document_id)
            safe = (PDFOpenError, vector_store.IndexMismatchError, converters.ConversionError)
            message = str(exc) if isinstance(exc, safe) \
                else "Processing failed unexpectedly. Check the server log for details."
            if db.get_document(document_id):
                db.update_document_status(document_id, "failed", error=message)


def _process(document_id: str) -> None:
    settings = get_settings()
    document = db.get_document(document_id)
    if document is None:
        return  # deleted while waiting in the queue
    db.update_document_status(document_id, "processing")
    bm25_cache.invalidate()  # its old chunks must not be searched while reprocessing
    timings: dict[str, float] = {}

    # Word, PowerPoint, text and audio: convert to a PDF first (only once).
    if not document_pdf_path(document_id).exists():
        started = time.perf_counter()
        page_count = converters.convert(document_id, document["filename"])
        db.update_document_status(document_id, "processing", page_count=page_count)
        timings["convert_s"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    extraction = extract_pdf(document_pdf_path(document_id))
    timings["extract_s"] = round(time.perf_counter() - started, 2)

    # Text and audio: index the exact text, not the text read back from the PDF.
    exact_texts = converters.load_page_texts(document_id)
    if exact_texts and len(exact_texts) == len(extraction.pages):
        for page, text in zip(extraction.pages, exact_texts):
            page.text, page.ocr_used, page.tables = text, False, []
            if page.kind == "duplicate":
                page.kind = "text"

    chunks, chunk_counts = build_chunks(document_id, extraction.pages)
    title = document_title(document["filename"], extraction.title)

    # --- Images: save, caption (limited), describe as text chunks -------------
    image_dir = document_image_dir(document_id)
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    image_rows: list[dict] = []
    pil_images: list[Image.Image] = []
    caption_stats = {"captioned": 0, "caption_failed": 0, "caption_skipped_limit": 0}
    captions_allowed = settings.caption_images and settings.llm_configured
    caption_error = None
    t0 = time.perf_counter()
    for index, image in enumerate(extraction.images):
        image_id = f"{document_id}-img{index}"
        path = image_dir / f"{image_id}.png"
        path.write_bytes(image.png_bytes)
        caption = None
        if captions_allowed and caption_error is None:
            if caption_stats["captioned"] < settings.max_captions_per_document:
                try:
                    caption = llm_service.caption_image(image.png_bytes)
                    caption_stats["captioned"] += 1
                except llm_service.LLMError as err:
                    caption_stats["caption_failed"] += 1
                    if not err.retryable:
                        caption_error = err.message  # stop asking (quota, key...)
            else:
                caption_stats["caption_skipped_limit"] += 1
        image_rows.append({
            "id": image_id, "document_id": document_id, "page_number": image.page_number,
            "path": str(path.relative_to(settings.data_dir)), "width": image.width,
            "height": image.height, "source": image.source, "caption": caption,
        })
        if caption:
            chunks.append({
                "id": f"{image_id}:caption", "document_id": document_id,
                "page_number": image.page_number, "chunk_index": 1000 + index,
                "content_type": "image_caption",
                "text": f"Figure on page {image.page_number}: {caption}",
                "image_id": image_id,
            })
        with Image.open(path) as img:
            pil_images.append(img.convert("RGB"))
    timings["caption_s"] = round(time.perf_counter() - t0, 2)

    # --- Embeddings -------------------------------------------------------------
    t0 = time.perf_counter()
    for chunk in chunks:
        chunk["context"] = f"{title}, page {chunk['page_number']}"
    text_vectors = models.embed_texts([search_text(c) for c in chunks])
    image_vectors = models.embed_images(pil_images)
    timings["embed_s"] = round(time.perf_counter() - t0, 2)

    if db.get_document(document_id) is None:  # deleted while we were working
        shutil.rmtree(image_dir, ignore_errors=True)
        return

    # --- Store: vectors first, then facts (a failure leaves status != ready) ------
    vector_store.delete_document_vectors(document_id)
    name = document["filename"]
    vector_store.add_items(
        vector_store.text_collection(),
        ids=[c["id"] for c in chunks],
        embeddings=text_vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[{"document_id": document_id, "document_name": name,
                    "page_number": c["page_number"], "content_type": c["content_type"],
                    "chunk_id": c["id"], "image_id": c["image_id"]} for c in chunks],
    )
    vector_store.add_items(
        vector_store.image_collection(),
        ids=[r["id"] for r in image_rows],
        embeddings=image_vectors,
        documents=None,
        metadatas=[{"document_id": document_id, "document_name": name,
                    "page_number": r["page_number"], "content_type": "image",
                    "image_id": r["id"], "image_path": r["path"]} for r in image_rows],
    )
    pages = [{"document_id": document_id, "page_number": p.page_number,
              "text_chars": len(p.text), "ocr_used": int(p.ocr_used), "kind": p.kind,
              "width": p.width, "height": p.height, "rotation": p.rotation}
             for p in extraction.pages]
    db.replace_document_content(document_id, pages, chunks, image_rows)

    warnings = list(extraction.warnings)
    if caption_error:
        warnings.append(f"Image captioning stopped: {caption_error}")
    if settings.caption_images and not settings.llm_configured and image_rows:
        warnings.append("Images were indexed with CLIP only (no LLM configured for captions).")
    if not chunks and not image_rows:
        warnings.append("No text, tables or images could be extracted from this PDF.")

    stats = {**extraction.stats, **chunk_counts, **caption_stats,
             "chunks": len(chunks), "timings": timings, "warnings": warnings,
             "model_device": models.device(), "title": title}
    db.update_document_status(document_id, "ready", stats=stats,
                              page_count=extraction.page_count)
    bm25_cache.invalidate()
    logger.info("Document %s ready: %d chunks, %d images", document_id, len(chunks),
                len(image_rows))
