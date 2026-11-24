"""
backend/ingestion/service.py

Document lifecycle used by the API: accept an upload, delete, reprocess.
Keeps the HTTP layer thin and makes the same operations usable from scripts.
"""

import logging
import shutil

from backend.config import get_settings
from backend.ingestion.pdf_extractor import PDFOpenError, inspect_pdf
from backend.ingestion.pipeline import document_image_dir
from backend.ingestion.upload import UploadRejected, save_upload
from backend.retrieval.retriever import bm25_cache
from backend.storage import database as db
from backend.storage import vector_store

logger = logging.getLogger(__name__)


class DuplicateDocument(Exception):
    def __init__(self, existing: dict):
        super().__init__("duplicate")
        self.existing = existing


def accept_upload(stream, filename: str | None) -> dict:
    """Validate, store and register an upload. Processing is scheduled by the caller."""
    saved = save_upload(stream, filename)

    existing = db.find_document_by_sha256(saved.sha256)
    if existing:
        saved.stored_path.unlink(missing_ok=True)  # keep only the first copy
        raise DuplicateDocument(existing)

    page_count = None  # Word, PowerPoint, text, audio: known after the conversion
    if not saved.needs_conversion:
        try:
            page_count = inspect_pdf(saved.stored_path)
        except PDFOpenError as err:
            saved.stored_path.unlink(missing_ok=True)
            raise UploadRejected(422, str(err)) from None

    db.insert_document(saved.document_id, saved.original_filename, saved.sha256,
                       saved.size_bytes, page_count)
    logger.info("Registered document %s (%s, %s pages, %d bytes)", saved.document_id,
                saved.file_type, page_count or "?", saved.size_bytes)
    return db.get_document(saved.document_id)


def delete_document(document_id: str) -> bool:
    """Remove a document everywhere: vectors, rows (cascade), PDF file, image files."""
    if db.get_document(document_id) is None:
        return False
    vector_store.delete_document_vectors(document_id)
    db.delete_document_rows(document_id)
    for path in get_settings().upload_dir.glob(f"{document_id}.*"):  # pdf, source, text
        path.unlink(missing_ok=True)
    shutil.rmtree(document_image_dir(document_id), ignore_errors=True)
    bm25_cache.invalidate()
    logger.info("Deleted document %s", document_id)
    return True
