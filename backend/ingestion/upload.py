"""
backend/ingestion/upload.py

Safely receive an uploaded file, confirm it is a PDF, and save it to data/uploads/.

Security rules applied here (see spec section 12):
  * File type is checked by CONTENT (the PDF signature), not by the file name.
  * File size is limited (MAX_UPLOAD_MB in .env, default 50 MB).
  * The file on disk is named <document_id>.pdf. The user's file name is never
    used as a path, so tricks like "../../evil.pdf" (path traversal) can't work.
  * The user's file name is cleaned and kept only as a display name.
  * Data is written to a temporary ".part" file and renamed only when complete,
    so an interrupted upload never leaves a half-written PDF behind.
  * The uploaded file is never executed or opened by anything except our code.

This module knows nothing about FastAPI or HTTP. It raises UploadRejected,
and the API layer turns that into an HTTP error response.
"""

import hashlib
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from backend.config import get_settings

CHUNK_SIZE = 1024 * 1024  # read the upload 1 MB at a time, never all at once
PDF_SIGNATURE = b"%PDF-"  # every PDF starts with this "magic number"
SIGNATURE_WINDOW = 1024  # the PDF standard allows it anywhere in the first 1024 bytes
MAX_FILENAME_CHARS = 100


class UploadRejected(Exception):
    """The upload is invalid. Carries the HTTP status code the API should return."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class SavedUpload:
    """What we know about a successfully saved upload."""

    document_id: str
    original_filename: str  # cleaned display name, e.g. "Research_Paper.pdf"
    stored_path: Path  # e.g. data/uploads/3f2a....pdf
    size_bytes: int
    sha256: str  # fingerprint of the content; used for duplicate detection later


def sanitize_filename(raw_name: str | None) -> str:
    """
    Turn whatever name the browser sent into a safe display name ending in .pdf.

    "../../etc/passwd"      -> "passwd.pdf"
    "my report (final).pdf" -> "my report _final_.pdf"
    ""                      -> "document.pdf"
    """
    # Keep only the last path component, whether "/" or "\" was used.
    name = (raw_name or "").replace("\\", "/").split("/")[-1]
    # Normalise look-alike Unicode characters (e.g. full-width letters).
    name = unicodedata.normalize("NFKC", name)
    # Allow letters (any language), digits, underscore, dot, hyphen and space.
    name = re.sub(r"[^\w.\- ]", "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")

    stem, dot, _extension = name.rpartition(".")
    if not dot:  # no extension at all
        stem = name
    stem = stem[: MAX_FILENAME_CHARS - len(".pdf")].strip(" .") or "document"
    return f"{stem}.pdf"


def save_pdf_upload(stream: BinaryIO, raw_filename: str | None) -> SavedUpload:
    """
    Stream the upload to disk while validating it.

    Raises UploadRejected for: not a PDF (415), too large (413), empty (400).
    """
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    document_id = uuid.uuid4().hex  # random, unguessable, unique
    final_path = settings.upload_dir / f"{document_id}.pdf"
    temp_path = final_path.with_suffix(".part")

    hasher = hashlib.sha256()
    size = 0

    try:
        with temp_path.open("wb") as out:
            while chunk := stream.read(CHUNK_SIZE):
                if size == 0 and PDF_SIGNATURE not in chunk[:SIGNATURE_WINDOW]:
                    raise UploadRejected(
                        415, "This file is not a PDF. Only PDF documents can be uploaded."
                    )
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise UploadRejected(
                        413, f"This file is larger than the {settings.max_upload_mb} MB upload limit."
                    )
                hasher.update(chunk)
                out.write(chunk)

        if size == 0:
            raise UploadRejected(400, "The uploaded file is empty.")

        temp_path.rename(final_path)  # only complete, valid files get the .pdf name
    except BaseException:
        # Any failure (rejection, disk full, client disconnect): leave nothing behind.
        temp_path.unlink(missing_ok=True)
        raise

    return SavedUpload(
        document_id=document_id,
        original_filename=sanitize_filename(raw_filename),
        stored_path=final_path,
        size_bytes=size,
        sha256=hasher.hexdigest(),
    )
