"""
backend/ingestion/upload.py

Safely receive an uploaded file and save it to data/uploads/.

Accepted: PDF; pictures (PNG, JPG, WEBP, TIFF, HEIC); Word (.docx, .doc);
PowerPoint (.pptx, .ppt); text (.txt, .md); audio (MP3, WAV, M4A, AAC, OGG/Opus,
FLAC, WEBM, and the sound of MP4 videos).

  * PDFs are stored as <id>.pdf.
  * Pictures are turned into a PDF right away (one page per picture, at the original
    resolution). The pipeline then reads them like scanned pages: OCR, an LLM
    description and a CLIP vector.
  * Word, PowerPoint, text and audio are stored as <id>.source.<ext> and converted to a
    PDF in the background (see converters.py), because that can take a while.

Security rules applied here (see spec section 12):
  * File type is checked by CONTENT (the file signature), not by the file name.
  * Images are decoded by Pillow with a pixel limit, so "image bombs" are rejected.
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
import io
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import zipfile

import pymupdf
from PIL import Image, ImageOps, ImageSequence

try:  # iPhone photos (HEIC); optional so the app still starts without the package
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

from backend.config import get_settings

CHUNK_SIZE = 1024 * 1024  # read the upload 1 MB at a time, never all at once
PDF_SIGNATURE = b"%PDF-"  # every PDF starts with this "magic number"
SIGNATURE_WINDOW = 1024  # the PDF standard allows it anywhere in the first 1024 bytes
MAX_FILENAME_CHARS = 100

IMAGE_TYPES = {"png": ".png", "jpeg": ".jpg", "webp": ".webp", "tiff": ".tiff",
               "heic": ".heic"}
# Converted in the background. Value: extension of the stored source file.
BACKGROUND_TYPES = {"docx": ".docx", "pptx": ".pptx", "doc": ".doc", "ppt": ".ppt",
                    "text": ".txt", "markdown": ".md", "audio": ".audio"}
MAX_IMAGE_PIXELS = 40_000_000  # about 6300 x 6300; larger images are refused
MAX_IMAGE_SIDE = 2000  # bigger pictures are scaled down (enough for OCR and the LLM)
IMAGE_PAGE_DPI = 110  # pages are rendered at this DPI later, so the image keeps its size
UNSUPPORTED_MESSAGE = ("This file type is not supported. Upload a PDF, a picture (PNG, JPG, "
                       "WEBP, TIFF, HEIC), Word, PowerPoint, text (.txt, .md) or audio.")
HEIC_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"heim", b"heis"}
OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # old .doc / .ppt / .xls


def _extension(filename: str | None) -> str:
    return (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""


def _looks_like_text(head: bytes) -> bool:
    if b"\x00" in head and not head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return False
    try:
        head.decode("utf-8")
        return True
    except UnicodeDecodeError as err:
        return err.start >= len(head) - 3  # only a character cut off at the chunk end


def detect_file_type(head: bytes, filename: str | None = None) -> str | None:
    """The file type from its first bytes (the name only decides between look-alikes)."""
    ext = _extension(filename)
    if PDF_SIGNATURE in head[:SIGNATURE_WINDOW]:
        return "pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    if head[4:8] == b"ftyp":  # ISO media: HEIC photo, or M4A/MP4/MOV audio-video
        return "heic" if head[8:12] in HEIC_BRANDS else "audio"
    if head.startswith(b"PK\x03\x04"):
        return "zip"  # .docx/.pptx are ZIP files; checked once the whole file is saved
    if head.startswith(OLE_SIGNATURE):
        return {"doc": "doc", "ppt": "ppt"}.get(ext)
    if ((head[:4] == b"RIFF" and head[8:12] == b"WAVE") or head.startswith(b"ID3")
            or head.startswith((b"OggS", b"fLaC", b"\x1aE\xdf\xa3"))
            or (head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"))
            or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)):
        return "audio"  # WAV, MP3, OGG/Opus, FLAC, WEBM, AIFF, MP3/AAC without tags
    if ext in ("txt", "md", "markdown") and _looks_like_text(head):
        return "markdown" if ext != "txt" else "text"
    return None


def _office_type_from_zip(path: Path) -> str | None:
    """'docx' or 'pptx' by looking at the names inside the ZIP (nothing is extracted)."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except (zipfile.BadZipFile, OSError):
        return None
    if "word/document.xml" in names:
        return "docx"
    if "ppt/presentation.xml" in names:
        return "pptx"
    return None


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
    original_filename: str  # cleaned display name, e.g. "Research_Paper.pdf" or "chart.png"
    stored_path: Path  # e.g. data/uploads/3f2a....pdf
    size_bytes: int
    sha256: str  # fingerprint of the content; used for duplicate detection later
    file_type: str = "pdf"  # pdf, png, jpeg, webp, tiff, heic, docx, pptx, doc, ppt,
    #                         text, markdown or audio

    @property
    def needs_conversion(self) -> bool:  # converted to PDF in the background
        return self.file_type in BACKGROUND_TYPES


def sanitize_filename(raw_name: str | None, extension: str = ".pdf") -> str:
    """
    Turn whatever name the browser sent into a safe display name ending in `extension`.

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
    stem = stem[: MAX_FILENAME_CHARS - len(extension)].strip(" .") or "document"
    return f"{stem}{extension}"


def _picture_to_png(img: Image.Image) -> tuple[bytes, int, int]:
    if img.width * img.height > MAX_IMAGE_PIXELS:
        raise UploadRejected(413, "This picture has too many pixels. "
                                  "Please upload a smaller image.")
    img = ImageOps.exif_transpose(img)  # phone photos: apply the rotation flag
    if img.mode in ("RGBA", "LA", "P"):  # transparent parts become white
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, "white")
        background.paste(img, mask=img.getchannel("A"))
        img = background
    else:
        img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))  # only ever shrinks
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue(), img.width, img.height


def image_to_pdf(source: Path, target: Path) -> None:
    """Save the picture in `source` as a PDF in `target`: one page per picture (a TIFF
    from a scanner can hold several pages)."""
    max_pages = get_settings().max_pages
    pages: list[tuple[bytes, int, int]] = []
    try:
        with Image.open(source) as img:
            if img.width * img.height > MAX_IMAGE_PIXELS:  # checked before decoding
                raise UploadRejected(413, "This picture has too many pixels. "
                                          "Please upload a smaller image.")
            frames = ImageSequence.Iterator(img) if img.format == "TIFF" else [img]
            for frame in frames:
                if len(pages) >= max_pages:
                    raise UploadRejected(422, f"This file has more than {max_pages} pages.")
                pages.append(_picture_to_png(frame))
    except UploadRejected:
        raise
    except Exception:
        raise UploadRejected(422, "This picture could not be read. It may be damaged.") from None

    points = 72 / IMAGE_PAGE_DPI
    pdf = pymupdf.open()
    for png, width, height in pages:
        page = pdf.new_page(width=width * points, height=height * points)
        page.insert_image(page.rect, stream=png)
    pdf.save(target, garbage=3, deflate=True)
    pdf.close()


def save_upload(stream: BinaryIO, raw_filename: str | None) -> SavedUpload:
    """
    Stream the upload to disk while validating it. Pictures are converted to a PDF now;
    Word, PowerPoint, text and audio are kept as <id>.source.<ext> for the background job.

    Raises UploadRejected for: unsupported type (415), too large (413), empty (400),
    unreadable picture (422).
    """
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    document_id = uuid.uuid4().hex  # random, unguessable, unique
    final_path = settings.upload_dir / f"{document_id}.pdf"
    temp_path = final_path.with_suffix(".part")

    hasher = hashlib.sha256()
    size = 0
    file_type = None

    try:
        with temp_path.open("wb") as out:
            while chunk := stream.read(CHUNK_SIZE):
                if size == 0:
                    file_type = detect_file_type(chunk, raw_filename)
                    if file_type is None:
                        raise UploadRejected(415, UNSUPPORTED_MESSAGE)
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise UploadRejected(
                        413, f"This file is larger than the {settings.max_upload_mb} MB upload limit."
                    )
                hasher.update(chunk)
                out.write(chunk)

        if size == 0:
            raise UploadRejected(400, "The uploaded file is empty.")

        if file_type == "zip":
            file_type = _office_type_from_zip(temp_path)
            if file_type is None:
                raise UploadRejected(415, UNSUPPORTED_MESSAGE)
        if file_type == "pdf":
            temp_path.rename(final_path)  # only complete, valid files get the .pdf name
        elif file_type in BACKGROUND_TYPES:
            final_path = settings.upload_dir / f"{document_id}.source{BACKGROUND_TYPES[file_type]}"
            temp_path.rename(final_path)
        else:
            image_to_pdf(temp_path, final_path)
            temp_path.unlink(missing_ok=True)
    except BaseException:
        # Any failure (rejection, disk full, client disconnect): leave nothing behind.
        temp_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise

    return SavedUpload(
        document_id=document_id,
        original_filename=sanitize_filename(raw_filename, _display_extension(file_type,
                                                                              raw_filename)),
        stored_path=final_path,
        size_bytes=size,
        sha256=hasher.hexdigest(),
        file_type=file_type,
    )


save_pdf_upload = save_upload  # old name, kept for existing callers


def _display_extension(file_type: str, raw_filename: str | None) -> str:
    """Keep the user's extension when it matches the real type (e.g. .jpeg, .m4a)."""
    ext = "." + _extension(raw_filename) if _extension(raw_filename) else ""
    if file_type == "audio":
        return ext if ext in (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".webm",
                              ".mp4", ".mov", ".aiff", ".aif", ".mpga", ".oga") else ".audio"
    if file_type == "jpeg":
        return ext if ext in (".jpg", ".jpeg") else ".jpg"
    if file_type == "tiff":
        return ext if ext in (".tif", ".tiff") else ".tiff"
    if file_type == "heic":
        return ext if ext in (".heic", ".heif") else ".heic"
    if file_type == "markdown":
        return ext if ext in (".md", ".markdown") else ".md"
    return {"pdf": ".pdf", "text": ".txt"}.get(file_type) or IMAGE_TYPES.get(file_type) \
        or BACKGROUND_TYPES.get(file_type, ".pdf")
