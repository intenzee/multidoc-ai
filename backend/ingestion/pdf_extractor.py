"""
backend/ingestion/pdf_extractor.py

Extract everything useful from a PDF, page by page, keeping page numbers:

  * Text            PyMuPDF get_text(sort=True): reading order, handles multi-column
  * Scanned pages   pages with almost no text are rendered at 300 DPI and OCR'd (Tesseract)
  * Tables          pdfplumber finds ruled tables and keeps headers + rows
  * Images          embedded raster images (photos, diagrams, scanned figures)
  * Vector charts   pages drawn with many vector operations (charts made in Excel etc.
                    contain no raster image) are rendered to PNG so CLIP and Gemini can see them
  * Clean-up        repeated headers/footers removed, blank and duplicate pages recorded

Nothing is skipped silently: every skipped page/image is counted in `stats`
and stored with the document, so the UI and the report can show it.
"""

import hashlib
import io
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (the old "fitz" name is deprecated)
import pdfplumber
from PIL import Image

from backend.config import get_settings

logger = logging.getLogger(__name__)


class PDFOpenError(Exception):
    """The PDF can't be processed at all (corrupted, encrypted, empty, too long)."""


@dataclass
class ExtractedImage:
    page_number: int
    png_bytes: bytes
    width: int
    height: int
    source: str  # "embedded" or "page_render"


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    kind: str  # text | scanned | blank | duplicate
    ocr_used: bool
    width: float
    height: float
    rotation: int
    tables: list[str] = field(default_factory=list)  # each table as Markdown


@dataclass
class ExtractionResult:
    page_count: int
    pages: list[ExtractedPage]
    images: list[ExtractedImage]
    stats: dict
    warnings: list[str]
    title: str = ""  # PDF metadata title, if the file has one


# --- Opening and validating -------------------------------------------------------


def open_pdf(path: Path) -> fitz.Document:
    """Open a PDF or raise PDFOpenError with a message a user can understand."""
    settings = get_settings()
    try:
        doc = fitz.open(path)
    except Exception as exc:  # PyMuPDF raises several types for damaged files
        raise PDFOpenError("The PDF is corrupted or unreadable.") from exc
    if doc.needs_pass:
        doc.close()
        raise PDFOpenError("The PDF is password-protected. Remove the password and upload again.")
    if doc.page_count == 0:
        doc.close()
        raise PDFOpenError("The PDF has no pages.")
    if doc.page_count > settings.max_pages:
        count = doc.page_count
        doc.close()
        raise PDFOpenError(f"The PDF has {count} pages; the limit is {settings.max_pages}.")
    return doc


def inspect_pdf(path: Path) -> int:
    """Quick check used at upload time. Returns the page count."""
    doc = open_pdf(path)
    try:
        doc.load_page(0)  # forces parsing of the first page; fails on badly broken files
        return doc.page_count
    except Exception as exc:
        raise PDFOpenError("The PDF is corrupted or unreadable.") from exc
    finally:
        doc.close()


# --- Text helpers -------------------------------------------------------------------


def clean_text(text: str) -> str:
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _line_signature(line: str) -> str:
    """'Page 3 of 10' and 'Page 4 of 10' get the same signature."""
    return re.sub(r"\d+", "#", line.strip().lower())


def remove_repeated_headers_footers(page_texts: list[str]) -> tuple[list[str], int]:
    """Drop first/last lines that repeat on at least 60% of pages (3+ pages only)."""
    if len(page_texts) < 3:
        return page_texts, 0
    counts: Counter[str] = Counter()
    for text in page_texts:
        lines = [l for l in text.splitlines() if l.strip()]
        edge = set(_line_signature(l) for l in lines[:2] + lines[-2:])
        counts.update(edge)
    threshold = 0.6 * len(page_texts)
    repeated = {sig for sig, n in counts.items() if n >= threshold and sig}
    if not repeated:
        return page_texts, 0

    removed = 0
    cleaned = []
    for text in page_texts:
        lines = text.splitlines()
        non_empty = [i for i, l in enumerate(lines) if l.strip()]
        edge_idx = set(non_empty[:2] + non_empty[-2:])
        kept = []
        for i, line in enumerate(lines):
            if i in edge_idx and _line_signature(line) in repeated:
                removed += 1
                continue
            kept.append(line)
        cleaned.append("\n".join(kept))
    return cleaned, removed


# --- OCR ----------------------------------------------------------------------------


def _tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_page(page: fitz.Page, language: str) -> str:
    import pytesseract

    pix = page.get_pixmap(dpi=300)  # Tesseract works best around 300 DPI
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(image, lang=language)


# --- Tables ---------------------------------------------------------------------------


def table_to_markdown(rows: list[list[str | None]]) -> str | None:
    """Convert pdfplumber rows to a Markdown table. Merged/empty cells become ''."""
    rows = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in rows if row]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:  # a header alone is not a useful table
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def extract_tables(path: Path, warnings: list[str]) -> dict[int, list[str]]:
    tables: dict[int, list[str]] = {}
    try:
        with pdfplumber.open(path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                try:
                    for raw in page.extract_tables():
                        markdown = table_to_markdown(raw)
                        if markdown:
                            tables.setdefault(index, []).append(markdown)
                except Exception:
                    warnings.append(f"Table extraction failed on page {index}.")
    except Exception:
        warnings.append("Table extraction could not read this PDF; tables are only in plain text.")
    return tables


# --- Images ----------------------------------------------------------------------------


def _to_png(image_bytes: bytes) -> tuple[bytes, int, int] | None:
    """Normalise any embedded image format (JPEG, JPX, CMYK...) to RGB PNG."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            out = io.BytesIO()
            img.save(out, format="PNG")
            return out.getvalue(), img.width, img.height
    except Exception:
        return None


def extract_page_images(doc: fitz.Document, page: fitz.Page, page_number: int,
                        seen_hashes: set[str], stats: Counter) -> list[ExtractedImage]:
    settings = get_settings()
    images: list[ExtractedImage] = []
    for info in page.get_images(full=True):
        xref = info[0]
        try:
            raw = doc.extract_image(xref)
        except Exception:
            stats["images_unreadable"] += 1
            continue
        if not raw or min(raw.get("width", 0), raw.get("height", 0)) < settings.min_image_side_px:
            stats["images_too_small_skipped"] += 1
            continue
        digest = hashlib.sha256(raw["image"]).hexdigest()
        if digest in seen_hashes:  # logos repeated on every page
            stats["images_duplicate_skipped"] += 1
            continue
        converted = _to_png(raw["image"])
        if converted is None:
            stats["images_unreadable"] += 1
            continue
        seen_hashes.add(digest)
        png, width, height = converted
        images.append(ExtractedImage(page_number, png, width, height, "embedded"))
    return images


def render_page(page: fitz.Page, dpi: int = 110) -> tuple[bytes, int, int]:
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png"), pix.width, pix.height


# --- Main entry point --------------------------------------------------------------------


def extract_pdf(path: Path) -> ExtractionResult:
    settings = get_settings()
    stats: Counter = Counter()
    warnings: list[str] = []
    doc = open_pdf(path)

    try:
        ocr_available = settings.ocr_enabled and _tesseract_available()
        if settings.ocr_enabled and not ocr_available:
            warnings.append("Tesseract OCR is not installed; scanned pages could not be read.")

        raw_texts: list[str] = []
        ocr_flags: list[bool] = []
        images: list[ExtractedImage] = []
        seen_hashes: set[str] = set()
        geometry = []

        for index in range(doc.page_count):
            page_number = index + 1
            try:
                page = doc.load_page(index)
            except Exception:
                warnings.append(f"Page {page_number} could not be read and was skipped.")
                raw_texts.append("")
                ocr_flags.append(False)
                geometry.append((0.0, 0.0, 0))
                stats["pages_unreadable"] += 1
                continue

            geometry.append((page.rect.width, page.rect.height, page.rotation))
            text = page.get_text("text", sort=True)
            page_images = extract_page_images(doc, page, page_number, seen_hashes, stats)
            used_ocr = False

            if len(text.strip()) < settings.ocr_min_text_chars:
                # Little or no selectable text: probably a scan (or a blank page).
                if ocr_available:
                    try:
                        ocr_text = ocr_page(page, settings.ocr_language)
                        if len(ocr_text.strip()) > len(text.strip()):
                            text, used_ocr = ocr_text, True
                            stats["pages_ocr"] += 1
                            # On a scanned page the embedded image IS the page. Store one
                            # rendered page image instead, so visual questions still work.
                            png, width, height = render_page(page)
                            page_images = [ExtractedImage(page_number, png, width, height,
                                                          "page_render")]
                    except Exception:
                        warnings.append(f"OCR failed on page {page_number}.")
                        stats["pages_ocr_failed"] += 1
                elif page_images:
                    stats["pages_scanned_unread"] += 1

            # Vector charts have no embedded image: render the page so it can be "seen".
            if (settings.render_vector_graphics and not page_images and not used_ocr):
                try:
                    if len(page.get_drawings()) >= settings.vector_drawing_threshold:
                        png, width, height = render_page(page)
                        page_images.append(ExtractedImage(page_number, png, width, height,
                                                          "page_render"))
                        stats["pages_rendered_as_image"] += 1
                except Exception:
                    warnings.append(f"Could not render vector graphics on page {page_number}.")

            images.extend(page_images)
            raw_texts.append(text)
            ocr_flags.append(used_ocr)

        texts, removed = remove_repeated_headers_footers(raw_texts)
        if removed:
            stats["header_footer_lines_removed"] = removed

        tables = extract_tables(path, warnings)

        pages: list[ExtractedPage] = []
        seen_page_hashes: set[str] = set()
        for index, text in enumerate(texts):
            page_number = index + 1
            text = clean_text(text)
            width, height, rotation = geometry[index]
            if not text and not tables.get(page_number):
                kind = "blank"
                stats["pages_blank"] += 1
            else:
                digest = hashlib.sha256(text.encode()).hexdigest()
                if text and digest in seen_page_hashes:
                    kind = "duplicate"
                    stats["pages_duplicate"] += 1
                else:
                    seen_page_hashes.add(digest)
                    kind = "scanned" if ocr_flags[index] else "text"
            pages.append(ExtractedPage(
                page_number=page_number, text=text, kind=kind, ocr_used=ocr_flags[index],
                width=width, height=height, rotation=rotation,
                tables=tables.get(page_number, []),
            ))

        stats["pages"] = doc.page_count
        stats["tables"] = sum(len(t) for t in tables.values())
        stats["images"] = len(images)
        title = " ".join(((doc.metadata or {}).get("title") or "").split())[:200]
        return ExtractionResult(doc.page_count, pages, images, dict(stats), warnings, title)
    finally:
        doc.close()
