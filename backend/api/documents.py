"""
backend/api/documents.py

Document endpoints:
    POST   /api/documents/upload              upload a PDF or a picture (processed in background)
    GET    /api/documents                     list documents with status
    GET    /api/documents/{id}                details, statistics, warnings
    DELETE /api/documents/{id}                delete file, vectors, rows and images
    POST   /api/documents/{id}/reprocess      run processing again
    GET    /api/documents/{id}/file           the original PDF
    GET    /api/documents/{id}/page/{n}       page rendered as PNG (?highlight=text)
    GET    /api/images/{image_id}             an extracted image
"""

import logging
from typing import Annotated, Any

import pymupdf
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from backend.config import get_settings
from backend.ingestion import service
from backend.ingestion.pipeline import document_pdf_path, process_document
from backend.ingestion.upload import UploadRejected
from backend.storage import database as db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])
images_router = APIRouter(prefix="/api/images", tags=["documents"])


class DocumentOut(BaseModel):
    id: str
    filename: str
    size_bytes: int
    page_count: int | None
    status: str  # uploaded | processing | ready | failed
    error: str | None
    created_at: str
    updated_at: str


class DocumentDetail(DocumentOut):
    stats: dict[str, Any] | None
    counts: dict[str, int]
    pages: list[dict[str, Any]]


def _out(doc: dict) -> DocumentOut:
    return DocumentOut(**{k: doc[k] for k in DocumentOut.model_fields})


def _get_or_404(document_id: str) -> dict:
    doc = db.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED,
             responses={400: {"description": "Empty file"},
                        409: {"description": "The same file was already uploaded"},
                        413: {"description": "File larger than MAX_UPLOAD_MB"},
                        415: {"description": "Not a PDF or a PNG/JPG/WEBP picture"},
                        422: {"description": "Corrupted, encrypted, empty, unreadable picture or too many pages"}})
def upload_document(background: BackgroundTasks,
                    file: Annotated[UploadFile, File(description="A PDF, or a picture/chart (PNG, JPG, WEBP)")]):
    """Validate and store a PDF or picture, then process it in the background (poll its status).

    Pictures are stored as a one-page PDF, so everything downstream treats them alike.
    """
    try:
        doc = service.accept_upload(file.file, file.filename)
    except UploadRejected as rejection:
        raise HTTPException(status_code=rejection.status_code, detail=rejection.message) from None
    except service.DuplicateDocument as dup:
        return JSONResponse(status_code=409, content={
            "detail": f"This file was already uploaded as '{dup.existing['filename']}'.",
            "existing_document_id": dup.existing["id"]})
    background.add_task(process_document, doc["id"])
    return _out(doc)


@router.get("", response_model=list[DocumentOut])
def list_documents():
    return [_out(d) for d in db.list_documents()]


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(document_id: str):
    doc = _get_or_404(document_id)
    return DocumentDetail(**_out(doc).model_dump(), stats=doc.get("stats"),
                          counts=db.count_content(document_id), pages=db.get_pages(document_id))


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str):
    doc = _get_or_404(document_id)
    if doc["status"] == "processing":
        raise HTTPException(status_code=409,
                            detail="The document is being processed. Delete it when it is done.")
    service.delete_document(document_id)
    return Response(status_code=204)


@router.post("/{document_id}/reprocess", response_model=DocumentOut, status_code=202)
def reprocess_document(document_id: str, background: BackgroundTasks):
    doc = _get_or_404(document_id)
    if doc["status"] == "processing":
        raise HTTPException(status_code=409, detail="The document is already being processed.")
    db.update_document_status(document_id, "uploaded")
    background.add_task(process_document, document_id)
    return _out(db.get_document(document_id))


@router.get("/{document_id}/file")
def get_document_file(document_id: str):
    doc = _get_or_404(document_id)
    path = document_pdf_path(document_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="The PDF file is missing on the server.")
    name = doc["filename"]
    if not name.lower().endswith(".pdf"):  # a picture: it is served as its one-page PDF
        name = name.rsplit(".", 1)[0] + ".pdf"
    return FileResponse(path, media_type="application/pdf", filename=name,
                        content_disposition_type="inline")


@router.get("/{document_id}/page/{page_number}",
            responses={200: {"content": {"image/png": {}}}})
def get_page_image(document_id: str, page_number: int,
                   highlight: Annotated[str | None, Query(max_length=300)] = None,
                   dpi: Annotated[int, Query(ge=50, le=200)] = 110):
    """Render one page as PNG. `highlight` marks where that text appears on the page."""
    _get_or_404(document_id)
    path = document_pdf_path(document_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="The PDF file is missing on the server.")
    with pymupdf.open(path) as pdf:
        if not 1 <= page_number <= pdf.page_count:
            raise HTTPException(status_code=404,
                                detail=f"Page {page_number} does not exist "
                                       f"(the document has {pdf.page_count} pages).")
        page = pdf.load_page(page_number - 1)
        if highlight:
            # Try the whole snippet first, then its first line (search works per line).
            snippet = " ".join(highlight.split())
            rects = page.search_for(snippet) or page.search_for(snippet[:60])
            for rect in rects[:20]:
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=(1.0, 0.85, 0.1))
                annot.update()
        png = page.get_pixmap(dpi=dpi, annots=True).tobytes("png")
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "private, max-age=300"})


@images_router.get("/{image_id}", responses={200: {"content": {"image/png": {}}}})
def get_image(image_id: str):
    row = db.get_image(image_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    settings = get_settings()
    path = (settings.data_dir / row["path"]).resolve()
    if settings.data_dir.resolve() not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Image file not found.")
    return FileResponse(path, media_type="image/png")
