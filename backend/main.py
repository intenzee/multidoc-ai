"""
backend/main.py

Entry point of the MultiDoc AI backend.

Run from the project root:
    .venv/bin/uvicorn backend.main:app --reload --reload-dir backend
Then open:
    http://127.0.0.1:8000/api/health   (health check)
    http://127.0.0.1:8000/docs         (interactive API documentation)
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.api.chat import router as chat_router
from backend.api.documents import images_router
from backend.api.documents import router as documents_router
from backend.config import get_settings
from backend.storage import database as db

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpx2", "chromadb", "sentence_transformers", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("multidoc")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_data_dirs()
    db.init_db()
    interrupted = db.mark_interrupted_documents()
    if interrupted:
        logger.warning("%d document(s) were interrupted by a restart and marked failed. "
                       "Use 'Reprocess' to run them again.", interrupted)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multimodal RAG system for document question answering.",
    lifespan=lifespan,
)

# Only the React dev server may call the API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

app.include_router(documents_router)
app.include_router(images_router)
app.include_router(chat_router)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    """Log the full error on the server; send the browser a short message, never a trace."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500,
                        content={"detail": "Internal server error. See the server log."})


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    server_time_utc: datetime
    llm_provider: str
    llm_model: str | None
    llm_configured: bool  # key + model present (never the key itself)


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Fast, free check: does NOT call Gemini, the vector database or any model."""
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        server_time_utc=datetime.now(timezone.utc),
        llm_provider=settings.llm_provider,
        llm_model=settings.gemini_model or None,
        llm_configured=settings.llm_configured,
    )
