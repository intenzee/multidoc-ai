"""
backend/config.py

Central configuration for MultiDoc AI.

Every setting lives here, in ONE place. Values come from (highest priority first):
  1. Real environment variables (e.g. exported in your shell)
  2. The project's .env file
  3. The defaults written below

Other modules never call os.getenv() themselves. They do:
    from backend.config import get_settings
    settings = get_settings()
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/config.py -> backend/ -> project root.
# Using an absolute path means the app finds .env and data/ no matter which
# folder you start it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All app settings. Field names match .env variables, case-insensitively
    (gemini_model <-> GEMINI_MODEL)."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unknown variables in .env are ignored, not errors
    )

    # --- App -----------------------------------------------------------------
    app_name: str = "MultiDoc AI"
    app_version: str = "1.0.0"
    # Comma-separated list of browser origins allowed to call the API (CORS).
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    log_level: str = "INFO"

    # --- LLM (used only by generation; everything else works without it) -----
    llm_provider: str = "gemini"
    # SecretStr hides the value when printed or logged: it shows '**********'.
    gemini_api_key: SecretStr | None = None
    gemini_model: str | None = None
    llm_timeout_seconds: int = Field(default=30, ge=5, le=120)
    llm_max_retries: int = Field(default=2, ge=0, le=5)  # retries for 5xx/timeouts only
    llm_temperature: float = Field(default=0.2, ge=0.0, le=1.0)

    # --- Storage -------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"

    # --- Upload limits -------------------------------------------------------
    max_upload_mb: int = Field(default=50, ge=1, le=500)
    max_pages: int = Field(default=300, ge=1, le=5000)

    # --- Ingestion -----------------------------------------------------------
    chunk_size: int = Field(default=800, ge=100, le=4000)  # characters per chunk
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    ocr_enabled: bool = True
    ocr_language: str = "eng"  # Tesseract language code(s), e.g. "eng+hin"
    ocr_min_text_chars: int = 50  # pages with less extractable text get OCR
    min_image_side_px: int = 80  # smaller embedded images (icons, bullets) are skipped
    render_vector_graphics: bool = True  # render pages with vector charts as images
    vector_drawing_threshold: int = 30  # drawing operations needed to count as a chart
    caption_images: bool = True  # ask the LLM to describe each image (uses quota)
    max_captions_per_document: int = Field(default=10, ge=0, le=200)

    # --- Other file types (converted to PDF before processing) ----------------
    # Word/PowerPoint files are converted with LibreOffice (headless, no macros).
    office_timeout_s: int = Field(default=180, ge=10, le=1800)
    # Audio is transcribed locally with Whisper on the Apple GPU (mlx). Free, offline,
    # no LLM quota. A bigger model is more accurate but slower:
    # mlx-community/whisper-large-v3-turbo (about 1.6 GB).
    whisper_model: str = "mlx-community/whisper-small-mlx"
    max_audio_minutes: int = Field(default=60, ge=1, le=600)
    transcript_page_seconds: int = Field(default=120, ge=30, le=900)  # audio per "page"

    # --- Models (all run locally, free) ---------------------------------------
    text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_embedding_model: str = "sentence-transformers/clip-ViT-B-32"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Where the local models run. auto = Apple GPU (mps) or NVIDIA GPU (cuda) if present,
    # otherwise CPU. Set MODEL_DEVICE=cpu if the GPU gives trouble.
    model_device: Literal["auto", "mps", "cuda", "cpu"] = "auto"

    # --- Retrieval defaults (each can be overridden per request) --------------
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    use_reranker: bool = True
    multimodal: bool = True
    top_k_text: int = Field(default=8, ge=1, le=50)  # candidates per text retriever
    top_k_images: int = Field(default=3, ge=0, le=20)
    final_top_k: int = Field(default=6, ge=1, le=30)  # evidence items sent to the LLM
    # Calibrated on evaluation/dataset.sample.json (see evaluation/results/results.md):
    # chart questions scored 0.34-0.40 with CLIP, non-visual questions at most 0.30.
    image_min_similarity: float = 0.30  # CLIP cosine similarity needed to keep an image
    # Cross-encoder logits: > 0 means "relevant". Below this, retrieval is flagged as weak
    # and figures are not attached via their captions. It does NOT block the answer:
    # broad questions ("summarise this") score as low as off-topic ones.
    rerank_min_score: float = 0.0
    max_images_to_llm: int = Field(default=3, ge=0, le=10)
    max_context_chars: int = Field(default=12000, ge=1000, le=100000)
    history_turns: int = Field(default=3, ge=0, le=10)

    # --- Derived values (computed from the settings above) -------------------
    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def image_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "multidoc.sqlite3"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def llm_configured(self) -> bool:
        """True if a key AND a model are set. Says nothing about whether they work."""
        has_key = bool(self.gemini_api_key and self.gemini_api_key.get_secret_value())
        return has_key and bool(self.gemini_model)

    def ensure_data_dirs(self) -> None:
        """Create the storage folders if they don't exist yet."""
        for folder in (self.upload_dir, self.image_dir, self.chroma_dir):
            folder.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse the same object everywhere (cached)."""
    return Settings()
