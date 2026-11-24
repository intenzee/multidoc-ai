"""
backend/storage/database.py

SQLite metadata store: documents, pages, chunks, images, chat sessions, messages.

Why SQLite next to ChromaDB?
  * ChromaDB stores VECTORS for similarity search.
  * SQLite stores the FACTS: which documents exist, their status, every chunk's
    exact text and page, image file paths, and chat history.
  Keyword search (BM25), citations and deletes all need the facts.

Each function opens its own short-lived connection. That is safe with FastAPI's
thread pool and with background processing, and simple to reason about.
Foreign keys use ON DELETE CASCADE, so deleting a document row also deletes
its pages, chunks and images rows.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from backend.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    sha256        TEXT NOT NULL UNIQUE,
    size_bytes    INTEGER NOT NULL,
    page_count    INTEGER,
    status        TEXT NOT NULL,          -- uploaded | processing | ready | failed
    error         TEXT,
    stats_json    TEXT,                   -- processing statistics and warnings
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number   INTEGER NOT NULL,       -- 1-based, as humans count pages
    text_chars    INTEGER NOT NULL,
    ocr_used      INTEGER NOT NULL,
    kind          TEXT NOT NULL,          -- text | scanned | blank | duplicate
    width         REAL,
    height        REAL,
    rotation      INTEGER,
    PRIMARY KEY (document_id, page_number)
);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number   INTEGER NOT NULL,
    chunk_index   INTEGER NOT NULL,
    content_type  TEXT NOT NULL,          -- text | ocr | table | image_caption
    text          TEXT NOT NULL,
    image_id      TEXT,                   -- set for image_caption chunks
    context       TEXT                    -- "document title, page N": used for search only
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE TABLE IF NOT EXISTS images (
    id            TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number   INTEGER NOT NULL,
    path          TEXT NOT NULL,          -- relative to the data directory
    width         INTEGER,
    height        INTEGER,
    source        TEXT NOT NULL,          -- embedded | page_render
    caption       TEXT
);
CREATE INDEX IF NOT EXISTS idx_images_document ON images(document_id);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role          TEXT NOT NULL,          -- user | assistant
    content       TEXT NOT NULL,
    payload_json  TEXT,                   -- citations, evidence, timings (assistant only)
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit on success, roll back on error, always close."""
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")  # readers don't block the writer
        conn.executescript(SCHEMA)
        # Databases created before the "context" column existed get it added.
        columns = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
        if "context" not in columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN context TEXT")


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


# --- Documents ------------------------------------------------------------------


def _decode_document(row: sqlite3.Row | None) -> dict[str, Any] | None:
    doc = _row(row)
    if doc is not None:
        doc["stats"] = json.loads(doc.pop("stats_json") or "{}")
    return doc


def insert_document(document_id: str, filename: str, sha256: str, size_bytes: int,
                    page_count: int | None) -> None:
    stamp = now_iso()
    with connect() as conn:
        conn.execute(
            "INSERT INTO documents (id, filename, sha256, size_bytes, page_count, status,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'uploaded', ?, ?)",
            (document_id, filename, sha256, size_bytes, page_count, stamp, stamp),
        )


def get_document(document_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return _decode_document(row)


def find_document_by_sha256(sha256: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
    return _decode_document(row)


def list_documents() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    return [_decode_document(r) for r in rows]


def update_document_status(document_id: str, status: str, error: str | None = None,
                           stats: dict | None = None, page_count: int | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE documents SET status = ?, error = ?,"
            " stats_json = COALESCE(?, stats_json), page_count = COALESCE(?, page_count),"
            " updated_at = ? WHERE id = ?",
            (status, error, json.dumps(stats) if stats is not None else None,
             page_count, now_iso(), document_id),
        )


def delete_document_rows(document_id: str) -> bool:
    with connect() as conn:
        cursor = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    return cursor.rowcount > 0


def mark_interrupted_documents() -> int:
    """At startup: documents left 'processing' by a crash/restart are marked failed."""
    with connect() as conn:
        cursor = conn.execute(
            "UPDATE documents SET status = 'failed',"
            " error = 'Processing was interrupted (server restarted). Use reprocess to try again.',"
            " updated_at = ? WHERE status IN ('processing', 'uploaded')",
            (now_iso(),),
        )
    return cursor.rowcount


# --- Pages, chunks, images --------------------------------------------------------


def replace_document_content(document_id: str, pages: list[dict], chunks: list[dict],
                             images: list[dict]) -> None:
    """Store all extracted content for a document in ONE transaction (all or nothing)."""
    with connect() as conn:
        for table in ("pages", "chunks", "images"):
            conn.execute(f"DELETE FROM {table} WHERE document_id = ?", (document_id,))
        conn.executemany(
            "INSERT INTO pages (document_id, page_number, text_chars, ocr_used, kind, width,"
            " height, rotation) VALUES (:document_id, :page_number, :text_chars, :ocr_used,"
            " :kind, :width, :height, :rotation)",
            pages,
        )
        conn.executemany(
            "INSERT INTO images (id, document_id, page_number, path, width, height, source,"
            " caption) VALUES (:id, :document_id, :page_number, :path, :width, :height,"
            " :source, :caption)",
            images,
        )
        conn.executemany(
            "INSERT INTO chunks (id, document_id, page_number, chunk_index, content_type,"
            " text, image_id, context) VALUES (:id, :document_id, :page_number, :chunk_index,"
            " :content_type, :text, :image_id, :context)",
            [{"context": None, **c} for c in chunks],
        )


def get_chunks(document_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """All chunks (optionally only for some documents), with the document name."""
    sql = ("SELECT c.*, d.filename AS document_name FROM chunks c"
           " JOIN documents d ON d.id = c.document_id WHERE d.status = 'ready'")
    params: list[Any] = []
    if document_ids:
        sql += f" AND c.document_id IN ({','.join('?' * len(document_ids))})"
        params.extend(document_ids)
    with connect() as conn:
        rows = conn.execute(sql + " ORDER BY c.document_id, c.page_number, c.chunk_index",
                            params).fetchall()
    return [dict(r) for r in rows]


def get_chunks_by_ids(chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Chunks by id, only from READY documents (stale vectors are ignored)."""
    if not chunk_ids:
        return {}
    with connect() as conn:
        rows = conn.execute(
            "SELECT c.*, d.filename AS document_name FROM chunks c JOIN documents d"
            f" ON d.id = c.document_id WHERE c.id IN ({','.join('?' * len(chunk_ids))})"
            " AND d.status = 'ready'",
            chunk_ids,
        ).fetchall()
    return {r["id"]: dict(r) for r in rows}


def get_image(image_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT i.*, d.filename AS document_name FROM images i JOIN documents d"
            " ON d.id = i.document_id WHERE i.id = ?",
            (image_id,),
        ).fetchone()
    return _row(row)


def get_images_by_ids(image_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Images by id, only from READY documents."""
    if not image_ids:
        return {}
    with connect() as conn:
        rows = conn.execute(
            "SELECT i.*, d.filename AS document_name FROM images i JOIN documents d"
            f" ON d.id = i.document_id WHERE i.id IN ({','.join('?' * len(image_ids))})"
            " AND d.status = 'ready'",
            image_ids,
        ).fetchall()
    return {r["id"]: dict(r) for r in rows}


def get_pages(document_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM pages WHERE document_id = ? ORDER BY page_number", (document_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def count_content(document_id: str) -> dict[str, int]:
    with connect() as conn:
        chunks = conn.execute(
            "SELECT content_type, COUNT(*) AS n FROM chunks WHERE document_id = ?"
            " GROUP BY content_type", (document_id,)
        ).fetchall()
        images = conn.execute(
            "SELECT COUNT(*) FROM images WHERE document_id = ?", (document_id,)
        ).fetchone()[0]
    counts = {f"{r['content_type']}_chunks": r["n"] for r in chunks}
    counts["images"] = images
    return counts


# --- Chat -------------------------------------------------------------------------


def create_session(title: str) -> str:
    session_id = new_id()
    with connect() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (id, title, created_at) VALUES (?, ?, ?)",
            (session_id, title[:80], now_iso()),
        )
    return session_id


def session_exists(session_id: str) -> bool:
    with connect() as conn:
        return conn.execute(
            "SELECT 1 FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone() is not None


def list_sessions() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT s.*, COUNT(m.id) AS message_count FROM chat_sessions s"
            " LEFT JOIN chat_messages m ON m.session_id = s.id"
            " GROUP BY s.id ORDER BY s.created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> bool:
    with connect() as conn:
        cursor = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    return cursor.rowcount > 0


def add_message(session_id: str, role: str, content: str, payload: dict | None = None) -> str:
    message_id = new_id()
    with connect() as conn:
        conn.execute(
            "INSERT INTO chat_messages (id, session_id, role, content, payload_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, session_id, role, content,
             json.dumps(payload) if payload is not None else None, now_iso()),
        )
    return message_id


def get_messages(session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
    messages = []
    for r in rows:
        m = dict(r)
        m["payload"] = json.loads(m.pop("payload_json") or "null")
        messages.append(m)
    return messages[-limit:] if limit else messages
