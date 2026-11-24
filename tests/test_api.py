"""API behaviour for uploads and documents. Background processing is replaced by a
no-op here (processing itself is checked by scripts/smoke_test.py)."""

import io

import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend.api import documents as documents_api
from backend.main import app


def make_pdf(text: str = "Hello from a test PDF.", pages: int = 1) -> bytes:
    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"{text} Page {i + 1}.")
    return doc.tobytes()


@pytest.fixture(scope="module")
def client():
    documents_api.process_document = lambda document_id: None  # don't load ML models
    with TestClient(app) as c:
        yield c


def upload(client, data: bytes, name: str = "test.pdf"):
    return client.post("/api/documents/upload",
                       files={"file": (name, io.BytesIO(data), "application/pdf")})


def test_health_does_not_need_llm(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["llm_configured"] is False


def test_rejects_non_pdf(client):
    r = upload(client, b"just some text", "notes.pdf")
    assert r.status_code == 415


def test_rejects_empty_file(client):
    assert upload(client, b"").status_code == 400


def test_rejects_corrupted_pdf(client):
    r = upload(client, b"%PDF-1.7\n this is not really a pdf \n%%EOF")
    assert r.status_code == 422


def test_rejects_encrypted_pdf(client):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    r = upload(client, data, "locked.pdf")
    assert r.status_code == 422
    assert "password" in r.json()["detail"].lower()


def test_upload_list_get_page_delete(client):
    data = make_pdf(pages=2)
    r = upload(client, data, "../../evil name.pdf")
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["filename"] == "evil name.pdf"  # directories stripped, name kept
    assert doc["page_count"] == 2

    assert any(d["id"] == doc["id"] for d in client.get("/api/documents").json())
    assert client.get(f"/api/documents/{doc['id']}").status_code == 200

    page = client.get(f"/api/documents/{doc['id']}/page/1?highlight=Hello")
    assert page.status_code == 200 and page.headers["content-type"] == "image/png"
    assert client.get(f"/api/documents/{doc['id']}/page/3").status_code == 404

    dup = upload(client, data, "copy.pdf")  # same bytes, different name
    assert dup.status_code == 409 and dup.json()["existing_document_id"] == doc["id"]

    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert client.get(f"/api/documents/{doc['id']}").status_code == 404


def test_unknown_ids_return_404(client):
    assert client.get("/api/documents/nope").status_code == 404
    assert client.get("/api/documents/nope/page/1").status_code == 404
    assert client.get("/api/images/nope").status_code == 404
    assert client.get("/api/chat/history?session_id=nope").status_code == 404


def test_chat_without_documents_is_a_clear_error(client):
    r = client.post("/api/chat", json={"question": "Anything?"})
    assert r.status_code == 400
    assert "upload" in r.json()["detail"].lower()


def test_blank_question_rejected(client):
    assert client.post("/api/chat", json={"question": "   "}).status_code == 422
