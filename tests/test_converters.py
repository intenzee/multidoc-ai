"""File-type detection and the fast converters (text, multi-page TIFF). Word, PowerPoint
and audio need LibreOffice / ffmpeg + Whisper and are checked end to end by hand."""

import io
import zipfile

import pytest
from PIL import Image

from backend.ingestion.converters import text_pages
from backend.ingestion.upload import _office_type_from_zip, detect_file_type


@pytest.mark.parametrize("head,name,expected", [
    (b"%PDF-1.7 ...", "x.pdf", "pdf"),
    (b"\x89PNG\r\n\x1a\n....", "x.png", "png"),
    (b"\xff\xd8\xff\xe0....", "x.jpg", "jpeg"),
    (b"II*\x00....", "scan.tif", "tiff"),
    (b"\x00\x00\x00\x18ftypheic....", "IMG_0001.HEIC", "heic"),
    (b"\x00\x00\x00\x20ftypM4A ....", "voice.m4a", "audio"),
    (b"ID3\x04\x00....", "talk.mp3", "audio"),
    (b"RIFF\x24\x08\x00\x00WAVEfmt ", "rec.wav", "audio"),
    (b"OggS\x00\x02....", "note.opus", "audio"),
    (b"PK\x03\x04....", "slides.pptx", "zip"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1....", "old.ppt", "ppt"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1....", "budget.xls", None),  # spreadsheets: no
    ("नमस्ते, notes".encode(), "notes.txt", "text"),
    (b"# Title\n\ntext", "notes.md", "markdown"),
    (b"# Title\n\ntext", "script.py", None),  # text only with a text extension
    (b"\x00\x01\x02binary", "fake.txt", None),
    (b"MZ\x90\x00 windows exe", "virus.pdf", None),
])
def test_detect_file_type(head, name, expected):
    assert detect_file_type(head, name) == expected


def test_office_type_comes_from_the_zip_contents(tmp_path):
    def make(names):
        path = tmp_path / "f.zip"
        with zipfile.ZipFile(path, "w") as z:
            for n in names:
                z.writestr(n, "x")
        return path
    assert _office_type_from_zip(make(["word/document.xml"])) == "docx"
    assert _office_type_from_zip(make(["ppt/presentation.xml"])) == "pptx"
    assert _office_type_from_zip(make(["xl/workbook.xml"])) is None
    assert _office_type_from_zip(make(["evil.exe"])) is None


def test_text_pages_split_at_line_breaks():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(200))
    pages = text_pages(text, limit=1000)
    assert len(pages) > 5 and all(len(p) <= 1000 for p in pages)
    assert "\n".join(pages) == text  # nothing lost or reordered


def test_multipage_tiff_upload():
    from fastapi.testclient import TestClient

    from backend.api import documents as documents_api
    from backend.main import app

    documents_api.process_document = lambda document_id: None
    frames = [Image.new("RGB", (300, 200), c) for c in ("white", "gray", "black")]
    data = io.BytesIO()
    frames[0].save(data, format="TIFF", save_all=True, append_images=frames[1:])
    with TestClient(app) as client:
        r = client.post("/api/documents/upload",
                        files={"file": ("scan.tiff", data.getvalue(), "image/tiff")})
        assert r.status_code == 201, r.text
        assert r.json()["page_count"] == 3
        client.delete(f"/api/documents/{r.json()['id']}")
