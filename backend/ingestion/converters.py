"""
backend/ingestion/converters.py

Turn non-PDF uploads into a PDF, so the rest of the pipeline (extraction, chunking,
citations with page numbers, the page viewer) works the same for every file type.

  * Word / PowerPoint (.docx .doc .pptx .ppt)  -> LibreOffice, headless, no macros
  * Text (.txt .md)                              -> one PDF page per ~2,500 characters
  * Audio (.mp3 .wav .m4a .ogg .flac .webm ...)  -> transcribed locally with Whisper
                                                    (mlx, Apple GPU), one PDF "page"
                                                    per 2 minutes, with timestamps

For text and audio the exact text is also saved next to the PDF ("<id>.pages.json")
and indexed instead of the text read back from the PDF. Reading text back out of a
PDF garbles some scripts (e.g. Hindi conjuncts); the side file keeps it exact.

Uploaded files are untrusted: LibreOffice runs with its own profile, a time limit,
and is killed (with all its child processes) if it hangs.
"""

import html
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import pymupdf

from backend.config import get_settings

logger = logging.getLogger(__name__)

TEXT_PAGE_CHARS = 2500
OFFICE_APP = "/Applications/LibreOffice.app/Contents/MacOS/soffice"


class ConversionError(Exception):
    """The file could not be converted. The message is safe to show to the user."""


# --- Paths ------------------------------------------------------------------------------


def source_path(document_id: str, extension: str) -> Path:
    return get_settings().upload_dir / f"{document_id}.source{extension}"


def find_source(document_id: str) -> Path | None:
    matches = sorted(get_settings().upload_dir.glob(f"{document_id}.source.*"))
    return matches[0] if matches else None


def pages_path(document_id: str) -> Path:
    return get_settings().upload_dir / f"{document_id}.pages.json"


def load_page_texts(document_id: str) -> list[str] | None:
    path = pages_path(document_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --- Entry point ------------------------------------------------------------------------


def convert(document_id: str, display_name: str) -> int:
    """Convert the stored source file to <id>.pdf. Returns the number of pages."""
    from backend.ingestion.pipeline import document_pdf_path  # avoid an import cycle

    source = find_source(document_id)
    if source is None:
        raise ConversionError("The uploaded file is missing on the server. Please upload it again.")
    target = document_pdf_path(document_id)
    kind = source.suffix.lower()
    title = display_name.rsplit(".", 1)[0]

    if kind in (".docx", ".doc", ".pptx", ".ppt"):
        office_to_pdf(source, target)
        page_texts = None
    elif kind in (".txt", ".md"):
        page_texts = text_pages(read_text_file(source))
        texts_to_pdf(page_texts, target, title)
    else:
        page_texts = transcribe_to_pages(source)
        texts_to_pdf(page_texts, target, title)

    if page_texts is not None:
        pages_path(document_id).write_text(json.dumps(page_texts, ensure_ascii=False),
                                           encoding="utf-8")
    with pymupdf.open(target) as pdf:
        page_count = pdf.page_count
    if page_count == 0:
        raise ConversionError("The file has no content that could be read.")
    if page_count > get_settings().max_pages:
        raise ConversionError(f"The converted file has {page_count} pages; the limit is "
                              f"{get_settings().max_pages}.")
    return page_count


# --- Word / PowerPoint ------------------------------------------------------------------


def _office_binary() -> str:
    found = shutil.which("soffice") or (OFFICE_APP if Path(OFFICE_APP).exists() else None)
    if not found:
        raise ConversionError("Word and PowerPoint files need LibreOffice. Install it with: "
                              "brew install --cask libreoffice")
    return found


def office_to_pdf(source: Path, target: Path) -> None:
    settings = get_settings()
    profile = settings.data_dir / "libreoffice_profile"  # separate from the user's own
    with tempfile.TemporaryDirectory() as out_dir:
        command = [_office_binary(), "--headless", "--norestore", "--nolockcheck",
                   "--nodefault", f"-env:UserInstallation={profile.resolve().as_uri()}",
                   "--convert-to", "pdf", "--outdir", out_dir, str(source)]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            process.wait(timeout=settings.office_timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)  # LibreOffice starts helper processes
            process.wait()
            raise ConversionError("Converting this file took too long and was stopped.") from None
        produced = list(Path(out_dir).glob("*.pdf"))
        if process.returncode != 0 or not produced:
            raise ConversionError("This Word/PowerPoint file could not be converted. "
                                  "It may be damaged or password-protected.")
        shutil.move(str(produced[0]), target)


# --- Text -------------------------------------------------------------------------------


def read_text_file(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            if encoding == "utf-16" and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
                continue
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("cp1252", errors="replace")  # old Windows text files


def text_pages(text: str, limit: int = TEXT_PAGE_CHARS) -> list[str]:
    """Split text into page-sized parts at paragraph or line breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ConversionError("The text file is empty.")
    pages, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:  # one enormous line: cut it
            if current:
                pages.append(current)
                current = ""
            pages.append(line[:limit])
            line = line[limit:]
        if current and len(current) + len(line) + 1 > limit:
            pages.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    if current.strip():
        pages.append(current)
    return pages


def texts_to_pdf(page_texts: list[str], target: Path, title: str) -> None:
    """One PDF page per text part (for viewing; indexing uses the exact text)."""
    pdf = pymupdf.open()
    for text in page_texts:
        page = pdf.new_page()  # A4
        body = html.escape(text).replace("\n", "<br>")
        page.insert_htmlbox(page.rect + (50, 50, -50, -50),
                            f'<div style="font-size:11px; line-height:1.4">{body}</div>')
    pdf.set_metadata({"title": title})
    pdf.save(target, garbage=3, deflate=True)
    pdf.close()


# --- Audio ------------------------------------------------------------------------------


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    return (f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours
            else f"{rest // 60:02d}:{rest % 60:02d}")


def audio_duration_seconds(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ConversionError("Audio files need ffmpeg. Install it with: brew install ffmpeg")
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
             str(path)], capture_output=True, text=True, timeout=60)
        return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        raise ConversionError("This audio file could not be read. It may be damaged.") from None


def transcribe_to_pages(path: Path) -> list[str]:
    settings = get_settings()
    duration = audio_duration_seconds(path)
    if duration > settings.max_audio_minutes * 60:
        raise ConversionError(f"This recording is {duration / 60:.0f} minutes long; the limit "
                              f"is {settings.max_audio_minutes} minutes.")
    try:
        import mlx_whisper
    except ImportError:
        raise ConversionError("Audio transcription is not installed (pip install mlx-whisper).") \
            from None

    logger.info("Transcribing %.0f s of audio with %s", duration, settings.whisper_model)
    try:
        result = mlx_whisper.transcribe(str(path), path_or_hf_repo=settings.whisper_model,
                                        verbose=None, condition_on_previous_text=False)
    except Exception:
        logger.exception("Whisper failed")
        raise ConversionError("This audio file could not be transcribed.") from None

    segments = [s for s in result.get("segments", []) if s.get("text", "").strip()]
    if not segments:
        raise ConversionError("No speech was found in this recording.")

    window = settings.transcript_page_seconds
    pages: dict[int, list[str]] = {}
    for segment in segments:
        part = int(segment["start"] // window)
        pages.setdefault(part, []).append(f"[{_clock(segment['start'])}] {segment['text'].strip()}")
    language = result.get("language") or "unknown"
    texts = []
    for part in sorted(pages):
        end = min((part + 1) * window, duration)
        header = (f"Audio transcript, {_clock(part * window)} to {_clock(end)} "
                  f"(language: {language})")
        texts.append(header + "\n" + "\n".join(pages[part]))
    return texts
