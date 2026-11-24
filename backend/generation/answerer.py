"""
backend/generation/answerer.py

question -> retrieval -> grounded prompt (text evidence + real image bytes)
         -> LLM -> refusal detection -> citation validation -> answer

The LLM SEES the original images (sent as PNG bytes), not only their captions.
If the LLM fails (quota, network...), the retrieved evidence is still returned,
so the user can read the sources themselves.
"""

import logging
import re
import time

from backend.config import get_settings
from backend.generation import llm_service
from backend.generation.prompts import (INSUFFICIENT_MARKER, SYSTEM_PROMPT, build_user_prompt,
                                        format_evidence_block, format_image_note)
from backend.retrieval.retriever import RetrievalOptions, retrieve

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]")
FOLLOW_UP_HINTS = {"it", "its", "that", "this", "they", "them", "those", "these", "he", "she",
                   "there", "same", "above", "previous", "more", "else", "also"}

NO_EVIDENCE_TEXT = ("I could not find information about this in the selected documents. "
                    "Try rephrasing the question or selecting other documents.")


def retrieval_query(question: str, history: list[tuple[str, str]]) -> str:
    """Short follow-ups ("what about Europe?", "and its price?") borrow the previous question."""
    words = re.findall(r"[a-z']+", question.lower())
    previous = [text for role, text in history if role == "user"]
    if previous and (len(words) <= 6 or FOLLOW_UP_HINTS.intersection(words)):
        return f"{previous[-1]} {question}"
    return question


def parse_citations(text: str, valid_labels: set[str]) -> tuple[str, list[str], list[str]]:
    """Return (text with invalid citations removed, valid labels used, invalid labels)."""
    used: list[str] = []
    invalid: list[str] = []

    def replace(match: re.Match) -> str:
        labels = [l.strip() for l in re.split(r"[,;]", match.group(1))]
        keep = []
        for label in labels:
            if label in valid_labels:
                keep.append(label)
                if label not in used:
                    used.append(label)
            elif label not in invalid:
                invalid.append(label)
        return "".join(f"[{l}]" for l in keep)

    cleaned = CITATION_RE.sub(replace, text)
    return cleaned, used, invalid


def _source_dict(label: str, item) -> dict:
    data = item.to_dict()
    data["label"] = label
    return data


def answer_question(question: str, document_ids: list[str] | None = None,
                    options: RetrievalOptions | None = None,
                    history: list[tuple[str, str]] | None = None) -> dict:
    settings = get_settings()
    history = history or []
    started = time.perf_counter()
    query = retrieval_query(question, history)

    result = retrieve(query, document_ids, options)
    timings = {"retrieval_ms": result.timings_ms.get("total", 0.0),
               "retrieval_detail_ms": result.timings_ms}

    # --- Label sources: S1..Sn text, then images; enforce the context budget ----------
    sources: list[dict] = []
    blocks: list[str] = []
    images: list[llm_service.ImageInput] = []
    budget = settings.max_context_chars
    for item in result.text_items:
        if budget <= 0:
            break
        label = f"S{len(sources) + 1}"
        text = item.text[:budget]
        budget -= len(text)
        sources.append(_source_dict(label, item))
        blocks.append(format_evidence_block(label, sources[-1], text))
    for item in result.image_items:
        relative = _image_path(item.image_id)
        path = settings.data_dir / relative if relative else None
        if path is None or not path.exists():
            continue
        label = f"S{len(sources) + 1}"
        sources.append(_source_dict(label, item))
        blocks.append(format_image_note(label, sources[-1]))
        images.append(llm_service.ImageInput(path.read_bytes(), f"[{label}]"))

    response = {
        "question": question, "retrieval_query": query, "answer": "", "status": "",
        "citations": [], "sources": sources, "warnings": [], "llm": None,
        "llm_error": None, "timings": timings,
    }

    # --- No evidence at all: answer honestly WITHOUT spending an LLM call ---------------
    # Weak relevance scores do NOT stop the LLM: broad questions ("summarise", "list the
    # subjects") score as low as off-topic ones, so the LLM decides from the evidence and
    # refuses with INSUFFICIENT_EVIDENCE when it cannot answer.
    if not sources:
        response.update(status="insufficient_evidence", answer=NO_EVIDENCE_TEXT, sources=[])
        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return response

    prompt = build_user_prompt(question, blocks, history[-2 * settings.history_turns:],
                               overview=result.overview)
    try:
        llm = llm_service.get_llm()
        reply = llm.generate(prompt, system=SYSTEM_PROMPT, images=images)
    except llm_service.LLMError as err:
        response.update(status="llm_error", llm_error={"code": err.code, "message": err.message},
                        answer="The answer could not be generated, but the most relevant "
                               "passages are listed below.")
        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return response

    timings["llm_ms"] = round(reply.latency_ms, 1)
    response["llm"] = {"model": reply.model, "attempts": reply.attempts, "usage": reply.usage,
                       "images_sent": len(images)}
    text = reply.text.strip()

    if text.upper().startswith(INSUFFICIENT_MARKER) or INSUFFICIENT_MARKER in text[:80].upper():
        reason = text.split(":", 1)[1].strip() if ":" in text else ""
        response.update(status="insufficient_evidence",
                        answer=reason or NO_EVIDENCE_TEXT)
    else:
        valid = {s["label"] for s in sources}
        cleaned, used, invalid = parse_citations(text, valid)
        response["answer"] = cleaned
        response["status"] = "answered"
        response["citations"] = [s for s in sources if s["label"] in used]
        if invalid:
            response["warnings"].append(
                f"Removed citations to sources that do not exist: {', '.join(invalid)}.")
        if not used:
            response["warnings"].append(
                "The answer has no citations, so it may not be grounded in the documents.")

    timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return response


def _image_path(image_id: str | None):
    from backend.storage import database as db

    if not image_id:
        return None
    row = db.get_image(image_id)
    return row["path"] if row else None
