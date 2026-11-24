"""
backend/ingestion/chunking.py

Split page text into overlapping chunks for embedding.

Why chunk at all? Embedding models have an input limit (all-MiniLM-L6-v2 reads
at most 256 word-pieces, about 1,000 characters) and a smaller passage gives a
more precise vector: one idea per chunk retrieves better than ten ideas averaged.

Rules used here:
  * A chunk never crosses a page boundary, so every chunk has ONE exact page
    number for citations.
  * We split at the largest natural boundary that fits: paragraph, then line,
    then sentence, then word. Only as a last resort is a word cut.
  * Consecutive chunks overlap by `overlap` characters, so a sentence split
    between two chunks still appears whole in at least one of them.
"""

import re

SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+", " "]


def _split(text: str, separator: str) -> list[str]:
    """Split, but keep the separator on each part so joining restores the text."""
    if separator.startswith("(?"):  # sentence regex: whitespace after . ! ?
        parts = re.split(separator, text)
        joiner = " "
    else:
        parts = text.split(separator)
        joiner = separator
    return [p + joiner for p in parts[:-1]] + parts[-1:]


def _pieces(text: str, size: int, level: int = 0) -> list[str]:
    """Break text into pieces no longer than `size`, preferring big boundaries."""
    if len(text) <= size:
        return [text]
    if level >= len(SEPARATORS):
        return [text[i:i + size] for i in range(0, len(text), size)]
    pieces: list[str] = []
    for part in _split(text, SEPARATORS[level]):
        if len(part) <= size:
            pieces.append(part)
        else:
            pieces.extend(_pieces(part, size, level + 1))
    return pieces


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    overlap = min(overlap, size // 2)
    chunks: list[str] = []
    current = ""
    for piece in _pieces(text, size):
        if len(current) + len(piece) <= size:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        # Start the next chunk with the tail of the previous one (the overlap),
        # cut at a word boundary so we don't begin mid-word.
        tail = current[-overlap:] if overlap else ""
        if " " in tail:
            tail = tail[tail.index(" ") + 1:]
        current = (tail + piece) if len(tail) + len(piece) <= size else piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def split_table(markdown: str, size: int) -> list[str]:
    """Split a long Markdown table into parts, repeating the header row in each."""
    lines = markdown.splitlines()
    if len(markdown) <= size or len(lines) <= 3:
        return [markdown]
    header = lines[:2]
    parts, current = [], list(header)
    for row in lines[2:]:
        if len("\n".join(current + [row])) > size and len(current) > 2:
            parts.append("\n".join(current))
            current = list(header)
        current.append(row)
    if len(current) > 2:
        parts.append("\n".join(current))
    return parts
