"""Structure-aware chunking for retrieval.

``layout_section`` keeps section boundaries and windows long sections by an
approximate token budget (words * 1.3). ``fixed_window`` ignores structure.
"""
from __future__ import annotations

from typing import Iterator

from pydantic import BaseModel

from dip.parsing.docling_parser import ParsedDoc

_WORDS_PER_TOKEN = 1 / 1.3


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    order: int
    section: str | None
    text: str


def _windows(words: list[str], size: int, overlap: int) -> Iterator[str]:
    if not words:
        return
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if not piece:
            break
        yield " ".join(piece)
        if start + size >= len(words):
            break


def chunk_parsed(
    parsed: ParsedDoc,
    *,
    strategy: str = "layout_section",
    max_tokens: int = 512,
    overlap: int = 64,
    min_chars: int = 40,
) -> list[Chunk]:
    max_words = max(32, int(max_tokens * _WORDS_PER_TOKEN))
    overlap_words = max(0, int(overlap * _WORDS_PER_TOKEN))

    chunks: list[Chunk] = []
    order = 0

    if strategy == "fixed_window":
        blocks = [(None, parsed.text)]
    else:
        blocks = [(s.heading, s.text) for s in parsed.sections]

    for heading, text in blocks:
        text = (text or "").strip()
        if len(text) < min_chars:
            continue
        for window in _windows(text.split(), max_words, overlap_words):
            if len(window) < min_chars:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{parsed.doc_id}::{order:04d}",
                    doc_id=parsed.doc_id,
                    order=order,
                    section=heading,
                    text=window,
                )
            )
            order += 1

    if not chunks and parsed.text.strip():
        chunks.append(
            Chunk(
                chunk_id=f"{parsed.doc_id}::0000",
                doc_id=parsed.doc_id,
                order=0,
                section=None,
                text=parsed.text.strip()[: max_words * 8],
            )
        )
    return chunks
