"""Layout-aware document parsing via Docling, with a JSON cache.

Docling handles PDF and image (OCR) inputs. Plain-text inputs (CUAD contracts)
skip Docling and use a light heading splitter. Parsed output is cached to
``data/cache/{doc_id}.json`` so the expensive parse runs once.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from dip.config import get_settings

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+[A-Z].*"          # "1.2 Something"
    r"|ARTICLE\s+[IVXLC0-9]+.*"
    r"|SECTION\s+\d+.*"
    r"|[A-Z][A-Z0-9 ,'/&\-]{4,}"                   # ALL-CAPS line
    r")\s*$"
)


class Section(BaseModel):
    heading: str | None = None
    text: str
    page: int | None = None


class ParsedDoc(BaseModel):
    doc_id: str
    source_path: str
    source_format: str          # pdf | image | text
    parser: str                 # docling | raw_text
    text: str
    sections: list[Section]
    page_count: int | None = None

    @property
    def char_len(self) -> int:
        return len(self.text)


# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _converter():
    """DocumentConverter is expensive to construct (loads layout/OCR models)."""
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


def _sections_from_markdown(md: str) -> list[Section]:
    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []
    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            if buf and "".join(buf).strip():
                sections.append(Section(heading=heading, text="\n".join(buf).strip()))
            heading = m.group(2).strip()
            buf = []
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        sections.append(Section(heading=heading, text="\n".join(buf).strip()))
    if not sections:
        sections = [Section(heading=None, text=md.strip())]
    return sections


def _sections_from_text(text: str) -> list[Section]:
    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line and _HEADING_RE.match(line) and len(line) < 90:
            if buf and "".join(buf).strip():
                sections.append(Section(heading=heading, text="\n".join(buf).strip()))
            heading = line.strip()
            buf = []
        else:
            buf.append(raw)
    if buf and "".join(buf).strip():
        sections.append(Section(heading=heading, text="\n".join(buf).strip()))
    return sections or [Section(heading=None, text=text.strip())]


def _parse_text_file(path: Path, doc_id: str) -> ParsedDoc:
    text = path.read_text(encoding="utf-8", errors="replace")
    return ParsedDoc(
        doc_id=doc_id,
        source_path=str(path),
        source_format="text",
        parser="raw_text",
        text=text,
        sections=_sections_from_text(text),
    )


def _parse_with_docling(path: Path, doc_id: str) -> ParsedDoc:
    result = _converter().convert(str(path))
    doc = result.document
    md = doc.export_to_markdown()
    try:
        page_count = len(doc.pages) or None
    except Exception:  # noqa: BLE001
        page_count = None
    fmt = "image" if path.suffix.lower() in _IMAGE_EXTS else "pdf"
    return ParsedDoc(
        doc_id=doc_id,
        source_path=str(path),
        source_format=fmt,
        parser="docling",
        text=md,
        sections=_sections_from_markdown(md),
        page_count=page_count,
    )


# --------------------------------------------------------------------------- #
def parse_document(
    file_path: str | Path,
    doc_id: str,
    *,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    force: bool = False,
) -> ParsedDoc:
    path = Path(file_path)
    cache_dir = Path(cache_dir or get_settings().cache_dir)
    cache_path = cache_dir / f"{doc_id}.json"

    if use_cache and not force and cache_path.exists():
        return ParsedDoc.model_validate_json(cache_path.read_text(encoding="utf-8"))

    if path.suffix.lower() == ".txt":
        parsed = _parse_text_file(path, doc_id)
    else:
        parsed = _parse_with_docling(path, doc_id)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
    return parsed
