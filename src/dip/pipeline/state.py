from __future__ import annotations

from typing import TypedDict


class DocState(TypedDict, total=False):
    # inputs
    doc_id: str
    file_path: str
    forced_doc_type: str | None

    # parse
    parsed: dict | None            # ParsedDoc.model_dump()

    # classify
    doc_type: str | None
    doc_type_confidence: float | None

    # extract
    extraction: dict
    field_confidences: dict[str, float]
    doc_confidence: float
    confidence_method: str

    # validate / route
    validation: list[dict]
    critical_failures: list[str]
    auto_accept: bool

    # index
    chunks_indexed: int

    # bookkeeping
    status: str                    # processing | auto_accepted | needs_review | indexed | failed
    errors: list[str]
