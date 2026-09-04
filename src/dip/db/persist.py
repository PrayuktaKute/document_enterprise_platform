"""Persist a pipeline DocState into Postgres."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from dip.db.models import (
    AuditLog,
    Document,
    Extraction,
    FieldConfidence,
    ReviewQueue,
    ValidationResult,
)
from dip.db.session import get_session


def persist_state(state: dict) -> int:
    """Insert/update a document + a new extraction row. Returns extraction id."""
    with get_session() as s:
        doc = s.get(Document, state["doc_id"])
        if doc is None:
            doc = Document(
                id=state["doc_id"],
                filename=Path(state["file_path"]).name,
                file_path=str(state["file_path"]),
            )
            s.add(doc)
        doc.doc_type = state.get("doc_type")
        doc.doc_type_confidence = state.get("doc_type_confidence")
        doc.status = state.get("status", "processing")

        ex = Extraction(
            document_id=doc.id,
            payload=state.get("extraction") or {},
            doc_confidence=state.get("doc_confidence"),
            source="model",
        )
        s.add(ex)
        s.flush()

        method = state.get("confidence_method", "none")
        for name, conf in (state.get("field_confidences") or {}).items():
            s.add(FieldConfidence(extraction_id=ex.id, field_name=name,
                                  confidence=float(conf), method=method))

        for v in state.get("validation") or []:
            s.add(ValidationResult(
                extraction_id=ex.id, rule_name=v["rule_name"], passed=bool(v["passed"]),
                is_critical=bool(v.get("is_critical")), message=v.get("message"),
            ))

        if state.get("status") == "needs_review":
            reason = ", ".join(state.get("critical_failures") or []) or \
                f"low confidence ({state.get('doc_confidence', 0):.2f})"
            open_item = (
                s.query(ReviewQueue)
                .filter(ReviewQueue.document_id == doc.id, ReviewQueue.resolved.is_(False))
                .first()
            )
            if open_item is None:
                s.add(ReviewQueue(document_id=doc.id, reason=reason))

        s.add(AuditLog(
            document_id=doc.id, event="pipeline_run",
            detail={"status": state.get("status"), "errors": state.get("errors") or [],
                    "auto_accept": state.get("auto_accept")},
        ))
        s.flush()
        return ex.id


def resolve_review(doc_id: str, corrected: dict, note: str = "") -> None:
    """Human correction: new 'human' extraction + close the review item."""
    with get_session() as s:
        doc = s.get(Document, doc_id)
        if doc is None:
            raise KeyError(doc_id)
        ex = Extraction(document_id=doc_id, payload=corrected, doc_confidence=1.0, source="human")
        s.add(ex)
        for item in s.query(ReviewQueue).filter(
            ReviewQueue.document_id == doc_id, ReviewQueue.resolved.is_(False)
        ):
            item.resolved = True
            item.resolved_at = datetime.now(timezone.utc)
        doc.status = "indexed"
        s.add(AuditLog(document_id=doc_id, event="human_review", detail={"note": note}))
