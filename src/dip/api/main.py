"""FastAPI surface for the platform."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select

from dip.config import get_settings
from dip.db.models import Document, Extraction, FieldConfidence, ReviewQueue, ValidationResult
from dip.db.persist import persist_state, resolve_review
from dip.db.session import get_session, init_db
from dip.pipeline import run_document
from dip.retrieval.indexing import index_document
from dip.retrieval.search import semantic_search

app = FastAPI(title="Enterprise Document Intelligence Platform", version="0.1.0")

UPLOAD_DIR = get_settings().data_dir / "uploads"
ART = Path(get_settings().artifacts_dir)


@app.on_event("startup")
def _startup() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    init_db()


# --------------------------------------------------------------------------- #
def _process(doc_id: str, path: str, doc_type: str | None) -> None:
    state = run_document(doc_id, path, forced_doc_type=doc_type, index=True)
    persist_state(state)


def _document_view(doc_id: str) -> dict:
    with get_session() as s:
        doc = s.get(Document, doc_id)
        if doc is None:
            raise HTTPException(404, f"unknown document {doc_id}")
        ex = s.scalars(
            select(Extraction).where(Extraction.document_id == doc_id)
            .order_by(Extraction.id.desc())
        ).first()
        payload, conf, fconf, vres, src = {}, None, [], [], None
        if ex is not None:
            payload, conf, src = ex.payload, ex.doc_confidence, ex.source
            fconf = [
                {"field": f.field_name, "confidence": f.confidence, "method": f.method}
                for f in s.scalars(
                    select(FieldConfidence).where(FieldConfidence.extraction_id == ex.id)
                )
            ]
            vres = [
                {"rule": v.rule_name, "passed": v.passed, "critical": v.is_critical, "message": v.message}
                for v in s.scalars(
                    select(ValidationResult).where(ValidationResult.extraction_id == ex.id)
                )
            ]
        return {
            "doc_id": doc.id,
            "filename": doc.filename,
            "doc_type": doc.doc_type,
            "doc_type_confidence": doc.doc_type_confidence,
            "status": doc.status,
            "extraction": payload,
            "extraction_source": src,
            "doc_confidence": conf,
            "field_confidences": fconf,
            "validation": vres,
        }


# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(
    background: BackgroundTasks, file: UploadFile, doc_type: str | None = Form(default=None)
) -> dict:
    doc_id = f"upload_{uuid4().hex[:12]}"
    dest = UPLOAD_DIR / f"{doc_id}{Path(file.filename or '').suffix or '.bin'}"
    dest.write_bytes(await file.read())
    with get_session() as s:
        s.add(Document(id=doc_id, filename=file.filename or dest.name,
                       file_path=str(dest), status="processing"))
    background.add_task(_process, doc_id, str(dest), doc_type)
    return {"doc_id": doc_id, "status": "processing"}


@app.get("/documents")
def list_documents(status: str | None = None, limit: int = 100) -> list[dict]:
    with get_session() as s:
        q = select(Document).order_by(Document.created_at.desc()).limit(limit)
        if status:
            q = select(Document).where(Document.status == status).limit(limit)
        return [
            {"doc_id": d.id, "filename": d.filename, "doc_type": d.doc_type, "status": d.status}
            for d in s.scalars(q)
        ]


@app.get("/documents/{doc_id}")
def get_document(doc_id: str) -> dict:
    return _document_view(doc_id)


class ExtractionUpdate(BaseModel):
    payload: dict
    note: str = ""


@app.put("/documents/{doc_id}/extraction")
def correct_extraction(doc_id: str, body: ExtractionUpdate) -> dict:
    with get_session() as s:
        doc = s.get(Document, doc_id)
        if doc is None:
            raise HTTPException(404, f"unknown document {doc_id}")
        file_path, doc_type = doc.file_path, doc.doc_type
    resolve_review(doc_id, body.payload, note=body.note)
    try:
        n = index_document(doc_id, file_path, doc_type)
    except Exception as exc:  # noqa: BLE001
        n = -1
        print(f"reindex failed for {doc_id}: {exc}")
    return {"doc_id": doc_id, "status": "indexed", "chunks_indexed": n}


@app.post("/documents/{doc_id}/approve")
def approve_document(doc_id: str) -> dict:
    view = _document_view(doc_id)
    resolve_review(doc_id, view["extraction"], note="approved as-is")
    return {"doc_id": doc_id, "status": "indexed"}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_type: str | None = None


@app.post("/search")
def search(body: SearchRequest) -> dict:
    return {"query": body.query, "results": semantic_search(body.query, body.top_k, body.doc_type)}


@app.get("/metrics")
def metrics() -> dict:
    def _load(name: str) -> dict:
        p = ART / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    return {"extraction": _load("metrics.json"), "retrieval": _load("retrieval_metrics.json")}
