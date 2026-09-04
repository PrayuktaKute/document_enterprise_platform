"""LangGraph node functions. Each takes DocState and returns a partial update.

Nodes are pure w.r.t. external stores except ``index`` (writes Qdrant). DB
persistence is handled by the caller (batch runner / API), not here.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from dip.config import get_doc_type_configs, get_pipeline_config, resolve_confidence
from dip.llm.extract import extract_fields
from dip.parsing.docling_parser import ParsedDoc, parse_document
from dip.pipeline.classify import classify_document
from dip.pipeline.state import DocState
from dip.validation.rules import run_rules

_TERMINAL = {"failed"}


def ingest_node(state: DocState) -> dict:
    p = Path(state["file_path"])
    if not p.exists():
        return {"status": "failed", "errors": [f"file not found: {p}"]}
    return {"status": "processing", "errors": list(state.get("errors", []))}


def parse_node(state: DocState) -> dict:
    if state.get("status") in _TERMINAL:
        return {}
    try:
        parsed = parse_document(state["file_path"], state["doc_id"])
        return {"parsed": parsed.model_dump()}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "errors": state.get("errors", []) + [f"parse: {exc}"]}


def classify_node(state: DocState) -> dict:
    if state.get("status") in _TERMINAL:
        return {}
    if state.get("forced_doc_type"):
        return {"doc_type": state["forced_doc_type"], "doc_type_confidence": 1.0}
    text = (state.get("parsed") or {}).get("text", "")
    dt, conf = classify_document(text)
    return {"doc_type": dt, "doc_type_confidence": conf}


def extract_node(state: DocState) -> dict:
    if state.get("status") in _TERMINAL:
        return {}
    dt = state.get("doc_type")
    if dt not in get_doc_type_configs():
        return {"status": "failed", "errors": state.get("errors", []) + [f"unknown doc_type: {dt!r}"]}
    parsed = ParsedDoc.model_validate(state["parsed"])
    res = extract_fields(parsed, dt)
    errs = state.get("errors", []) + ([res.error] if res.error else [])
    return {
        "extraction": res.data,
        "field_confidences": res.field_confidences,
        "doc_confidence": res.doc_confidence,
        "confidence_method": res.confidence_method,
        "errors": errs,
    }


def validate_node(state: DocState) -> dict:
    if state.get("status") in _TERMINAL:
        return {"auto_accept": False, "critical_failures": ["pipeline_failed"]}
    dt = state["doc_type"]
    outcomes = run_rules(dt, state.get("extraction") or {})
    crit = [o.rule_name for o in outcomes if o.is_critical and not o.passed]
    _, field_thr, doc_thr = resolve_confidence(dt)
    tol = get_pipeline_config().confidence.max_low_conf_fields
    fconf = state.get("field_confidences", {}) or {}
    low = [f for f, c in fconf.items() if c < field_thr]
    auto = (
        not crit
        and state.get("doc_confidence", 0.0) >= doc_thr
        and len(low) <= tol
    )
    return {
        "validation": [asdict(o) for o in outcomes],
        "critical_failures": crit,
        "auto_accept": bool(auto),
    }


def route_decision(state: DocState) -> str:
    if state.get("status") in _TERMINAL:
        return "review"
    return "index" if state.get("auto_accept") else "review"


def index_node(state: DocState) -> dict:
    from dip.retrieval.indexing import index_parsed

    parsed = ParsedDoc.model_validate(state["parsed"])
    n = index_parsed(state["doc_id"], parsed, state["doc_type"])
    return {"chunks_indexed": n, "status": "indexed" if n else "auto_accepted"}


def review_node(state: DocState) -> dict:
    return {"status": "failed" if state.get("status") in _TERMINAL else "needs_review"}
