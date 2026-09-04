"""LangGraph assembly and a convenience runner.

    ingest -> parse -> classify -> extract -> validate --(auto_accept)--> index -> END
                                                       \--(else)--------> review -> END
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, StateGraph

from dip.pipeline.nodes import (
    classify_node,
    extract_node,
    index_node,
    ingest_node,
    parse_node,
    review_node,
    route_decision,
    validate_node,
)
from dip.pipeline.state import DocState


@lru_cache(maxsize=2)
def build_pipeline(index: bool = True):
    g = StateGraph(DocState)
    g.add_node("ingest", ingest_node)
    g.add_node("parse", parse_node)
    g.add_node("classify", classify_node)
    g.add_node("extract", extract_node)
    g.add_node("validate", validate_node)
    g.add_node("review", review_node)
    g.set_entry_point("ingest")
    g.add_edge("ingest", "parse")
    g.add_edge("parse", "classify")
    g.add_edge("classify", "extract")
    g.add_edge("extract", "validate")

    if index:
        g.add_node("index", index_node)
        g.add_conditional_edges("validate", route_decision, {"index": "index", "review": "review"})
        g.add_edge("index", END)
    else:
        g.add_conditional_edges("validate", route_decision, {"index": "review", "review": "review"})
    g.add_edge("review", END)
    return g.compile()


def run_document(
    doc_id: str,
    file_path: str,
    *,
    forced_doc_type: str | None = None,
    index: bool = True,
) -> DocState:
    app = build_pipeline(index=index)
    init: DocState = {
        "doc_id": doc_id,
        "file_path": file_path,
        "forced_doc_type": forced_doc_type,
        "errors": [],
    }
    return app.invoke(init)  # type: ignore[return-value]
