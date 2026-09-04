"""Semantic search entry point used by the API and the retrieval eval."""
from __future__ import annotations

from dip.retrieval.embed import embed_query
from dip.retrieval.store import VectorStore


def semantic_search(
    query: str, top_k: int = 5, doc_type: str | None = None, store: VectorStore | None = None
) -> list[dict]:
    store = store or VectorStore.from_config()
    hits = store.search(embed_query(query), top_k=top_k, doc_type=doc_type)
    return [
        {
            "rank": i + 1,
            "score": round(h.get("score", 0.0), 4),
            "doc_id": h.get("doc_id"),
            "doc_type": h.get("doc_type"),
            "section": h.get("section"),
            "text": h.get("text", ""),
            "chunk_id": h.get("chunk_id"),
        }
        for i, h in enumerate(hits)
    ]
