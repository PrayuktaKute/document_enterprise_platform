"""Qdrant vector store: collection lifecycle, chunk upsert, filtered search, snapshot."""
from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from dip.config import get_pipeline_config, get_settings
from dip.parsing.chunker import Chunk

_NS = uuid.UUID("6f4d3c2b-1a09-4e8d-9c7b-000000000001")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


class VectorStore:
    def __init__(
        self, url: str | None, collection: str, dim: int = 1024, path: str | None = None
    ) -> None:
        self.client = QdrantClient(path=path) if path else QdrantClient(url=url)
        self.embedded = bool(path)
        self.collection = collection
        self.dim = dim

    @classmethod
    def from_config(cls) -> "VectorStore":
        s = get_settings()
        pc = get_pipeline_config().embedding
        return cls(s.qdrant_url, s.qdrant_collection, pc.dense_dim, path=s.qdrant_path or None)

    # ------------------------------------------------------------------ #
    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            )
            for field in ("doc_type", "doc_id"):
                try:
                    self.client.create_payload_index(
                        self.collection, field, qm.PayloadSchemaType.KEYWORD
                    )
                except Exception:  # noqa: BLE001  (embedded mode may not support this)
                    pass

    def upsert_chunks(
        self, chunks: list[Chunk], vectors: list[list[float]], extra_payload: dict | None = None
    ) -> int:
        points = []
        for ch, vec in zip(chunks, vectors):
            payload: dict[str, Any] = {
                "chunk_id": ch.chunk_id,
                "doc_id": ch.doc_id,
                "order": ch.order,
                "section": ch.section,
                "text": ch.text,
            }
            if extra_payload:
                payload.update(extra_payload)
            points.append(qm.PointStruct(id=_point_id(ch.chunk_id), vector=vec, payload=payload))
        if points:
            self.client.upsert(self.collection, points=points, wait=True)
        return len(points)

    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])
            ),
        )

    def search(
        self, query_vec: list[float], top_k: int = 5, doc_type: str | None = None
    ) -> list[dict]:
        flt = None
        if doc_type:
            flt = qm.Filter(
                must=[qm.FieldCondition(key="doc_type", match=qm.MatchValue(value=doc_type))]
            )
        res = self.client.query_points(
            self.collection, query=query_vec, limit=top_k, query_filter=flt, with_payload=True
        ).points
        return [{"score": p.score, **(p.payload or {})} for p in res]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    # ------------------------------------------------------------------ #
    def snapshot(self) -> str | None:
        if self.embedded:
            return None
        snap = self.client.create_snapshot(self.collection)
        return snap.name if snap else None

    def list_snapshots(self) -> list[str]:
        if self.embedded:
            return []
        return [s.name for s in self.client.list_snapshots(self.collection)]
