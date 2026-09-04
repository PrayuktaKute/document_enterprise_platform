"""BAAI BGE-M3 dense embeddings (FlagEmbedding). Heavy import -- lazy-loaded."""
from __future__ import annotations

from functools import lru_cache

from dip.config import get_pipeline_config, get_settings


@lru_cache(maxsize=1)
def _model():
    from FlagEmbedding import BGEM3FlagModel

    name = get_settings().embed_model or get_pipeline_config().embedding.model
    return BGEM3FlagModel(name, use_fp16=False)


def embed_texts(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    if not texts:
        return []
    cfg = get_pipeline_config().embedding
    out = _model().encode(
        texts,
        batch_size=batch_size or cfg.batch_size,
        max_length=8192,
    )["dense_vecs"]
    return [v.tolist() for v in out]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
