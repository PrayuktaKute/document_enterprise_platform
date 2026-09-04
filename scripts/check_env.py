"""Phase 0 smoke test: config loads, DB reachable, Qdrant reachable, LLM answers
and returns token logprobs.

    python scripts/check_env.py
"""
from __future__ import annotations


def main() -> None:
    from dip.config import get_doc_type_configs, get_pipeline_config, get_settings

    s = get_settings()
    pc = get_pipeline_config()
    print("=== settings ===")
    print(f"  LLM      {s.llm_base_url}  model={s.llm_model}")
    print(f"  DB       {s.database_url}")
    print(f"  Qdrant   {s.qdrant_url}  collection={s.qdrant_collection}")
    print(f"  logprobs={pc.llm.logprobs} top_logprobs={pc.llm.top_logprobs} "
          f"confidence={pc.confidence.method}")

    print("=== doc types ===")
    for name, cfg in get_doc_type_configs().items():
        cls = cfg.schema_cls()
        print(f"  {name:15s} -> {cls.__name__:16s} fields={cls.extraction_fields()}")
        print(f"  {'':15s}    strategy={cfg.extraction.strategy} "
              f"critical={cfg.validation.critical_rules}")

    print("=== postgres ===")
    try:
        from dip.db import init_db

        init_db()
        print("  tables created / verified OK")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}")

    print("=== qdrant ===")
    try:
        from qdrant_client import QdrantClient

        qc = QdrantClient(url=s.qdrant_url)
        cols = [c.name for c in qc.get_collections().collections]
        print(f"  reachable; collections={cols}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}")

    print("=== llm + logprobs ===")
    try:
        from dip.llm import LLMClient

        client = LLMClient.from_config()
        resp = client.chat(
            [{"role": "user", "content": "Reply with exactly one word: pong"}],
            max_tokens=8,
            logprobs=True,
            top_logprobs=5,
        )
        print(f"  text={resp.text.strip()!r} finish={resp.finish_reason}")
        print(f"  logprobs_available={resp.has_logprobs} n_tokens={len(resp.tokens)}")
        for t in resp.tokens[:6]:
            print(f"    {t.token!r:12s} logprob={t.logprob:.4f} p={2.718281828 ** t.logprob:.4f}")
        if not resp.has_logprobs:
            print("  -> WARNING: no logprobs. Use CONFIDENCE_METHOD=self_consistency fallback.")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}")


if __name__ == "__main__":
    main()
