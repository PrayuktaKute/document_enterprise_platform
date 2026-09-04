"""Phase 3: parse -> chunk -> BGE-M3 embed -> upsert every manifest doc into Qdrant.

Runs independently of extraction quality (all docs are retrievable). Optionally
writes a Qdrant snapshot to restore on the laptop.

    python scripts/build_index.py --recreate --snapshot
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dip.config import get_doc_type_config
from dip.parsing import chunk_parsed, parse_document
from dip.retrieval.embed import embed_texts
from dip.retrieval.store import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "eval" / "manifest.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    rows = [json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        rows = rows[: args.limit]

    store = VectorStore.from_config()
    store.ensure_collection(recreate=args.recreate)

    all_chunks = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        try:
            parsed = parse_document(str(REPO_ROOT / r["file_path"]), r["doc_id"])
        except Exception as exc:  # noqa: BLE001
            print(f"  parse fail {r['doc_id']}: {exc}")
            continue
        ck = get_doc_type_config(r["doc_type"]).chunking
        chunks = chunk_parsed(parsed, strategy=ck.strategy, max_tokens=ck.max_tokens, overlap=ck.overlap)
        for c in chunks:
            all_chunks.append((c, r["doc_type"]))
        print(f"  [{i:3d}/{len(rows)}] {r['doc_id']:20s} {len(chunks):2d} chunks")

    print(f"embedding {len(all_chunks)} chunks with BGE-M3 ...")
    vectors = embed_texts([c.text for c, _ in all_chunks], batch_size=args.batch)

    by_type: dict[str, list] = {}
    for (c, dt), v in zip(all_chunks, vectors):
        by_type.setdefault(dt, []).append((c, v))
    total = 0
    for dt, items in by_type.items():
        n = store.upsert_chunks([c for c, _ in items], [v for _, v in items],
                                extra_payload={"doc_type": dt})
        total += n
    print(f"upserted {total} points; collection count = {store.count()}  ({time.time() - t0:.0f}s)")

    if args.snapshot:
        name = store.snapshot()
        print(f"snapshot: {name}  (server storage: /qdrant/snapshots/{store.collection}/{name})")


if __name__ == "__main__":
    main()
