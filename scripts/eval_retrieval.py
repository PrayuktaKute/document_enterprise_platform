"""Phase 4: retrieval eval -- recall@k and MRR over data/eval/retrieval_queries.jsonl.

    python scripts/eval_retrieval.py --k 5
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dip.retrieval.search import semantic_search
from dip.retrieval.store import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERIES = REPO_ROOT / "data" / "eval" / "retrieval_queries.jsonl"
ART = REPO_ROOT / "artifacts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--filter-type", action="store_true", help="pass doc_type filter to the query")
    args = ap.parse_args()

    rows = [json.loads(x) for x in QUERIES.read_text(encoding="utf-8").splitlines() if x.strip()]
    graded = [r for r in rows if r.get("relevant_doc_ids")]

    store = VectorStore.from_config()
    hit = 0
    rr_sum = 0.0
    per_type = defaultdict(lambda: [0, 0])
    samples = []

    for r in graded:
        dt = r["doc_type"] if args.filter_type else None
        results = semantic_search(r["query"], top_k=args.k, doc_type=dt, store=store)
        got = [h["doc_id"] for h in results]
        rel = set(r["relevant_doc_ids"])
        rank = next((i + 1 for i, d in enumerate(got) if d in rel), None)
        ok = rank is not None
        hit += int(ok)
        rr_sum += (1.0 / rank) if rank else 0.0
        per_type[r["doc_type"]][0] += int(ok)
        per_type[r["doc_type"]][1] += 1
        if len(samples) < 8:
            samples.append({"query": r["query"], "want": r["relevant_doc_ids"], "got": got, "rank": rank})

    n = len(graded)
    metrics = {
        "n_queries_graded": n,
        "n_queries_total": len(rows),
        f"recall_at_{args.k}": round(hit / n, 4) if n else 0.0,
        f"mrr_at_{args.k}": round(rr_sum / n, 4) if n else 0.0,
        "collection_points": store.count(),
        f"recall_at_{args.k}_by_type": {
            t: round(c[0] / c[1], 4) for t, c in per_type.items() if c[1]
        },
        "samples": samples,
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "retrieval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"graded queries      : {n}")
    print(f"indexed chunks      : {metrics['collection_points']}")
    print(f"recall@{args.k}            : {metrics[f'recall_at_{args.k}']:.1%}")
    print(f"MRR@{args.k}               : {metrics[f'mrr_at_{args.k}']:.3f}")
    for t, v in metrics[f"recall_at_{args.k}_by_type"].items():
        print(f"  {t:15s} {v:.1%}")
    print(f"-> {ART / 'retrieval_metrics.json'}")


if __name__ == "__main__":
    main()
