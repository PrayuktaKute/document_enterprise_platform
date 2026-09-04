"""Phase 3: run the LangGraph pipeline over the eval manifest.

Writes one JSON line per document to artifacts/pipeline_results.jsonl with the
prediction plus the ground truth, ready for scripts/eval_extraction.py.

    python scripts/run_pipeline_batch.py --workers 3
    python scripts/run_pipeline_batch.py --limit 8 --types invoice,purchase_order --persist
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dip.pipeline import run_document

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "eval" / "manifest.jsonl"
OUT = REPO_ROOT / "artifacts" / "pipeline_results.jsonl"

_KEYS = (
    "doc_type", "doc_type_confidence", "extraction", "field_confidences",
    "doc_confidence", "confidence_method", "validation", "critical_failures",
    "auto_accept", "status", "errors",
)


def _run_one(row: dict, index: bool) -> dict:
    t0 = time.time()
    fp = str(REPO_ROOT / row["file_path"])
    try:
        state = run_document(row["doc_id"], fp, index=index)
        rec = {k: state.get(k) for k in _KEYS}
        rec["ok"] = state.get("status") != "failed"
    except Exception as exc:  # noqa: BLE001
        rec = {k: None for k in _KEYS}
        rec.update(status="failed", errors=[f"crash: {exc}"], ok=False)
    rec.update(
        doc_id=row["doc_id"],
        gt_doc_type=row["doc_type"],
        ground_truth=row["ground_truth"],
        seconds=round(time.time() - t0, 1),
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--types", type=str, default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--index", action="store_true", help="also embed+upsert to Qdrant per doc")
    ap.add_argument("--persist", action="store_true", help="write rows to Postgres")
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    # Preflight: fail fast + loud if the LLM endpoint is unreachable, instead of
    # 120 documents each burning through the client retry budget.
    try:
        from dip.llm import LLMClient

        r = LLMClient.from_config().chat(
            [{"role": "user", "content": "ping"}], max_tokens=1
        )
        print(f"LLM preflight ok (model responded, finish={r.finish_reason})")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"\nLLM preflight FAILED: {exc}\n"
            f"Is Ollama running and the model pulled? "
            f"Check: curl http://localhost:11434/api/version ; ollama list\n"
        )

    rows = [json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.types:
        keep = set(args.types.split(","))
        rows = [r for r in rows if r["doc_type"] in keep]
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    persist_fn = None
    if args.persist:
        from dip.db import init_db
        from dip.db.persist import persist_state

        init_db()
        persist_fn = persist_state

    print(f"running {len(rows)} docs  workers={args.workers} index={args.index} persist={args.persist}")
    t0 = time.time()
    done = 0
    with out_path.open("w", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, r, args.index): r for r in rows}
        for fut in as_completed(futs):
            rec = fut.result()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if persist_fn:
                try:
                    persist_fn(rec)
                except Exception as exc:  # noqa: BLE001
                    print(f"  persist {rec['doc_id']}: {exc}")
            done += 1
            print(f"  [{done:3d}/{len(rows)}] {rec['doc_id']:20s} "
                  f"pred={rec['doc_type']} conf={rec.get('doc_confidence')} "
                  f"status={rec['status']} {rec['seconds']}s")

    print(f"done in {time.time() - t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
