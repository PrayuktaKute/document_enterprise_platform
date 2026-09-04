"""Replay artifacts/pipeline_results.jsonl into Postgres so the API / Streamlit
review queue work without re-running the pipeline (e.g. after a Colab eval run).

    python scripts/load_results_to_db.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT = REPO_ROOT / "artifacts" / "pipeline_results.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(DEFAULT))
    args = ap.parse_args()

    from dip.db import init_db
    from dip.db.persist import persist_state

    init_db()
    rows = [json.loads(x) for x in Path(args.results).read_text(encoding="utf-8").splitlines() if x.strip()]
    n_review = 0
    for r in rows:
        # normalise file_path to absolute for the Document row
        r = {**r, "file_path": str(REPO_ROOT / _guess_path(r))}
        persist_state(r)
        n_review += int(r.get("status") == "needs_review")
    print(f"loaded {len(rows)} documents ({n_review} in review queue)")


def _guess_path(rec: dict) -> str:
    folder = {
        "invoice": "invoices", "purchase_order": "purchase_orders",
        "medical_report": "medical_reports", "contract": "contracts",
    }.get(rec.get("gt_doc_type") or rec.get("doc_type"), "invoices")
    ext = ".jpg" if folder == "invoices" else (".txt" if folder == "contracts" else ".pdf")
    return f"data/raw/{folder}/{rec['doc_id']}{ext}"


if __name__ == "__main__":
    main()
