"""Phase 1c: unify raw data + ground truth into a single eval manifest.

Reads data/raw/<type>/ (documents + ground_truth.json) and writes
data/eval/manifest.jsonl with one row per document:

    {"doc_id", "doc_type", "file_path", "ground_truth": {...}}

    python scripts/prepare_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"
EVAL = REPO_ROOT / "data" / "eval"

TYPE_DIRS = {
    "invoice": "invoices",
    "purchase_order": "purchase_orders",
    "medical_report": "medical_reports",
    "contract": "contracts",
}
DOC_EXTS = {".pdf", ".jpg", ".jpeg", ".png", ".txt", ".tiff"}


def main() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    summary: dict[str, int] = {}

    for doc_type, folder in TYPE_DIRS.items():
        d = RAW / folder
        if not d.exists():
            print(f"  {doc_type}: no folder {d}, skipping")
            continue
        gt_path = d / "ground_truth.json"
        gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}

        count = 0
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in DOC_EXTS or f.name == "ground_truth.json":
                continue
            doc_id = f.stem
            rows.append({
                "doc_id": doc_id,
                "doc_type": doc_type,
                "file_path": str(f.relative_to(REPO_ROOT)).replace("\\", "/"),
                "ground_truth": gt.get(doc_id, {}),
            })
            count += 1
        summary[doc_type] = count

    manifest = EVAL / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows -> {manifest}")
    for k, v in summary.items():
        missing = sum(1 for r in rows if r["doc_type"] == k and not r["ground_truth"])
        print(f"  {k:15s} {v:3d} docs  ({missing} without ground truth)")


if __name__ == "__main__":
    main()
