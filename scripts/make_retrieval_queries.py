"""Build data/eval/retrieval_queries.jsonl from ground truth.

Each row: {"query": str, "relevant_doc_ids": [doc_id, ...], "doc_type": str}
~2 queries per document (natural-language, not keyword-identical) plus a few
cross-document thematic queries.

    python scripts/make_retrieval_queries.py --per-type 20
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "eval" / "manifest.jsonl"
OUT = REPO_ROOT / "data" / "eval" / "retrieval_queries.jsonl"


def _first_party(parties) -> str | None:
    if isinstance(parties, list) and parties:
        return str(parties[0])
    if isinstance(parties, str) and parties:
        import re

        bits = [b.strip(' "\'') for b in re.split(r";|,|\band\b", parties) if b.strip()]
        return bits[0] if bits else None
    return None


def queries_for(row: dict) -> list[str]:
    gt, t = row["ground_truth"], row["doc_type"]
    out: list[str] = []
    if t == "invoice":
        comp = (gt.get("company") or "").strip()
        addr = (gt.get("address") or "").strip()
        if comp:
            out.append(f"invoice or receipt from {comp}")
        if comp and addr:
            # last address segment is usually the city / region
            tail = [s.strip() for s in re.split(r",|\n", addr) if s.strip()]
            loc = tail[-1] if tail else ""
            if loc:
                out.append(f"receipt from {comp} located in {loc}")
    elif t == "purchase_order":
        if gt.get("vendor"):
            out.append(f"purchase order sent to vendor {gt['vendor']}")
        if gt.get("line_items"):
            desc = gt["line_items"][0].get("description")
            if desc:
                out.append(f"order that includes {desc}")
    elif t == "medical_report":
        if gt.get("modality"):
            out.append(f"{gt['modality']} report and its impression")
        if gt.get("diagnoses"):
            out.append(f"report describing {gt['diagnoses'][0]}")
        elif gt.get("body_site"):
            out.append(f"imaging of the {gt['body_site']} with no acute abnormality")
    elif t == "contract":
        p = _first_party(gt.get("parties"))
        if p:
            out.append(f"agreement involving {p}")
        if gt.get("governing_law"):
            out.append(f"contract governed by the laws of {gt['governing_law']}")
    return [q for q in out if q]


THEMATIC = [
    ("pneumonia found on a chest x-ray", "medical_report"),
    ("distributor agreement between two companies", "contract"),
    ("purchase order with tax applied", "purchase_order"),
    ("grocery or food receipt from a store", "invoice"),
    ("contract that can be renewed or extended", "contract"),
    ("MRI of the brain", "medical_report"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    random.seed(args.seed)

    rows = [json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r["doc_type"], []).append(r)

    queries: list[dict] = []
    for t, items in by_type.items():
        random.shuffle(items)
        for r in items[: args.per_type]:
            for q in queries_for(r):
                queries.append({"query": q, "relevant_doc_ids": [r["doc_id"]], "doc_type": t})
    for q, t in THEMATIC:
        queries.append({"query": q, "relevant_doc_ids": [], "doc_type": t})

    OUT.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in queries) + "\n", encoding="utf-8")
    print(f"wrote {len(queries)} queries -> {OUT}")
    for t in by_type:
        print(f"  {t:15s} {sum(1 for q in queries if q['doc_type'] == t)}")


if __name__ == "__main__":
    main()
