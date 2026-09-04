"""Phase 4: score artifacts/pipeline_results.jsonl against ground truth.

Produces:
  artifacts/metrics.json      structured numbers
  artifacts/eval_report.md    tables
  artifacts/calibration.png   confidence vs. accuracy

Metrics: field-level extraction accuracy (overall / per type / per field),
classification accuracy, auto-accept rate (= manual-verification reduction) with
accuracy inside/outside the auto-accepted set, and confidence calibration (ECE).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rapidfuzz import fuzz

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts"
RESULTS = ART / "pipeline_results.jsonl"

_STR_THRESHOLD = 85
_LIST_F1_THRESHOLD = 0.7
_DATE_FMTS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y",
              "%d %b %Y", "%d/%m/%y", "%m/%d/%y", "%Y/%m/%d")


# --------------------------------------------------------------------------- #
def _is_empty(v) -> bool:
    return v is None or (isinstance(v, (str, list, dict)) and len(v) == 0)


def _num(v):
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        import re

        m = re.search(r"-?\d+\.?\d*", s)
        return float(m.group(0)) if m else None


def _date(v):
    if not isinstance(v, str):
        return None
    for f in _DATE_FMTS:
        try:
            return datetime.strptime(v.strip(), f).date().isoformat()
        except ValueError:
            continue
    return None


def _kind(field: str, gt, pred) -> str:
    if "date" in field.lower():
        return "date"
    sample = gt if not _is_empty(gt) else pred
    if isinstance(sample, list):
        return "list"
    if isinstance(sample, (int, float)):
        return "number"
    if field in {"total", "subtotal", "tax", "amount", "unit_price", "quantity"}:
        return "number"
    return "string"


def _str_match(a: str, b: str) -> bool:
    a, b = str(a).lower().strip(), str(b).lower().strip()
    if not a or not b:
        return False
    return max(fuzz.token_sort_ratio(a, b), fuzz.partial_ratio(a, b)) >= _STR_THRESHOLD


def _list_f1(gt: list, pred: list) -> float:
    g = [str(x).lower().strip() for x in gt if str(x).strip()]
    p = [str(x).lower().strip() for x in pred if str(x).strip()]
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    tp = sum(1 for x in g if any(fuzz.token_sort_ratio(x, y) >= _STR_THRESHOLD for y in p))
    prec = tp / len(p)
    rec = tp / len(g)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def compare(field: str, gt, pred) -> str:
    """Return one of: correct | incorrect | missing | spurious | skip."""
    ge, pe = _is_empty(gt), _is_empty(pred)
    if ge and pe:
        return "skip"
    if ge and not pe:
        return "spurious"
    if pe and not ge:
        return "missing"

    kind = _kind(field, gt, pred)
    if kind == "number":
        a, b = _num(gt), _num(pred)
        if a is None or b is None:
            return "incorrect"
        return "correct" if abs(a - b) <= max(0.01, 0.01 * max(abs(a), abs(b))) else "incorrect"
    if kind == "date":
        a, b = _date(gt) or str(gt).strip(), _date(pred) or str(pred).strip()
        return "correct" if a == b else "incorrect"
    if kind == "list":
        gl = gt if isinstance(gt, list) else [gt]
        pl = pred if isinstance(pred, list) else [pred]
        return "correct" if _list_f1(gl, pl) >= _LIST_F1_THRESHOLD else "incorrect"
    return "correct" if _str_match(gt, pred) else "incorrect"


# --------------------------------------------------------------------------- #
def _acc(counts: dict) -> float:
    denom = counts["correct"] + counts["incorrect"] + counts["missing"] + counts["spurious"]
    return counts["correct"] / denom if denom else 0.0


def ece(pairs: list[tuple[float, int]], bins: int = 10) -> tuple[float, list[dict]]:
    if not pairs:
        return 0.0, []
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for conf, ok in pairs:
        idx = min(bins - 1, max(0, int(conf * bins)))
        buckets[idx].append((conf, ok))
    total = len(pairs)
    err = 0.0
    rows = []
    for b, items in enumerate(buckets):
        if not items:
            continue
        mc = sum(c for c, _ in items) / len(items)
        ma = sum(o for _, o in items) / len(items)
        err += (len(items) / total) * abs(mc - ma)
        rows.append({"bucket": f"{b/bins:.1f}-{(b+1)/bins:.1f}", "n": len(items),
                     "mean_conf": round(mc, 3), "accuracy": round(ma, 3)})
    return err, rows


def _plot_calibration(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    if not rows:
        return
    xs = [r["mean_conf"] for r in rows]
    ys = [r["accuracy"] for r in rows]
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    ax.scatter(xs, ys, s=[max(20, n) for n in ns], alpha=0.7)
    ax.plot(xs, ys, color="#2f4b7c")
    ax.set_xlabel("mean field confidence")
    ax.set_ylabel("field accuracy")
    ax.set_title("Confidence calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    args = ap.parse_args()

    recs = [json.loads(x) for x in Path(args.results).read_text(encoding="utf-8").splitlines() if x.strip()]
    ART.mkdir(parents=True, exist_ok=True)

    overall = defaultdict(int)
    per_type: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    per_field: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    auto_counts = defaultdict(int)
    review_counts = defaultdict(int)
    calib_pairs: list[tuple[float, int]] = []
    cls_correct = cls_total = 0
    auto_docs = auto_docs_clean = 0
    n_auto = n_docs = 0
    latencies = []

    for r in recs:
        n_docs += 1
        gt_type = r["gt_doc_type"]
        cls_total += 1
        cls_correct += int(r.get("doc_type") == gt_type)
        if r.get("seconds"):
            latencies.append(r["seconds"])

        ext = r.get("extraction") or {}
        gt = r.get("ground_truth") or {}
        fconf = r.get("field_confidences") or {}
        is_auto = bool(r.get("auto_accept"))
        n_auto += int(is_auto)
        doc_has_error = False

        for field, gval in gt.items():
            verdict = compare(field, gval, ext.get(field))
            if verdict == "skip":
                continue
            overall[verdict] += 1
            per_type[gt_type][verdict] += 1
            per_field[f"{gt_type}.{field}"][verdict] += 1
            (auto_counts if is_auto else review_counts)[verdict] += 1
            if verdict != "correct":
                doc_has_error = True
            c = fconf.get(field)
            if isinstance(c, (int, float)):
                calib_pairs.append((float(c), int(verdict == "correct")))

        if is_auto:
            auto_docs += 1
            auto_docs_clean += int(not doc_has_error)

    ece_val, calib_rows = ece(calib_pairs)
    _plot_calibration(calib_rows, ART / "calibration.png")

    metrics = {
        "n_documents": n_docs,
        "field_accuracy_overall": round(_acc(overall), 4),
        "field_accuracy_by_type": {t: round(_acc(c), 4) for t, c in per_type.items()},
        "field_counts_overall": dict(overall),
        "classification_accuracy": round(cls_correct / cls_total, 4) if cls_total else 0.0,
        "auto_accept_rate": round(n_auto / n_docs, 4) if n_docs else 0.0,
        "manual_verification_reduction_pct": round(100 * n_auto / n_docs, 1) if n_docs else 0.0,
        "field_accuracy_auto_accepted": round(_acc(auto_counts), 4),
        "field_accuracy_needs_review": round(_acc(review_counts), 4),
        "auto_accepted_docs": auto_docs,
        "auto_accepted_docs_error_free": auto_docs_clean,
        "auto_accepted_doc_precision": round(auto_docs_clean / auto_docs, 4) if auto_docs else 0.0,
        "calibration_ece": round(ece_val, 4),
        "calibration_buckets": calib_rows,
        "field_accuracy_by_field": {f: round(_acc(c), 4) for f, c in sorted(per_field.items())},
        "mean_latency_s": round(sum(latencies) / len(latencies), 1) if latencies else None,
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ---- report ----
    lines = ["# Extraction Evaluation", ""]
    lines.append(f"- Documents scored: **{n_docs}**")
    lines.append(f"- **Field-level extraction accuracy: {metrics['field_accuracy_overall']:.1%}**")
    lines.append(f"- Classification accuracy: {metrics['classification_accuracy']:.1%}")
    lines.append(f"- **Manual-verification reduction (auto-accept rate): "
                 f"{metrics['manual_verification_reduction_pct']:.0f}%**")
    lines.append(f"- Accuracy within auto-accepted: {metrics['field_accuracy_auto_accepted']:.1%} "
                 f"(vs {metrics['field_accuracy_needs_review']:.1%} in the review queue)")
    lines.append(f"- Auto-accepted docs error-free: {metrics['auto_accepted_doc_precision']:.1%} "
                 f"({auto_docs_clean}/{auto_docs})")
    lines.append(f"- Confidence calibration ECE: {metrics['calibration_ece']:.3f}")
    if metrics["mean_latency_s"]:
        lines.append(f"- Mean pipeline latency: {metrics['mean_latency_s']}s/doc")
    lines += ["", "## Field accuracy by document type", "", "| type | accuracy |", "|---|---|"]
    for t, a in metrics["field_accuracy_by_type"].items():
        lines.append(f"| {t} | {a:.1%} |")
    lines += ["", "## Field counts", "",
              "| verdict | n |", "|---|---|"]
    for k in ("correct", "incorrect", "missing", "spurious"):
        lines.append(f"| {k} | {overall.get(k, 0)} |")
    lines += ["", "## Calibration buckets", "", "| conf bucket | n | mean conf | accuracy |", "|---|---|---|---|"]
    for row in calib_rows:
        lines.append(f"| {row['bucket']} | {row['n']} | {row['mean_conf']} | {row['accuracy']} |")
    lines += ["", "## Per-field accuracy", "", "| type.field | accuracy |", "|---|---|"]
    for f, a in metrics["field_accuracy_by_field"].items():
        lines.append(f"| {f} | {a:.1%} |")
    (ART / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"field accuracy   : {metrics['field_accuracy_overall']:.1%}")
    print(f"classification   : {metrics['classification_accuracy']:.1%}")
    print(f"auto-accept rate : {metrics['manual_verification_reduction_pct']:.0f}%")
    print(f"auto acc / review: {metrics['field_accuracy_auto_accepted']:.1%} / "
          f"{metrics['field_accuracy_needs_review']:.1%}")
    print(f"calibration ECE  : {metrics['calibration_ece']:.3f}")
    print(f"-> {ART/'metrics.json'} , {ART/'eval_report.md'} , {ART/'calibration.png'}")


if __name__ == "__main__":
    main()
