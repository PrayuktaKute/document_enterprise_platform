#!/usr/bin/env bash
# Full eval run. Assumes data/ is populated (fetch_data.py + gen_synthetic.py + prepare_eval.py)
# and Ollama + Qdrant are up. Produces artifacts/.
set -euo pipefail

WORKERS="${WORKERS:-4}"

echo "== 1/5 pipeline batch (120 docs, workers=$WORKERS) =="
python scripts/run_pipeline_batch.py --workers "$WORKERS" --out artifacts/pipeline_results.jsonl

echo "== 2/5 build vector index + snapshot =="
python scripts/build_index.py --recreate --snapshot

echo "== 3/5 build retrieval queries =="
python scripts/make_retrieval_queries.py --per-type 20

echo "== 4/5 extraction eval =="
python scripts/eval_extraction.py

echo "== 5/5 retrieval eval =="
python scripts/eval_retrieval.py --k 5

echo "== package artifacts =="
SNAP_DIR=$(python -c "import glob,os;print(next(iter(sorted(glob.glob('/content/qdrant_storage/snapshots/documents/*'))),''))" 2>/dev/null || true)
mkdir -p artifacts/qdrant_snapshot
[ -n "${SNAP_DIR:-}" ] && cp "$SNAP_DIR" artifacts/qdrant_snapshot/ || echo "(no snapshot file found to copy)"
cp -r data/cache artifacts/parsed_cache 2>/dev/null || true
zip -qr artifacts_bundle.zip artifacts
echo "-> artifacts_bundle.zip  ($(du -h artifacts_bundle.zip | cut -f1))"
