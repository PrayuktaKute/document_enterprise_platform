#!/usr/bin/env bash
# Full eval run. Assumes data/ is populated (fetch_data.py + gen_synthetic.py + prepare_eval.py)
# and Ollama + Qdrant are up. Produces artifacts/.
set -euo pipefail

WORKERS="${WORKERS:-4}"

echo "== 0/5 preflight =="
curl -sf http://localhost:11434/api/version >/dev/null || {
  echo "ERROR: Ollama is not reachable on :11434. Run the 'Start Ollama' notebook cell first."; exit 1; }
ollama list | grep -q "qwen2.5:3b" || {
  echo "ERROR: model qwen2.5:3b-instruct-q4_K_M not pulled."; exit 1; }
echo "ollama + model ok"

echo "== 1/5 pipeline batch (120 docs, workers=$WORKERS) =="
python scripts/run_pipeline_batch.py --workers "$WORKERS" --out artifacts/pipeline_results.jsonl

echo "== 2/5 build vector index =="
python scripts/build_index.py --recreate

echo "== 3/5 build retrieval queries =="
python scripts/make_retrieval_queries.py --per-type 20

echo "== 4/5 extraction eval =="
python scripts/eval_extraction.py

echo "== 5/5 retrieval eval =="
python scripts/eval_retrieval.py --k 5

echo "== package artifacts =="
cp -r data/cache artifacts/parsed_cache 2>/dev/null || true
cp data/eval/retrieval_queries.jsonl artifacts/ 2>/dev/null || true
zip -qr artifacts_bundle.zip artifacts
echo "-> artifacts_bundle.zip  ($(du -h artifacts_bundle.zip | cut -f1))"
