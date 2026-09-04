#!/usr/bin/env bash
# Colab bootstrap: deps + Ollama (GPU) + Qdrant server. Run from the repo root.
set -euo pipefail

echo "== python deps =="
pip -q install -r requirements-ml.txt
pip -q install -e . --no-deps

echo "== ollama =="
curl -fsSL https://ollama.com/install.sh | sh
export OLLAMA_NUM_PARALLEL=4          # serve concurrent requests (GPU has headroom)
export OLLAMA_KEEP_ALIVE=30m
nohup env OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=30m ollama serve > /content/ollama.log 2>&1 &
sleep 5
ollama pull qwen2.5:3b-instruct-q4_K_M

echo "== .env  (embedded Qdrant -- no server needed on Colab) =="
cat > .env <<'EOF'
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:3b-instruct-q4_K_M
EMBED_MODEL=BAAI/bge-m3
DATABASE_URL=postgresql+psycopg://unused
QDRANT_URL=
QDRANT_PATH=/content/qdrant_local
QDRANT_COLLECTION=documents
CONFIDENCE_METHOD=logprob_min
DATA_DIR=./data
ARTIFACTS_DIR=./artifacts
CACHE_DIR=./data/cache
EOF

curl -s http://localhost:11434/api/version && echo " ollama ok"
python - <<'PY'
from qdrant_client import QdrantClient
QdrantClient(path="/content/qdrant_local"); print("embedded qdrant ok")
PY
echo "bootstrap done"
