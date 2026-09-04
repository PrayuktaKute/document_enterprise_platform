#!/usr/bin/env bash
# Colab bootstrap: deps + Ollama (GPU) + Qdrant server. Run from the repo root.
set -euo pipefail

echo "== python deps =="
pip -q install -r requirements-ml.txt
pip -q install -e . --no-deps

echo "== ollama =="
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > /content/ollama.log 2>&1 &
sleep 5
ollama pull qwen2.5:3b-instruct-q4_K_M

echo "== qdrant server =="
QDRANT_VER=v1.15.4
if [ ! -x /content/qdrant/qdrant ]; then
  mkdir -p /content/qdrant
  curl -fsSL -o /content/qdrant.tar.gz \
    "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VER}/qdrant-x86_64-unknown-linux-gnu.tar.gz"
  tar -xzf /content/qdrant.tar.gz -C /content/qdrant
fi
mkdir -p /content/qdrant_storage
( cd /content/qdrant_storage && nohup /content/qdrant/qdrant > /content/qdrant.log 2>&1 & )
sleep 5

echo "== .env =="
cat > .env <<'EOF'
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:3b-instruct-q4_K_M
EMBED_MODEL=BAAI/bge-m3
DATABASE_URL=sqlite://
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents
CONFIDENCE_METHOD=logprob_min
DATA_DIR=./data
ARTIFACTS_DIR=./artifacts
CACHE_DIR=./data/cache
EOF

curl -s http://localhost:11434/api/version && echo " ollama ok"
curl -s http://localhost:6333/ | head -c 120 && echo " qdrant ok"
echo "bootstrap done"
