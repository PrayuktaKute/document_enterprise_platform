#!/usr/bin/env bash
# Colab bootstrap: python deps + Ollama binary + .env (embedded Qdrant).
# NOTE: Ollama is *started* from a Python cell in the notebook (Popen) so it
# survives for the kernel session -- a backgrounded process here would be reaped
# when this cell's shell exits.
set -euo pipefail

echo "== python deps =="
pip -q install -r requirements-ml.txt
pip -q install -e . --no-deps

echo "== ollama binary =="
if ! which ollama >/dev/null 2>&1; then
  # the Ollama install tarball is zstd-compressed; Colab's base image lacks zstd
  apt-get install -y -qq zstd 2>/dev/null || sudo apt-get install -y zstd || true
  if which zstd >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
  else
    # fallback: pull the raw linux-amd64 binary directly
    curl -fsSL -o /usr/local/bin/ollama https://ollama.com/download/ollama-linux-amd64
    chmod +x /usr/local/bin/ollama
  fi
fi
ollama --version || true

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

python - <<'PY'
from qdrant_client import QdrantClient
QdrantClient(path="/content/qdrant_local"); print("embedded qdrant ok")
PY
echo "bootstrap done (start Ollama in the next cell)"
