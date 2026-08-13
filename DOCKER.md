
Application Docker image (inference stays on the host)
The image contains React + FastAPI + Hybrid RAG + FAISS/BM25 + embedding/reranker
libraries. It does not contain Ollama, vLLM, CUDA, or Qwen weights.

Host	App talks to
Windows	http://host.docker.internal:11434/v1 → Ollama (qwen3:8b)
HPC (later)	VLLM_BASE_URL → external vLLM (Qwen3.6-27B on A40)
Single Uvicorn worker. Same-origin frontend (/api). Bind-mount data/index/models.

1. Build
   From the repo root (audit/):

bat

docker compose build
or

bat

docker build -t incois-audit-app:local .
2. Run
Put the Hybrid RAG index under storage/hybrid_rag and BGE-M3 under models/bge-m3
(or let the first query download into the /models mount). Start Ollama on Windows
and ollama pull qwen3:8b before starting the container.

bat

docker compose up -d
3. Stop
bat

docker compose down
4. Logs
bat

docker compose logs -f app
5. Health
bat

curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
ready needs the mounted index and host Ollama reachable at the configured
VLLM_BASE_URL.

6. Host Ollama from the container
   bat

docker compose exec app python -c "import httpx; r=httpx.get('http://host.docker.internal:11434/api/tags', timeout=5); print(r.status_code, r.text[:200])"
docker compose exec app python -c "import httpx; r=httpx.get('http://host.docker.internal:11434/v1/models', timeout=5); print(r.status_code, r.text[:300])"
7. Model dropdown (vLLM + host Ollama)
Compose sets MODEL_CATALOG=/app/config/models.docker.yaml. That file is a
Docker-only copy of the production catalogue plus one vLLM family:

Qwen 3 8B (host Ollama) → qwen3:8b (think_mode: none).

HPC Qwen3.6-27B / 30B-A3B stay in the same file. Production/HPC without
MODEL_CATALOG still loads
models.yaml
 (unchanged).

Pick Qwen 3 8B (host Ollama) in the vLLM dropdown for the local smoke test.

8. Frontend
   Open http://localhost:8000/ — FastAPI serves frontend/dist. Keep the UI
   provider on the OpenAI-compatible / vLLM setting so traffic uses
   VLLM_BASE_URL (host Ollama /v1). Native “Ollama” in the header still
   targets localhost:11434 inside the container (existing app behavior; no
   code change).

Notes
Do not pass --workers > 1.
Do not copy corpus/index into the image; use the three bind mounts.
Prompt debug (if any) is written under APP_DATA_DIR, never CWD.
This document is not an HPC or multi-user certification.
