# ── Stage 1: frontend build ───────────────────────────────────────────────
FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: app (FROM base — no pip install here) ───────────────────────
FROM incois-audit-base:latest AS app

COPY src /app/src
COPY config /app/config
COPY pyproject.toml /app/pyproject.toml
COPY --from=frontend /frontend/dist /app/frontend/dist
COPY frontend/src/utils/grounding_aliases.json \
     /app/frontend/src/utils/grounding_aliases.json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health/live || exit 1

CMD ["python", "-m", "src.scripts.startup"]