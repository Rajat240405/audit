# INCOIS Audit Pro — Frontend

Production-grade React 19 + TypeScript + Vite + Tailwind v4 + shadcn-style UI
workstation for the Parliamentary Audit assistant. Designed as a scientific
**audit workstation** (drafting workspace + evidence + pipeline + versions),
not a chatbot.

## Stack

- **React 19** + **TypeScript** (strict)
- **Vite** (dev server proxies `/api` → FastAPI backend)
- **Tailwind CSS v4** (`@tailwindcss/vite`)
- **shadcn/ui-style primitives** (Radix-based: tabs, select, dialog, scroll-area)
- **React Router** (`/workspace`, `/`, future pages)
- **TanStack Query** (server state: status, models, graph build)
- **Zustand** (client state: app config, draft, sessions, pipeline)
- **Framer Motion** (minimal animations)
- **SSE streaming** from the start (`/api/chat/stream`, `/api/edit`)

## Architecture

```
frontend/
├── src/
│   ├── api/          # API layer (chat, model, graph, retrieval) + SSE client
│   ├── services/     # business logic (sse, markdown, export, grounding)
│   ├── store/        # Zustand stores (app, draft, session, pipeline)
│   ├── hooks/        # useChatStream, useSessions, useEditDraft, useBackendStatus
│   ├── components/
│   │   ├── layout/   # Header, Sidebar, RightPanel, MainLayout
│   │   ├── chat/     # ConversationList, ChatInput, Message, StreamingMessage
│   │   ├── workspace/# DraftCanvas, Toolbar, VersionPanel, ExportMenu
│   │   ├── evidence/ # EvidencePanel, EvidenceCard, CitationViewer
│   │   ├── pipeline/ # PipelineView, StageProgress, Metrics
│   │   ├── graph/    # GraphPlaceholder, GraphLegend, BuildGraphModal
│   │   ├── common/   # Loading, Modal, Toast
│   │   └── ui/       # shadcn-style primitives (button, card, badge, tabs, …)
│   ├── pages/        # Workspace, Dashboard, Settings
│   ├── types/        # domain types (shared with the API contract)
│   ├── utils/        # cn, formatters, evidence helpers
│   ├── App.tsx
│   └── main.tsx
├── vite.config.ts    # /api proxy → backend; allowedHosts for previews
└── package.json
```

## Run (dev, on your machine)

1. Start the FastAPI backend:
   ```powershell
   python -m src.retrieval.frontend.server
   ```
2. In another terminal:
   ```powershell
   cd frontend
   npm install
   npm run dev
   ```
3. Open http://localhost:5173 — Vite proxies `/api` to the backend, so
   streaming works without CORS.

## Run (production)

```powershell
cd frontend
npm run build
```
The FastAPI backend automatically serves `frontend/dist` at `/` (it falls back
to the legacy `index.html` when the build is absent). Just restart the backend.

## Key UX flows implemented

- **Query → Retrieval → Evidence → Drafting → AI Editing → Verification → Export**
- Live **SSE streaming** into the drafting canvas (token-by-token), with
  pipeline stage progress (BM25 / Dense / RRF / Rerank / Generate).
- **Evidence tab**: per-doc cards with doc ID, confidence, component scores
  (dense / BM25 / RRF / rerank), expandable transcripts, click-to-highlight.
- **Pipeline tab**: live stage visualization, mode-aware (Hybrid vs Graph).
- **Drafting workspace**: markdown rendering, grounding badge, AI editing
  toolbar (quick actions + free-form instruction), version history, export
  (MD / TXT / DOCX via backend).
- **Header**: provider + model selectors, Load Model, Build Graph modal,
  Execution Profile (Fast/Deep), Draft Style, GPU status.
- **Left sidebar**: sessions (new / search / pin / delete).
- **Retrieval mode selector**: Hybrid RAG | GraphRAG — mode-aware UI.
- **Graph tab**: reserved placeholder.

## Extensibility

- New retrieval engines → add to `retrieval_mode` union + `api/retrieval.ts`.
- New LLM providers → extend `api/model.ts` + backend `/api/provider`.
- Graph visualization → replace `GraphPlaceholder` with an interactive canvas.
- Auth / multi-user / HPC → add routes + API modules without restructuring.
