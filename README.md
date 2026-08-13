```
audit2
├─ benchmarks
│  └─ default.json
├─ config
│  └─ ingestion.yaml
├─ data
├─ frontend
│  ├─ index.html
│  ├─ package-lock.json
│  ├─ package.json
│  ├─ src
│  │  ├─ api
│  │  │  ├─ chat.ts
│  │  │  ├─ client.ts
│  │  │  ├─ graph.ts
│  │  │  ├─ model.ts
│  │  │  └─ retrieval.ts
│  │  ├─ App.tsx
│  │  ├─ components
│  │  │  ├─ activity
│  │  │  │  └─ ModelActivityPanel.tsx
│  │  │  ├─ chat
│  │  │  │  ├─ ChatInput.tsx
│  │  │  │  ├─ ConversationList.tsx
│  │  │  │  ├─ Message.tsx
│  │  │  │  └─ StreamingMessage.tsx
│  │  │  ├─ common
│  │  │  │  ├─ Loading.tsx
│  │  │  │  ├─ Modal.tsx
│  │  │  │  └─ Toast.tsx
│  │  │  ├─ evidence
│  │  │  │  ├─ CitationViewer.tsx
│  │  │  │  ├─ DocViewerModal.tsx
│  │  │  │  ├─ EvidenceCard.tsx
│  │  │  │  └─ EvidencePanel.tsx
│  │  │  ├─ graph
│  │  │  │  ├─ BuildGraphModal.tsx
│  │  │  │  ├─ GraphLegend.tsx
│  │  │  │  └─ GraphPlaceholder.tsx
│  │  │  ├─ layout
│  │  │  │  ├─ Header.tsx
│  │  │  │  ├─ MainLayout.tsx
│  │  │  │  ├─ Sidebar.tsx
│  │  │  │  └─ SourceFilter.tsx
│  │  │  ├─ pipeline
│  │  │  │  ├─ Metrics.tsx
│  │  │  │  └─ PipelineView.tsx
│  │  │  ├─ ui
│  │  │  │  ├─ badge.tsx
│  │  │  │  ├─ button.tsx
│  │  │  │  ├─ card.tsx
│  │  │  │  ├─ dialog.tsx
│  │  │  │  ├─ input.tsx
│  │  │  │  ├─ scroll-area.tsx
│  │  │  │  ├─ select.tsx
│  │  │  │  └─ tabs.tsx
│  │  │  └─ workspace
│  │  │     ├─ DraftCanvas.tsx
│  │  │     ├─ ExportMenu.tsx
│  │  │     ├─ HistoryPanel.tsx
│  │  │     ├─ NotesEditor.tsx
│  │  │     └─ Toolbar.tsx
│  │  ├─ hooks
│  │  │  ├─ useBackendStatus.ts
│  │  │  ├─ useChatStream.ts
│  │  │  ├─ useEditDraft.ts
│  │  │  └─ useSessions.ts
│  │  ├─ index.css
│  │  ├─ lib
│  │  │  └─ sourceFilter.ts
│  │  ├─ main.tsx
│  │  ├─ pages
│  │  │  ├─ Dashboard.tsx
│  │  │  ├─ Settings.tsx
│  │  │  └─ Workspace.tsx
│  │  ├─ services
│  │  │  ├─ export.ts
│  │  │  ├─ grounding.ts
│  │  │  ├─ markdown.ts
│  │  │  └─ sse.ts
│  │  ├─ store
│  │  │  ├─ useActivityStore.ts
│  │  │  ├─ useAppStore.ts
│  │  │  ├─ useChatActionsStore.ts
│  │  │  ├─ useDocViewerStore.ts
│  │  │  ├─ useDraftStore.ts
│  │  │  ├─ useEditStore.ts
│  │  │  ├─ useNotesStore.ts
│  │  │  ├─ usePipelineStore.ts
│  │  │  ├─ useSessionStore.ts
│  │  │  ├─ useThemeStore.ts
│  │  │  └─ useToastStore.ts
│  │  ├─ types
│  │  │  ├─ api.ts
│  │  │  └─ index.ts
│  │  ├─ utils
│  │  │  ├─ cn.ts
│  │  │  ├─ evidence.ts
│  │  │  ├─ formatters.ts
│  │  │  └─ grounding_aliases.json
│  │  └─ vite-env.d.ts
│  ├─ tsconfig.app.json
│  ├─ tsconfig.json
│  ├─ tsconfig.node.json
│  └─ vite.config.ts
├─ pyproject.toml
├─ requirements.txt
├─ RUNNING.md
├─ script.ipynb
├─ src
│  ├─ data
│  │  ├─ enricher.py
│  │  ├─ ingestion_pipeline.py
│  │  ├─ loader.py
│  │  ├─ scraper.py
│  │  ├─ validator.py
│  │  └─ __init__.py
│  ├─ generation
│  │  ├─ client.py
│  │  ├─ defaults.py
│  │  ├─ generator.py
│  │  ├─ registry.py
│  │  └─ __init__.py
│  ├─ graphrag
│  │  ├─ checkpoint.py
│  │  ├─ cli.py
│  │  ├─ config.py
│  │  ├─ display.py
│  │  ├─ embeddings.py
│  │  ├─ extractor.py
│  │  ├─ llm.py
│  │  ├─ models.py
│  │  ├─ neo4j_client.py
│  │  ├─ pipeline.py
│  │  ├─ query.py
│  │  ├─ verify.py
│  │  └─ __init__.py
│  ├─ models
│  │  ├─ qa_record.py
│  │  ├─ statistics.py
│  │  └─ __init__.py
│  ├─ retrieval
│  │  ├─ cli.py
│  │  ├─ evaluation
│  │  │  ├─ cli.py
│  │  │  ├─ comparison.py
│  │  │  ├─ metrics.py
│  │  │  ├─ reporter.py
│  │  │  └─ runner.py
│  │  ├─ frontend
│  │  │  ├─ index.html
│  │  │  ├─ org_tree.py
│  │  │  └─ server.py
│  │  ├─ graph
│  │  │  ├─ cli.py
│  │  │  ├─ retriever.py
│  │  │  ├─ store.py
│  │  │  └─ __init__.py
│  │  ├─ hybrid
│  │  │  ├─ bm25_index.py
│  │  │  ├─ embedder.py
│  │  │  ├─ fusion.py
│  │  │  ├─ pipeline.py
│  │  │  ├─ query_expansion.py
│  │  │  ├─ reranker.py
│  │  │  ├─ vector_store.py
│  │  │  └─ __init__.py
│  │  ├─ result.py
│  │  └─ __init__.py
│  ├─ scripts
│  │  ├─ convert_sirs_knowledge.py
│  │  ├─ coverage_report.py
│  │  ├─ crawl_annual_reports.py
│  │  ├─ crawl_incois_reports.py
│  │  ├─ crawl_moes_reports.py
│  │  ├─ detect_doc_type.py
│  │  ├─ diagnose_sources.py
│  │  ├─ extract_structured_content.py
│  │  ├─ finetune_merge.py
│  │  ├─ finetune_prepare_data.py
│  │  ├─ finetune_train.py
│  │  ├─ gpu_check.py
│  │  ├─ ingest_all.py
│  │  ├─ ingest_folder.py
│  │  ├─ ingest_inbox.py
│  │  ├─ ocr_pdfs.py
│  │  ├─ purge_corpus.py
│  │  ├─ repair_corpus.py
│  │  └─ sync_sources.py
│  ├─ utils
│  │  └─ project_scope.py
│  └─ __init__.py
└─ tests
   ├─ test_data_ingestion.py
   ├─ test_evaluation.py
   ├─ test_graphrag.py
   └─ test_hybrid_rag.py
```
