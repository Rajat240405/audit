# Parliamentary RAG Project

An enterprise-grade proof-of-concept knowledge retrieval system using the public Lok Sabha Questions & Answers dataset.

## Features

- **Phase 1: Ingestion Pipeline**: Scrapes, validates, deduplicates, and enriches parliamentary Q&A data.
- **Phase 2: Hybrid RAG**: Implements dense vector search (FAISS) + lexical search (BM25) with Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking.
- **Phase 3: GraphRAG**: Under development.

## Setup & Installation

Install the package and dependencies:

```bash
pip install -e .
```

Or install via `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running the CLIs

### Phase 1: Data Ingestion

```bash
ingest ingest --count 3500 --strategy mock
```

### Phase 2: Hybrid RAG Retrieval

Build indices:

```bash
retrieve build
```

Query the system:

```bash
retrieve query "What measures address malaria in rural areas?"
```

```
audit2
├─ config
│  └─ ingestion.yaml
├─ data
├─ diagnose_pipeline.py
├─ generation_prompt_debug.txt
├─ pyproject.toml
├─ README.md
├─ requirements.txt
├─ RUNNING.md
├─ smoke_test.md
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
│  │  ├─ generator.py
│  │  ├─ registry.py
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
│  │  │  ├─ reranker.py
│  │  │  ├─ vector_store.py
│  │  │  └─ __init__.py
│  │  ├─ result.py
│  │  └─ __init__.py
│  └─ __init__.py
├─ test.pdf
└─ tests
   ├─ test_data_ingestion.py
   ├─ test_evaluation.py
   ├─ test_graphrag.py
   └─ test_hybrid_rag.py

```
