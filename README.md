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
