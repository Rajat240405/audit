# Parliamentary RAG Project — Setup and Execution Guide

This guide explains how to set up, install, and run the Parliamentary RAG system on your local machine, covering both **Phase 1 (Data Ingestion)** and **Phase 2 (Hybrid RAG Retrieval and Generation)**.

---

## Prerequisites

Before setting up the project, make sure you have:
1. **Python 3.11 or 3.12** installed.
2. **Pip** (Python package installer) upgraded.
3. *(Optional)* **Ollama** installed locally (required to generate grounded answers using local LLMs).

---

## 1. Environment Setup

It is highly recommended to use a Python virtual environment to avoid package conflicts.

### Step 1: Create a Virtual Environment
Navigate to the root directory of your cloned repository and run:
```bash
python -m venv .venv
```

### Step 2: Activate the Virtual Environment
* **On macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```
* **On Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **On Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

---

## 2. Dependency Installation

You can install dependencies in editable package mode or using the flat `requirements.txt` file.

### Option A: Editable Package Mode (Recommended)
This registers the command-line commands `ingest` and `retrieve` globally inside your virtual environment.

To install the CPU-only version of PyTorch first (strongly recommended to save download size (~150MB instead of 500MB+) and disk space):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the package and its requirements:
```bash
pip install -e .
```
*(If you wish to install formatting and testing tools like Pytest and Ruff, run `pip install -e ".[dev]"`).*

### Option B: Flat Requirements Installation
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 3. Phase 1 — Running Data Ingestion

The ingestion pipeline scrapes parliamentary Q&A data, validates the schema, deduplicates records, enriches them with extracted metadata (ministry, subject, date), and writes them to a JSONL knowledge base.

You can run these commands using the globally registered `ingest` script, or via python module mode `python -m src.data.ingestion_pipeline`.

### Run Ingestion with Mock Data Strategy (Fast & Failsafe)
Generates 3,500 highly realistic parliamentary Q&A records offline:
```bash
ingest ingest --count 3500 --strategy mock
```

### Run Ingestion with Active Web Scraping
Attempts to scrape actual live Q&A records from the official website:
```bash
ingest ingest --count 100 --strategy httpx
```

### Show Latest Ingestion Statistics
```bash
ingest stats
```

### Validate Ingested Datasets
```bash
ingest validate
```

*All ingested data files are written to: `data/raw/`, `data/processed/`, and `data/enriched/`.*

---

## 4. Phase 2 — Running the Hybrid RAG CLI

The Hybrid RAG pipeline uses dense vector search (FAISS) and lexical search (BM25) combined through Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking to find and ground knowledge retrieval.

You can run these commands using the globally registered `retrieve` script, or via python module mode `python -m src.retrieval.cli`.

### Step 1: Build Search Indices
Builds the FAISS vector store and the BM25 lexical index using the latest enriched dataset from Phase 1:
```bash
retrieve build
```
*(To force a rebuild, add the `--rebuild` flag: `retrieve build --rebuild`)*

### Step 2: Query the System
To query the system and retrieve the most relevant historical Q&A records:
```bash
retrieve query "What measures has the government taken to address malaria in rural areas?"
```

#### Optional Query Customizations:
* **Show Stage Timings**: `--show-trace` (prints precise latency breakdown)
* **Adjust returned results count**: `--top-k 3` (retrieves top-3 instead of top-5)
* **Skip cross-encoder reranking**: `--no-rerank` (retrieves raw RRF results)
* **Skip generation step**: `--no-generate` (useful for evaluating search alone)

---

## 5. Integrating Local LLM (Ollama)

To enable Phase 2's grounded generation step (grounding answers inside retrieved context without hallucinations), integrate the local Ollama LLM provider.

### Step 1: Install Ollama
Download and install Ollama from [ollama.com](https://ollama.com).

### Step 2: Start the Ollama Service
Ensure the Ollama application is active and running:
```bash
ollama serve
```

### Step 3: Pull the Target Model
Pull the project's default model (`Qwen 2.5: 7B`):
```bash
ollama pull qwen2.5:7b
```
*(You can also pull lighter models like `llama3.2:3b` if running on a memory-constrained machine, then run queries with `retrieve query "..." --llm-model llama3.2:3b`)*

### Step 4: Run Grounded Q&A
Run the query command with the LLM step active:
```bash
retrieve query "What schemes are active for digital education in schools?"
```
The model will output a structured, concise response with quote attributions from the retrieved parliamentary context!

---

## 6. Interactive Mode & Benchmarking

### Launch Interactive Shell
Enter an interactive console where you can query repeatedly without reloading models:
```bash
retrieve interactive
```

### Run Latency Benchmark
Evaluate retrieval latencies across a standard benchmark set of 10 complex queries:
```bash
retrieve benchmark
```

---

## 7. Running Unit Tests
To verify everything is working exactly as designed:
```bash
python -m pytest
```
All 85 end-to-end integration and units tests should pass.
