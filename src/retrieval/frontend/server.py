"""
FastAPI Backend Server for Phase 8 Interactive Chat Frontend.
Serves static client assets and exposes REST API endpoints for Hybrid RAG and GraphRAG.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from src.retrieval.hybrid.pipeline import HybridRAGPipeline
from src.retrieval.graph.store import GraphStore
from src.retrieval.graph.retriever import GraphRetriever
from src.generation.client import LLMClient
from src.generation.generator import AnswerGenerator

app = FastAPI(
    title="Parliamentary & Audit Assistant API",
    description="REST API backing the Interactive Chat Frontend for Phase 8."
)

# ─────────────────────────────────────────────────────────────────────────────
# State Initialization
# ─────────────────────────────────────────────────────────────────────────────

# Load pipelines (document / chunk / graph)
index_dir = Path("storage/hybrid_rag")
graph_dir = Path("storage/graphrag")

# Instantiate and cache pipelines
pipeline = HybridRAGPipeline()
if index_dir.exists():
    pipeline.load(index_dir)

graph_store = GraphStore(storage_dir=str(graph_dir))
if graph_store.graph_file.exists():
    graph_store.load()
graph_retriever = GraphRetriever(store=graph_store)

# Instantiate generation components
llm_client = LLMClient()
generator = AnswerGenerator(llm_client=llm_client)

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    mode: str = "hybrid"  # "hybrid" (Baseline chunked) or "graph" (GraphRAG traversal)


class SourceItem(BaseModel):
    doc_id: str
    ministry: Optional[str] = None
    subject: Optional[str] = None
    score: float
    question: str
    answer: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    retrieval_latency_ms: float
    generation_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    is_fallback: bool = False
    is_graph_result: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_informative_summary(answer_text: str, subject_text: str) -> str:
    """
    Cleans up official parliamentary answers by removing boilerplate headings,
    minister signatures, salutations, and returns the first 1-2 informative sentences.
    Falls back to the document's subject if no informative text remains.
    """
    import re
    # 1. Standardize whitespace and remove newlines for clean regex matching
    text = re.sub(r'\s+', ' ', answer_text).strip()
    
    # 2. Remove ANSWER/REPLY heading
    text = re.sub(r'(?i)^ANSWER\s*[:\-]?\s*', '', text).strip()
    text = re.sub(r'(?i)^REPLY\s*[:\-]?\s*', '', text).strip()
    
    # 3. Remove MINISTER signatures and departments (e.g. MINISTER OF STATE (SHRI...) IN THE MINISTRY OF...)
    text = re.sub(r'(?i)^MINISTER\s+OF\s+STATE\s*\([^)]*\)\s*(?:IN\s+THE\s+MINISTRY\s+OF\s+[\w\s&,]+)?\s*', '', text).strip()
    text = re.sub(r'(?i)^(?:THE\s+HON’BLE\s+)?MINISTER\s+OF\s+[\w\s&,]+\s*\([^)]*\)\s*', '', text).strip()
    text = re.sub(r'(?i)^(?:THE\s+HON’BLE\s+)?MINISTER\s+OF\s+[\w\s&,]+\s*', '', text).strip()
    
    # 4. Remove leading list indices like (a) to (c) or (a) & (b) or (a)
    text = re.sub(r'(?i)^\([a-g]\)\s*(?:to|&|and)?\s*(?:\([a-g]\))?\s*[:\-\.]?\s*', '', text).strip()
    
    # 5. Remove any residual leading punctuation/dashes
    text = re.sub(r'^[\s\-\:\.\*]+', '', text).strip()

    # Split into sentences
    sentences = re.split(r'\.\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    informative_sentences = []
    for s in sentences:
        s_lower = s.lower()
        if len(s) < 15:
            continue
        # Skip purely meta-text lines
        if any(term in s_lower for term in ['referred to', 'is enclosed', 'laid on the table']):
            continue
        informative_sentences.append(s)

    if informative_sentences:
        summary = '. '.join(informative_sentences[:2])
        if not summary.endswith('.'):
            summary += '.'
        if len(summary) > 250:
            summary = summary[:247] + '...'
        return summary

    return subject_text if subject_text else 'No detailed summary available.'


def is_semantic_synthesis_query(query: str) -> bool:
    """
    Detects if the query is requesting summarization, explanation, comparison,
    reasoning, synthesis, or drafting over the documents (requiring LLM generation).
    Uses strict whole-word set intersection to prevent substring collisions (e.g. 'show' matching 'how').
    """
    import re
    query_clean = re.sub(r"[^\w\s]", " ", query.lower()).strip()
    words = set(query_clean.split())
    
    synthesis_triggers = {
        "summarize", "summarise", "explain", "compare", "reason", "why", "synthesize",
        "synthesise", "how", "relation", "contrast", "summary", "analyze", "analyse",
        "draft"
    }
    return bool(words.intersection(synthesis_triggers))


def is_metadata_query(query: str) -> bool:
    """
    Detects if the query is requesting document listings, ministry lookups,
    subject lookups, session/date filtering, graph stats, or other metadata lookups.
    """
    import re
    query_clean = re.sub(r"[^\w\s]", " ", query.lower()).strip()
    words = set(query_clean.split())
    
    metadata_triggers = {
        "list", "find", "lookup", "show", "get", "statistics", "stats", "mps", 
        "members", "session", "date", "document", "question", "questions"
    }
    
    # Also check multi-word phrase triggers
    has_phrase = any(phrase in query_clean for phrase in ["who asked", "which ministry", "what subjects", "where is"])
    
    has_metadata_trigger = bool(words.intersection(metadata_triggers)) or has_phrase
    has_synthesis_trigger = is_semantic_synthesis_query(query)
    
    return has_metadata_trigger and not has_synthesis_trigger


# ─────────────────────────────────────────────────────────────────────────────
# REST Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the single-page HTML chat client."""
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html template not found")
    with open(html_path, encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat route routing requests to Hybrid RAG or GraphRAG traversals,
    evaluating, and executing grounded generation.
    """
    query = request.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query message cannot be empty")

    mode = request.mode.lower()
    sources: List[SourceItem] = []
    
    t_ret_start = time.perf_counter()

    # ── Path Selection Matrix ──
    # Check if the query is a deterministic metadata lookup, bypassing the LLM entirely (Point 1)
    is_graph_result = is_metadata_query(query) or (mode == "graph" and not is_semantic_synthesis_query(query))

    # Temporary Logging (Point 4)
    print("METADATA:", is_graph_result)

    if is_graph_result:
        # ── PATH A: DETERMINISTIC METADATA QUERY PATH ──
        # Bypasses LLM, AnswerGenerator, and Ollama health checks entirely.
        print("PATH: GRAPH")
        print("RETURNING GRAPH RESPONSE")
        
        results = graph_retriever.retrieve(query, top_k=5)
        ret_latency = (time.perf_counter() - t_ret_start) * 1000
        
        # Format Retrieved Results into Source Items
        for r in results:
            sources.append(SourceItem(
                doc_id=r.doc_id,
                ministry=r.metadata.get("ministry") or "-",
                subject=r.metadata.get("subject") or "-",
                score=float(r.score),
                question=r.question,
                answer=r.answer
            ))

        if results:
            # Build structured metadata cards with short summaries
            header = "### 🕸️ GraphRAG: Document Explorer\n"
            header += "The following real parliamentary document cards were resolved directly from the metadata graph relationships:\n\n"
            
            cards = []
            for r in results:
                # Extract clean, informative summary directly from official PDF text
                summary = extract_informative_summary(r.answer, r.metadata.get("subject", ""))
                
                card = (
                    f"#### 📄 **Document: {r.doc_id}**\n"
                    f"* 🏢 **Ministry**: {r.metadata.get('ministry') or '-'}\n"
                    f"* 🏷️ **Subject**: {r.metadata.get('subject') or '-'}\n"
                    f"* 📅 **Date**: {r.metadata.get('date') or '-'}\n"
                    f"* ❓ **Question Type**: {r.metadata.get('question_type') or 'Unstarred'}\n"
                    f"* 💡 **Summary**: {summary}\n"
                )
                cards.append(card)
            
            answer = header + "\n---\n".join(cards)
        else:
            answer = (
                "### 🕸️ GraphRAG: Document Explorer\n\n"
                "No matching parliamentary metadata relationships were resolved for your query in the graph."
            )
            
        prompt_tok = 0
        comp_tok = len(answer) // 4
        total_tok = comp_tok
        is_fallback = False

        # Return response immediately (Point 2 & 3)
        return ChatResponse(
            answer=answer,
            sources=sources,
            retrieval_latency_ms=ret_latency,
            generation_latency_ms=0.0,
            prompt_tokens=0,
            completion_tokens=comp_tok,
            total_tokens=total_tok,
            is_fallback=is_fallback,
            is_graph_result=True
        )

    else:
        # ── PATH B: SEMANTIC / SYNTHESIS QUERY PATH ──
        # Performs standard retrieval and invokes LLM/health checks
        print("PATH: LLM")
        
        if mode == "graph":
            results = graph_retriever.retrieve(query, top_k=5)
            ret_latency = (time.perf_counter() - t_ret_start) * 1000
        else:
            results, timings = pipeline.retrieve(query, top_k=5)
            ret_latency = (time.perf_counter() - t_ret_start) * 1000

        # Format Retrieved Results into Source Items
        for r in results:
            sources.append(SourceItem(
                doc_id=r.doc_id,
                ministry=r.metadata.get("ministry") or "-",
                subject=r.metadata.get("subject") or "-",
                score=float(r.score),
                question=r.question,
                answer=r.answer
            ))

        t_gen_start = time.perf_counter()
        
        # Check LLM Availability
        llm_available = llm_client.check_health()

        if llm_available and results:
            gen_res = generator.generate(query, results)
            gen_latency = gen_res.generation_latency_ms
            answer = gen_res.answer
            prompt_tok = gen_res.prompt_tokens
            comp_tok = gen_res.completion_tokens
            total_tok = gen_res.total_tokens
            is_fallback = False
        else:
            # LLM is Offline - Report unavailability explicitly for semantic queries
            is_fallback = True
            gen_latency = (time.perf_counter() - t_gen_start) * 1000
            answer = (
                "**[System Notice: Local LLM Generation Offline]**\n\n"
                "The local LLM service (Ollama) is currently offline or unreachable. "
                "Because this query requires synthesis, explanation, or comparison, a complete "
                "response cannot be compiled. Please start Ollama using `ollama serve` to enable "
                "local LLM generation."
            )
            prompt_tok = 0
            comp_tok = len(answer) // 4
            total_tok = comp_tok

    return ChatResponse(
        answer=answer,
        sources=sources,
        retrieval_latency_ms=ret_latency,
        generation_latency_ms=gen_latency,
        prompt_tokens=prompt_tok,
        completion_tokens=comp_tok,
        total_tokens=total_tok,
        is_fallback=is_fallback,
        is_graph_result=is_graph_result
    )


def start_server(port: int = 4000) -> None:
    """Run the FastAPI application on host 0.0.0.0."""
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    start_server()
