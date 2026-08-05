"""
FastAPI Backend Server for Phase 10 Multi-Provider LLM Platform.
Serves static client assets, manages runtime provider/model/mode switches,
and exposes REST API endpoints for Hybrid RAG, GraphRAG, and Graph Explorer lookups.
Features the Provider and Model Registries for intelligent, zero-override model matching.
Refactored to treat Fast/Deep as execution profiles rather than separate model variants.
Supports true runtime dynamic model discovery for Ollama.
Symmetric quality-optimized parameters applied for both standard and thinking models.
Ensures perfect, secure propagation of in-memory API keys across runtime boundaries.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.retrieval.hybrid.pipeline import HybridRAGPipeline
from src.retrieval.graph.store import GraphStore
from src.retrieval.graph.retriever import GraphRetriever
from src.generation.client import LLMClient
from src.generation.generator import AnswerGenerator
from src.generation.registry import model_registry, provider_registry

app = FastAPI(
    title="Parliamentary & Audit Assistant Multi-Provider API",
    description="Provider-agnostic API backing the Interactive Chat Frontend for Phase 10."
)

# ─────────────────────────────────────────────────────────────────────────────
# In-Memory Active Configuration State (Thread-safe single-session storage)
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_CONFIG = {
    "provider": "ollama",
    "model_family": "qwen2.5", # Default family (Objective 3)
    "model": "qwen2.5:7b",     # Resolved model
    "mode": "fast",            # "fast" or "deep"
    "retrieval_mode": "hybrid",# "hybrid" or "graph"
    "groq_api_key": None
}

# Load parent pipelines
index_dir = Path("storage/hybrid_rag")
graph_dir = Path("storage/graphrag")

pipeline = HybridRAGPipeline()
if index_dir.exists():
    pipeline.load(index_dir)
    # Phase 11 required runtime logging
    chunks_count = len(pipeline._chunk_map) if pipeline.use_chunking else len(pipeline._doc_map)
    print("=" * 60)
    print(f"Embedding Model : {pipeline.embedder.model_name}")
    print(f"Embedding Dimension : {pipeline.embedder.embedding_dim}")
    print(f"Documents Indexed : {len(pipeline._doc_map)}")
    print(f"Chunks Indexed : {chunks_count}")
    print("FAISS Index Built Successfully")
    print("=" * 60)

graph_store = GraphStore(storage_dir=str(graph_dir))
if graph_store.graph_file.exists():
    graph_store.load()
graph_retriever = GraphRetriever(store=graph_store)

# Resolve default starting configuration dynamically from registry
_default_fam = model_registry.get(ACTIVE_CONFIG["model_family"])
if _default_fam:
    ACTIVE_CONFIG["model"] = _default_fam.model_name
    _num_ctx = _default_fam.context_window
else:
    _num_ctx = 8192

# Instantiate the active client and generator router
llm_client = LLMClient(
    provider=ACTIVE_CONFIG["provider"],
    model=ACTIVE_CONFIG["model"],
    num_ctx=_num_ctx
)
generator = AnswerGenerator(llm_client=llm_client)

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ProviderSwitchRequest(BaseModel):
    provider: str
    model: str # Now serves as the model family ID (e.g. "qwen3")
    api_key: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    mode: str = "fast"            # Execution Mode: "fast" or "deep"
    retrieval_mode: str = "hybrid" # Retrieval Mode: "hybrid" or "graph"


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
    active_provider: str
    active_model: str # Concrete model name
    active_mode: str
    # Expanded runtime metrics (Objective 7)
    model_family: str
    resolved_model: str
    context_window: int
    prompt_budget: int
    network_latency_ms: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_semantic_synthesis_query(query: str) -> bool:
    """
    Detects if the query is requesting summarization, explanation, comparison,
    reasoning, synthesis, or drafting over the documents (requiring LLM generation).
    Uses strict whole-word set intersection to prevent substring collisions.
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
    
    # 3. Remove MINISTER signatures and departments
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


@app.get("/api/providers")
async def get_providers():
    """Get list of active provider backends."""
    return ["ollama", "groq"]


@app.get("/api/models")
async def get_models(provider: str):
    """
    Get available model families dynamically for the selected provider from the Model Registry.
    For Ollama, queries the local service dynamically and performs automated family mapping
    or dynamic family generation for un-registered local models (Objective 1).
    """
    prov = provider.lower().strip()
    if prov == "groq":
        families = model_registry.list_by_provider("groq")
        return [
            {
                "id": f.id,
                "display_name": f.display_name,
                "provider": f.provider,
                "model_name": f.model_name,
                "context_window": f.context_window,
                "thinking_capable": f.thinking_capable,
                "recommended_execution_mode": f.recommended_execution_mode
            }
            for f in families
        ]
    elif prov == "ollama":
        try:
            # Query local Ollama service dynamically
            with httpx.Client(timeout=3.0) as client:
                r = client.get("http://localhost:11434/api/tags")
                if r.status_code == 200:
                    ollama_data = r.json()
                    ollama_models = [m["name"] for m in ollama_data.get("models", [])]
                else:
                    ollama_models = []
        except Exception:
            # Raise connection exception if Ollama is offline
            raise HTTPException(status_code=503, detail="Ollama local service is offline or unreachable")

        if not ollama_models:
            # If Ollama is running but has no models installed, return empty list
            return []

        # Perform mapping or dynamic registration
        resolved_families = []
        for model_tag in ollama_models:
            # Try to find a registered family that matches this tag
            matched = None
            for f in model_registry.list_by_provider("ollama"):
                if f.model_name == model_tag or f.model_name in model_tag or model_tag in f.model_name:
                    matched = f
                    break
            
            if matched:
                resolved_families.append({
                    "id": matched.id,
                    "display_name": matched.display_name,
                    "provider": "ollama",
                    "model_name": matched.model_name,
                    "context_window": matched.context_window,
                    "thinking_capable": matched.thinking_capable,
                    "recommended_execution_mode": matched.recommended_execution_mode
                })
            else:
                # Dynamically register a new family on the fly (unregistered local model)
                base_name = model_tag.split(":")[0] if ":" in model_tag else model_tag
                friendly_name = base_name.replace("-", " ").replace("_", " ").title()
                tag_suffix = f" ({model_tag.split(':')[1]})" if ":" in model_tag and model_tag.split(':')[1] != "latest" else ""
                display_name = f"{friendly_name}{tag_suffix}"
                
                # Clean ID
                fam_id = model_tag.replace(":", "_").replace(".", "_").replace("-", "_")
                
                # Check thinking capability from substrings
                thinking_capable = any(t in model_tag.lower() for t in ["thinking", "r1", "reason", "o1", "o3"])
                
                # Guess context window
                context_window = 8192
                if "llama3" in model_tag.lower() or "llama-3" in model_tag.lower():
                    context_window = 8192
                elif "qwen" in model_tag.lower():
                    context_window = 32768 if "2.5" in model_tag else 8192
                
                # Register in memory
                from src.generation.registry import ModelFamily
                new_fam = ModelFamily(
                    id=fam_id,
                    display_name=display_name,
                    provider="ollama",
                    model_name=model_tag,
                    context_window=context_window,
                    thinking_capable=thinking_capable,
                    recommended_execution_mode="GPU"
                )
                model_registry.register(new_fam)

                resolved_families.append({
                    "id": fam_id,
                    "display_name": display_name,
                    "provider": "ollama",
                    "model_name": model_tag,
                    "context_window": context_window,
                    "thinking_capable": thinking_capable,
                    "recommended_execution_mode": "GPU"
                })

        return resolved_families
    else:
        raise HTTPException(status_code=400, detail="Unknown provider requested")


@app.post("/api/provider")
async def switch_provider(request: ProviderSwitchRequest):
    """
    Switches active provider and model family in memory without restarting FastAPI.
    Also registers the Groq API key in-memory for the current session.
    """
    prov = request.provider.lower().strip()
    if prov not in ["ollama", "groq"]:
        raise HTTPException(status_code=400, detail="Unknown provider")

    family_id = request.model
    family = model_registry.get(family_id)
    if not family:
        raise HTTPException(status_code=400, detail="Model family not found in registry")

    ACTIVE_CONFIG["provider"] = prov
    ACTIVE_CONFIG["model_family"] = family.id
    ACTIVE_CONFIG["groq_api_key"] = request.api_key

    # Resolve concrete model parameters based on execution mode profile
    exec_mode = ACTIVE_CONFIG["mode"]
    exec_params = family.get_execution_params(exec_mode)
    resolved_model = family.model_name

    ACTIVE_CONFIG["model"] = resolved_model

    # Re-initialize the active inference client and generator
    llm_client.provider = prov
    llm_client.model = resolved_model
    llm_client.num_ctx = family.context_window
    llm_client.temperature = exec_params["temperature"]
    llm_client.max_tokens = exec_params["max_tokens"]
    llm_client.api_key = request.api_key # Securely store session API key in client memory!
    generator.max_context_docs = exec_params["max_context_docs"]
    generator.max_doc_chars = exec_params["max_doc_chars"]
    generator.llm_client = llm_client

    return {
        "status": "success",
        "active_provider": prov,
        "active_model_family": family.display_name,
        "resolved_model": resolved_model,
        "context_window": family.context_window,
        "prompt_budget": int(family.context_window * 0.80),
        "thinking_capable": family.thinking_capable,
        "recommended_execution_mode": family.recommended_execution_mode,
        "is_connected": llm_client.check_health(api_key=request.api_key)
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    API routing chat requests dynamically based on execution mode and pathways.
    """
    query = request.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query message cannot be empty")

    ret_mode = request.retrieval_mode.lower()
    exec_mode = request.mode.lower() # "fast" or "deep"
    ACTIVE_CONFIG["mode"] = exec_mode

    # Resolve Model Family dynamically from Registry
    family_id = ACTIVE_CONFIG["model_family"]
    family = model_registry.get(family_id) or model_registry.get("qwen2.5")
    
    # Get adjusted execution parameters for the requested profile (Objective 1)
    exec_params = family.get_execution_params(exec_mode)
    resolved_model = family.model_name

    # Dynamically bind properties and session key to client
    llm_client.model = resolved_model
    llm_client.num_ctx = family.context_window
    llm_client.temperature = exec_params["temperature"]
    llm_client.max_tokens = exec_params["max_tokens"]
    llm_client.api_key = ACTIVE_CONFIG["groq_api_key"] # Propagate cached API key dynamically!
    ACTIVE_CONFIG["model"] = resolved_model

    # Configure document and context sizes dynamically from execution profile (Objective 1)
    generator.max_doc_chars = exec_params["max_doc_chars"]
    generator.max_context_docs = exec_params["max_context_docs"]

    # Calculate token budgets dynamically: Prompt Budget = 0.80 * Context Window (Objective 5)
    prompt_budget = int(family.context_window * 0.80)
    generator.context_budget_ratio = 0.80

    t_ret_start = time.perf_counter()
    sources: List[SourceItem] = []

    # ── Path Selection Matrix ──
    is_graph_result = (ret_mode == "graph" and not is_semantic_synthesis_query(query))

    if is_graph_result:
        # ── PATH A: DETERMINISTIC METADATA QUERY PATH ──
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
            header = "### 🕸️ GraphRAG: Document Explorer\n"
            header += "The following real parliamentary document cards were resolved directly from the metadata graph relationships:\n\n"
            
            cards = []
            for r in results:
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
            
        comp_tok = len(answer) // 4

        return ChatResponse(
            answer=answer,
            sources=sources,
            retrieval_latency_ms=ret_latency,
            generation_latency_ms=0.0,
            prompt_tokens=0,
            completion_tokens=comp_tok,
            total_tokens=comp_tok,
            is_fallback=False,
            is_graph_result=True,
            active_provider=ACTIVE_CONFIG["provider"],
            active_model=resolved_model,
            active_mode=exec_mode,
            model_family=family.display_name,
            resolved_model=resolved_model,
            context_window=family.context_window,
            prompt_budget=prompt_budget,
            network_latency_ms=0.0
        )

    else:
        # ── PATH B: SEMANTIC / SYNTHESIS QUERY PATH ──
        if ret_mode == "graph":
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
        
        api_key = ACTIVE_CONFIG["groq_api_key"]
        llm_available = llm_client.check_health(api_key=api_key)

        network_latency_ms = 0.0

        if llm_available and results:
            gen_res = generator.generate(query, results)
            gen_latency = gen_res.generation_latency_ms
            answer = gen_res.answer
            prompt_tok = gen_res.prompt_tokens
            comp_tok = gen_res.completion_tokens
            total_tok = gen_res.total_tokens
            is_fallback = False

            # Extract exact network overhead for Groq (Objective 7)
            if ACTIVE_CONFIG["provider"] == "groq" and gen_res.raw_response:
                usage = gen_res.raw_response.get("usage", {})
                total_time_sec = usage.get("total_time", 0.0)
                if total_time_sec > 0:
                    model_processing_ms = total_time_sec * 1000
                    network_latency_ms = max(0.0, gen_latency - model_processing_ms)
                else:
                    network_latency_ms = gen_latency * 0.15  # Fallback 15% overhead estimate
        else:
            is_fallback = True
            gen_latency = (time.perf_counter() - t_gen_start) * 1000
            provider_label = "Ollama (Local)" if ACTIVE_CONFIG["provider"] == "ollama" else "Groq (Cloud)"
            
            if ACTIVE_CONFIG["provider"] == "groq" and not api_key:
                err_cause = "Authentication Failed: Groq API Key was not supplied or saved."
            elif ACTIVE_CONFIG["provider"] == "groq":
                err_cause = "Authentication Failed: The supplied Groq API Key is invalid or expired."
            else:
                err_cause = "Ollama Offline: The local service is currently offline or unreachable."

            answer = (
                f"**[System Notice: {provider_label} Generation Offline]**\n\n"
                f"The active LLM service is currently offline or unauthorized.\n"
                f"* **Reason**: {err_cause}\n\n"
                f"Because this query requires cognitive synthesis, explanation, or comparison, a complete "
                f"response cannot be compiled. Please switch back to connected providers or start the local service."
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
            is_graph_result=is_graph_result,
            active_provider=ACTIVE_CONFIG["provider"],
            active_model=resolved_model,
            active_mode=exec_mode,
            model_family=family.display_name,
            resolved_model=resolved_model,
            context_window=family.context_window,
            prompt_budget=prompt_budget,
            network_latency_ms=network_latency_ms
        )


def start_server(port: int = 4000) -> None:
    """Run the FastAPI application on host 0.0.0.0."""
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    start_server()
