"""
Answer generator — grounded generation using adaptive context budgeting.

Design Decisions
----------------
1. ADAPTIVE CONTEXT BUDGETING: Build full context first, estimate tokens,
   and selectively apply compression only when the prompt exceeds a configurable
   safety threshold (e.g., 80% of num_ctx). Small-to-medium queries retain
   complete, uncompressed details for superior answer quality, while massive
   audit queries stay safe from context window overflow.

2. We use a structured prompt with clear sections:
   - Task description (grounded Q&A answering)
   - Retrieved context (up to k Q&A pairs)
   - User question
   - Answer instruction (cite sources, don't hallucinate)

3. The prompt is designed to minimize hallucination risk:
   - Explicit instruction: "Answer based ONLY on the provided context"
   - Citation requirement: "Quote the relevant passage"
   - Uncertainty handling: "If insufficient information, say so"
   - No knowledge cutoff reminder — the model shouldn't rely on training data

4. The system prompt is kept concise (under 200 tokens) because:
   - Qwen2.5:7B-Q4_K_M at 2048 context has limited headroom
   - More tokens for retrieved context = better grounding
   - We can include more Q&A pairs this way

5. The generator returns the full response with timing, token counts,
   and source attribution — all needed for the evaluation framework.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from rich.console import Console

from src.generation.client import LLMClient
from src.retrieval.result import RetrievedResult

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert parliamentary research assistant answering questions about Indian government policies, schemes, and administrative matters based ONLY on the provided Question & Answer context.

RULES:
1. Answer using ONLY the information in the provided context below.
2. If the context does not contain enough information to answer, say: "The provided context does not contain sufficient information to answer this question."
3. Do NOT hallucinate or make up facts, statistics, or claims not present in the context.
4. Quote or paraphrase relevant passages from the context in your answer.
5. If multiple context items are relevant, synthesize information from all of them.
6. Keep your answer concise, factual, and directly responsive to the question."""


def extract_relevant_evidence(text: str, query: str, max_chars: int = 1500) -> str:
    """
    Intelligently extracts the most relevant paragraphs or blocks of text from a
    retrieved answer matching query keywords, strictly staying within the character budget.
    """
    if len(text) <= max_chars:
        return text

    # Extract keywords from the query
    keywords = [w.lower() for w in re.sub(r"[^\w\s]", " ", query).split() if len(w) > 3]
    if not keywords:
        return text[:max_chars] + " ... [Truncated to fit context budget]"

    # Split answer text into paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    matched_paragraphs = []
    
    for p in paragraphs:
        p_lower = p.lower()
        if any(kw in p_lower for kw in keywords):
            matched_paragraphs.append(p)

    if matched_paragraphs:
        assembled = ""
        for p in matched_paragraphs:
            if len(assembled) + len(p) + 2 <= max_chars:
                assembled += p + "\n\n"
            else:
                remaining = max_chars - len(assembled)
                if remaining > 100:
                    assembled += p[:remaining] + " ... [Truncated to fit context budget]"
                break
        return assembled.strip() or text[:max_chars] + " ... [Truncated to fit context budget]"

    return text[:max_chars] + " ... [Truncated to fit context budget]"


def build_user_prompt(
    question: str,
    retrieved_results: list[RetrievedResult],
    max_doc_chars: int = 999999,  # High default to support uncompressed assembly
) -> str:
    """
    Build the user prompt with retrieved context.
    """
    parts = [
        "Below is the most relevant parliamentary Question & Answer context retrieved for your question.",
        "",
        "=" * 70,
        f"RETRIEVED CONTEXT ({len(retrieved_results)} records):",
        "=" * 70,
        "",
    ]

    for i, result in enumerate(retrieved_results, start=1):
        parts.append(f"[Source {i}] (ID: {result.doc_id})")
        if result.metadata.get("ministry"):
            parts.append(f"Ministry: {result.metadata['ministry']}")
        if result.metadata.get("subject"):
            parts.append(f"Subject: {result.metadata['subject']}")
        parts.append("")

        q_text = result.question
        a_text = result.answer
        if len(a_text) > max_doc_chars:
            a_text = extract_relevant_evidence(a_text, question, max_doc_chars)

        parts.append(f"QUESTION: {q_text}")
        parts.append(f"ANSWER: {a_text}")
        parts.append("")
        parts.append("-" * 70)

    parts.extend([
        "",
        "=" * 70,
        "USER QUESTION:",
        "=" * 70,
        question,
        "",
        "=" * 70,
        "ANSWER:",
        "=" * 70,
    ])

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Generation Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Result of answer generation."""

    answer: str
    model: str
    sources_used: list[str]  # doc_ids
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_latency_ms: float
    prompt: str  # The full prompt (for debugging/evaluation)

    @property
    def estimated_cost_usd(self) -> float:
        """
        Estimate cost in USD based on OpenAI GPT-4o-mini rates.
        This is for comparison purposes only — actual cost is $0
        since we're using Ollama locally.
        """
        # GPT-4o-mini rates (approximate)
        PROMPT_COST_PER_1K = 0.00015  # $0.15/1M tokens
        COMPLETION_COST_PER_1K = 0.00060  # $0.60/1M tokens
        return (
            self.prompt_tokens * PROMPT_COST_PER_1K / 1000
            + self.completion_tokens * COMPLETION_COST_PER_1K / 1000
        )


# ─────────────────────────────────────────────────────────────────────────────
# Answer Generator
# ─────────────────────────────────────────────────────────────────────────────

class AnswerGenerator:
    """
    Grounded answer generation using adaptive context budgeting.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        system_prompt: str | None = None,
        max_context_docs: int = 5,
        max_doc_chars: int = 1500,  # Fallback compression target per document
        context_budget_ratio: float = 0.80,  # Threshold ratio of context window
        compression_enabled: bool = True,  # Globally enable/disable adaptive budgeting
    ) -> None:
        """
        Parameters
        ----------
        llm_client : LLMClient, optional
            LLM client. Created with defaults if not provided.
        max_doc_chars : int
            Fallback compression characters limit per document if budget is exceeded.
        context_budget_ratio : float
            Threshold of context window before compression is triggered (e.g., 0.80).
        compression_enabled : bool
            Whether adaptive context budgeting is globally active.
        """
        self.llm_client = llm_client or LLMClient()
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.max_context_docs = max_context_docs
        self.max_doc_chars = max_doc_chars
        self.context_budget_ratio = context_budget_ratio
        self.compression_enabled = compression_enabled

    def generate(
        self,
        question: str,
        retrieved_results: list[RetrievedResult],
    ) -> GenerationResult:
        """
        Generate a grounded answer from retrieved context.
        """
        # Limit context to max_context_docs
        context = retrieved_results[: self.max_context_docs]

        if not context:
            return GenerationResult(
                answer=(
                    "No relevant documents were retrieved from the knowledge base. "
                    "I cannot answer this question based on the available context."
                ),
                model=self.llm_client.model,
                sources_used=[],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                generation_latency_ms=0.0,
                prompt="",
            )

        # ── 1. Assemble Uncompressed Prompt First ──
        uncompressed_prompt = build_user_prompt(question, context, max_doc_chars=999999)
        uncompressed_total_text = f"SYSTEM PROMPT:\n{self.system_prompt}\n\nUSER PROMPT:\n{uncompressed_prompt}"
        
        # Estimate token count
        uncompressed_tokens = len(uncompressed_total_text) // 4
        
        # Determine Context Window Boundaries
        effective_window = getattr(self.llm_client, "num_ctx", 8192)
        if not isinstance(effective_window, (int, float)):
            effective_window = 8192
            
        budget_threshold = int(effective_window * self.context_budget_ratio)

        # ── 2. Adaptive Decision Matrix ──
        compression_applied = False
        reason = "Prompt fits within budget"

        if self.compression_enabled and uncompressed_tokens > budget_threshold:
            compression_applied = True
            reason = "Prompt exceeded threshold"
            # Build compressed, high-relevance prompt
            user_prompt = build_user_prompt(question, context, self.max_doc_chars)
            total_prompt_text = f"SYSTEM PROMPT:\n{self.system_prompt}\n\nUSER PROMPT:\n{user_prompt}"
            final_tokens = len(total_prompt_text) // 4
        else:
            user_prompt = uncompressed_prompt
            total_prompt_text = uncompressed_total_text
            final_tokens = uncompressed_tokens

        # ── 3. Structured Budget Logging (Requested) ──
        console.print(f"\n[bold cyan]Context Budget Manager[/bold cyan]")
        console.print(f"Estimated Prompt Tokens : {final_tokens:,}")
        console.print(f"Maximum Context Window  : {effective_window:,}")
        console.print(f"Budget Threshold        : {budget_threshold:,}")
        console.print(f"Compression Applied     : {'YES' if compression_applied else 'NO'}")
        console.print(f"Reason                  : {reason}\n")

        # Save exact prompt sent to LLM
        try:
            with open("generation_prompt_debug.txt", "w", encoding="utf-8") as f:
                f.write(total_prompt_text)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save prompt to debug file: {e}[/yellow]")

        # Generate
        start_time = time.monotonic()
        response = self.llm_client.generate(
            prompt=user_prompt,
            system=self.system_prompt,
        )
        generation_latency_ms = (time.monotonic() - start_time) * 1000

        return GenerationResult(
            answer=response.text,
            model=response.model,
            sources_used=[r.doc_id for r in context],
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            generation_latency_ms=generation_latency_ms,
            prompt=user_prompt,  # Include for evaluation/debugging
        )

    def check_health(self) -> bool:
        """Check if the LLM service is available."""
        return self.llm_client.check_health()
