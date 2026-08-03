"""
Answer generator — grounded generation using retrieved context.

Design Decisions
----------------
1. The generator uses ONLY retrieved context — it never has access to
   the raw knowledge base. This enforces strict retrieval-grounding.

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

6. Generation is a SEPARATE step from retrieval — they can be timed
   independently and swapped independently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from src.generation.client import LLMClient, LLMResponse
from src.retrieval.result import RetrievedResult


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


def build_user_prompt(
    question: str,
    retrieved_results: list[RetrievedResult],
) -> str:
    """
    Build the user prompt with retrieved context.

    The prompt includes all retrieved Q&A pairs formatted as structured blocks.
    Each block is clearly labelled so the LLM can reference specific sources.

    Parameters
    ----------
    question : str
        The user's question.
    retrieved_results : list[RetrievedResult]
        Top-K retrieved Q&A records.

    Returns
    -------
    str
        The complete user prompt with context.
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
        parts.append(f"QUESTION: {result.question}")
        parts.append(f"ANSWER: {result.answer}")
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
    Grounded answer generation using retrieved context + LLM.

    Takes retrieved Q&A records and generates a grounded answer
    using the configured LLM.

    Usage
    -----
    ```python
    generator = AnswerGenerator(llm_client=llm_client)

    result = generator.generate(
        question="What measures address malaria?",
        retrieved_results=retrieved,
    )
    print(result.answer)
    print(f"Tokens: {result.total_tokens}, Latency: {result.generation_latency_ms}ms")
    ```
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        system_prompt: Optional[str] = None,
        max_context_docs: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        llm_client : LLMClient, optional
            LLM client. Created with defaults if not provided.
        system_prompt : str, optional
            Custom system prompt. Uses the default if not provided.
        max_context_docs : int
            Maximum number of retrieved docs to include in context.
            More docs = better coverage but higher prompt tokens.
            Default 5 is a good balance.
        """
        self.llm_client = llm_client or LLMClient()
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.max_context_docs = max_context_docs

    def generate(
        self,
        question: str,
        retrieved_results: list[RetrievedResult],
    ) -> GenerationResult:
        """
        Generate a grounded answer from retrieved context.

        Parameters
        ----------
        question : str
            The user's question.
        retrieved_results : list[RetrievedResult]
            Retrieved Q&A records from the retrieval pipeline.

        Returns
        -------
        GenerationResult
            Generated answer with source attribution and metadata.
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

        # Build prompt
        user_prompt = build_user_prompt(question, context)

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
