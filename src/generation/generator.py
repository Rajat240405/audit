"""
Answer generator — grounded generation using adaptive context budgeting.

Design Decisions
----------------
1. DYNAMIC EVIDENCE BUDGETING (Task 3): when an ExecutionPlan is attached
   (server/CLI always do), the prompt is assembled by
   src.generation.evidence: the plan's reserve-based evidence budget
   (num_ctx − max_tokens − scaffold − margin, minus the ACTUAL system prompt
   and question) is spent admitting as many relevant candidates as fit —
   whole documents first, semantic blocks second, at most one marked
   sentence truncation as the last resort. No fixed document-count quota and
   no unconditional per-document char cap apply in this path.

   A LEGACY path (no plan attached — standalone/notebook use) keeps the old
   behavior exactly: count slice (max_context_docs), per-doc char caps
   (max_doc_chars), 0.80 threshold check. Small-to-medium queries retain
   complete, uncompressed details either way, and massive audit queries stay
   safe from context window overflow.

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

4. The generator returns the full response with timing, token counts,
   and source attribution — all needed for the evaluation framework.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
import httpx
from rich.console import Console

from src.generation.client import LLMClient
from src.retrieval.result import RetrievedResult
from src.generation.registry import model_registry
from src.generation.defaults import default_num_ctx

# Task-3 evidence engine. clean/truncate/extract helpers moved here for
# module cohesion; they are re-exported so existing import sites
# (pipeline.py, tests, notebooks) keep working unchanged.
from src.generation.evidence import (  # noqa: F401  (re-exports)
    AGGRESSIVE_HEAL_POOL,
    AGGRESSIVE_HEAL_TOKENS,
    Allocation,
    allocate_evidence,
    assemble_budgeted_prompt,
    clean_parliament_text,
    estimate_tokens,
    extract_relevant_evidence,
    render_user_prompt,
    truncate_at_sentence,
)

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert parliamentary research assistant for Indian government policy, schemes, and administrative matters. Answer the question using ONLY the provided Question & Answer context.

ANSWER STYLE:
1. If a TONE block is present later in this system message, it is authoritative for register, layout, and verbosity (including how long or short to write). Follow that TONE. Do not apply a fixed character budget.
2. If no TONE block is present: write in the third person, official and factual (e.g. "The Government has...", "The Ministry provides..."). Never use first-person ("I", "we", "my"). Give a complete answer without padding or repetition — do not target a character count.
3. If the question has multiple parts (a), (b), (c)... and the TONE does not specify another layout, structure the answer with matching (a), (b), (c) sub-sections.
4. If a part of the question is not addressed in the context, state "Does not arise." or "The provided documents do not address this." — do not invent an answer for it.
5. Synthesize across ALL provided documents when multiple are relevant; do not just copy one document verbatim. Quote or paraphrase the key passages.

GROUNDING RULES (critical — audit transparency):
6. Every proper noun — organization name, programme/scheme name, acronym, institute, city, ministry — MUST appear EXACTLY as written in the retrieved context (same spelling, same abbreviation). NEVER expand an abbreviation (e.g. keep "NIOT", never write "National Institute of Ocean Technology"), NEVER substitute a modern or official alternative name, NEVER use a name from your general knowledge.
7. Every number, figure, date, budget amount, percentage, and measurement MUST be copied VERBATIM from the retrieved context. NEVER supply a statistic from memory or training data (e.g. sea-level rise rates, radar counts, year ranges).
8. If a name, programme, or figure is NOT in the retrieved context, OMIT it. An omitted detail is always better than an invented one. If the retrieved documents are silent on a point, state that the documents do not address it.
9. Do NOT include any [Source N] tags or citation markers in your final answer. The answer must read as a clean, final response with no inline citations visible to the user. Source attribution is handled separately by the system.
10. Do NOT hallucinate or make up facts, statistics, or claims that are not supported by the provided context.

If the provided context is insufficient to answer the question, clearly state:
1. That the available context is insufficient.
2. Which specific topic, entity, fact, or part of the question is missing from the context.
3. What portion of the question can be answered from the available context, if any.

Do not use a generic refusal when you can identify the missing information.

For example, if the user asks "Compare INCOIS and Virat Kohli" and the context contains information about INCOIS but nothing about Virat Kohli, say that the context contains information about INCOIS but does not contain sufficient information about Virat Kohli to make the comparison.

Never fill the missing information using your own general knowledge."""



def compact_documents_with_llm(
    question: str,
    retrieved_results: list[RetrievedResult],
    max_doc_chars: int = 1500,
    llm_client=None,
) -> str:
    """LLM-based compaction: condense each retrieved document to its key
    facts (names, figures, dates) instead of lossy keyword-truncation.

    Falls back to extract_relevant_evidence if no LLM client is available
    or the compaction call fails — never raises.
    """
    if not llm_client:
        return build_user_prompt(question, retrieved_results, max_doc_chars)

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

        q_text = clean_parliament_text(result.question)
        a_text = clean_parliament_text(result.answer)
        # NEVER send a 400k parent to the LLM to "compress" it. Bound first.
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



def build_user_prompt(
    question: str,
    retrieved_results: list[RetrievedResult],
    max_doc_chars: int = 999999,
) -> str:
    """LEGACY prompt builder — fixed per-document char cap, used by the
    no-plan fallback path (standalone AnswerGenerator) and existing callers.

    The budgeted Task-3 path assembles via evidence.allocate_evidence +
    assemble_budgeted_prompt instead; both share the ONE renderer
    (evidence.render_user_prompt), so the visual prompt contract
    ([Source N] blocks, QUESTION/ANSWER sections, tail) is identical.
    """
    items = []
    for result in retrieved_results:
        q_text = clean_parliament_text(result.question)
        a_text = clean_parliament_text(result.answer)
        if len(a_text) > max_doc_chars:
            a_text = extract_relevant_evidence(a_text, question, max_doc_chars)
        items.append((result, q_text, a_text))
    return render_user_prompt(question, items)


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
    raw_response: dict | None = None

    @property
    def estimated_cost_usd(self) -> float:
        """
        Estimate cost in USD based on OpenAI GPT-4o-mini rates.
        This is for comparison purposes only — actual cost is $0
        since we're using Ollama locally.
        """
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
        # Task 3: the ExecutionPlan attached by the server/CLI plan-binding
        # (_apply_execution_plan / _apply_cli_plan). None → legacy behavior.
        self.plan = None
        self.last_allocation: Allocation | None = None
        self._ctx_cache: tuple | None = None

    # ─────────────────────────────────────────────────────────────────────
    # Task-3 budgeted context assembly (shared by generate + generate_stream
    # + the server's "what the model received" sources emission)
    # ─────────────────────────────────────────────────────────────────────
    def prepare_context(
        self,
        question: str,
        retrieved_results: list[RetrievedResult],
        *,
        pool_override: int | None = None,
        budget_override: int | None = None,
    ) -> tuple[str, list[str], dict]:
        """Assemble the user prompt under the attached ExecutionPlan.

        Pool = initial candidates (plan.retrieval_top_k); budget =
        plan.evidence_budget_tokens − actual system-prompt/question tokens.
        Admission itself is budget-driven (evidence.allocate_evidence), never
        count-driven. Returns (prompt, admitted_doc_ids, diagnostics).
        Cached by (question, doc-id sequence, overrides, system-prompt) so the
        non-stream generate(), the streaming path and the server sources
        emission all reuse ONE assembly — stream/non-stream parity.
        """
        plan = self.plan
        if plan is None:
            raise RuntimeError("prepare_context requires an attached ExecutionPlan")
        pool = retrieved_results[: max(1, int(pool_override or plan.retrieval_top_k))]
        key = (
            question,
            tuple(r.doc_id for r in pool),
            pool_override,
            budget_override,
            self.system_prompt or "",  # TONE suffixes mutate this per request
        )
        if self._ctx_cache and self._ctx_cache[0] == key:
            return self._ctx_cache[1]
        overhead = estimate_tokens((self.system_prompt or "") + "\n" + question)
        base = budget_override if budget_override is not None else plan.evidence_budget_tokens
        budget = max(0, int(base) - overhead)
        alloc = allocate_evidence(pool, question, budget)
        prompt = assemble_budgeted_prompt(question, alloc.admissions)
        ids = alloc.admitted_ids
        diag = {
            "pool": len(pool),
            "admitted": len(ids),
            "evidence_budget_tokens": budget,
            "evidence_used_tokens": alloc.used_tokens,
            "prompt_est_tokens": estimate_tokens(prompt),
            "skipped_doc_ids": list(alloc.skipped_doc_ids),
            "truncated_doc_ids": list(alloc.truncated_doc_ids),
        }
        self.last_allocation = alloc
        self._ctx_cache = (key, (prompt, ids, diag))
        return prompt, ids, diag

    def _no_context_result(self) -> GenerationResult:
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

    def _provider_model(self) -> tuple[str, str]:
        provider = getattr(self.llm_client, "provider", "ollama")
        if not isinstance(provider, str):
            provider = "ollama"
        model_name = getattr(self.llm_client, "model", "qwen2.5:7b")
        if not isinstance(model_name, str):
            model_name = "qwen2.5:7b"
        return provider, model_name

    def _write_debug_prompt(self, total_prompt_text: str) -> None:
        """Optional debug dump — under APP_DATA_DIR (never process CWD)."""
        try:
            from src.utils.app_paths import data_dir, prompt_debug_path

            data_dir().mkdir(parents=True, exist_ok=True)
            dest = prompt_debug_path()
            dest.write_text(total_prompt_text, encoding="utf-8")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save prompt to debug file: {e}[/yellow]")

    def _generation_result(self, response, source_ids, latency_ms, prompt) -> GenerationResult:
        return GenerationResult(
            answer=response.text,
            model=response.model,
            sources_used=list(source_ids),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            generation_latency_ms=latency_ms,
            prompt=prompt,
            raw_response=response.raw_response,
        )

    def _graceful_context_fail_result(self, source_ids, prompt) -> GenerationResult:
        return GenerationResult(
            answer=(
                "### ⚠️ Context Size Exceeded\n\n"
                "The retrieved parliamentary context was too large for the LLM provider, "
                "and our automated self-healing retry also exceeded request limits.\n\n"
                "**Suggestions**:\n"
                "1. Please try a more specific question with more precise keywords.\n"
                "2. Try switching the Execution Mode to **⚡ Fast Mode** to use a lighter context budget."
            ),
            model=self.llm_client.model,
            sources_used=list(source_ids),
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            generation_latency_ms=0.0,
            prompt=prompt,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Task-3 budgeted generate (plan attached)
    # ─────────────────────────────────────────────────────────────────────
    def _generate_budgeted(
        self,
        question: str,
        retrieved_results: list[RetrievedResult],
    ) -> GenerationResult:
        if not retrieved_results:
            return self._no_context_result()

        provider, model_name = self._provider_model()
        user_prompt, source_ids, diag = self.prepare_context(question, retrieved_results)
        total_prompt_text = f"SYSTEM PROMPT:\n{self.system_prompt}\n\nUSER PROMPT:\n{user_prompt}"

        console.print(
            f"[context] pool={diag['pool']} admitted={diag['admitted']} "
            f"evidence≈{diag['evidence_used_tokens']}/{diag['evidence_budget_tokens']} tok "
            f"prompt≈{diag['prompt_est_tokens']} tok ids={source_ids}"
            + (f" truncated={diag['truncated_doc_ids']}" if diag["truncated_doc_ids"] else "")
            + (f" skipped={diag['skipped_doc_ids']}" if diag["skipped_doc_ids"] else "")
        )
        self._write_debug_prompt(total_prompt_text)

        try:
            start_time = time.monotonic()
            response = self.llm_client.generate(prompt=user_prompt, system=self.system_prompt)
            latency_ms = (time.monotonic() - start_time) * 1000
            return self._generation_result(response, source_ids, latency_ms, user_prompt)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 413:
                raise
            # Budget-conform self-heal: same allocator, aggressive budget.
            console.print("[bold yellow]⚠️ HTTP 413 — retrying with aggressive budget-conform rebuild...[/bold yellow]")
            heal_prompt, heal_ids, hdiag = self.prepare_context(
                question,
                retrieved_results,
                pool_override=AGGRESSIVE_HEAL_POOL,
                budget_override=AGGRESSIVE_HEAL_TOKENS,
            )
            console.print(
                f"[bold yellow]Retry pool={hdiag['pool']} admitted={hdiag['admitted']} "
                f"prompt≈{hdiag['prompt_est_tokens']} tok.[/bold yellow]"
            )
            try:
                start_time = time.monotonic()
                response = self.llm_client.generate(prompt=heal_prompt, system=self.system_prompt)
                latency_ms = (time.monotonic() - start_time) * 1000
                return self._generation_result(response, heal_ids, latency_ms, heal_prompt)
            except Exception as retry_err:  # noqa: BLE001
                console.print(f"[bold red]❌ Self-healing retry failed: {retry_err}[/bold red]")
                return self._graceful_context_fail_result(source_ids, heal_prompt)

    def generate(
        self,
        question: str,
        retrieved_results: list[RetrievedResult],
    ) -> GenerationResult:
        """
        Generate a grounded answer from retrieved context with provider-aware budgeting.
        """
        # Task 3: budget-driven assembly when a plan is attached (server/CLI)
        if self.plan is not None:
            return self._generate_budgeted(question, retrieved_results)

        # ── LEGACY path (no plan attached) — unchanged behavior ──
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

        # ── 1. Determine active provider, model, and effective context window ──
        provider = getattr(self.llm_client, "provider", "ollama")
        if not isinstance(provider, str):
            provider = "ollama"
        model_name = getattr(self.llm_client, "model", "qwen2.5:7b")
        if not isinstance(model_name, str):
            model_name = "qwen2.5:7b"

        # Dynamically resolve from registry
        family = model_registry.get(model_name)
        if family:
            effective_window = family.context_window
        else:
            effective_window = getattr(self.llm_client, "num_ctx", default_num_ctx())
            if not isinstance(effective_window, (int, float)):
                effective_window = 8192

        operational_window = effective_window

        budget_threshold = int(operational_window * self.context_budget_ratio)

        # HARD per-doc cap BEFORE any LLM call (never assemble 400k+ parents).
        user_prompt = build_user_prompt(question, context, max_doc_chars=self.max_doc_chars)
        total_prompt_text = f"SYSTEM PROMPT:\n{self.system_prompt}\n\nUSER PROMPT:\n{user_prompt}"
        final_tokens = len(total_prompt_text) // 4
        compression_applied = False
        reason = "Per-doc hard cap applied before LLM"
        if self.compression_enabled and final_tokens > budget_threshold:
            compression_applied = True
            reason = "Prompt exceeded window budget — extra extract"
            user_prompt = compact_documents_with_llm(
                question, context, self.max_doc_chars, llm_client=None
            )
            total_prompt_text = f"SYSTEM PROMPT:\n{self.system_prompt}\n\nUSER PROMPT:\n{user_prompt}"
            final_tokens = len(total_prompt_text) // 4

        # ── 4. Simulate serialized outbound payload size for logging ──
        temp_val = getattr(self.llm_client, "temperature", 0.2)
        tokens_val = getattr(self.llm_client, "max_tokens", 2048)

        # These attributes come from the client object; coerce anything
        # non-numeric (e.g. a misconfigured stand-in) to the fallback so the
        # simulated payload always serializes. Generic type check only — no
        # test-framework types belong in production code.
        if isinstance(temp_val, bool) or not isinstance(temp_val, (int, float)):
            temp_val = 0.2
        if isinstance(tokens_val, bool) or not isinstance(tokens_val, int):
            tokens_val = 2048

        simulated_payload = {
            "model": model_name,
            "prompt": total_prompt_text,
            "stream": False,
            "options": {
                "temperature": temp_val,
                "num_predict": tokens_val,
                "num_ctx": effective_window,
            },
        }

        try:
            # default=str stringifies any leftover non-JSON value instead
            # of crashing the stats log.
            serialized_bytes = len(json.dumps(simulated_payload, default=str).encode("utf-8"))
        except Exception:
            serialized_bytes = len(str(simulated_payload).encode("utf-8"))

        # Log request statistics before sending
        console.print(f"\n[bold magenta]🚀 Pre-Request Outbound Stats[/bold magenta]")
        console.print(f"Provider                : {provider.upper()}")
        console.print(f"Model Name              : {model_name}")
        console.print(f"Prompt Character Count  : {len(total_prompt_text):,}")
        console.print(f"Estimated Prompt Tokens : {final_tokens:,}")
        console.print(f"Retrieved Documents Count: {len(context)}")
        console.print(f"Serialized Payload Size : {serialized_bytes:,} bytes")
        console.print(f"Maximum Context Window  : {effective_window:,}")
        console.print(f"Budget Threshold        : {budget_threshold:,}")
        console.print(f"Compression Applied     : {'YES' if compression_applied else 'NO'}")
        console.print(f"Reason                  : {reason}\n")

        # Optional debug dump — under APP_DATA_DIR (never process CWD).
        try:
            from src.utils.app_paths import data_dir, prompt_debug_path

            data_dir().mkdir(parents=True, exist_ok=True)
            dest = prompt_debug_path()
            dest.write_text(total_prompt_text, encoding="utf-8")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save prompt to debug file: {e}[/yellow]")

        # ── 5. Generate with self-healing HTTP 413 retry ──
        try:
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
                prompt=user_prompt,
                raw_response=response.raw_response,
            )
        except httpx.HTTPStatusError as e:
            raise  # re-raise all HTTP errors — the legacy 2-doc/500-char heuristic was removed

    # ─────────────────────────────────────────────────────────────────────
    # Task-3 budgeted stream (plan attached) — mirrors _generate_budgeted's
    # assembly exactly (same prepare_context cache) for stream parity.
    # ─────────────────────────────────────────────────────────────────────
    def _generate_stream_budgeted(self, question: str, context: list):
        if not context:
            yield {"type": "tokens", "text": (
                "No relevant documents were retrieved from the knowledge base. "
                "I cannot answer this question based on the available context."
            )}
            return

        provider, model_name = self._provider_model()
        stream_prompt, source_ids, diag = self.prepare_context(question, context)
        print(
            f"[context] pool={diag['pool']} admitted={diag['admitted']} "
            f"evidence≈{diag['evidence_used_tokens']}/{diag['evidence_budget_tokens']} tok "
            f"ids={source_ids}"
        )

        start_time = time.monotonic()
        full_text: list[str] = []
        try:
            retried_413 = False
            # Fast mode resilience: some Ollama/qwen3 builds IGNORE think:false
            # and spend the whole token budget on reasoning, leaving content
            # EMPTY. We detect that (0 visible tokens from a think=false run)
            # and retry once with thinking ON so the user always gets an answer.
            think_disabled = getattr(self.llm_client, "think", None) is False
            empty_retried = False
            while True:
                try:
                    saw_token = False
                    for chunk in self.llm_client.generate_stream(
                        prompt=stream_prompt, system=self.system_prompt
                    ):
                        if isinstance(chunk, dict):
                            ev_type = chunk.get("type", "tokens")
                            if ev_type == "tokens":
                                text = chunk.get("text", "")
                                saw_token = True
                                full_text.append(text)
                                yield {"type": "tokens", "text": text}
                            elif ev_type == "reasoning":
                                # Application-boundary think-gate (invariant):
                                # reasoning may reach consumers ONLY when the
                                # resolved plan says thinking is ON. Standard
                                # resolves thinking=False, and some providers
                                # still stream reasoning (several Ollama/qwen3
                                # builds ignore think:false) — the application
                                # drops those events here instead of trusting
                                # the provider flag. Plan-less consumers keep
                                # legacy behavior (no plan → not gated).
                                if self.plan is None or self.plan.thinking:
                                    yield {"type": "reasoning", "text": chunk.get("text", "")}
                            elif ev_type == "answer_start":
                                yield {"type": "answer_start"}
                            elif ev_type == "done":
                                break
                        else:
                            saw_token = True
                            full_text.append(chunk)
                            yield {"type": "tokens", "text": chunk}
                    if not saw_token and think_disabled and not empty_retried:
                        empty_retried = True
                        self.llm_client.think = True
                        full_text.clear()
                        print("[gen] think=false gave no answer — retrying with thinking ON")
                        continue
                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 413 and not retried_413:
                        console.print("[bold yellow]⚠️ HTTP 413 during stream — aggressive budget-conform rebuild...[/bold yellow]")
                        retried_413 = True
                        stream_prompt, source_ids, _ = self.prepare_context(
                            question,
                            context,
                            pool_override=AGGRESSIVE_HEAL_POOL,
                            budget_override=AGGRESSIVE_HEAL_TOKENS,
                        )
                        continue
                    raise  # re-raise other HTTP errors or second 413
        finally:
            latency_ms = (time.monotonic() - start_time) * 1000

        text = "".join(full_text)
        prompt_tokens = estimate_tokens(
            f"SYSTEM PROMPT:\n{self.system_prompt}\n\nUSER PROMPT:\n{stream_prompt}"
        )
        completion_tokens = estimate_tokens(text)
        yield {
            "type": "meta",
            "model": model_name,
            "provider": provider,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "generation_latency_ms": latency_ms,
            "sources_used": source_ids,
        }

    def generate_stream(self, question: str, context: list, **overrides):
        """Stream a grounded answer token-by-token.

        Mirrors ``generate``'s context assembly (doc limit + adaptive
        compression) but yields text chunks as the LLM produces them, so the
        frontend drafting workspace can render live (ChatGPT-style).
        Yields dicts: {"type": "tokens", "text": ...} then a final
        {"type": "meta", ...} with token estimates + latency.
        """
        # Task 3: budget-driven assembly when a plan is attached (server/CLI)
        if self.plan is not None:
            yield from self._generate_stream_budgeted(question, context)
            return

        # ── LEGACY path (no plan attached) — unchanged behavior ──
        context = context[: self.max_context_docs]
        if not context:
            yield {"type": "tokens", "text": (
                "No relevant documents were retrieved from the knowledge base. "
                "I cannot answer this question based on the available context."
            )}
            return

        provider = getattr(self.llm_client, "provider", "ollama")
        if not isinstance(provider, str):
            provider = "ollama"
        model_name = getattr(self.llm_client, "model", "qwen2.5:7b")
        if not isinstance(model_name, str):
            model_name = "qwen2.5:7b"

        family = model_registry.get(model_name)
        if family:
            effective_window = family.context_window
        else:
            effective_window = getattr(self.llm_client, "num_ctx", default_num_ctx())
            if not isinstance(effective_window, (int, float)):
                effective_window = 8192

        operational_window = effective_window

        budget_threshold = int(operational_window * self.context_budget_ratio)
        user_prompt = build_user_prompt(question, context, max_doc_chars=self.max_doc_chars)
        prompt_chars = len(self.system_prompt or "") + len(user_prompt)
        print(
            f"[context] docs={len(context)} chars={prompt_chars} "
            f"tokens~{prompt_chars // 4} ids={[r.doc_id for r in context]} "
            f"per={[ (r.doc_id, len(r.answer or '')) for r in context ]}"
        )
        if self.compression_enabled and (prompt_chars // 4) > budget_threshold:
            user_prompt = compact_documents_with_llm(
                question, context, self.max_doc_chars, llm_client=None
            )

        start_time = time.monotonic()
        full_text = []
        try:
            stream_prompt = user_prompt
            stream_context = context
            # Fast mode resilience: some Ollama/qwen3 builds IGNORE think:false
            # and spend the whole token budget on reasoning, leaving content
            # EMPTY. We detect that (0 visible tokens from a think=false run)
            # and retry once with thinking ON so the user always gets an answer.
            think_disabled = getattr(self.llm_client, "think", None) is False
            empty_retried = False
            while True:
                try:
                    saw_token = False
                    for chunk in self.llm_client.generate_stream(
                        prompt=stream_prompt, system=self.system_prompt
                    ):
                        # Structured events from the client: forward reasoning
                        # + answer_start to the UI, keep visible tokens.
                        if isinstance(chunk, dict):
                            ev_type = chunk.get("type", "tokens")
                            if ev_type == "tokens":
                                text = chunk.get("text", "")
                                saw_token = True
                                full_text.append(text)
                                yield {"type": "tokens", "text": text}
                            elif ev_type == "reasoning":
                                # Same application-boundary think-gate as the
                                # budgeted path: no plan (this legacy branch)
                                # keeps legacy behavior; a plan gates by
                                # plan.thinking. Some providers stream
                                # reasoning even against think=false.
                                if self.plan is None or self.plan.thinking:
                                    yield {"type": "reasoning", "text": chunk.get("text", "")}
                            elif ev_type == "answer_start":
                                yield {"type": "answer_start"}
                            elif ev_type == "done":
                                break
                        else:
                            # legacy string chunk (shouldn't happen after the
                            # client event migration, but keep it safe)
                            saw_token = True
                            full_text.append(chunk)
                            yield {"type": "tokens", "text": chunk}
                    # think:false produced NO visible answer -> reasoning ate
                    # the budget. Retry once with thinking enabled.
                    if not saw_token and think_disabled and not empty_retried:
                        empty_retried = True
                        self.llm_client.think = True
                        full_text.clear()
                        print("[gen] think=false gave no answer — retrying with thinking ON")
                        continue
                    break  # completed normally
                except httpx.HTTPStatusError:
                    raise  # re-raise all HTTP errors
        finally:
            latency_ms = (time.monotonic() - start_time) * 1000

        text = "".join(full_text)
        prompt_tokens = len(
            f"SYSTEM PROMPT:\n{self.system_prompt}\n\nUSER PROMPT:\n{stream_prompt}"
        ) // 4
        completion_tokens = len(text) // 4
        yield {
            "type": "meta",
            "model": model_name,
            "provider": provider,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "generation_latency_ms": latency_ms,
            "sources_used": [r.doc_id for r in stream_context],
        }

    def check_health(self) -> bool:
        """Check if the LLM service is available."""
        return self.llm_client.check_health()
