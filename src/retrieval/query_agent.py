"""Query-understanding + routing agent (orchestration only).

ONE lightweight LLM call in front of retrieval that answers four questions:

    * what language is the user writing in?
    * can the SERVING model actually read it?
    * what is the English-normalized query to retrieve with?
    * what language/style should the FINAL answer use?

In AUTO mode the same call also picks the retrieval route. In MANUAL mode the
user's selection is authoritative and NO routing decision is made or used.

Deliberate non-goals (see the architecture constraints):
  * this module contains no Hybrid RAG or GraphRAG logic — it only prepares
    and routes the query;
  * it creates no model/provider configuration of its own. The client is
    supplied by the caller, which builds it from the SAME
    ``resolve_active_stack()`` mechanism the rest of the app uses, so if the
    serving model changes the agent follows it with no code change;
  * it never raises for model misbehaviour — malformed output degrades to a
    safe default so a request is never lost to a bad agent response.

English is the INTERNAL RETRIEVAL language only. ``response_language`` is
carried separately so the answer can be produced in the user's own language.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ROUTES",
    "ROUTE_HYBRID",
    "ROUTE_GRAPH",
    "ROUTE_BOTH",
    "QueryPlan",
    "normalize_route",
    "run_query_agent",
]

ROUTE_HYBRID = "HYBRID"
ROUTE_GRAPH = "GRAPH"
ROUTE_BOTH = "HYBRID_AND_GRAPH"
#: The ONLY routes that may ever be executed.
ROUTES = (ROUTE_HYBRID, ROUTE_GRAPH, ROUTE_BOTH)

#: Returned verbatim when the serving model cannot read the user's language.
UNSUPPORTED_LANGUAGE_MESSAGE = (
    "I'm unable to read or understand the language in your query. "
    "Please provide your query in English."
)

_SYSTEM_PROMPT = """You are a query-understanding component of a document retrieval system for Indian parliamentary and ocean-science records.

Return STRICT JSON only. No markdown, no commentary, no explanation of your reasoning.

Shape:
{
  "language": "<English name of the language the user wrote in, e.g. English, Hindi, Hinglish>",
  "supported": true,
  "english_query": "<the query translated/normalized into English for retrieval>",
  "response_language": "<the language the ANSWER should be written in>",
  "route": "HYBRID"
}

Rules:
- "supported" is false ONLY if you genuinely cannot read the query's language. If you can read it, it is true.
- "english_query" MUST be English, even when the user wrote another language. Keep proper nouns, organisation names and acronyms as-is.
- "response_language" is the language the user would expect the answer in — normally the language they wrote in. For mixed Hindi/English (Hinglish), use "Hinglish" so the answer follows their style.
- "route" must be exactly one of: HYBRID, GRAPH, HYBRID_AND_GRAPH.

Route guidance:
- HYBRID: document-level information, facts, statements, dates, measures, summaries, parliamentary/document questions. This is the DEFAULT.
- GRAPH: relationships between entities — organisation/programme links, ministry/organisation structure, facility ownership, multi-hop connections, questions where traversing a knowledge graph is materially required.
- HYBRID_AND_GRAPH: the question materially needs BOTH document evidence AND relationship evidence.

IMPORTANT: merely MENTIONING an organisation or programme does NOT mean GRAPH. If the query is ambiguous or broad, choose HYBRID."""

_USER_TEMPLATE = """User query:
{query}

Return the JSON now."""


@dataclass
class QueryPlan:
    """Validated result of the agent call.

    ``route`` is meaningful only in AUTO mode; manual callers ignore it.
    """

    original_query: str
    language: str = "English"
    supported: bool = True
    english_query: str = ""
    response_language: str = "English"
    route: str = ROUTE_HYBRID
    #: True when the agent call failed or returned unusable output.
    degraded: bool = False
    #: Short, user-safe reason for degradation (never a stack trace).
    notes: list[str] = field(default_factory=list)

    @property
    def retrieval_query(self) -> str:
        """The query to retrieve with — English when we have it."""
        return (self.english_query or self.original_query).strip()

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "supported": self.supported,
            "english_query": self.english_query,
            "response_language": self.response_language,
            "route": self.route,
            "degraded": self.degraded,
        }


def normalize_route(value: object, *, default: str = ROUTE_HYBRID) -> str:
    """Coerce agent output to a legal route, else ``default``.

    Anything unrecognised — a made-up route, wrong case, None, a non-string —
    falls back rather than raising, so a bad agent response can never crash a
    request or execute an unknown pipeline.
    """
    if not isinstance(value, str):
        return default
    token = value.strip().upper().replace("-", "_").replace(" ", "_")
    if token in ROUTES:
        return token
    # tolerate a few obvious spellings without inventing new routes
    if token in ("BOTH", "HYBRID+GRAPH", "GRAPH_AND_HYBRID"):
        return ROUTE_BOTH
    return default


def _extract_json_object(raw: str) -> Optional[dict]:
    """Parse the agent payload, reusing the hardened GraphRAG repair chain.

    The extraction parser already handles the two failure modes seen in
    production (literal control characters inside strings, and lone
    backslashes), so that logic is reused here rather than duplicated.
    """
    from src.graphrag.extract import SemanticExtractor

    text = (raw or "").strip()
    if not text:
        return None
    # tolerate a single markdown fence even though the prompt forbids it
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # models sometimes prepend a sentence; take the outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start > 0 or (end != -1 and end < len(text) - 1):
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    for candidate in (
        text,
        SemanticExtractor._escape_control_chars_in_strings(text),
        SemanticExtractor._repair_invalid_escapes(text),
        SemanticExtractor._repair_invalid_escapes(
            SemanticExtractor._escape_control_chars_in_strings(text)
        ),
    ):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return data if isinstance(data, dict) else None
    return None


def _coerce_bool(value: object, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("true", "yes", "1"):
            return True
        if token in ("false", "no", "0"):
            return False
    return default


def _clean_text(value: object, fallback: str = "") -> str:
    if not isinstance(value, str):
        return fallback
    cleaned = value.strip()
    return cleaned or fallback


def run_query_agent(
    query: str,
    llm_client,
    *,
    auto: bool,
    log: logging.Logger = logger,
) -> QueryPlan:
    """Run the single understanding (+ routing, when ``auto``) call.

    ``llm_client`` MUST be the application's active client (built from
    ``resolve_active_stack``) — this function never selects a model or provider.

    Never raises: any transport or parsing failure yields a degraded plan that
    keeps the request alive with the original query and the HYBRID default.
    """
    original = (query or "").strip()
    plan = QueryPlan(original_query=original, english_query=original)

    if not original:
        return plan
    if llm_client is None:
        plan.degraded = True
        plan.notes.append("no LLM client available")
        return plan

    try:
        resp = llm_client.generate(
            _USER_TEMPLATE.format(query=original), system=_SYSTEM_PROMPT
        )
        raw = getattr(resp, "text", "") or ""
    except Exception as e:  # noqa: BLE001 — degraded mode is the contract
        log.warning("[agent] call failed (%s); falling back to HYBRID", type(e).__name__)
        plan.degraded = True
        plan.notes.append(f"agent call failed: {type(e).__name__}")
        return plan

    data = _extract_json_object(raw)
    if data is None:
        log.warning("[agent] unparseable output; falling back to HYBRID")
        plan.degraded = True
        plan.notes.append("agent returned unparseable JSON")
        return plan

    plan.language = _clean_text(data.get("language"), "English")
    plan.supported = _coerce_bool(data.get("supported"), True)
    plan.response_language = _clean_text(
        data.get("response_language"), plan.language or "English"
    )
    # An empty/missing english_query must not blank the retrieval query.
    plan.english_query = _clean_text(data.get("english_query"), original)

    if auto:
        raw_route = data.get("route")
        plan.route = normalize_route(raw_route)
        if normalize_route(raw_route, default="") == "":
            plan.notes.append(f"invalid route {raw_route!r}; using HYBRID")
    else:
        # MANUAL: the caller's selection is authoritative. No routing here.
        plan.route = ROUTE_HYBRID

    if not plan.supported:
        # Nothing downstream may run; the caller stops the pipeline.
        plan.notes.append("language not supported by the serving model")

    return plan
