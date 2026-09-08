"""Verification Authority package (Phase 3 / WS3).

ONE centralized verification engine for every retrieval mode. Hybrid RAG and
GraphRAG both normalize their evidence into the shared contract
(:mod:`src.verification.contract`) via the adapters in
:mod:`src.verification.sources`, and both are verified by
:class:`~src.verification.authority.VerificationAuthority`. There is no
per-mode verifier.

Public surface::

    from src.verification import VerificationAuthority, hybrid_sources
    report = VerificationAuthority(llm_client=client).verify(
        hybrid_sources(sources), answer, depth="light")

The deterministic text core (:mod:`src.verification.text_support`) is exposed
both as a module (``from src.verification import text_support as ts``) and
through the re-exported functions below; the server binds its legacy private
names to these so the Phase 1 frozen behavior runs the same code path.

The LLM client is always INJECTED — this package hardcodes no provider and no
model, and registers nothing of its own.
"""
from __future__ import annotations

from src.verification import text_support
from src.verification.authority import VerificationAuthority
from src.verification.contract import (
    MODE_GRAPH,
    MODE_HYBRID,
    STATUS_INSUFFICIENT,
    STATUS_REJECTED,
    STATUS_VERIFIED,
    Claim,
    ClaimVerdict,
    EvidenceItem,
    EvidenceSet,
    GraphProvenance,
    VerificationReport,
)
from src.verification.sources import graph_results, hybrid_sources
from src.verification.text_support import (
    ALIAS_GROUPS,
    apply_citation_filter,
    claim_candidates,
    claim_supported,
    extract_claims,
    grounding_report,
    normalize,
    remove_rejected_sentences,
    singularize,
)

__all__ = [
    # engine
    "VerificationAuthority",
    # adapters
    "hybrid_sources",
    "graph_results",
    # contract
    "Claim",
    "ClaimVerdict",
    "EvidenceItem",
    "EvidenceSet",
    "GraphProvenance",
    "VerificationReport",
    "STATUS_VERIFIED",
    "STATUS_REJECTED",
    "STATUS_INSUFFICIENT",
    "MODE_HYBRID",
    "MODE_GRAPH",
    # deterministic text core
    "text_support",
    "ALIAS_GROUPS",
    "extract_claims",
    "normalize",
    "singularize",
    "claim_candidates",
    "claim_supported",
    "grounding_report",
    "apply_citation_filter",
    "remove_rejected_sentences",
]
