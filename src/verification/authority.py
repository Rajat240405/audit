"""Verification Authority (Phase 3 / WS3).

The ONE centralized verification engine for every retrieval mode and every
entry point:

  * ``/api/verify`` (post-answer and Cross-Verify) -> ``verify_answer``
  * ``/api/investigate`` (one claim)               -> ``investigate_claim``
  * GraphRAG / EvidenceSet                          -> ``verify``

All three use the same verdict rules (``semantic.py``): SUPPORTED ->
``verified``, CONTRADICTED -> ``rejected``, INSUFFICIENT_EVIDENCE ->
``insufficient``. Verdicts are evidence-grounded: exact containment is a
deterministic fast path, and the LLM judge decides semantic support from
passages selected from the supplied records. Every passage ID, source ID and
quote the judge cites is validated against the supplied records before it is
used.

Deterministic term checks (``text_support.grounding_report``) still run for
entities, figures and acronyms. They report presence only and never count as
fact verification.

Automatic rewrite (Deep mode only, unchanged behaviour): sentences that carry a
CONTRADICTED claim are removed or rewritten. Insufficient evidence never
removes or rewrites anything.

Graph mode adds integrity checks on the resolved provenance (``graph_checks``).

Model/provider agnostic: the LLM client is INJECTED. Light depth makes zero LLM
calls.
"""
from __future__ import annotations

import re
from typing import Optional

from src.verification import semantic as sem
from src.verification import text_support as ts
from src.verification.markdown_structure import (
    remove_list_and_heading_for_claims,
    remove_table_rows_for_claims,
    validate_rewrite,
)
from src.verification.contract import (
    MODE_GRAPH,
    MODE_HYBRID,
    STATUS_INSUFFICIENT,
    STATUS_REJECTED,
    STATUS_VERIFIED,
    ClaimVerdict,
    EvidenceItem,
    EvidenceSet,
    VerificationReport,
)

__all__ = ["VerificationAuthority"]

_INVESTIGATION_STATUS = {
    STATUS_VERIFIED: "supported",
    STATUS_REJECTED: "contradicted",
    STATUS_INSUFFICIENT: "insufficient",
}


class VerificationAuthority:
    """One verification engine for all retrieval modes and entry points."""

    def __init__(self, llm_client=None) -> None:
        # llm_client: anything with .generate(prompt=..., system=...) -> obj.text
        # (the project's LLMClient satisfies this; tests use scripted fakes).
        self._llm = llm_client

    # ── shared core ───────────────────────────────────────────────────────

    def _check(self, answer: str, sources: list[dict], depth: str, client):
        """Fact rows (semantic) + term rows (presence). Returns (rows, stats)."""
        judge = client if depth == "full" else None
        fact_rows, stats = sem.verify_factual_sentences(
            answer, sources, client=judge, depth=depth)
        for r in fact_rows:
            r["kind"] = "fact"
            r["found"] = r["status"] == STATUS_VERIFIED
        term_rows = []
        for g in ts.grounding_report(answer, sources):
            present = bool(g["found"])
            term_rows.append({
                "text": g["text"], "sentence": None, "kind": "term",
                "status": None, "found": present,
                "method": "text",
                "sources": [g["source"]] if g.get("source") else [],
                "quotes": [],
                "note": ("term present in the cited records (presence check, "
                         "not a fact check)") if present else
                        "term not found in the cited records",
                "rationale": "",
            })
        return fact_rows + term_rows, stats

    @staticmethod
    def _wire_row(r: dict) -> dict:
        """Response row for /api/verify. ``found`` = verified fact or present term."""
        sources = list(r.get("sources") or [])
        return {
            "text": r["text"],
            "found": bool(r.get("found")),
            "status": r.get("status"),
            "kind": r.get("kind", "fact"),
            "source": sources[0] if sources else None,
            "sources": sources,
            "quotes": list(r.get("quotes") or []),
            "sentence": r.get("sentence"),
            "method": r.get("method", "text"),
            "note": r.get("note", ""),
            "rationale": r.get("rationale", ""),
        }

    @staticmethod
    def _rejected_sentences(rows: list[dict]) -> list[str]:
        out: list[str] = []
        for r in rows:
            if r.get("kind") == "fact" and r.get("status") == STATUS_REJECTED:
                s = r.get("sentence") or r["text"]
                if s not in out:
                    out.append(s)
        return out

    def _rewrite(self, answer: str, rows: list[dict], sources: list[dict],
                 client) -> tuple[str, bool]:
        """Deep-mode rewrite of CONTRADICTED sentences. Returns (text, rewritten).

        A failing rewrite keeps the original answer (never silently drops text).
        """
        rejected = self._rejected_sentences(rows)
        if not rejected or client is None or not answer.strip():
            return answer, False
        try:
            rewrite = self.rewrite_answer(answer, rejected, sources, client)
        except Exception as e:  # noqa: BLE001 - keep the answer on any failure
            print(f"[verify] rewrite skipped ({type(e).__name__}: {e})")
            return answer, False
        if rewrite and rewrite.strip():
            final, _ = ts.remove_rejected_sentences(rewrite, rejected)
            return final, True
        return answer, False

    # ── public: normalized contract (graph and hybrid EvidenceSet) ────────

    def verify(
        self,
        evidence: EvidenceSet,
        answer: str,
        depth: str = "light",
        llm_client=None,
    ) -> VerificationReport:
        """Verify an answer against a normalized EvidenceSet.

        ``depth``: "light" = deterministic only (no LLM); "full" = + semantic
        judge and rewrite. Returns a VerificationReport.
        """
        client = llm_client or self._llm
        sources = evidence.as_sources()
        report = VerificationReport(method=depth)
        if not answer or not sources:
            return report

        rows, _stats = self._check(answer, sources, depth, client)
        for r in rows:
            report.claims.append(ClaimVerdict(
                claim=r["text"],
                status=(r["status"] if r.get("kind") == "fact"
                        else (STATUS_VERIFIED if r["found"] else STATUS_INSUFFICIENT)),
                source_doc_id=(r.get("sources") or [None])[0],
                method=r.get("method", "text"),
                note=r.get("note", ""),
            ))

        # citation filter (informational; frozen behavior)
        _filtered, dropped = ts.apply_citation_filter(answer, sources)
        report.citation_dropped = dropped

        # rewrite (full depth) — only CONTRADICTED sentences
        if depth == "full":
            final, rewritten = self._rewrite(answer, rows, sources, client)
            report.final_text = final
            report.judge_rewritten = rewritten
            report.judge_removed = [r["text"] for r in rows
                                    if r.get("kind") == "fact"
                                    and r.get("status") == STATUS_REJECTED]

        # citation/provenance integrity: a verified claim must cite a source
        # that is actually in the evidence set
        doc_ids = evidence.doc_ids()
        for c in report.claims:
            if c.status == STATUS_VERIFIED and c.source_doc_id not in doc_ids:
                c.status = STATUS_INSUFFICIENT
                c.note = (c.note + " | citation not in evidence set").strip(" |")
        report.provenance_ok = all(
            c.source_doc_id in doc_ids
            for c in report.claims if c.status == STATUS_VERIFIED)

        # graph integrity checks (graph mode only)
        if evidence.mode == MODE_GRAPH:
            report.graph_checks = self._graph_checks(evidence, report)
            if not report.graph_checks.get("ok", True):
                self._reject_on_broken_provenance(evidence, report)
        return report

    # ── public: /api/verify response shape ────────────────────────────────

    def verify_answer(
        self,
        answer: str,
        sources: list[dict],
        depth: str = "light",
        llm_client=None,
    ) -> dict:
        """The /api/verify engine.

        Returns {text, grounding, judge_rewritten, judge_removed_count,
        judge_removed, citation_dropped_count, verification_stats, method} on
        success, or the {"error": ...} shape on missing input or internal
        failure. ``grounding`` rows carry ``kind`` ("fact" | "term") and, for facts,
        ``status`` ("verified" | "rejected" | "insufficient"), validated ``sources``
        and ``quotes``, and a ``note``.
        """
        client = llm_client or self._llm
        if not answer or not sources:
            return {"text": answer, "grounding": [], "judge_rewritten": False,
                    "judge_removed_count": 0, "error": "missing answer or sources"}
        try:
            rows, stats = self._check(answer, sources, depth, client)
            _filtered, citation_dropped = ts.apply_citation_filter(answer, sources)
            final_text, judge_rewritten = answer, False
            if depth == "full":
                final_text, judge_rewritten = self._rewrite(answer, rows, sources, client)
            rejected = [r for r in rows
                        if r.get("kind") == "fact" and r.get("status") == STATUS_REJECTED]
            return {
                "text": final_text,
                "grounding": [self._wire_row(r) for r in rows],
                "judge_rewritten": judge_rewritten,
                "judge_removed_count": len(rejected),
                "judge_removed": [r["text"] for r in rejected][:20],
                "citation_dropped_count": len(citation_dropped),
                "verification_stats": stats.to_dict(),
                "method": depth,
            }
        except Exception as e:  # noqa: BLE001 - never break the app
            import traceback
            print(f"[api/verify] failed ({type(e).__name__}: {e})")
            print(traceback.format_exc(limit=3))
            return {"text": answer, "grounding": [], "judge_rewritten": False,
                    "judge_removed_count": 0,
                    "error": f"{type(e).__name__}: {str(e)[:200]}"}

    # ── public: /api/investigate (one claim, same verdict rules) ──────────

    def investigate_claim(self, claim: str, sources: list[dict],
                          llm_client=None) -> dict:
        """Judge ONE claim with the same rules as /api/verify.

        Returns {status: supported|contradicted|insufficient, rationale, note,
        sources, quotes, method}. Report-only: nothing is rewritten.
        """
        client = llm_client or self._llm
        row, _stats = sem.verify_single_claim(claim, sources, client)
        return {
            "status": _INVESTIGATION_STATUS.get(row["status"], "insufficient"),
            "rationale": row.get("rationale") or row.get("note") or "",
            "note": row.get("note", ""),
            "sources": list(row.get("sources") or []),
            "quotes": list(row.get("quotes") or []),
            "method": row.get("method", ""),
        }

    def rewrite_answer(self, answer: str, rejected_claims: list[str],
                       sources: list[dict], client=None) -> str:
        """LLM rewrite of the answer WITHOUT the judge-rejected claims.

        Public API (WS3): the server's ``_llm_rewrite_answer`` delegates here.
        ``client`` defaults to the injected constructor client. Raises on
        failure — the caller falls back to the original."""
        client = client or self._llm
        if client is None:
            raise RuntimeError("no LLM client for rewrite")

        # STEP 1 — deterministic, structure-aware removal FIRST. Claims that
        # map unambiguously to a single markdown table row are dropped without
        # involving the LLM at all, so a table can never be regenerated (and
        # corrupted) just to delete one row. Header/delimiter rows are never
        # candidates; ambiguous claims fall through to the LLM step.
        working, applied, remaining = remove_table_rows_for_claims(
            answer, rejected_claims)
        # Claims the table pass could not place are offered to the list /
        # heading pass, which applies the same one-unit-only rule. Its
        # `allowed` signatures are what let validate_rewrite accept a
        # DELIBERATE heading removal while still rejecting an LLM that
        # silently flattens one.
        allowed_removals: set[tuple] = set()
        if remaining:
            lr = remove_list_and_heading_for_claims(working, remaining)
            if lr.applied:
                working = lr.text
                applied = list(applied) + list(lr.applied)
                allowed_removals |= set(lr.allowed)
            remaining = list(lr.unmapped)
        if applied and not remaining:
            ok, why = validate_rewrite(answer, working,
                                       allowed_removals=allowed_removals)
            if ok:
                return working.strip()
            # deterministic edit somehow broke structure — signal "no safe
            # rewrite" ("" ) so the caller keeps the original verbatim and
            # does NOT report judge_rewritten.
            print(f"[verify] deterministic removal rejected: {why}")
            return ""
        rejected_claims = remaining or rejected_claims

        src_blocks = []
        for i, s in enumerate(sources[:6], start=1):
            ans = (s.get("answer") or "")[:1500]
            src_blocks.append(f"[Source {i}] {s.get('doc_id','')}\n{ans}")
        src_text = "\n\n".join(src_blocks)

        rejected_lines = "\n".join(f"- {c}" for c in rejected_claims)
        prompt = (
            "You are an evidence auditor. Below is a draft answer and a list of "
            "claims that were REJECTED because they are NOT supported by the source "
            "documents.\n\n"
            f"REJECTED CLAIMS:\n{rejected_lines}\n\n"
            f"DRAFT ANSWER:\n{working}\n\n"
            f"SOURCES:\n{src_text}\n\n"
            "Rewrite the draft answer so that it:\n"
            "1. Removes every statement based on a rejected claim.\n"
            "2. Keeps all supported statements verbatim where possible.\n"
            "3. Does NOT add any new facts, names, or figures.\n"
            "4. Preserves markdown formatting and keeps the answer free of "
            "[Source N] citation markers (attribution is handled separately).\n"
            "If everything was rejected, say the context does not support the "
            "claim.\n"
            "Return ONLY the rewritten answer, no commentary."
        )
        resp = client.generate(
            prompt=prompt,
            system=(
                "You are an evidence-verification assistant. Rewrite the answer to "
                "remove unsupported claims. Never invent facts."
            ),
        )
        candidate = (resp.text or "").strip()

        # STEP 3 — post-condition. A prompt asking the model to "preserve
        # markdown" is not a guarantee; this is. If the rewrite reshaped a
        # table, touched fenced code, or otherwise changed block structure
        # beyond dropping whole blocks, DISCARD it and keep the original —
        # a slightly over-inclusive answer beats a corrupted one.
        ok, why = validate_rewrite(working, candidate)
        if not ok:
            print(f"[verify] rewrite discarded (structure): {why}")
            # "" = no safe rewrite. Returning `working` here would look like a
            # successful rewrite to the caller and flip judge_rewritten, even
            # though the text is unchanged.
            return "" if working.strip() == answer.strip() else working.strip()
        return candidate

    # Backward-compatibility alias for the pre-WS3 private name.
    def _rewrite_answer(self, answer: str, rejected_claims: list[str],
                        sources: list[dict], client=None) -> str:
        return self.rewrite_answer(answer, rejected_claims, sources, client)

    # ── graph integrity (WS2 provenance re-verified here) ─────────────────

    def _graph_checks(self, evidence: EvidenceSet,
                      report: VerificationReport) -> dict:
        """Validate the resolved graph provenance carried by the evidence.

        Checks (per spec §12/§13 + Phase 2 guarantees):
          * every cited fact exists and is resolved
          * LLM-origin facts carry verbatim evidence in >=1 support
          * zero-support facts are never grounded
          * supports reference documents known to this run
          * deterministic facts name a source field
        Returns {"ok": bool, "problems": [{item, fact_key, issue}]}.
        """
        problems: list[dict] = []
        known_docs = evidence.doc_ids()
        for item in evidence.items:
            if item.mode != MODE_GRAPH or item.graph is None:
                continue
            # A cited fact whose provenance could NOT be resolved through the
            # store is missing — never silently ignored, never trusted.
            resolved = {f.get("fact_key") for f in item.graph.facts}
            for fk in item.graph.fact_keys:
                if fk not in resolved:
                    problems.append({"item": item.doc_id, "fact_key": fk,
                                     "issue": "fact_missing"})
            for fact in item.graph.facts:
                fk = fact.get("fact_key")
                if not fk:
                    problems.append({"item": item.doc_id, "fact_key": None,
                                     "issue": "fact_missing"})
                    continue
                if int(fact.get("doc_count", 0)) <= 0:
                    problems.append({"item": item.doc_id, "fact_key": fk,
                                     "issue": "unsupported_fact"})
                    continue
                supports = fact.get("supports") or []
                if fact.get("origin") == "llm" and not any(
                        (s.get("evidence") or "").strip() for s in supports):
                    problems.append({"item": item.doc_id, "fact_key": fk,
                                     "issue": "llm_fact_ungrounded"})
                if fact.get("origin") == "deterministic" and \
                        not (fact.get("source_field") or "").strip():
                    problems.append({"item": item.doc_id, "fact_key": fk,
                                     "issue": "missing_source_field"})
                for s in supports:
                    if s.get("doc_key") not in known_docs:
                        problems.append({"item": item.doc_id, "fact_key": fk,
                                         "issue": "unknown_support_doc",
                                         "doc_key": s.get("doc_key")})
        return {"ok": not problems, "problems": problems}

    def _reject_on_broken_provenance(self, evidence: EvidenceSet,
                                     report: VerificationReport) -> None:
        """A graph claim whose provenance is broken is NOT presented as
        grounded: downgrade verified→rejected for claims citing only broken
        facts."""
        broken_items = {p["item"] for p in
                        (report.graph_checks or {}).get("problems", [])}
        if not broken_items:
            return
        for c in report.claims:
            if c.status == STATUS_VERIFIED and c.source_doc_id in broken_items:
                c.status = STATUS_REJECTED
                c.note = (c.note + " | graph provenance broken").strip(" |")
