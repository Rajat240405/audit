"""Verification Authority (Phase 3 / WS3).

The ONE centralized verification engine for every retrieval mode:
Hybrid RAG and GraphRAG evidence are normalized into the shared contract
(``contract.py``) and verified here — there is no per-mode verifier.

Deterministic core: claim extraction + strict normalized/alias-aware support
(``text_support``). Optional LLM pass (``depth="full"``): the judge re-checks
only the claims the regex could NOT verify, with the FROZEN verbatim-trust
gate — a judge "supported" is accepted only when the cited source actually
contains the claim (same normalized matcher); on any judge failure the regex
verdicts stand. Optional rewrite removes judge-rejected claims, followed by
the deterministic rejected-sentence safety net.

Model/provider agnostic: the LLM client is INJECTED (the server passes its
active client built by the shared generation architecture; tests pass a
scripted fake; light depth makes zero LLM calls).

Graph mode adds integrity checks on the resolved provenance
(``graph_checks``): every cited fact must exist, LLM-origin facts must carry
verbatim evidence in at least one support, zero-support facts are never
presented as grounded, supports must reference known documents, and
deterministic facts must name their source field.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from src.verification import text_support as ts
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

_JUDGE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class VerificationAuthority:
    """One verification engine for all retrieval modes (see module docstring)."""

    def __init__(self, llm_client=None) -> None:
        # llm_client: anything with .generate(prompt=..., system=...) -> obj.text
        # (the project's LLMClient satisfies this; tests use scripted fakes).
        self._llm = llm_client

    # ── public: normalized contract (both modes) ──────────────────────────

    def verify(
        self,
        evidence: EvidenceSet,
        answer: str,
        depth: str = "light",
        llm_client=None,
    ) -> VerificationReport:
        """Verify an answer against a normalized EvidenceSet.

        ``depth``: "light" = deterministic only (no LLM); "full" = + LLM
        judge/rewrite. Returns a VerificationReport (claims + sufficiency +
        provenance + graph checks)."""
        client = llm_client or self._llm
        sources = evidence.as_sources()
        report = VerificationReport(method=depth)
        if not answer or not sources:
            return report

        # 1) deterministic grounding (shared core)
        grounding = ts.grounding_report(answer, sources)
        for g in grounding:
            if g["found"]:
                status, method = STATUS_VERIFIED, "text"
            elif str(g.get("note", "")).startswith("rejected by LLM judge"):
                status, method = STATUS_REJECTED, "llm_judge"
            else:
                status, method = STATUS_INSUFFICIENT, "text"
            report.claims.append(ClaimVerdict(
                claim=g["text"], status=status,
                source_doc_id=g.get("source"), method=method,
                note=g.get("note", "")))

        # 2) citation filter (informational; frozen behavior)
        _filtered, dropped = ts.apply_citation_filter(answer, sources)
        report.citation_dropped = dropped

        # 3) LLM judge — full depth only, only claims the regex missed
        if depth == "full" and any(not g.get("found") for g in grounding):
            grounding = self._judge_claims(grounding, sources, client)
            report.claims = self._claims_from_grounding(grounding)

        # 4) rewrite (full depth) — remove judge-rejected claims
        rejected = [
            c.claim for c in report.claims
            if c.status == STATUS_REJECTED
            and str(c.note).startswith("rejected by LLM judge")
        ]
        if depth == "full" and rejected:
            rewrite = self._rewrite_answer(answer, rejected, sources, client)
            if rewrite and rewrite.strip():
                report.final_text = rewrite
                report.judge_rewritten = True
                report.final_text, _ = ts.remove_rejected_sentences(
                    report.final_text, rejected)
            report.judge_removed = rejected

        # 5) citation/provenance integrity: a verified claim must cite a
        #    source that is actually in the evidence set
        doc_ids = evidence.doc_ids()
        for c in report.claims:
            if c.status == STATUS_VERIFIED and c.source_doc_id not in doc_ids:
                c.status = STATUS_INSUFFICIENT
                c.note = (c.note + " | citation not in evidence set").strip(" |")
        report.provenance_ok = all(
            c.source_doc_id in doc_ids
            for c in report.claims if c.status == STATUS_VERIFIED)

        # 6) graph integrity checks (graph mode only)
        if evidence.mode == MODE_GRAPH:
            report.graph_checks = self._graph_checks(evidence, report)
            if not report.graph_checks.get("ok", True):
                # facts presented as grounded but broken → reject their claims
                self._reject_on_broken_provenance(evidence, report)
        return report

    # ── public: the frozen /api/verify engine shape ───────────────────────

    def verify_answer(
        self,
        answer: str,
        sources: list[dict],
        depth: str = "light",
        llm_client=None,
    ) -> dict:
        """Behavior-identical to the server's pre-Phase-3 verify_answer.

        Same input/output contract (frozen in Phase 1 WS0): returns
        {text, grounding, judge_rewritten, judge_removed_count, judge_removed,
        citation_dropped_count} on success, or the {"error": ...} shape on
        missing input / internal failure. The server delegates here."""
        client = llm_client or self._llm
        if not answer or not sources:
            return {"text": answer, "grounding": [], "judge_rewritten": False,
                    "judge_removed_count": 0, "error": "missing answer or sources"}
        try:
            grounding = ts.grounding_report(answer, sources)
            _filtered, citation_dropped = ts.apply_citation_filter(answer, sources)
            if depth == "full" and grounding and any(not c.get("found") for c in grounding):
                grounding = self._judge_claims(grounding, sources, client)
            rejected_claims = [
                c["text"] for c in grounding
                if (not c.get("found"))
                and str(c.get("note", "")).startswith("rejected by LLM judge")
            ]
            final_text = answer
            judge_rewritten = False
            if depth == "full" and rejected_claims and answer.strip():
                rewrite = self._rewrite_answer(answer, rejected_claims, sources, client)
                if rewrite and rewrite.strip():
                    final_text = rewrite
                    judge_rewritten = True
                    # deterministic safety net
                    final_text, _ = ts.remove_rejected_sentences(
                        final_text, rejected_claims)
            return {
                "text": final_text,
                "grounding": grounding,
                "judge_rewritten": judge_rewritten,
                "judge_removed_count": len(rejected_claims),
                "judge_removed": rejected_claims[:20],
                "citation_dropped_count": len(citation_dropped),
            }
        except Exception as e:  # noqa: BLE001 - never break the app
            import traceback
            print(f"[api/verify] failed ({type(e).__name__}: {e})")
            print(traceback.format_exc(limit=3))
            return {"text": answer, "grounding": [], "judge_rewritten": False,
                    "judge_removed_count": 0,
                    "error": f"{type(e).__name__}: {str(e)[:200]}"}

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _claims_from_grounding(grounding: list[dict]) -> list[ClaimVerdict]:
        out = []
        for g in grounding:
            if g["found"]:
                status = STATUS_VERIFIED
            elif str(g.get("note", "")).startswith("rejected by LLM judge"):
                status = STATUS_REJECTED
            else:
                status = STATUS_INSUFFICIENT
            method = "llm_judge" if g.get("note") else "text"
            out.append(ClaimVerdict(
                claim=g["text"], status=status, source_doc_id=g.get("source"),
                method=method, note=g.get("note", "")))
        return out

    def _judge_claims(self, claims: list[dict], sources: list[dict],
                      client) -> list[dict]:
        """Verify flagged claims with the LLM judge against the sources.

        Returns the claims list with updated ``found`` / ``source`` / ``note``.
        On any failure (LLM offline, parse error) returns the original claims
        unchanged — the regex verdicts remain authoritative."""
        if client is None:
            return claims  # no LLM available: regex verdicts stand
        if not claims or not sources:
            return claims
        # Only judge claims the regex could NOT verify (the interesting ones).
        pending = [c for c in claims if not c.get("found")]
        if not pending:
            return claims

        try:
            # Compact source text: id + answer (truncated) per doc.
            src_blocks = []
            for i, s in enumerate(sources[:6], start=1):
                ans = (s.get("answer") or "")[:1500]
                src_blocks.append(f"[Source {i}] {s.get('doc_id','')}\n{ans}")
            src_text = "\n\n".join(src_blocks)

            claim_lines = "\n".join(
                f"{i}. {c['text']}" for i, c in enumerate(pending, start=1))
            prompt = (
                "You are a strict evidence auditor. Below are source documents and a "
                "list of claims. For EACH claim decide whether it is SUPPORTED by the "
                "sources — supported means the claim's fact appears in the source text "
                "verbatim or is directly implied. If a claim names an organization, "
                "programme, or figure that does not appear in the sources, it is NOT "
                "supported.\n"
                "Return ONLY JSON (no prose):\n"
                '{"verdicts":[{"index":1,"supported":true,"source":"18-3-2571"},'
                '{"index":2,"supported":false,"source":null}]}\n\n'
                f"SOURCES:\n{src_text}\n\nCLAIMS:\n{claim_lines}"
            )
            resp = client.generate(
                prompt=prompt,
                system=(
                    "You are an evidence-verification assistant. Answer strictly with "
                    "the requested JSON. Never invent claims or sources."
                ),
            )
            raw = resp.text
            m = _JUDGE_JSON_RE.search(raw or "")
            if not m:
                return claims
            data = json.loads(m.group(0))
            verdicts = data.get("verdicts") or []

            by_index = {}
            for v in verdicts:
                try:
                    by_index[int(v.get("index"))] = v
                except (TypeError, ValueError):
                    continue

            out = []
            pending_i = 0
            for c in claims:
                if c.get("found"):
                    out.append(c)  # already verified verbatim by regex — keep
                    continue
                pending_i += 1  # index within the pending list the judge saw
                v = by_index.get(pending_i)
                if v is None:
                    out.append(c)  # judge gave no verdict — keep regex verdict
                    continue
                supported = bool(v.get("supported"))
                src = v.get("source") or None
                # CRITICAL: the judge's "supported" is only trusted if it can
                # name a source that ACTUALLY contains the claim. The judge is
                # the same model that hallucinated the claim (e.g. "VSSC" — it
                # "remembers" VSSC is ISRO's space centre), so a bare
                # "supported: true" must NOT override a verbatim miss. Only
                # accept the judge's verdict when the claim text appears in
                # the cited source's text (same normalized+alias matcher).
                judge_trusted = False
                if supported and src:
                    for s in sources:
                        if s.get("doc_id") == src:
                            stxt = ts.normalize(
                                f"{s.get('question','')} {s.get('answer','')}")
                            if ts.claim_supported(c["text"], stxt):
                                judge_trusted = True
                            break
                if supported and not judge_trusted:
                    out.append({
                        "text": c["text"],
                        "found": False,
                        "source": None,
                        "note": "rejected by LLM judge (no verbatim source support)",
                    })
                else:
                    out.append({
                        "text": c["text"],
                        "found": judge_trusted,
                        "source": src if judge_trusted else None,
                        "note": ("verified by LLM judge" if judge_trusted
                                 else "rejected by LLM judge"),
                    })
            return out
        except Exception as e:  # noqa: BLE001 - judge must never break the flow
            import traceback
            print(f"[llm-judge] failed ({type(e).__name__}: {e}) — using regex verdicts")
            print(traceback.format_exc(limit=3))
            return claims

    def _rewrite_answer(self, answer: str, rejected_claims: list[str],
                        sources: list[dict], client) -> str:
        """LLM rewrite of the answer WITHOUT the judge-rejected claims.

        Raises on failure — the caller falls back to the original."""
        if client is None:
            raise RuntimeError("no LLM client for rewrite")
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
            f"DRAFT ANSWER:\n{answer}\n\n"
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
        return (resp.text or "").strip()

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
