"""Evidence-grounded semantic verification of factual claims.

Replaces the verbatim-only judge gate. A factual sentence is no longer
"unsupported" just because the LLM paraphrased or combined several records.
Instead:

  1. Exact containment is kept as a deterministic FAST PATH. A sentence (or an
     atomic claim) that appears verbatim in a cited record is SUPPORTED with
     no LLM call. Exact text is also the test used to validate quotes.
  2. Non-verbatim sentences are split into atomic claims (one LLM call for all
     sentences, validated; on any doubt the whole sentence is kept as one
     claim, and the judge is asked to check every component).
  3. For each claim, relevant passages are SELECTED from the supplied records
     by deterministic lexical scoring. No corpus is sent to the model. A claim
     with no lexically related passage is INSUFFICIENT without an LLM call.
  4. A batched judge call returns a verdict per claim with passage IDs and
     quotes. Every passage ID is checked against the passages shown for that
     claim, every source ID against the supplied records, and every quote
     against the ORIGINAL record text (normalized containment). The generated
     claim itself need not be verbatim.
  5. Deterministic guards apply to SUPPORTED only (see ``supported_guards``).
     They are conservative: a mismatch downgrades to INSUFFICIENT, never to
     CONTRADICTED.

Verdicts (contract names): SUPPORTED -> ``verified``, CONTRADICTED ->
``rejected``, INSUFFICIENT_EVIDENCE -> ``insufficient``. Missing or failed
judgement is always INSUFFICIENT, never CONTRADICTED.

Model/provider agnostic: the client is injected and only ``.generate(prompt=,
system=)`` is used. No provider or model is named here.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.verification import text_support as ts

__all__ = [
    "Passage",
    "VerifyStats",
    "build_passages",
    "select_passages",
    "decompose_sentences",
    "verify_claims",
    "verify_factual_sentences",
    "verify_single_claim",
    "supported_guards",
    "parse_verdicts",
    "VERIFIED", "REJECTED", "INSUFFICIENT",
]

# Contract status names (see src/verification/contract.py).
VERIFIED = "verified"
REJECTED = "rejected"
INSUFFICIENT = "insufficient"

# ─────────────────────────────────────────────────────────────────────────────
# Limits. Each one bounds latency or prompt size. None of them changes a
# verdict on its own: exceeding a limit yields INSUFFICIENT, never SUPPORTED.
# ─────────────────────────────────────────────────────────────────────────────

#: Atomic claims judged per answer. Rationale: keeps a long answer to a few
#: judge calls. Claims over the limit are reported as not checked.
MAX_CLAIMS = int(os.environ.get("VERIFY_MAX_CLAIMS", "24"))
#: Claims per judge call. Rationale: 5 claims x <=4 passages x <=1200 chars
#: keeps each prompt small enough for a local model's context.
CLAIMS_PER_JUDGE_CALL = 5
#: Sentences per decomposition call.
SENTENCES_PER_DECOMPOSE_CALL = 10
#: Passages shown per claim, and the character budget per claim.
MAX_PASSAGES_PER_CLAIM = 4
MAX_EVIDENCE_CHARS_PER_CLAIM = 2400
#: Passages (sentences and 2-sentence windows) are capped at this length.
MAX_PASSAGE_CHARS = 1200
#: Wall-clock budget for all judge calls of one answer, and the per-call
#: timeout applied to the client while a call is in flight.
JUDGE_BUDGET_SECONDS = float(os.environ.get("VERIFY_JUDGE_BUDGET_SECONDS", "120"))
JUDGE_CALL_TIMEOUT_SECONDS = int(os.environ.get("VERIFY_JUDGE_CALL_TIMEOUT_SECONDS", "60"))
#: A SUPPORTED verdict must be anchored in most of the claim's content words.
#: Rationale: a claim sharing only one keyword with a quote (an acronym, a
#: place name) is not supported by that quote even if the judge says so. The
#: value 0.5 is a conservative lower bound chosen by hand, not calibrated. It
#: can downgrade a valid heavily-paraphrased claim to INSUFFICIENT (a false
#: negative), but it cannot turn an unrelated claim into SUPPORTED.
MIN_CLAIM_COVERAGE = 0.5
#: A quote must have this many normalized tokens (and >= 12 characters).
#: Rationale: a one-word quote such as an acronym is not evidence of a fact.
MIN_QUOTE_TOKENS = 3
MIN_QUOTE_CHARS = 12

_VERDICTS = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}

# Words whose presence or absence can flip a claim's meaning.
_NEGATION = {"not", "no", "never", "none", "neither", "nor", "without",
             "cannot", "unable", "lack", "lacks", "nothing"}
_QUALIFIERS = {"all", "every", "each", "some", "most", "majority", "many", "few",
               "only", "least", "more", "less", "over", "under", "above", "below",
               "approximately", "about", "around", "nearly", "almost", "up",
               "exactly", "minimum", "maximum"}
_UNITS = {"km", "m", "mm", "cm", "kg", "tonne", "tonnes", "percent", "crore",
          "lakh", "million", "billion", "hour", "hours", "day", "days", "year",
          "years", "mw", "gw", "kw", "rupee", "rupees", "degree", "degrees",
          "hectare", "hectares", "sq"}
_STOPWORDS = {"the", "of", "and", "in", "to", "for", "is", "are", "was", "were",
              "by", "with", "on", "at", "from", "that", "this", "as", "be", "has",
              "have", "had", "its", "it", "their", "which", "a", "an", "or", "into",
              "also", "such", "been", "being", "they", "these", "those", "there"}

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


# ─────────────────────────────────────────────────────────────────────────────
# Passages
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Passage:
    pid: str            # "P1", "P2", ... unique within one verification run
    doc_id: str
    text: str           # shown to the judge (truncated to MAX_PASSAGE_CHARS)
    tokens: frozenset = field(default_factory=frozenset)


@dataclass
class Source:
    doc_id: str
    full_norm: str      # normalized question + answer, for quote validation


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut or text[:limit]


def content_tokens(text: str) -> set[str]:
    """Normalized content tokens (singular, no stopwords). Numbers are kept."""
    out: set[str] = set()
    for tok in ts.normalize(text).split():
        if tok in _STOPWORDS:
            continue
        if not tok.isdigit() and len(tok) < 3:
            continue
        out.add(ts.singularize(tok) if not tok.isdigit() else tok)
    return out


def build_passages(sources: list[dict]) -> tuple[list[Passage], dict[str, Source]]:
    """Split each supplied record into sentence passages and 2-sentence windows.

    Only the supplied records are used. Returns the passages and a map of
    doc_id -> Source (the full normalized record text, for quote validation).
    """
    passages: list[Passage] = []
    docs: dict[str, Source] = {}
    n = 0
    for s in sources:
        doc_id = str(s.get("doc_id") or "")
        if not doc_id:
            continue
        raw = f"{s.get('question', '') or ''}\n{s.get('answer', '') or ''}"
        docs[doc_id] = Source(doc_id=doc_id, full_norm=ts.normalize(raw))
        sents = [x.strip() for x in _SENT_SPLIT_RE.split(raw) if x and x.strip()]
        sents = [x for x in sents if len(ts.normalize(x).split()) >= 3]
        # single sentences, then 2-sentence windows (joint statements)
        texts = sents + [sents[i] + " " + sents[i + 1] for i in range(len(sents) - 1)]
        for text in texts:
            shown = _truncate(text, MAX_PASSAGE_CHARS)
            n += 1
            passages.append(Passage(
                pid=f"P{n}", doc_id=doc_id, text=shown,
                tokens=frozenset(content_tokens(shown))))
    return passages, docs


def select_passages(claim: str, passages: list[Passage]) -> list[Passage]:
    """Passages most related to the claim, spread across documents.

    Score = sum of IDF weights of shared content tokens. A shared number counts
    double, because figures are the most decisive part of a claim. The best
    passage of each document is taken first, so joint support from several
    records is visible to the judge. Passages that are substrings of an already
    chosen passage are skipped. Empty result = no lexical candidate.
    """
    ctoks = content_tokens(claim)
    if not ctoks or not passages:
        return []
    n_passages = len(passages)
    df: dict[str, int] = {}
    for p in passages:
        for t in p.tokens:
            df[t] = df.get(t, 0) + 1

    def weight(t: str) -> float:
        base = math.log((n_passages + 1) / (df.get(t, 0) + 1)) + 1.0
        return base * (2.0 if t.isdigit() else 1.0)

    scored: list[tuple[float, Passage]] = []
    for p in passages:
        shared = ctoks & p.tokens
        if shared:
            scored.append((sum(weight(t) for t in shared), p))
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1].pid))

    chosen: list[Passage] = []
    chars = 0

    def take(p: Passage) -> bool:
        nonlocal chars
        if len(chosen) >= MAX_PASSAGES_PER_CLAIM:
            return False
        if chars + len(p.text) > MAX_EVIDENCE_CHARS_PER_CLAIM and chosen:
            return False
        norm = ts.normalize(p.text)
        if any(norm in ts.normalize(c.text) or ts.normalize(c.text) in norm
               for c in chosen):
            return False
        chosen.append(p)
        chars += len(p.text)
        return True

    seen_docs: set[str] = set()
    for _score, p in scored:           # pass 1: best passage per document
        if p.doc_id not in seen_docs:
            if take(p):
                seen_docs.add(p.doc_id)
    for _score, p in scored:           # pass 2: fill by score
        if len(chosen) >= MAX_PASSAGES_PER_CLAIM:
            break
        take(p)
    return chosen


# ─────────────────────────────────────────────────────────────────────────────
# LLM plumbing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VerifyStats:
    decompose_calls: int = 0
    judge_calls: int = 0
    claims_exact: int = 0
    claims_judged: int = 0
    claims_no_candidates: int = 0
    claims_not_checked: int = 0
    prompt_chars: int = 0
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "decompose_calls", "judge_calls", "claims_exact", "claims_judged",
            "claims_no_candidates", "claims_not_checked", "prompt_chars",
            "elapsed_ms")}


def _parse_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    body = raw.strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.MULTILINE).strip()
    m = _JSON_OBJ_RE.search(body)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


class _TimeoutScope:
    """Temporarily lower the client's per-call timeout while a judge call runs.

    Clients without a ``timeout_seconds`` attribute (scripted test clients) are
    left alone. The original value is always restored.
    """

    def __init__(self, client: Any) -> None:
        self.client = client
        self.saved: Any = None

    def __enter__(self):
        if hasattr(self.client, "timeout_seconds"):
            try:
                self.saved = self.client.timeout_seconds
                current = int(self.saved or JUDGE_CALL_TIMEOUT_SECONDS)
                self.client.timeout_seconds = min(current, JUDGE_CALL_TIMEOUT_SECONDS)
            except (TypeError, ValueError, AttributeError):
                self.saved = None
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            try:
                self.client.timeout_seconds = self.saved
            except AttributeError:
                pass
        return False


def _generate(client: Any, prompt: str, system: str, stats: VerifyStats) -> str:
    stats.prompt_chars += len(prompt)
    with _TimeoutScope(client):
        resp = client.generate(prompt=prompt, system=system)
    return (getattr(resp, "text", None) or "") if resp is not None else ""


# ─────────────────────────────────────────────────────────────────────────────
# Decomposition: sentence -> atomic claims (validated)
# ─────────────────────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = (
    "You split factual sentences into independently checkable claims. "
    "You never add facts. Answer strictly with the requested JSON."
)


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", ts.normalize(text)))


def _has_negation(text: str) -> bool:
    toks = re.findall(r"[a-z']+", text.lower())
    return any(t in _NEGATION or t.endswith("n't") for t in toks)


def _validate_atoms(sentence: str, atoms: list[str]) -> Optional[list[str]]:
    """Return the atoms if they faithfully cover the sentence, else None.

    Checks: non-empty, at most 6 claims, no number that the sentence does not
    contain, every sentence number kept, and negation kept if the sentence has
    it. Any failure returns None and the caller keeps the whole sentence.
    """
    clean: list[str] = []
    for a in atoms:
        if not isinstance(a, str):
            return None
        a = " ".join(a.split())
        if not a or len(a) > 300:
            return None
        if a not in clean:
            clean.append(a)
    if not clean or len(clean) > 6:
        return None
    sent_nums = _numbers(sentence)
    atom_nums: set[str] = set()
    for a in clean:
        nums = _numbers(a)
        if not nums <= sent_nums:
            return None            # the model introduced a figure
        atom_nums |= nums
    if not sent_nums <= atom_nums:
        return None                # the model dropped a figure
    if _has_negation(sentence) and not any(_has_negation(a) for a in clean):
        return None                # the model dropped a negation
    return clean


def decompose_sentences(sentences: list[str], client: Any,
                        stats: VerifyStats) -> dict[str, list[str]]:
    """One batched call (in chunks) to split sentences into atomic claims.

    Returns sentence -> atomic claims. A sentence whose decomposition is
    missing, malformed or fails validation maps to ``[sentence]``.
    """
    out: dict[str, list[str]] = {s: [s] for s in sentences}
    for start in range(0, len(sentences), SENTENCES_PER_DECOMPOSE_CALL):
        chunk = sentences[start:start + SENTENCES_PER_DECOMPOSE_CALL]
        lines = "\n".join(f"S{i}: {s}" for i, s in enumerate(chunk, start=1))
        prompt = (
            "TASK: DECOMPOSE\n"
            "Split each sentence into the independently checkable factual claims it "
            "states. Rules:\n"
            "- Split only where the sentence joins separate facts. A single fact "
            "stays one claim.\n"
            "- Each claim must be understandable alone and keep every name, number, "
            "date, unit, negation and qualifier (all, some, only, up to, ...) from "
            "the sentence.\n"
            "- Do not add facts. Do not generalise. Do not drop any part.\n"
            "- Skip headings, citations and non-factual text.\n"
            'Return ONLY JSON: {"sentences":[{"id":1,"claims":["...","..."]}]}\n\n'
            f"SENTENCES:\n{lines}"
        )
        stats.decompose_calls += 1
        try:
            data = _parse_json(_generate(client, prompt, _DECOMPOSE_SYSTEM, stats))
        except Exception:  # noqa: BLE001 - keep the whole sentence on any failure
            data = None
        if not data:
            continue
        for item in data.get("sentences") or []:
            try:
                idx = int(item.get("id")) - 1
            except (TypeError, ValueError, AttributeError):
                continue
            if not (0 <= idx < len(chunk)):
                continue
            atoms = _validate_atoms(chunk[idx], item.get("claims") or [])
            if atoms:
                out[chunk[idx]] = atoms
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Judge: batched verdicts with validated evidence
# ─────────────────────────────────────────────────────────────────────────────

_VERIFY_SYSTEM = (
    "You are a careful evidence auditor. You judge whether the cited passages "
    "support each claim, by meaning, not by wording. You never use outside "
    "knowledge. Answer strictly with the requested JSON."
)


def _build_verify_prompt(claims: list[dict], passage_table: dict[str, Passage]) -> str:
    lines = ["TASK: VERIFY", "PASSAGES (use only these; cite by passage ID):"]
    for pid, p in passage_table.items():
        lines.append(f"[{pid}] (record {p.doc_id}) {p.text}")
    lines.append("")
    lines.append("CLAIMS (each lists the only passage IDs you may cite for it):")
    for i, c in enumerate(claims, start=1):
        cands = ", ".join(p.pid for p in c["passages"]) or "none"
        lines.append(f"C{i}: {c['text']}")
        lines.append(f"   candidate passages: {cands}")
    lines.append("")
    lines.append(
        "RULES:\n"
        "- SUPPORTED: the passages establish the claim, possibly jointly across "
        "several passages or records. Faithful paraphrase is allowed. Every "
        "component of the claim must be covered. Check numbers, quantities, units, "
        "dates and time periods, negation and direction, qualifiers (all, some, "
        "majority, up to), causal links and entity roles.\n"
        "- CONTRADICTED: a cited passage explicitly states something that conflicts "
        "with the claim (for example a different figure, date or negation).\n"
        "- INSUFFICIENT_EVIDENCE: the passages neither establish nor contradict the "
        "claim. A shared name, acronym or single keyword is NOT support. If any "
        "component is missing, the claim is INSUFFICIENT_EVIDENCE. If unsure, choose "
        "INSUFFICIENT_EVIDENCE.\n"
        "- Quotes must be copied exactly, word for word, from a cited passage. "
        "Never invent a quote or a passage ID.\n"
        "Return ONLY JSON:\n"
        '{"results":[{"claim":"C1","verdict":"SUPPORTED","evidence":'
        '[{"passage":"P3","quote":"exact words from P3"}],"rationale":"one short sentence"}]}'
    )
    return "\n".join(lines)


def supported_guards(claim: str, quotes: list[str]) -> Optional[str]:
    """Deterministic checks a SUPPORTED verdict must pass against its quotes.

    Returns None when the guards pass, otherwise a reason. Each rule is
    documented with its rationale:
      * every number in the claim appears in a quote (figures, dates, counts);
      * every unit in the claim appears in a quote (km vs m, crore vs lakh);
      * negation is present in the claim if and only if it is present in a
        quote (polarity); a negation mismatch is a direction error;
      * every qualifier in the claim appears in a quote (all vs some, up to);
      * the quotes cover at least ``MIN_CLAIM_COVERAGE`` of the claim's content
        words, so a verdict cannot rest on one shared keyword.
    A failure downgrades to INSUFFICIENT_EVIDENCE. It never produces a
    contradiction, because the guard cannot tell which side is wrong.
    """
    joined = " ".join(quotes)
    qnums = _numbers(joined)
    missing = sorted(_numbers(claim) - qnums)
    if missing:
        return f"figure(s) {', '.join(missing)} not present in the cited quotes"
    raw_claim = re.findall(r"[a-z%]+", claim.lower())
    raw_quote = set(re.findall(r"[a-z%]+", joined.lower()))
    if "%" in claim and "%" not in joined and "percent" not in raw_quote:
        return "unit % not present in the cited quotes"
    units = {t for t in raw_claim if t in _UNITS}
    missing_units = sorted(u for u in units if u not in raw_quote and
                           (u + "s") not in raw_quote and u.rstrip("s") not in raw_quote)
    if missing_units:
        return f"unit(s) {', '.join(missing_units)} not present in the cited quotes"
    if _has_negation(claim) != _has_negation(joined):
        return "negation differs between the claim and the cited quotes"
    quals = {t for t in raw_claim if t in _QUALIFIERS}
    missing_q = sorted(q for q in quals if q not in raw_quote)
    if missing_q:
        return f"qualifier(s) {', '.join(missing_q)} not present in the cited quotes"
    ctoks = content_tokens(claim)
    if ctoks:
        covered = ctoks & content_tokens(joined)
        coverage = len(covered) / len(ctoks)
        if coverage < MIN_CLAIM_COVERAGE:
            return (f"the cited quotes cover only {len(covered)} of {len(ctoks)} "
                    f"content words of the claim")
    return None


def _validate_quote(quote: str, doc: Source) -> bool:
    qn = ts.normalize(quote)
    if len(qn.split()) < MIN_QUOTE_TOKENS or len(quote.strip()) < MIN_QUOTE_CHARS:
        return False
    return bool(qn) and f" {qn} " in f" {doc.full_norm} "


def parse_verdicts(raw: str, claims: list[dict], passage_table: dict[str, Passage],
                   docs: dict[str, Source]) -> dict[int, dict]:
    """Validate the judge output for one batch.

    Returns claim index (1-based) -> {status, method, sources, quotes, note,
    rationale}. Anything that fails validation is INSUFFICIENT. Source IDs,
    passage IDs and quotes are each checked against the supplied material.
    """
    data = _parse_json(raw)
    results: dict[int, dict] = {}
    if data is None:
        return {i: _insufficient("judge response could not be parsed")
                for i in range(1, len(claims) + 1)}
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("claim") or "").strip().lstrip("Cc")
        try:
            idx = int(cid)
        except ValueError:
            continue
        if not (1 <= idx <= len(claims)) or idx in results:
            continue
        claim = claims[idx - 1]
        verdict = str(item.get("verdict") or "").strip().upper()
        rationale = " ".join(str(item.get("rationale") or "").split())[:400]
        if verdict not in _VERDICTS:
            results[idx] = _insufficient("unrecognised verdict from the judge",
                                         rationale)
            continue

        allowed = {p.pid for p in claim["passages"]}
        valid: list[tuple[str, str]] = []   # (doc_id, quote)
        rejected_items = 0
        for ev in item.get("evidence") or []:
            if not isinstance(ev, dict):
                rejected_items += 1
                continue
            pid = str(ev.get("passage") or "").strip()
            quote = str(ev.get("quote") or "").strip()
            passage = passage_table.get(pid)
            if passage is None or pid not in allowed:
                rejected_items += 1           # unknown or not shown for this claim
                continue
            src_id = str(ev.get("source") or "").strip()
            if src_id and src_id != passage.doc_id:
                rejected_items += 1           # source ID does not match passage
                continue
            doc = docs.get(passage.doc_id)
            if doc is None or not _validate_quote(quote, doc):
                rejected_items += 1           # quote not found in the record
                continue
            valid.append((passage.doc_id, quote))
        valid = list(dict.fromkeys(valid))
        quotes = [{"source": d, "text": q} for d, q in valid]
        sources = list(dict.fromkeys(d for d, _ in valid))
        suffix = (f" ({rejected_items} evidence item(s) rejected: could not be "
                  f"validated)") if rejected_items else ""

        if verdict == "SUPPORTED":
            if not valid:
                results[idx] = _insufficient(
                    "judge's supporting quote could not be validated" + suffix,
                    rationale)
                continue
            reason = supported_guards(claim["text"], [q for _, q in valid])
            if reason:
                results[idx] = _insufficient(
                    "judge said supported, but " + reason, rationale,
                    sources=sources, quotes=quotes)
                continue
            results[idx] = {"status": VERIFIED, "method": "llm_judge",
                            "sources": sources, "quotes": quotes,
                            "note": "supported by the cited passages" + suffix,
                            "rationale": rationale}
        elif verdict == "CONTRADICTED":
            if not valid:
                results[idx] = _insufficient(
                    "judge reported a conflict without a validated quote" + suffix,
                    rationale)
                continue
            results[idx] = {"status": REJECTED, "method": "llm_judge",
                            "sources": sources, "quotes": quotes,
                            "note": "conflicting evidence in the cited passages" + suffix,
                            "rationale": rationale}
        else:
            results[idx] = _insufficient(
                "the cited records do not establish this claim" + suffix,
                rationale)
    for i in range(1, len(claims) + 1):
        results.setdefault(i, _insufficient("no verdict returned for this claim"))
    return results


def _insufficient(note: str, rationale: str = "", sources: Optional[list] = None,
                  quotes: Optional[list] = None) -> dict:
    return {"status": INSUFFICIENT, "method": "llm_judge",
            "sources": sources or [], "quotes": quotes or [],
            "note": note, "rationale": rationale}


def verify_claims(claims: list[str], passages: list[Passage], docs: dict[str, Source],
                  client: Any, stats: VerifyStats,
                  deadline: Optional[float] = None) -> list[dict]:
    """Judge atomic claims in batches. One result dict per claim, in order.

    Claims without a lexical candidate passage are INSUFFICIENT with no LLM
    call. A failed or malformed batch makes its claims INSUFFICIENT. Once the
    time budget is spent the remaining claims are INSUFFICIENT and not checked.
    """
    results: list[Optional[dict]] = [None] * len(claims)
    pending: list[tuple[int, dict]] = []
    for i, text in enumerate(claims):
        cands = select_passages(text, passages)
        if not cands:
            stats.claims_no_candidates += 1
            results[i] = _insufficient(
                "no passage in the cited records shares terms with this claim")
            continue
        pending.append((i, {"text": text, "passages": cands}))

    for start in range(0, len(pending), CLAIMS_PER_JUDGE_CALL):
        batch = pending[start:start + CLAIMS_PER_JUDGE_CALL]
        if deadline is not None and time.monotonic() > deadline:
            for i, _ in batch:
                stats.claims_not_checked += 1
                results[i] = _insufficient(
                    "not checked: verification time budget reached")
            continue
        table: dict[str, Passage] = {}
        for _, c in batch:
            for p in c["passages"]:
                table.setdefault(p.pid, p)
        prompt = _build_verify_prompt([c for _, c in batch], table)
        stats.judge_calls += 1
        try:
            raw = _generate(client, prompt, _VERIFY_SYSTEM, stats)
            verdicts = parse_verdicts(raw, [c for _, c in batch], table, docs)
        except Exception as e:  # noqa: BLE001 - judge failure is INSUFFICIENT
            note = f"judge unavailable ({type(e).__name__})"
            verdicts = {k: _insufficient(note) for k in range(1, len(batch) + 1)}
        for k, (i, _) in enumerate(batch, start=1):
            stats.claims_judged += 1
            results[i] = verdicts[k]
    return [r if r is not None else _insufficient("not checked") for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point: factual sentences of an answer
# ─────────────────────────────────────────────────────────────────────────────

def verify_factual_sentences(answer: str, sources: list[dict], client: Any = None,
                             depth: str = "light") -> tuple[list[dict], VerifyStats]:
    """Verify every factual sentence of ``answer`` against ``sources``.

    ``depth="light"`` makes no LLM call: exact containment only, and any other
    sentence is INSUFFICIENT with a note that Deep mode checks it. ``depth="full"``
    runs decomposition and judging (``client`` required; without it the sentence is
    INSUFFICIENT with a note).

    Returns (rows, stats). Each row:
      {text, sentence, status, method, sources, quotes, note, rationale}
    ``text`` is the atomic claim (equal to ``sentence`` when the sentence was not
    split). ``sentence`` is the answer sentence the claim came from.
    """
    stats = VerifyStats()
    started = time.monotonic()
    sentences = ts.extract_factual_sentences(answer or "")
    passages, docs = build_passages(sources or [])
    src_norm = {d: s.full_norm for d, s in docs.items()}

    def exact_doc(text: str) -> Optional[str]:
        for doc_id, norm in src_norm.items():
            if ts.sentence_contained(text, norm):
                return doc_id
        return None

    rows: list[dict] = []
    pending: list[str] = []
    for s in sentences:
        doc = exact_doc(s) if ts.is_factual_length(s) else None
        if doc is not None:
            stats.claims_exact += 1
            rows.append({"text": s, "sentence": s, "status": VERIFIED,
                         "method": "exact", "sources": [doc],
                         "quotes": [{"source": doc, "text": s}],
                         "note": "appears verbatim in the cited record",
                         "rationale": ""})
        else:
            pending.append(s)
    if not pending:
        return _in_answer_order(rows, sentences, started, stats)

    if depth != "full" or client is None:
        reason = ("not verbatim in the cited records; the semantic check runs in "
                  "Deep mode" if depth != "full" else
                  "not verbatim in the cited records; no judge is available")
        for s in pending:
            stats.claims_not_checked += 1
            rows.append({"text": s, "sentence": s, "status": INSUFFICIENT,
                         "method": "not_checked", "sources": [], "quotes": [],
                         "note": reason, "rationale": ""})
        return _in_answer_order(rows, sentences, started, stats)

    decomposed = decompose_sentences(pending, client, stats)
    atoms: list[tuple[str, str]] = []           # (claim, sentence), in answer order
    for s in pending:
        for a in decomposed.get(s, [s]):
            atoms.append((a, s))

    # Atomic claims that appear verbatim are supported without the judge.
    # ``slots`` keeps answer order: a finished row, or None (= judge slot).
    slots: list[Optional[dict]] = []
    to_judge: list[tuple[str, str]] = []
    for a, s in atoms:
        doc = exact_doc(a) if ts.is_factual_length(a) else None
        if doc is not None:
            stats.claims_exact += 1
            slots.append({"text": a, "sentence": s, "status": VERIFIED,
                          "method": "exact", "sources": [doc],
                          "quotes": [{"source": doc, "text": a}],
                          "note": "appears verbatim in the cited record",
                          "rationale": ""})
        else:
            slots.append(None)
            to_judge.append((a, s))

    deadline = time.monotonic() + JUDGE_BUDGET_SECONDS
    judge_inputs = to_judge[:MAX_CLAIMS]        # claims over the cap are not judged
    judged = verify_claims([a for a, _ in judge_inputs], passages, docs, client,
                           stats, deadline=deadline)

    j = 0                                       # position in to_judge
    for (a, s), slot in zip(atoms, slots):
        if slot is not None:
            rows.append(slot)
            continue
        if j >= len(judge_inputs):
            stats.claims_not_checked += 1
            rows.append({"text": a, "sentence": s, "status": INSUFFICIENT,
                         "method": "not_checked", "sources": [], "quotes": [],
                         "note": f"not checked: more than {MAX_CLAIMS} claims in this answer",
                         "rationale": ""})
        else:
            v = judged[j]
            rows.append({"text": a, "sentence": s, "status": v["status"],
                         "method": v["method"], "sources": v["sources"],
                         "quotes": v["quotes"], "note": v["note"],
                         "rationale": v.get("rationale", "")})
        j += 1
    return _in_answer_order(rows, sentences, started, stats)


def _in_answer_order(rows: list[dict], sentences: list[str], started: float,
                     stats: VerifyStats) -> tuple[list[dict], VerifyStats]:
    """Stable sort: rows follow the order of their sentence in the answer."""
    order = {s: i for i, s in enumerate(sentences)}
    rows.sort(key=lambda r: order.get(r["sentence"], len(order)))
    stats.elapsed_ms = int((time.monotonic() - started) * 1000)
    return rows, stats


def verify_single_claim(claim: str, sources: list[dict], client: Any = None,
                        ) -> tuple[dict, VerifyStats]:
    """Verify ONE claim with the same verdict rules as ``verify_factual_sentences``.

    Used by /api/investigate. No decomposition: the claim is judged as given,
    and the judge must still cover every component of it.
    """
    stats = VerifyStats()
    started = time.monotonic()
    passages, docs = build_passages(sources or [])
    doc = None
    if ts.is_factual_length(claim):
        for doc_id, s in docs.items():
            if ts.sentence_contained(claim, s.full_norm):
                doc = doc_id
                break
    if doc is not None:
        stats.claims_exact += 1
        row = {"text": claim, "sentence": claim, "status": VERIFIED,
               "method": "exact", "sources": [doc],
               "quotes": [{"source": doc, "text": claim}],
               "note": "appears verbatim in the cited record", "rationale": ""}
    elif client is None:
        row = {"text": claim, "sentence": claim, "status": INSUFFICIENT,
               "method": "not_checked", "sources": [], "quotes": [],
               "note": "no judge is available to check this claim", "rationale": ""}
    else:
        deadline = time.monotonic() + JUDGE_BUDGET_SECONDS
        v = verify_claims([claim], passages, docs, client, stats, deadline=deadline)[0]
        row = {"text": claim, "sentence": claim, "status": v["status"],
               "method": v["method"], "sources": v["sources"], "quotes": v["quotes"],
               "note": v["note"], "rationale": v.get("rationale", "")}
    stats.elapsed_ms = int((time.monotonic() - started) * 1000)
    return row, stats
