"""
Temporal-awareness probe — proves, using the ACTUAL production code, how the
AWS-query scenario (2008/2011/2015/2022/2026 sources with conflicting counts)
is handled by retrieval metadata plumbing and Task-3 context assembly.

Read-only: imports only src/generation/evidence.py + src/retrieval/result.py
(stdlib). The heavy `src.retrieval` package __init__ (faiss/sentence-
transformers) is stubbed so this runs anywhere, unchanged against the repo.

Run:  python3 investigation_temporal/probe.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── stub the heavy src.retrieval package so evidence.py imports cleanly ──
pkg = types.ModuleType("src.retrieval")
pkg.__path__ = [str(ROOT / "src" / "retrieval")]
sys.modules["src.retrieval"] = pkg
resmod = types.ModuleType("src.retrieval.result")
resmod.__file__ = str(ROOT / "src" / "retrieval" / "result.py")
sys.modules["src.retrieval.result"] = resmod  # register BEFORE exec (dataclasses looks up __module__)
exec(compile(open(ROOT / "src" / "retrieval" / "result.py").read(), resmod.__file__, "exec"), resmod.__dict__)

import src.generation.evidence as ev  # noqa: E402
RetrievedResult = resmod.RetrievedResult  # noqa: E402

# ── the AWS scenario: same question asked in 5 different years ───────────
TABLE = "\n".join(
    ["ANNEXURE-II", "State-wise details of Automatic Weather Stations (AWS)",
     "State                       Current   Planned"] +
    [f"State {i:02d}                    {3 + i}        {10 + i}" for i in range(1, 38)]
)

def mk(doc_id, date, answer):
    return RetrievedResult(
        doc_id=doc_id,
        question="Will the Minister state the number of Automatic Weather Stations in the country, State-wise?",
        answer=answer,
        score=1.0,
        retrieval_method="rrf_fusion",
        metadata={"ministry": "EARTH SCIENCES", "subject": "Automatic Weather Stations",
                  "date": date, "document_type": "parliamentary_qa"},
    )

D2008 = mk("3-1204",    "2008-07-24", "As per records, 125 Automatic Weather Stations (AWS) have been installed in the country so far.")
D2011 = mk("15-2-2211", "2011-08-10", "The India Meteorological Department has installed about 550 Automatic Weather Stations (AWS) across the country. State-wise details are given at Annexure.")
D2015 = mk("16-6-2409", "2015-12-02", "The network comprises 675 Automatic Weather Stations in the country at present. A statement is enclosed.")
D2022 = mk("17-8-3300", "2022-03-15", "There are 702 Automatic Weather Stations (AWS) operational in the country as of now.")
D2026 = mk("18-4-4267", "2026-07-29",
           "The Government has approved augmentation of the observational network. "
           "The State-wise details of current AWS and the stations planned to be installed "
           "are given at Annexure-II.\n" + TABLE)

QUESTION = "How many automatic weather stations are currently installed in India, state-wise, and how many more are planned?"
RESULTS = [D2011, D2026, D2015, D2022, D2008]  # a plausible rerank order — 2011 phrasing edge

sep = "=" * 72
print(sep); print("PROBE 1 — What the LLM receives (dates present in metadata?)"); print(sep)
prompt = ev.render_user_prompt(QUESTION, [(r, ev.clean_parliament_text(r.question), ev.clean_parliament_text(r.answer)) for r in RESULTS])
head = prompt.split("USER QUESTION")[0]
print(head[:1400])
print("\n[check] metadata was present on every retrieved record:")
for r in RESULTS:
    print(f"  {r.doc_id:12s} metadata.date = {r.metadata['date']!r}")
print("\n[check] occurrences of each year inside the rendered prompt:")
for y in ("2008", "2011", "2015", "2022", "2026"):
    import re
    n = len(re.findall(rf"\b{y}\b", prompt))
    print(f"  {y}: {n}  ->", "HEADER field" if f"Date: {y}" in prompt else ("inside body text only" if n else "ABSENT"))
hdr_ok = any(f"Date: {y}" in prompt for y in ("2008", "2011", "2015", "2022", "2026"))
note_ok = any(l.startswith("NOTE:") for l in prompt.splitlines())
print(f"\n  => any 'Date:' line in any source header? {hdr_ok}")
print(f"  => temporal-conflict NOTE present? {note_ok}")
if hdr_ok and note_ok:
    print("  STATUS: POST-REMEDIATION TREE (R1/R2 applied) — pre-fix results are frozen in probe_output.txt")
else:
    print("  STATUS: PRE-REMEDIATION TREE — matches probe_output.txt")
    assert not hdr_ok, "investigation premise changed: headers now carry dates"

print(); print(sep); print("PROBE 2 — Tight budget: which sources survive, and in what shape?"); print(sep)
alloc = ev.allocate_evidence(RESULTS, QUESTION, budget_tokens=120)
print(f"budget=120 tokens -> admitted={[a.result.doc_id for a in alloc.admissions]}, skipped={alloc.skipped_doc_ids}")
for a in alloc.admissions:
    print(f"\n--- [Source: {a.result.doc_id}] whole={a.whole} omitted_units={a.omitted_units} truncated={a.truncated} ---")
    print(a.evidence_text[:400])
assert all(a.result.metadata.get("date") for a in alloc.admissions), "dates still available on results"
print("\n  => admission order is rerank/score order only; dates never consulted.")

print(); print(sep); print("PROBE 3 — In-text date carriers stripped by cleaning"); print(sep)
raw = "GOVERNMENT OF INDIA\nMINISTRY OF EARTH SCIENCES\nLOK SABHA\nUNSTARRED QUESTION NO. 4267\nTO BE ANSWERED ON 29.07.2026\nAUTOMATIC WEATHER STATIONS\n4267. SHRI X:\nWill the Minister state the number of AWS stations?\nANSWER\nTHE MINISTER OF STATE\n(DR. JITENDRA SINGH)\n(a) The details are given at Annexure-II."
cleaned = ev.clean_parliament_text(raw)
print("BEFORE (has 'TO BE ANSWERED ON 29.07.2026'):", "29.07.2026" in raw)
print("AFTER  (date line survives?):              ", "29.07.2026" in cleaned)
print("--- cleaned text ---"); print(cleaned)
assert "29.07.2026" not in cleaned, "cleaner no longer strips the in-text answer-date line"
print("\n  => the document's own answer-date line is dropped as boilerplate.")

print(); print(sep); print("PROBE 4 — Keyword-driven block selection vs. the 37-row table"); print(sep)
kw = ev.query_keywords(QUESTION)
blocks = ev.segment_blocks(ev.clean_parliament_text(D2026.answer))
print(f"question keywords: {kw}")
for i, b in enumerate(blocks):
    print(f"  block {i}: kind={b.kind:9s} keyword_hits={ev.block_relevance(b.text, kw)} tokens={ev.estimate_tokens(b.text)} preview={b.text.splitlines()[0][:70]!r}")
sel = ev._select_blocks(blocks, kw, avail_tokens=80)
assert sel is not None
emitted, omitted, truncated, cost = sel
print(f"\navail=80 -> emitted cost={cost} omitted={omitted} truncated={truncated}")
print("--- emitted evidence for Source 2026 ---"); print(emitted)
table_dropped = "State 01" not in emitted
print(f"\n  => the 37-row current-vs-planned table survived? {not table_dropped}  (omission marker present: {'omitted' in emitted})")
print(sep); print("ALL ASSERTIONS HELD — probe complete.")
