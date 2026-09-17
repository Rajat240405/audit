"""Repair the corpus encoding damage.

Strategy (deliberately NOT a QARecord round-trip):
  * metadata.core_text_stats and metadata.record_id are NOT QARecordMetadata
    fields and Pydantic's extra policy is "ignore", so model_dump() would
    silently delete the V2 provenance from 211 rows. We therefore repair the
    raw dicts in place and touch nothing else.
  * ftfy.fix_text on every string value: evidence-based, handles mixed
    genuine-Unicode + mojibake, and never applies a blanket latin1->utf8.
  * question_id is never passed to ftfy (identities are immutable).
  * content_hash and document_content are RECOMPUTED from the repaired text via
    the canonical QARecord computed fields, so the raw file becomes internally
    consistent again.
  * Output: UTF-8, NO BOM, LF, sort_keys=True (matching the original file),
    written atomically.

Usage: python3 repair_corpus.py [--apply]
"""
import json
import shutil
import sys
from pathlib import Path

import ftfy

sys.path.insert(0, "/tmp/repo")
from src.models.qa_record import QARecord  # noqa: E402

SRC = Path("/tmp/audit_in/corpus_dl.bin")
OUT = Path("/tmp/audit_in/corpus_reports_REPAIRED.jsonl")
FIXKW = dict(normalization=None, uncurl_quotes=False, fix_latin_ligatures=False,
             fix_character_width=False)

APPLY = "--apply" in sys.argv

raw = SRC.read_bytes()
had_bom = raw[:3] == b"\xef\xbb\xbf"
records = [json.loads(l) for l in raw.decode("utf-8-sig").splitlines() if l.strip()]


def fix_strings(obj, skip_keys=()):
    """Recursively ftfy every string, leaving non-strings and skipped keys alone."""
    if isinstance(obj, dict):
        return {k: (v if k in skip_keys else fix_strings(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [fix_strings(v) for v in obj]
    if isinstance(obj, str):
        return fix_symbol_pua(ftfy.fix_text(obj, **FIXKW))
    return obj


#: Microsoft Symbol-font glyphs that arrived as cp1252-decoded UTF-8 and that
#: ftfy deliberately leaves alone, because decoding them yields a Private Use
#: Area codepoint rather than real text:
#:   "ï‚·" = EF 82 B7 -> U+F0B7 (Symbol font bullet)
#:   "ï‚§" = EF 82 A7 -> U+F0A7 (Symbol font small filled square)
#: Both occur exclusively as list-item markers in this corpus (verified in
#: context: "These include:\n• Doppler Weather Radar ..."), so both map to the
#: ordinary bullet. This is a targeted two-sequence replacement backed by that
#: evidence — not a blanket codec conversion.
SYMBOL_PUA = {"ï‚·": "\u2022", "ï‚§": "\u2022"}


def fix_symbol_pua(s: str) -> str:
    for bad, good in SYMBOL_PUA.items():
        s = s.replace(bad, good)
    return s


changed_strings = 0
qid_changed = []
repaired = []

for rec in records:
    fixed = fix_strings(rec, skip_keys=("question_id",))
    if fixed.get("question_id") != rec.get("question_id"):
        qid_changed.append((rec.get("question_id"), fixed.get("question_id")))

    # Canonical recomputation of the two computed fields only.
    model = QARecord.model_validate(fixed)
    fixed["content_hash"] = model.content_hash
    fixed["document_content"] = model.document_content
    repaired.append(fixed)

    for k in ("answer_text", "question_text"):
        if rec.get(k) != fixed.get(k):
            changed_strings += 1

print(f"BOM present in source   : {had_bom}")
print(f"records                 : {len(repaired)}")
print(f"question_id altered     : {len(qid_changed)}   <-- must be 0")
print(f"answer/question changed : {changed_strings}")

if not APPLY:
    print("\n[dry-run] nothing written. Re-run with --apply to write.")
    sys.exit(0)

# ── backup, then atomic write ─────────────────────────────────────────────────
backup = SRC.with_name("corpus_reports_ARENA_CURRENT.BACKUP.jsonl")
shutil.copy2(SRC, backup)
print(f"\nbackup written          : {backup} ({backup.stat().st_size} bytes)")

payload = "".join(
    json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in repaired
).encode("utf-8")                      # UTF-8, no BOM
tmp = OUT.with_name(OUT.name + ".tmp")
with open(tmp, "wb") as fh:
    fh.write(payload)
    fh.flush()
    import os
    os.fsync(fh.fileno())
tmp.replace(OUT)
print(f"repaired corpus written : {OUT} ({OUT.stat().st_size} bytes)")
print(f"starts with BOM?        : {OUT.read_bytes()[:3] == b'\xef\xbb\xbf'}")
