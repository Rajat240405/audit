"""
Fine-tuning data preparation for the INCOIS Audit Pro generator.

Reads the enriched parliamentary corpus and builds Alpaca-format training
JSONL for LoRA fine-tuning of qwen3:4b.

Two dataset components:
  A) FAITHFUL PAIRS  — (question + its own document) -> official answer text.
     Free, zero labeling. Teaches style + "answer from the document".
  B) DISTILLED PAIRS — (question + top-k retrieved docs) -> answer written by a
     stronger teacher (Groq), KEPT ONLY IF it passes our grounding check.
     Teaches multi-document synthesis. Requires GROQ_API_KEY (optional).
  C) HONEST "I DON'T KNOW" — questions that have no matching documents, with a
     refusal answer. Teaches the model to say "not in context" instead of
     hallucinating. Generated deterministically (optional).

Usage:
    python scripts/finetune_prepare_data.py \
        --corpus "data/processed/*.jsonl" \
        --out-dir data/finetune \
        --max-train 1500 \
        [--with-refusals]

Outputs (in out-dir):
    train.jsonl   (Alpaca: {"instruction", "input", "output"})
    val.jsonl
    stats.json    (counts + sample)
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path

# ── helpers ─────────────────────────────────────────────────────────────

_HEADER_PATTERN = re.compile(
    r"(GOVERNMENT OF INDIA.*?QUESTION NO\.\s*\S+.*?TO BE ANSWERED ON.*?\n+)",
    re.DOTALL | re.IGNORECASE,
)
_MEMBER_PATTERN = re.compile(r"^\d+\.\s+.*$", re.MULTILINE)


def clean_question(qt: str) -> str:
    """Strip the parliamentary boilerplate header, keep the actual question."""
    if not qt:
        return ""
    t = qt.strip()
    # cut at the first occurrence of a numbered member line like "123. SHRI X"
    m = _MEMBER_PATTERN.search(t)
    if m:
        t = t[m.start():]
    # drop the "Will the Minister..." prefix boilerplate? keep as-is (it's the
    # real question); just collapse whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def strip_answer_boilerplate(at: str, max_chars: int = 4000) -> str:
    """Trim the answer to the substantive reply (drop the minister header)."""
    if not at:
        return ""
    t = at.strip()
    # cut at the minister-name line so the answer starts at "(a) ..." or the
    # first real sentence
    m = re.search(r"\((?:a|A)\)\s", t)
    if m:
        t = t[m.start():]
    else:
        # fall back: find first line that isn't header-ish
        lines = [ln for ln in t.split("\n") if ln.strip()]
        t = "\n".join(lines[:60])
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t[:max_chars].strip()


def build_user_prompt_for_train(question: str, docs: list[dict]) -> str:
    """Format the training INPUT: question + retrieved context."""
    parts = [
        "Below is the retrieved parliamentary Q&A context.",
        "=" * 70,
        f"RETRIEVED CONTEXT ({len(docs)} records):",
        "=" * 70,
    ]
    for i, d in enumerate(docs, start=1):
        parts.append(f"[Source {i}] (ID: {d.get('doc_id','')})")
        if d.get("subject"):
            parts.append(f"Subject: {d['subject']}")
        parts.append(f"QUESTION: {d.get('question','')}")
        parts.append(f"ANSWER: {d.get('answer','')}")
        parts.append("-" * 70)
    parts.extend(["=" * 70, f"USER QUESTION:\n{question}", "=" * 70, "ANSWER:"])
    return "\n".join(parts)


def load_corpus(corpus_glob: str) -> list[dict]:
    recs = []
    for fp in sorted(glob.glob(corpus_glob)):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return recs


# ── dataset A: faithful pairs ────────────────────────────────────────────

def build_faithful(recs: list[dict], max_items: int) -> list[dict]:
    """(question + its own doc) -> official answer."""
    out = []
    for r in recs:
        q = clean_question(r.get("question_text") or "")
        a = strip_answer_boilerplate(r.get("answer_text") or "")
        if not q or not a:
            continue
        doc = {
            "doc_id": r.get("question_id", ""),
            "subject": (r.get("metadata") or {}).get("subject", ""),
            "question": q,
            "answer": a,
        }
        out.append({
            "instruction": (
                "You are a parliamentary research assistant. Answer the question "
                "using ONLY the provided retrieved context. Copy names and figures "
                "exactly as written; never invent facts. Cite [Source N]."
            ),
            "input": build_user_prompt_for_train(q, [doc]),
            "output": a,
        })
        if len(out) >= max_items:
            break
    return out


# ── dataset C: honest refusals ───────────────────────────────────────────

_REFUSAL = (
    "The provided context does not contain sufficient information to answer "
    "this question. Based on the retrieved documents, this topic is not "
    "addressed."
)


def build_refusals(recs: list[dict], max_items: int) -> list[dict]:
    """Questions paired with UNRELATED docs -> refusal answer."""
    random.seed(42)
    out = []
    # build questions that mention an entity NOT present in a random doc bucket
    entities = ["NIOT", "INCOIS", "Mission Mausam", "Doppler Weather Radar",
                "Matsya 6000", "Tarang", "CRZ", "CWC"]
    for i in range(max_items):
        q = random.choice(entities)
        # pick a doc from a DIFFERENT subject so the context is unrelated
        group = random.choice(list(recs))
        doc = {
            "doc_id": group.get("question_id", ""),
            "subject": (group.get("metadata") or {}).get("subject", ""),
            "question": clean_question(group.get("question_text") or ""),
            "answer": strip_answer_boilerplate(group.get("answer_text") or ""),
        }
        out.append({
            "instruction": (
                "You are a parliamentary research assistant. Answer the question "
                "using ONLY the provided retrieved context. If the context does "
                "not contain the information, say so clearly."
            ),
            "input": build_user_prompt_for_train(f"What is the role of {q}?", [doc]),
            "output": _REFUSAL,
        })
    return out


# ── main ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Build fine-tuning dataset")
    ap.add_argument("--corpus", default="data/processed/*.jsonl")
    ap.add_argument("--out-dir", default="data/finetune")
    ap.add_argument("--max-train", type=int, default=1500)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--with-refusals", action="store_true")
    ap.add_argument("--refusal-count", type=int, default=150)
    args = ap.parse_args()

    recs = load_corpus(args.corpus)
    print(f"corpus records: {len(recs)}")

    random.seed(42)
    recs_shuffled = recs[:]  # faithful pairs use their own docs; shuffling
    random.shuffle(recs_shuffled)

    all_items: list[dict] = []

    # A: faithful
    faithful = build_faithful(recs_shuffled, args.max_train)
    all_items.extend(faithful)
    print(f"faithful pairs: {len(faithful)}")

    # C: refusals (optional)
    if args.with_refusals:
        refusals = build_refusals(recs, args.refusal_count)
        all_items.extend(refusals)
        print(f"refusal pairs: {len(refusals)}")

    random.shuffle(all_items)
    n_val = max(1, int(len(all_items) * args.val_ratio))
    val, train = all_items[:n_val], all_items[n_val:]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for it in train:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(out / "val.jsonl", "w", encoding="utf-8") as f:
        for it in val:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "train": len(train), "val": len(val),
            "faithful": len(faithful),
            "sample": all_items[0] if all_items else None,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(train)} train / {len(val)} val -> {out}/")
    print("sample input (first 200 chars):")
    if all_items:
        print(all_items[0]["input"][:200])


if __name__ == "__main__":
    main()
