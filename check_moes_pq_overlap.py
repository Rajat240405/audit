import hashlib
import json
from pathlib import Path

MOES = Path("data/.moes-website/press-release")
PARL = Path("data/parliamentary-qa")

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Parliamentary SHA index
print("Indexing parliamentary PDFs...")

parl_hashes = {}

for p in PARL.rglob("*.pdf"):
    h = sha256(p)
    parl_hashes.setdefault(h, []).append(p)

manifest = json.loads(
    (MOES / "manifest.json").read_text(encoding="utf-8")
)

# 101 PQ records
pq_ids = {
    r["id"]
    for r in manifest["records"]
    if r.get("title", "").strip().upper().startswith("PARLIAMENT QUESTION")
}

pq_docs = [
    d for d in manifest["documents"]
    if d.get("record_id") in pq_ids
]

# Build a lookup by filename because manifest paths are record-relative.
all_moes_pdfs = list(MOES.rglob("*.pdf"))

by_name = {}
for p in all_moes_pdfs:
    by_name.setdefault(p.name, []).append(p)

matches = []
nonmatches = []
uncomparable = []

for d in pq_docs:
    filename = Path(d["path"]).name
    candidates = by_name.get(filename, [])

    # Normally there should be exactly one.
    if not candidates:
        uncomparable.append((d["key"], "missing_file", filename))
        continue

    # Prefer exact suffix/path match if multiple candidates exist.
    path = candidates[0]

    if not d.get("sha256"):
        uncomparable.append((d["key"], "missing_sha256", str(path)))
        continue

    h = d["sha256"]

    if h in parl_hashes:
        matches.append((d, path, parl_hashes[h]))
    else:
        nonmatches.append((d, path))

print()
print("=" * 65)
print("MoES PQ -> Parliamentary Corpus SHA-256 Validation")
print("=" * 65)
print(f"PQ-titled MoES records : {len(pq_ids)}")
print(f"PQ documents compared  : {len(pq_docs)}")
print(f"Exact SHA-256 matches  : {len(matches)}")
print(f"Non-matches            : {len(nonmatches)}")
print(f"Uncomparable           : {len(uncomparable)}")
print(f"Total accounted        : {len(matches)+len(nonmatches)+len(uncomparable)}")

print()
print("=== EXACT SHA-256 MATCHES ===")

for d, moes_path, parl_paths in matches:
    print()
    print(f"KEY    : {d['key']}")
    print(f"SHA256 : {d['sha256']}")
    print(f"MOES   : {moes_path}")
    for p in parl_paths:
        print(f"PARL   : {p}")

print()
print("=== POTENTIALLY UNIQUE MoES PQ DOCUMENTS ===")
print(f"Count: {len(nonmatches)}")

for d, path in nonmatches:
    print(f"{d['key']} -> {path}")

print()
print("=== UNCOMPARABLE ===")
for x in uncomparable:
    print(f"{x[0]} -> {x[1]} -> {x[2]}")

print()
print("DONE — READ ONLY. No crawler/corpus files modified.")
