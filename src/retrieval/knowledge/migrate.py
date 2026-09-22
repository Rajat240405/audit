"""Non-destructive migration of legacy flat knowledge files to v2 records.

The legacy layout stored one file per question slug:

    user-knowledge/<slug>.json   {"question", "answer", "sources",
                                  "saved_at", "saved_by"}

The slug is derived from mutable question text and truncated to 60 characters,
so it is not a usable primary key — two different long questions collide onto
one file, and the second save silently destroys the first.

Migration therefore gives every legacy file a fresh UUID under ``records/`` and
**leaves the original in place**. Nothing is deleted, and re-running is a no-op,
so the process is safe to interrupt and repeat.

Same standard as ``src/graphrag/migrate.py``: explicit, idempotent,
non-destructive, dry-run capable.
"""
from __future__ import annotations

from src.retrieval.knowledge.store import (
    KnowledgeStore,
    content_hash,
    owner_id_from,
    trim_sources,
)

__all__ = ["plan_migration", "run_migration", "MIGRATED_SUFFIX"]

#: Originals are renamed, never deleted, and only when explicitly requested.
MIGRATED_SUFFIX = ".migrated"


def _legacy_identity(data: dict) -> tuple[str, str, str]:
    """(content_hash, owner_id, owner_name) for a legacy record."""
    question = str(data.get("question") or "")
    answer = str(data.get("answer") or "")
    owner_name = str(data.get("saved_by") or "").strip()
    return content_hash(question, answer), owner_id_from(owner_name), owner_name


def plan_migration(store: KnowledgeStore) -> dict:
    """Read-only preview: what migration WOULD do. Touches nothing."""
    existing = {
        (r.get("content_hash"), r.get("owner_id"))
        for r in store.all_records(include_archived=True)
    }
    actions = []
    for data in store.read_legacy():
        chash, oid, oname = _legacy_identity(data)
        question = str(data.get("question") or "").strip()
        answer = str(data.get("answer") or "").strip()
        if not question or not answer:
            actions.append({"file": data.get("_file", "?"), "action": "skip",
                            "reason": "missing question or answer"})
            continue
        if (chash, oid) in existing:
            actions.append({"file": data.get("_file", "?"), "action": "skip",
                            "reason": "already migrated", "content_hash": chash})
            continue
        actions.append({
            "file": data.get("_file", "?"), "action": "create",
            "content_hash": chash, "owner_name": oname, "owner_id": oid,
            "question_preview": question[:80],
        })
    counts = {"create": 0, "skip": 0}
    for a in actions:
        counts[a["action"]] = counts.get(a["action"], 0) + 1
    return {
        "legacy_files": len(store.read_legacy()),
        "existing_records": len(store.all_records(include_archived=True)),
        "actions": actions,
        "counts": counts,
        "destructive": False,
    }


def run_migration(store: KnowledgeStore, *, dry_run: bool = True,
                  rename_originals: bool = False,
                  index_path=None) -> dict:
    """Migrate legacy files into ``records/``.

    ``dry_run`` (the default) plans without writing anything.
    ``rename_originals`` appends ``.migrated`` to each successfully copied
    original so a later run skips it quickly — the data is still fully present
    and recoverable by removing the suffix.
    """
    plan = plan_migration(store)
    if dry_run:
        return {**plan, "dry_run": True, "migrated": 0, "renamed": 0}

    migrated = 0
    renamed = 0
    errors: list[dict] = []
    for action in plan["actions"]:
        if action["action"] != "create":
            continue
        name = action["file"]
        src = (index_path or store.root) / name
        try:
            import json
            data = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            errors.append({"file": name, "error": f"{type(e).__name__}: {e}"})
            continue
        question = str(data.get("question") or "").strip()
        answer = str(data.get("answer") or "").strip()
        owner_name = str(data.get("saved_by") or "").strip()
        ts = str(data.get("saved_at") or "") or None
        try:
            store.create(
                question=question, answer=answer,
                sources=trim_sources(data.get("sources")),
                owner_name=owner_name,
                owner_id=owner_id_from(owner_name),
                now=ts,
            )
            migrated += 1
        except Exception as e:  # noqa: BLE001 — one bad file must not stop the run
            errors.append({"file": name, "error": f"{type(e).__name__}: {e}"})
            continue
        if rename_originals:
            try:
                src.replace(src.with_name(src.name + MIGRATED_SUFFIX))
                renamed += 1
            except OSError as e:
                errors.append({"file": name,
                               "error": f"rename failed: {type(e).__name__}: {e}"})

    store.refresh_index()
    return {
        **plan, "dry_run": False, "migrated": migrated, "renamed": renamed,
        "errors": errors,
        "records_after": len(store.all_records(include_archived=True)),
    }
