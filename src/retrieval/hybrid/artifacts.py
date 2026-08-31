"""The ONE canonical "is the index complete?" contract (audit IW-11).

History: three independent marker definitions had drifted — ``retrieve``
required 6 artifacts while the ingestion engine and the server accepted a
4-artifact subset. A directory missing ``vector_store.ids`` /
``bm25_index.json`` would pass the lax check and then crash ``load()``.

Single source of truth: the writer. ``HybridRAGPipeline.save()`` persists
exactly the six artifacts below, so a usable index is defined as "all six
present". Every consumer (``src/retrieval/cli.py``,
``src/scripts/ingest_folder.py``, ``src/scripts/ingest_service.py``) imports
from here — do not grow per-caller copies again.
"""

from __future__ import annotations

from pathlib import Path

#: Artifacts a complete, loadable hybrid index consists of (what
#: ``HybridRAGPipeline.save()`` writes). Order is cosmetic; membership is the
#: contract.
INDEX_MARKER_FILES: tuple[str, ...] = (
    "pipeline_metadata.json",
    "doc_map.json",
    "vector_store.index",
    "vector_store.ids",
    "bm25_index.pkl",
    "bm25_index.json",
)


def index_is_complete(index_path: str | Path) -> bool:
    """True only if a complete, loadable pipeline index was saved at
    ``index_path``.

    Merely checking ``index_path.exists()`` is not enough: the directory can
    exist while being empty or partially populated (e.g. ``storage/hybrid_rag``
    created but the index files deleted, or an interrupted save). A partial
    directory must trigger a full build, never a doomed incremental load.
    """
    p = Path(index_path)
    if not p.exists() or not p.is_dir():
        return False
    return all((p / name).exists() for name in INDEX_MARKER_FILES)
