"""Regression guard: the Cross-Verified Facts feature stays removed.

The semantic-verification subsystem (``src/verification``), the ``/api/verify``
and ``/api/investigate`` routes, the verification fields of the ``final`` SSE
event, the GraphRAG ``verify_answer`` hooks and the ``verify_depth`` policy
setting were removed on purpose. These checks fail if any of them comes back,
and they confirm that importing the server starts none of it.

Retrieval provenance (matched chunks, score attribution) is covered by
``tests/test_matched_chunk_provenance.py`` and is not repeated here.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys

import pytest

import tests.conftest  # noqa: F401  (stubs torch/sentence-transformers)


def _server():
    import src.retrieval.frontend.server as srv

    return srv


def test_verification_package_is_not_importable():
    assert importlib.util.find_spec("src.verification") is None


def test_server_import_does_not_load_verification_code():
    _server()
    loaded = [m for m in sys.modules if m == "src.verification" or m.startswith("src.verification.")]
    assert loaded == [], f"verification modules loaded at server import: {loaded}"


def test_verify_and_investigate_routes_are_removed():
    srv = _server()
    paths = {getattr(route, "path", None) for route in srv.app.routes}
    assert "/api/verify" not in paths
    assert "/api/investigate" not in paths


def test_server_source_has_no_verification_payload_or_hooks():
    src_text = inspect.getsource(_server())
    for banned in (
        "src.verification",
        "verify_pending",
        "citation_dropped",
        "judge_removed",
        "judge_rewritten",
        "verify_depth",
        "_llm_rewrite_answer",
        "grounding_aliases",
    ):
        assert banned not in src_text, f"server.py still references {banned!r}"


def test_graph_capability_has_no_verify_hook():
    from src.graphrag.capability import DisabledGraphCapability, GraphCapability

    assert not hasattr(GraphCapability, "verify_answer")
    assert not hasattr(DisabledGraphCapability, "verify_answer")


def test_generation_plan_has_no_verify_depth():
    from src.generation.policy import resolve_execution
    from src.generation.registry import model_registry

    fam = model_registry.get("qwen3.6_35b_a3b_fp8")
    for mode in ("fast", "deep"):
        plan = resolve_execution(fam, mode, "vllm")
        assert not hasattr(plan, "verify_depth")
        assert "verify_depth" not in fam.get_execution_params(mode)


@pytest.mark.parametrize("name", ["verify_answer", "investigate_claim", "semantic_verify"])
def test_no_verification_entry_point_on_server(name):
    assert not hasattr(_server(), name)
