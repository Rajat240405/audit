"""Docker smoke catalogue is an overlay; production YAML is unchanged."""

from __future__ import annotations

from pathlib import Path

from src.generation.registry import ModelRegistry, load_model_catalog, populate_model_registry
from src.utils.app_paths import config_path


def test_production_catalog_vllm_is_hpc_only():
    data = load_model_catalog(str(config_path("models.yaml")))
    vllm = [f["model_name"] for f in data["providers"]["vllm"]["families"]]
    # Exact HPC list (extended in Task 2 with the Qwen3.8-27B-FP8 catalogue entry).
    assert vllm == ["Qwen3.6-27B", "Qwen3.6-30B-A3B", "Qwen3.6-35B-A3B-FP8",
                    "Qwen3.8-27B-FP8"]
    assert "qwen3:8b" not in vllm


def test_production_catalog_vllm_never_uses_name_suffix():
    """HPC thinking control is chat_template_kwargs only (no /think|/nothink)."""
    data = load_model_catalog(str(config_path("models.yaml")))
    families = data["providers"]["vllm"]["families"]
    assert all(f.get("think_mode") == "template" for f in families)
    fp8 = next(f for f in families if f["model_name"] == "Qwen3.6-35B-A3B-FP8")
    assert fp8["context_window"] == 32768
    assert fp8["thinking_capable"] is True


def test_docker_catalog_adds_host_ollama_and_keeps_hpc():
    path = config_path("models.docker.yaml")
    assert path.is_file()
    data = load_model_catalog(str(path))
    families = data["providers"]["vllm"]["families"]
    names = [f["model_name"] for f in families]
    assert names[0] == "qwen3:8b"
    assert "Qwen3.6-27B" in names
    assert "Qwen3.6-30B-A3B" in names
    assert "Qwen3.6-35B-A3B-FP8" in names
    assert "Qwen3.8-27B-FP8" in names          # Task 2 mirror of production
    host = next(f for f in families if f["id"] == "ollama_qwen3_8b")
    assert host["think_mode"] == "none"
    # The HPC overlay entries mirror production: template mode, never suffix.
    hpc = [f for f in families if f["id"] != "ollama_qwen3_8b"]
    assert all(f.get("think_mode") == "template" for f in hpc)


def test_docker_catalog_registers_for_vllm_dropdown(monkeypatch):
    monkeypatch.setenv("MODEL_CATALOG", str(config_path("models.docker.yaml")))
    data = load_model_catalog(str(Path(config_path("models.docker.yaml"))))
    reg = ModelRegistry()
    populate_model_registry(reg, data)
    vllm = reg.list_by_provider("vllm")
    ids = [f.id for f in vllm]
    assert "ollama_qwen3_8b" in ids
    assert "qwen3.6_27b" in ids
    assert "qwen3.6_30b_a3b" in ids
    assert "qwen3.6_35b_a3b_fp8" in ids
    assert "qwen3.8_27b_fp8" in ids            # Task 2 mirror of production
    host = next(f for f in vllm if f.id == "ollama_qwen3_8b")
    assert host.model_name == "qwen3:8b"
    assert host.think_mode == "none"
