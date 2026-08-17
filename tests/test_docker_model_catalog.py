"""Docker smoke catalogue is an overlay; production YAML is unchanged."""

from __future__ import annotations

from pathlib import Path

from src.generation.registry import ModelRegistry, load_model_catalog, populate_model_registry
from src.utils.app_paths import config_path


def test_production_catalog_vllm_is_hpc_only():
    data = load_model_catalog(str(config_path("models.yaml")))
    vllm = [f["model_name"] for f in data["providers"]["vllm"]["families"]]
    assert vllm == [
    "Qwen3.6-27B",
    "Qwen3.6-30B-A3B",
    "Qwen3.6-35B-A3B-FP8",
    ]
    assert "qwen3:8b" not in vllm


def test_docker_catalog_adds_host_ollama_and_keeps_hpc():
    path = config_path("models.docker.yaml")
    assert path.is_file()
    data = load_model_catalog(str(path))
    families = data["providers"]["vllm"]["families"]
    names = [f["model_name"] for f in families]
    assert names[0] == "qwen3:8b"
    assert "Qwen3.6-27B" in names
    assert "Qwen3.6-30B-A3B" in names
    host = next(f for f in families if f["id"] == "ollama_qwen3_8b")
    assert host["think_mode"] == "none"


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
    host = next(f for f in vllm if f.id == "ollama_qwen3_8b")
    assert host.model_name == "qwen3:8b"
    assert host.think_mode == "none"
