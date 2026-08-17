"""vLLM Qwen3.6 thinking control, token budgets, and PC/HPC provider parity.

Deployment contract under test
------------------------------
PC  : Ollama  + qwen3:8b                -> top-level ``think`` bool in /api/chat.
HPC : vLLM    + Qwen3.6-35B-A3B-FP8     -> ``chat_template_kwargs.enable_thinking``
      (32768 context)                     per request; the served model id is NEVER
                                          suffixed with /think or /nothink.
Mode mapping (single source: ModelFamily.get_execution_params):
      Standard/fast -> think OFF, max_tokens 4096
      Deep          -> think ON,  max_tokens 12288 (reasoning + answer budget)
"""

from __future__ import annotations

from pathlib import Path

import httpx

from src.generation.client import LLMClient, LLMResponse
from src.generation.registry import (
    ModelRegistry,
    OpenAICompatibleProvider,
    load_model_catalog,
    populate_model_registry,
    resolve_family_for_provider,
)
from src.utils.app_paths import config_path

FP8 = "Qwen3.6-35B-A3B-FP8"


# ── httpx capture doubles (same style as tests/test_openai_compat_url.py) ────

class _FakeChatResp:
    status_code = 200

    def __init__(self, reasoning: str = "", content: str = "ok") -> None:
        self._reasoning = reasoning
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        msg = {"content": self._content}
        if self._reasoning:
            msg["reasoning_content"] = self._reasoning
        return {
            "choices": [{"message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    def iter_lines(self):
        if self._reasoning:
            yield ('data: {"choices":[{"delta":{"reasoning_content":'
                   f'"{self._reasoning}"' + "}}]}")
        yield f'data: {{"choices":[{{"delta":{{"content":"{self._content}"}}}}]}}'
        yield "data: [DONE]"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _CapturingClient:
    """Captures the exact request bodies sent through httpx.Client."""

    posts: list = []
    streams: list = []
    reasoning: str = ""
    content: str = "ok"

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, **kw):
        type(self).posts.append({"url": url, "json": json})
        return _FakeChatResp(type(self).reasoning, type(self).content)

    def stream(self, method, url, json=None, **kw):
        type(self).streams.append({"url": url, "json": json})
        return _FakeChatResp(type(self).reasoning, type(self).content)

    @classmethod
    def reset(cls, reasoning: str = "", content: str = "ok"):
        cls.posts = []
        cls.streams = []
        cls.reasoning = reasoning
        cls.content = content


# ── helpers ──────────────────────────────────────────────────────────────────

def _fresh_prod_registry() -> ModelRegistry:
    reg = ModelRegistry()
    populate_model_registry(reg, load_model_catalog(str(config_path("models.yaml"))))
    return reg


def _fp8_family(reg: ModelRegistry):
    fam = reg.get("qwen3.6_35b_a3b_fp8")
    assert fam is not None, "HPC deployed family must exist in config/models.yaml"
    return fam


# ── 1. catalog / registry resolution ─────────────────────────────────────────

def test_hpc_fp8_family_template_mode_and_context():
    fam = _fp8_family(_fresh_prod_registry())
    assert fam.provider == "vllm"
    assert fam.model_name == FP8
    assert fam.think_mode == "template"
    assert fam.context_window == 32768
    assert fam.thinking_capable is True


def test_no_family_anywhere_uses_suffix_mode():
    """Hard guarantee: /think|/nothink name mangling is gone from ALL catalogs."""
    for catalog in ("models.yaml", "models.docker.yaml"):
        data = load_model_catalog(str(config_path(catalog)))
        for provider, cfg in data["providers"].items():
            for f in cfg["families"]:
                assert f.get("think_mode") != "suffix", f"{catalog}:{provider}:{f['id']}"


def test_vllm_entries_default_to_template_mode():
    """A vllm family with no explicit think_mode defaults to 'template'."""
    reg = ModelRegistry()
    populate_model_registry(reg, {"providers": {"vllm": {"families": [
        {"id": "x", "display_name": "x", "model_name": "X", "context_window": 1,
         "thinking_capable": True},
    ]}}})
    assert reg.get("x").think_mode == "template"


def test_legacy_suffix_entries_migrate_to_template():
    """Old catalog copies with think_mode: suffix are auto-migrated, never mangled."""
    reg = ModelRegistry()
    populate_model_registry(reg, {"providers": {"vllm": {"families": [
        {"id": "legacy", "display_name": "legacy", "model_name": "L",
         "context_window": 1, "thinking_capable": True, "think_mode": "suffix"},
    ]}}})
    assert reg.get("legacy").think_mode == "template"


# ── 2. payload shapes (unit level) ───────────────────────────────────────────

def _payload(provider, think, think_mode):
    return provider._payload(
        model=FP8, messages=[{"role": "user", "content": "q"}],
        temperature=0.0, max_tokens=4096, num_ctx=32768,
        stream=False, think=think, think_mode=think_mode,
    )


def test_payload_template_thinking_off_and_on():
    p = OpenAICompatibleProvider(base_url="http://x:1")
    off = _payload(p, False, "template")
    assert off["chat_template_kwargs"] == {"enable_thinking": False}
    assert off["model"] == FP8
    on = _payload(p, True, "template")
    assert on["chat_template_kwargs"] == {"enable_thinking": True}
    assert on["model"] == FP8


def test_payload_never_mangles_model_name():
    p = OpenAICompatibleProvider(base_url="http://x:1")
    for think in (None, False, True):
        for mode in ("template", "none", "key"):  # incl. unrecognized values
            body = _payload(p, think, mode)
            assert body["model"] == FP8
            assert "/think" not in body["model"] and "/nothink" not in body["model"]
            if think is None or mode != "template":
                assert "chat_template_kwargs" not in body


# ── 3. wire level through the provider (generate + stream) ───────────────────

def test_generate_sends_enable_thinking_and_keeps_model_verbatim(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _CapturingClient)
    p = OpenAICompatibleProvider(base_url="http://hpc:8001")
    for think, want in ((False, False), (True, True)):
        _CapturingClient.reset()
        p.generate(model=FP8, prompt="q", max_tokens=4096 if not think else 12288,
                   think=think, think_mode="template")
        body = _CapturingClient.posts[-1]["json"]
        assert body["model"] == FP8
        assert body["chat_template_kwargs"] == {"enable_thinking": want}
        assert body["max_tokens"] == (4096 if think is False else 12288)


def test_stream_sends_enable_thinking_and_keeps_model_verbatim(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _CapturingClient)
    p = OpenAICompatibleProvider(base_url="http://hpc:8001")
    _CapturingClient.reset()
    list(p.generate_stream(model=FP8, prompt="q", think=False, think_mode="template"))
    body = _CapturingClient.streams[-1]["json"]
    assert body["model"] == FP8
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["stream"] is True


# ── 4. end to end via LLMClient (catalog think_mode resolution) ──────────────

def test_llmclient_vllm_fp8_standard_off_deep_on(monkeypatch):
    """Standard -> enable_thinking False; Deep -> True; model id verbatim."""
    monkeypatch.setattr(httpx, "Client", _CapturingClient)
    c = LLMClient(provider="vllm", model=FP8, base_url="http://hpc:8001")
    assert c._family_think_mode() == "template"

    c.think = False  # Standard/fast
    _CapturingClient.reset()
    c.generate("q")
    body = _CapturingClient.posts[-1]["json"]
    assert body["model"] == FP8
    assert body["chat_template_kwargs"] == {"enable_thinking": False}

    c.think = True  # Deep
    _CapturingClient.reset()
    list(c.generate_stream("q"))
    body = _CapturingClient.streams[-1]["json"]
    assert body["model"] == FP8
    assert body["chat_template_kwargs"] == {"enable_thinking": True}


def test_llmclient_ollama_still_uses_top_level_think(monkeypatch):
    """PC parity: Ollama receives the top-level `think` bool (not kwargs)."""
    monkeypatch.setattr(httpx, "Client", _CapturingClient)
    c = LLMClient(provider="ollama", model="qwen3:8b", base_url="http://pc:11434")
    for think in (False, True):
        c.think = think
        _CapturingClient.reset()
        c.generate("q")
        body = _CapturingClient.posts[-1]["json"]
        assert body["model"] == "qwen3:8b"
        assert body["think"] is think
        assert "chat_template_kwargs" not in body


def test_llmclient_think_unset_leaves_server_default(monkeypatch):
    """think=None -> neither provider-side nor template-side forcing."""
    monkeypatch.setattr(httpx, "Client", _CapturingClient)
    c = LLMClient(provider="vllm", model=FP8, base_url="http://hpc:8001")
    _CapturingClient.reset()
    c.generate("q")
    body = _CapturingClient.posts[-1]["json"]
    assert "chat_template_kwargs" not in body
    assert body["model"] == FP8


# ── 5. mode → parameters (Standard 4096 / Deep 12288 locked) ─────────────────

def test_mode_params_standard_and_deep_locked():
    fam = _fp8_family(_fresh_prod_registry())
    fast = fam.get_execution_params("fast")
    deep = fam.get_execution_params("deep")
    assert fast["max_tokens"] == 4096
    assert fast["thinking"] is False
    assert fast["temperature"] == 0.0
    assert deep["max_tokens"] == 12288
    assert deep["thinking"] is True
    assert deep["temperature"] == 0.2
    # Deep budget exists because thinking consumes max_tokens before answering.
    assert deep["max_tokens"] > fast["max_tokens"]


def test_mode_params_flow_into_request_body(monkeypatch):
    """The 4096/12288 budgets reach the wire unchanged."""
    monkeypatch.setattr(httpx, "Client", _CapturingClient)
    c = LLMClient(provider="vllm", model=FP8, base_url="http://hpc:8001")
    for mode, want in (("fast", 4096), ("deep", 12288)):
        params = _fp8_family(_fresh_prod_registry()).get_execution_params(mode)
        c.max_tokens = params["max_tokens"]
        c.think = params["thinking"]
        _CapturingClient.reset()
        c.generate("q")
        body = _CapturingClient.posts[-1]["json"]
        assert body["max_tokens"] == want
        assert body["chat_template_kwargs"] == {"enable_thinking": params["thinking"]}


# ── 6. boot-time provider/family parity (PC vs HPC) ─────────────────────────

def test_resolve_family_pc_ollama_unchanged():
    reg = _fresh_prod_registry()
    fam = resolve_family_for_provider(reg, "ollama", "qwen3")
    assert fam.id == "qwen3" and fam.provider == "ollama"


def test_resolve_family_hpc_uses_vllm_model_env():
    reg = _fresh_prod_registry()
    fam = resolve_family_for_provider(reg, "vllm", "qwen3", preferred_model=FP8)
    assert fam.id == "qwen3.6_35b_a3b_fp8"
    assert fam.model_name == FP8
    assert fam.context_window == 32768


def test_resolve_family_hpc_without_env_falls_back_to_first_vllm_family():
    reg = _fresh_prod_registry()
    fam = resolve_family_for_provider(reg, "vllm", "qwen3", preferred_model=None)
    assert fam.provider == "vllm"
    assert fam is reg.list_by_provider("vllm")[0]


def test_resolve_family_keeps_family_already_on_provider():
    reg = _fresh_prod_registry()
    fam = resolve_family_for_provider(reg, "vllm", "qwen3.6_35b_a3b_fp8")
    assert fam.id == "qwen3.6_35b_a3b_fp8"


# ── 7. answer parity: reasoning never glued onto the visible answer ─────────

def test_nonstream_answer_excludes_reasoning(monkeypatch):
    """Ollama returns content only; the vLLM adapter must match that shape."""
    monkeypatch.setattr(httpx, "Client", _CapturingClient)
    _CapturingClient.reset(reasoning="internal chain of thought", content="final answer")
    p = OpenAICompatibleProvider(base_url="http://hpc:8001")
    resp = p.generate(model=FP8, prompt="q", think=True, think_mode="template")
    assert resp.text == "final answer"
    assert "chain of thought" not in resp.text
    # Reasoning is preserved for consumers that want it (raw_response).
    assert resp.raw_response["choices"][0]["message"]["reasoning_content"]


def test_stream_still_surfaces_reasoning_events(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _CapturingClient)
    _CapturingClient.reset(reasoning="thinking...", content="answer")
    p = OpenAICompatibleProvider(base_url="http://hpc:8001")
    events = list(p.generate_stream(model=FP8, prompt="q", think=True, think_mode="template"))
    kinds = [e["type"] for e in events]
    assert "reasoning" in kinds and "tokens" in kinds and "done" in kinds
    assert next(e for e in events if e["type"] == "reasoning")["text"] == "thinking..."


# ── 8. production code never depends on test doubles ────────────────────────

def test_production_code_has_no_mock_references():
    """unittest.mock is a test dependency; src/ must never reference it
    (regression guard for the generator.py MagicMock NameError)."""
    src = Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for py in src.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        text = py.read_text(encoding="utf-8")
        if "MagicMock" in text or "unittest.mock" in text or "unittest import" in text:
            offenders.append(str(py))
    assert offenders == []


def test_generator_runs_with_plain_duck_typed_client():
    """AnswerGenerator must work with ANY client exposing plain attributes —
    no mock-type special-casing required in production code."""
    from src.generation.generator import AnswerGenerator
    from src.retrieval.result import RetrievedResult

    class StubClient:
        provider = "vllm"
        model = FP8
        temperature = 0.0
        max_tokens = 4096
        num_ctx = 32768

        def generate(self, prompt, system=None, **kw):
            return LLMResponse(text="grounded answer", model=self.model,
                               prompt_tokens=10, completion_tokens=5,
                               total_tokens=15, latency_ms=1.0)

    gen = AnswerGenerator(llm_client=StubClient())
    result = gen.generate(
        "q",
        [RetrievedResult(doc_id="d1", question="Q", answer="A",
                         score=0.9, retrieval_method="rrf")],
    )
    assert result.answer == "grounded answer"
    assert result.model == FP8
    assert result.sources_used == ["d1"]
