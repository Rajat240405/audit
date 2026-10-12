"""Context-overflow recovery for vLLM/OpenAI-compatible generation.

Covers: overflow classification (helper), provider body retention (registry),
non-stream and stream recovery in AnswerGenerator, the per-request retry cap,
no duplicated streamed text, graceful exhaustion, and unchanged HTTP 413 and
unrelated-400 behaviour. Stubs only; no network, no index, no real model.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.generation.client import LLMResponse
from src.generation.context_overflow import (
    describe_http_error,
    error_text,
    is_context_overflow_error,
)
from src.generation.evidence import AGGRESSIVE_HEAL_POOL, AGGRESSIVE_HEAL_TOKENS
from src.generation.generator import (
    CONTEXT_TOO_LARGE_MESSAGE,
    OVERFLOW_RETRY_STEPS,
    AnswerGenerator,
)
from src.generation.policy import resolve_execution
from src.generation.registry import ModelFamily, OpenAICompatibleProvider
from src.retrieval.result import RetrievedResult

# ── fixtures / builders ───────────────────────────────────────────────────

VLLM_CTX_MESSAGE = (
    "This model's maximum context length is 32768 tokens. However, you requested "
    "12288 output tokens and your prompt contains at least 20481 input tokens, for a "
    "total of at least 32769 tokens. Please reduce the length of the input prompt or "
    "the number of requested output tokens."
)
VLLM_CTX_BODY = {"error": {"message": VLLM_CTX_MESSAGE, "type": "BadRequestError",
                           "param": "input_tokens", "code": 400}}
UNRELATED_BODY = {"error": {"message": "reasoning_effort 'high' is not supported by this model",
                            "type": "BadRequestError", "param": "reasoning_effort", "code": 400}}
URL = "http://vllm.test/v1/chat/completions"


def _http_error(status: int, body: dict | None = None, text: str | None = None) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", URL)
    content = text.encode() if text is not None else json.dumps(body or {}).encode()
    resp = httpx.Response(status, request=req, content=content,
                          headers={"content-type": "application/json"})
    return httpx.HTTPStatusError(f"HTTP {status}", request=req, response=resp)


def _plan(mode="deep", ctx=32768):
    fam = ModelFamily(
        id="t-plan", display_name="t", provider="vllm", model_name="t-model",
        context_window=ctx, thinking_capable=True,
    )
    return resolve_execution(fam, mode, "vllm")


def _res(i, answer):
    return RetrievedResult(
        doc_id=f"d{i}", question=f"Question {i}?", answer=answer,
        score=1.0 - i * 0.01, retrieval_method="rrf_fusion", metadata={},
    )


def _results(n=6):
    return [_res(i, "INCOIS " + f"content {i}. " * 60) for i in range(n)]


class _ScriptedLLM:
    """Each call consumes one script entry.

    generate(): an exception is raised; a string is the answer text.
    generate_stream(): a list of events; an exception inside the list is raised
    at that position (so earlier events have already been yielded).
    """

    def __init__(self, script):
        self.provider, self.model, self.think = "vllm", "stub", None
        self.script = list(script)
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt, system=None, **k):
        self.calls += 1
        self.prompts.append(prompt)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return LLMResponse(text=item, model="stub", prompt_tokens=1,
                           completion_tokens=1, total_tokens=2, latency_ms=1.0)

    def generate_stream(self, prompt, system=None, **k):
        self.calls += 1
        self.prompts.append(prompt)
        for ev in self.script.pop(0):
            if isinstance(ev, BaseException):
                raise ev
            yield ev


def _generator(llm, mode="deep"):
    gen = AnswerGenerator(llm_client=llm)
    gen.plan = _plan(mode)
    gen._write_debug_prompt = lambda text: None  # keep the app data dir untouched
    return gen


def _spy_prepare(gen):
    """Record the overrides each prepare_context() call received (no prompt text)."""
    calls: list[dict] = []
    original = gen.prepare_context

    def spy(question, results, **kwargs):
        calls.append(dict(kwargs))
        return original(question, results, **kwargs)

    gen.prepare_context = spy
    return calls


def _tokens(events) -> str:
    return "".join(e["text"] for e in events if e.get("type") == "tokens")


# ── 1. classification helper ──────────────────────────────────────────────

@pytest.mark.parametrize("body", [
    VLLM_CTX_BODY,
    {"error": {"message": "This model's maximum context length is 32768 tokens."}},
    {"error": {"message": "your prompt contains at least 28673 input tokens"}},
    {"error": {"message": "invalid request", "param": "input_tokens"}},
    {"error": {"message": "context_length_exceeded"}},
])
def test_vllm_context_messages_are_overflow(body):
    assert is_context_overflow_error(_http_error(400, body)) is True


@pytest.mark.parametrize("err", [
    _http_error(400, UNRELATED_BODY),            # bare validation 400
    _http_error(400, text=""),                   # 400 with no body
    _http_error(400, text="Bad Request"),        # plain-text 400
    _http_error(413, VLLM_CTX_BODY),             # 413 belongs to the existing heal path
    _http_error(500, VLLM_CTX_BODY),             # server error, not a 400
])
def test_unrelated_or_non_400_errors_are_not_overflow(err):
    assert is_context_overflow_error(err) is False


def test_wrapped_http_error_is_classified_through_cause():
    cause = _http_error(400, VLLM_CTX_BODY)
    try:
        try:
            raise cause
        except httpx.HTTPStatusError as inner:
            raise RuntimeError("OpenAI-compatible generate failed: 400") from inner
    except RuntimeError as wrapped:
        assert is_context_overflow_error(wrapped) is True
        assert is_context_overflow_error(RuntimeError("timed out")) is False


def test_unread_stream_body_fails_closed_without_raising():
    req = httpx.Request("POST", URL)
    resp = httpx.Response(400, request=req,
                          stream=httpx.ByteStream(json.dumps(VLLM_CTX_BODY).encode()))
    err = httpx.HTTPStatusError("400", request=req, response=resp)
    assert error_text(resp) == ""                # body not read yet: no exception
    assert is_context_overflow_error(err) is False


def test_log_summary_is_truncated_and_has_no_prompt_material():
    long_body = {"error": {"message": "maximum context length " + "x" * 5000}}
    summary = describe_http_error(_http_error(400, long_body).response)
    assert summary.startswith("HTTP 400: maximum context length")
    assert len(summary) <= 230


# ── 2. OpenAI-compatible provider: body retained, no double raise ─────────

@pytest.fixture
def vllm_http(monkeypatch):
    """Route the provider's own httpx.Client through a MockTransport."""
    real_client = httpx.Client
    state: dict = {"handler": None}

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(state["handler"])
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    return state


def _unread_stream_400(body: dict) -> httpx.Response:
    """A live-network-like response: the body is NOT loaded until read()."""
    return httpx.Response(400, stream=httpx.ByteStream(json.dumps(body).encode()),
                          headers={"content-type": "application/json"})


def test_provider_stream_400_retains_body_for_classification(vllm_http):
    vllm_http["handler"] = lambda req: _unread_stream_400(VLLM_CTX_BODY)
    provider = OpenAICompatibleProvider(base_url="http://vllm.test")
    events = []
    with pytest.raises(httpx.HTTPStatusError) as info:
        for ev in provider.generate_stream(model="m", prompt="p", base_url="http://vllm.test"):
            events.append(ev)
    assert events == []                                    # raised before any output
    assert "maximum context length" in info.value.response.text   # body was read
    assert is_context_overflow_error(info.value) is True


def test_provider_stream_unrelated_400_is_not_overflow(vllm_http):
    vllm_http["handler"] = lambda req: _unread_stream_400(UNRELATED_BODY)
    provider = OpenAICompatibleProvider(base_url="http://vllm.test")
    with pytest.raises(httpx.HTTPStatusError) as info:
        list(provider.generate_stream(model="m", prompt="p", base_url="http://vllm.test"))
    assert is_context_overflow_error(info.value) is False


def test_provider_non_stream_400_wrapped_error_is_overflow(vllm_http):
    vllm_http["handler"] = lambda req: httpx.Response(400, json=VLLM_CTX_BODY)
    provider = OpenAICompatibleProvider(base_url="http://vllm.test")
    with pytest.raises(RuntimeError) as info:
        provider.generate(model="m", prompt="p", base_url="http://vllm.test")
    assert is_context_overflow_error(info.value) is True


def test_provider_stream_success_still_yields_answer_tokens(vllm_http):
    sse = (
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    vllm_http["handler"] = lambda req: httpx.Response(
        200, content=sse, headers={"content-type": "text/event-stream"})
    provider = OpenAICompatibleProvider(base_url="http://vllm.test")
    events = list(provider.generate_stream(model="m", prompt="p", base_url="http://vllm.test"))
    assert _tokens(events) == "Hello world"


# ── 3. non-stream generation recovery ─────────────────────────────────────

@pytest.mark.parametrize("mode", ["fast", "deep"])
def test_normal_generation_is_single_call_for_fast_and_deep(mode):
    llm = _ScriptedLLM(["ok"])
    gen = _generator(llm, mode)
    spy = _spy_prepare(gen)
    out = gen.generate("INCOIS?", _results())
    assert out.answer == "ok"
    assert llm.calls == 1
    assert spy == [{}]                                     # no override on the normal path


def test_context_length_400_retries_with_smaller_evidence():
    llm = _ScriptedLLM([_http_error(400, VLLM_CTX_BODY), "recovered"])
    gen = _generator(llm)
    spy = _spy_prepare(gen)
    out = gen.generate("INCOIS?", _results())
    assert out.answer == "recovered"
    assert llm.calls == 2
    assert spy[1] == {"pool_override": 3, "budget_override": 4000}
    assert len(out.sources_used) <= 3
    assert len(llm.prompts[1]) < len(llm.prompts[0])       # smaller evidence was sent


def test_second_overflow_triggers_only_the_final_allowed_retry():
    llm = _ScriptedLLM([_http_error(400, VLLM_CTX_BODY),
                        _http_error(400, VLLM_CTX_BODY), "final"])
    gen = _generator(llm)
    spy = _spy_prepare(gen)
    out = gen.generate("INCOIS?", _results())
    assert out.answer == "final"
    assert llm.calls == 3
    assert spy == [{}, OVERFLOW_RETRY_STEPS[0], OVERFLOW_RETRY_STEPS[1]]
    assert spy[2] == {"pool_override": 2, "budget_override": AGGRESSIVE_HEAL_TOKENS}
    assert len(out.sources_used) <= 2


def test_413_during_overflow_retry_counts_as_failed_retry():
    llm = _ScriptedLLM([_http_error(400, VLLM_CTX_BODY), _http_error(413, text=""), "final"])
    gen = _generator(llm)
    spy = _spy_prepare(gen)
    out = gen.generate("INCOIS?", _results())
    assert out.answer == "final"
    assert llm.calls == 3 and spy[2] == OVERFLOW_RETRY_STEPS[1]


def test_413_on_every_overflow_retry_ends_gracefully_and_bounded():
    llm = _ScriptedLLM([_http_error(400, VLLM_CTX_BODY), _http_error(413, text=""),
                        _http_error(413, text="")])
    gen = _generator(llm)
    out = gen.generate("INCOIS?", _results())
    assert llm.calls == 3
    assert out.answer == CONTEXT_TOO_LARGE_MESSAGE


def test_exhausted_overflow_returns_graceful_message_not_traceback():
    llm = _ScriptedLLM([_http_error(400, VLLM_CTX_BODY)] * 3)
    gen = _generator(llm)
    spy = _spy_prepare(gen)
    out = gen.generate("INCOIS?", _results())
    assert llm.calls == 3 and len(spy) == 3               # exactly 1 + 2 retries
    assert out.answer == CONTEXT_TOO_LARGE_MESSAGE
    assert "Context Too Large" in out.answer
    assert "shorter or more specific question" in out.answer
    for leaked in ("Traceback", "HTTP 400", "maximum context length", "BadRequestError"):
        assert leaked not in out.answer
    assert out.sources_used == []


def test_unrelated_400_is_raised_not_retried():
    llm = _ScriptedLLM([_http_error(400, UNRELATED_BODY)])
    gen = _generator(llm)
    spy = _spy_prepare(gen)
    with pytest.raises(httpx.HTTPStatusError):
        gen.generate("INCOIS?", _results())
    assert llm.calls == 1 and len(spy) == 1


def test_unrelated_wrapped_runtime_error_is_raised_not_retried():
    cause = _http_error(400, UNRELATED_BODY)
    wrapped = RuntimeError("OpenAI-compatible generate failed")
    wrapped.__cause__ = cause
    llm = _ScriptedLLM([wrapped])
    gen = _generator(llm)
    spy = _spy_prepare(gen)
    with pytest.raises(RuntimeError):
        gen.generate("INCOIS?", _results())
    assert llm.calls == 1 and len(spy) == 1


def test_http_413_non_stream_still_heals_once_with_aggressive_budget():
    llm = _ScriptedLLM([_http_error(413, text=""), "healed"])
    gen = _generator(llm)
    spy = _spy_prepare(gen)
    out = gen.generate("INCOIS?", _results())
    assert out.answer == "healed"
    assert llm.calls == 2
    assert spy[1] == {"pool_override": AGGRESSIVE_HEAL_POOL,
                      "budget_override": AGGRESSIVE_HEAL_TOKENS}


def test_http_413_non_stream_exhausted_keeps_original_graceful_text():
    llm = _ScriptedLLM([_http_error(413, text=""), _http_error(413, text="")])
    gen = _generator(llm)
    out = gen.generate("INCOIS?", _results())
    assert llm.calls == 2
    assert "Context Size Exceeded" in out.answer          # unchanged 413 wording


# ── 4. streaming generation recovery ──────────────────────────────────────

@pytest.mark.parametrize("mode", ["fast", "deep"])
def test_normal_stream_is_single_pass_for_fast_and_deep(mode):
    llm = _ScriptedLLM([[{"type": "tokens", "text": "Answer."}, {"type": "done"}]])
    gen = _generator(llm, mode)
    events = list(gen.generate_stream("INCOIS?", _results()))
    assert _tokens(events) == "Answer."
    assert events[-1]["type"] == "meta"
    assert llm.calls == 1


def test_stream_overflow_before_output_retries_and_emits_once():
    llm = _ScriptedLLM([
        [_http_error(400, VLLM_CTX_BODY)],
        [{"type": "tokens", "text": "Answer."}, {"type": "done"}],
    ])
    gen = _generator(llm)
    spy = _spy_prepare(gen)
    events = list(gen.generate_stream("INCOIS?", _results()))
    assert _tokens(events) == "Answer."                    # not duplicated
    assert llm.calls == 2
    assert spy[1] == {"pool_override": 3, "budget_override": 4000}
    meta = events[-1]
    assert meta["type"] == "meta" and len(meta["sources_used"]) <= 3


def test_stream_second_overflow_triggers_only_final_retry():
    llm = _ScriptedLLM([
        [_http_error(400, VLLM_CTX_BODY)],
        [_http_error(400, VLLM_CTX_BODY)],
        [{"type": "tokens", "text": "Final."}, {"type": "done"}],
    ])
    gen = _generator(llm)
    spy = _spy_prepare(gen)
    events = list(gen.generate_stream("INCOIS?", _results()))
    assert _tokens(events) == "Final."
    assert llm.calls == 3
    assert spy == [{}, OVERFLOW_RETRY_STEPS[0], OVERFLOW_RETRY_STEPS[1]]
    assert len(events[-1]["sources_used"]) <= 2


def test_stream_overflow_after_partial_output_never_restarts():
    llm = _ScriptedLLM([
        [{"type": "tokens", "text": "Partial answer. "}, _http_error(400, VLLM_CTX_BODY)],
        [{"type": "tokens", "text": "SHOULD NOT APPEAR"}, {"type": "done"}],
    ])
    gen = _generator(llm)
    events = []
    with pytest.raises(httpx.HTTPStatusError):
        for ev in gen.generate_stream("INCOIS?", _results()):
            events.append(ev)
    assert llm.calls == 1                                  # no restart
    assert _tokens(events) == "Partial answer. "           # shown exactly once
    assert "SHOULD NOT APPEAR" not in _tokens(events)


def test_stream_reasoning_output_also_blocks_overflow_retry():
    llm = _ScriptedLLM([
        [{"type": "reasoning", "text": "thinking"}, _http_error(400, VLLM_CTX_BODY)],
        [{"type": "tokens", "text": "NOPE"}, {"type": "done"}],
    ])
    gen = _generator(llm, mode="deep")                     # plan.thinking → reasoning forwarded
    events = []
    with pytest.raises(httpx.HTTPStatusError):
        for ev in gen.generate_stream("INCOIS?", _results()):
            events.append(ev)
    assert llm.calls == 1 and _tokens(events) == ""


def test_stream_exhausted_overflow_yields_graceful_answer():
    llm = _ScriptedLLM([[_http_error(400, VLLM_CTX_BODY)]] * 3)
    gen = _generator(llm)
    events = list(gen.generate_stream("INCOIS?", _results()))
    assert llm.calls == 3
    assert _tokens(events) == CONTEXT_TOO_LARGE_MESSAGE    # non-empty, no traceback
    assert events[-1]["type"] == "meta" and events[-1]["sources_used"] == []


def test_stream_unrelated_400_is_raised_not_retried():
    llm = _ScriptedLLM([[_http_error(400, UNRELATED_BODY)]])
    gen = _generator(llm)
    with pytest.raises(httpx.HTTPStatusError):
        list(gen.generate_stream("INCOIS?", _results()))
    assert llm.calls == 1


def test_stream_413_still_heals_once_with_aggressive_budget():
    llm = _ScriptedLLM([[_http_error(413, text="")],
                        [{"type": "tokens", "text": "healed"}, {"type": "done"}]])
    gen = _generator(llm)
    spy = _spy_prepare(gen)
    events = list(gen.generate_stream("INCOIS?", _results()))
    assert _tokens(events) == "healed"
    assert llm.calls == 2
    assert spy[1] == {"pool_override": AGGRESSIVE_HEAL_POOL,
                      "budget_override": AGGRESSIVE_HEAL_TOKENS}
    assert len(events[-1]["sources_used"]) <= AGGRESSIVE_HEAL_POOL
