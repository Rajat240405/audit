"""Fast mode generation context is max_context_docs, not retrieval top_k."""

from src.generation.generator import AnswerGenerator
from src.generation.registry import ModelFamily
from src.retrieval.result import RetrievedResult


def test_fast_profile_max_context_docs_is_3():
    fam = ModelFamily(
        id="x",
        display_name="x",
        provider="ollama",
        model_name="m",
        context_window=8192,
        thinking_capable=False,
    )
    assert fam.get_execution_params("fast")["max_context_docs"] == 3
    assert fam.get_execution_params("deep")["max_context_docs"] == 5


def test_generate_stream_uses_only_max_context_docs():
    class _Stub:
        provider = "ollama"
        model = "stub"
        think = False

        def generate_stream(self, prompt, system=None):
            yield {"type": "tokens", "text": "ok"}
            yield {"type": "done"}

    hits = [
        RetrievedResult(
            doc_id=f"doc-{i}",
            question=f"Q{i}",
            answer=f"A{i}",
            score=1.0,
            retrieval_method="rrf",
        )
        for i in range(5)
    ]
    gen = AnswerGenerator(llm_client=_Stub(), max_context_docs=3, compression_enabled=False)
    events = list(gen.generate_stream("q", hits))
    meta = next(e for e in events if e.get("type") == "meta")
    assert meta["sources_used"] == ["doc-0", "doc-1", "doc-2"]
