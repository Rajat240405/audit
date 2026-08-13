"""Long-doc hits must not send the full parent annual report to the LLM."""

from __future__ import annotations

import numpy as np

from src.generation.generator import AnswerGenerator, build_user_prompt, compact_documents_with_llm
from src.generation.registry import ModelFamily
from src.models.qa_record import QARecord, QARecordMetadata
from src.retrieval.hybrid.pipeline import HybridRAGPipeline
from src.retrieval.result import RetrievedResult


class FakeEmbedder:
    model_name = "fake-test-embedder"
    embedding_dim = 8

    def embed(self, text: str) -> np.ndarray:
        seed = abs(hash(text)) % (2**31)
        rng = np.random.RandomState(seed)
        v = rng.randn(8).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v

    def embed_batch(self, texts, batch_size=1, show_progress=False):
        return np.stack([self.embed(t) for t in texts])


class FakeReranker:
    last_truncated_docs: set[str] = set()

    def rerank(self, query, candidates, k=5, doc_texts=None):
        return list(candidates)[:k]


class _RecordingLLM:
    """Records every prompt sent; never sees 400k bodies."""

    def __init__(self):
        self.prompts: list[str] = []
        self.provider = "ollama"
        self.model = "stub"
        self.think = False

    def generate(self, prompt, system=None, **k):
        self.prompts.append(prompt)
        from src.generation.client import LLMResponse

        return LLMResponse(text="ok", model="stub", prompt_tokens=1, completion_tokens=1, total_tokens=2, latency_ms=1.0)

    def generate_stream(self, prompt, system=None, **k):
        self.prompts.append(prompt)
        yield {"type": "tokens", "text": "ok"}
        yield {"type": "done"}


def _long_records():
    body = "\n".join([f"Paragraph {i} UNIQUE_TOKEN_{i} " + ("x" * 200) for i in range(2500)])
    assert len(body) > 400_000
    return [
        QARecord(
            question_id="incdoc-huge",
            question_text="Document: AR_2023-24_fake",
            answer_text=body,
            metadata=QARecordMetadata(
                ministry="EARTH SCIENCES",
                subject="Document: INCOIS Annual Report",
                document_type="annual_report",
            ),
        ),
        QARecord(
            question_id="18-0001",
            question_text="What is INCOIS?",
            answer_text="INCOIS is the Indian National Centre for Ocean Information Services.",
            metadata=QARecordMetadata(
                ministry="EARTH SCIENCES",
                subject="INCOIS",
                document_type="parliamentary_qa",
            ),
        ),
    ]


def test_A_retrieve_does_not_return_400k_parent():
    pipe = HybridRAGPipeline(
        records=_long_records(), embedder=FakeEmbedder(), reranker=FakeReranker()
    )
    hits, _ = pipe.retrieve("UNIQUE_TOKEN_55 What is INCOIS", top_k=5)
    assert any(h.doc_id == "incdoc-huge" for h in hits)
    huge = next(h for h in hits if h.doc_id == "incdoc-huge")
    assert len(huge.answer) < 20_000
    assert len(huge.answer) < len(_long_records()[0].answer_text) // 10


def test_B_E_chunk_evidence_keeps_parent_id():
    pipe = HybridRAGPipeline(
        records=_long_records(), embedder=FakeEmbedder(), reranker=FakeReranker()
    )
    hits, _ = pipe.retrieve("UNIQUE_TOKEN_55", top_k=5)
    huge = next(h for h in hits if h.doc_id == "incdoc-huge")
    assert huge.doc_id == "incdoc-huge"
    assert len(huge.answer) < len(_long_records()[0].answer_text)


def test_C_D_fast_deep_prompt_caps():
    fam = ModelFamily(
        id="t", display_name="t", provider="ollama", model_name="m",
        context_window=8192, thinking_capable=False,
    )
    assert fam.get_execution_params("fast")["max_doc_chars"] == 1000
    assert fam.get_execution_params("fast")["max_context_docs"] == 3
    assert fam.get_execution_params("deep")["max_doc_chars"] == 3000
    assert fam.get_execution_params("deep")["max_context_docs"] == 5

    huge = "INCOIS " + ("y" * 400_000)
    recs = [
        RetrievedResult("incdoc-huge", "Document: AR", huge, 1.0, "rrf"),
        RetrievedResult("18-1", "Q", "short INCOIS answer", 0.9, "rrf"),
    ]
    fast = build_user_prompt("What is INCOIS?", recs[:3], max_doc_chars=1000)
    deep = build_user_prompt("What is INCOIS?", recs[:5], max_doc_chars=3000)
    assert len(fast) < 20_000
    assert len(deep) < 40_000
    assert len(fast) <= len(deep) + 500


def test_F_compact_never_sends_full_parent():
    llm = _RecordingLLM()
    huge = "INCOIS centre " + ("z" * 400_000)
    recs = [RetrievedResult("incdoc-huge", "Document: AR", huge, 1.0, "rrf")]
    out = compact_documents_with_llm("What is INCOIS?", recs, max_doc_chars=1000, llm_client=llm)
    assert llm.prompts == []
    assert len(out) < 20_000
    assert "INCOIS" in out


def test_A_generate_stream_prompt_bounded():
    llm = _RecordingLLM()
    huge = "INCOIS " + ("w" * 400_000)
    recs = [
        RetrievedResult("incdoc-huge", "Document: AR", huge, 1.0, "rrf"),
        RetrievedResult("18-1", "What is INCOIS?", "INCOIS is a centre.", 0.8, "rrf"),
    ]
    gen = AnswerGenerator(llm_client=llm, max_context_docs=3, max_doc_chars=1000)
    list(gen.generate_stream("What is INCOIS?", recs))
    assert llm.prompts
    assert all(len(p) < 30_000 for p in llm.prompts)
    assert all("w" * 10_000 not in p for p in llm.prompts)


def test_G_short_qa_unchanged():
    recs = [
        RetrievedResult(
            "18-0001",
            "What is INCOIS?",
            "INCOIS is the Indian National Centre for Ocean Information Services.",
            1.0,
            "rrf",
        )
    ]
    prompt = build_user_prompt("What is INCOIS?", recs, max_doc_chars=1000)
    assert "Indian National Centre" in prompt
    assert "18-0001" in prompt
