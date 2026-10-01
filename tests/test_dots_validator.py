"""Tests for the PicoDet -> DOTS table pipeline.

Covers the four layers of the unified extractor:

1. **PicoDet detection** — "Table" label handling, the 0.25 default threshold,
   env-configurability, and the CPU-only inference path.
2. **The DOTS validator** — deterministic, non-AI, conservative. Rejects
   pathological output (empty / truncated / repetition-looped / malformed) and
   accepts ordinary messy parliamentary tables.
3. **Page routing** — a detected table goes to DOTS, legacy table strategies are
   bypassed, page order survives, and DOTS failure is a hard failure that never
   overwrites an existing record.
4. **Change detection** — ``qa_content_hash`` is authoritative, ``content_hash``
   never participates, and ``source_sha256`` / ``extractor_version`` cannot make
   an unchanged record look changed.

Everything here is hermetic: PicoDet is exercised through an injected backend
and DOTS through a stubbed transport, so no Paddle install, no GPU and no
network are required.
"""

from __future__ import annotations

import json

import pytest

from src.data import dots_client as dots
from src.data import pdf_table_extract as pte
from src.data import table_detect as td

# ── helpers ──────────────────────────────────────────────────────────────────


def _box(label="Table", score=0.9, bbox=(10.0, 20.0, 100.0, 200.0)):
    return {"label": label, "score": score, "coordinate": list(bbox)}


def _layout(*elements):
    """Serialise a DOTS layout response."""
    return json.dumps(list(elements))


def _table_el(html="<table><tr><td>A</td><td>B</td></tr></table>", bbox=(0, 0, 10, 10)):
    return {"bbox": list(bbox), "category": "Table", "text": html}


def _text_el(text="Some prose.", bbox=(0, 0, 10, 10)):
    return {"bbox": list(bbox), "category": "Text", "text": text}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Reset global detector/client state and env between tests."""
    td.reset_backend()
    dots.reset_client()
    for var in (
        "DOTS_ENABLED", "DOTS_BASE_URL", "DOTS_MODEL", "DOTS_PROMPT_MODE",
        "DOTS_VALIDATOR_STRICT", "DOTS_MAX_LINE_REPEAT", "DOTS_TABLE_FORMAT",
        "PICODET_SCORE_THRESHOLD", "PICODET_DEVICE", "PICODET_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)
    yield
    td.reset_backend()
    dots.reset_client()


# ── 1. PicoDet detection ─────────────────────────────────────────────────────


class TestPicoDetDetection:
    def test_default_threshold_is_025(self):
        """The validated production default, per project decision."""
        assert td.DEFAULT_SCORE_THRESHOLD == 0.25
        assert td.score_threshold() == 0.25

    def test_threshold_is_env_configurable(self, monkeypatch):
        monkeypatch.setenv("PICODET_SCORE_THRESHOLD", "0.6")
        assert td.score_threshold() == 0.6

    def test_malformed_threshold_falls_back_to_default(self, monkeypatch):
        """A typo in .env must never break a nightly run."""
        monkeypatch.setenv("PICODET_SCORE_THRESHOLD", "zero-point-six")
        assert td.score_threshold() == 0.25

    def test_device_defaults_to_cpu(self):
        """The GPU belongs to DOTS; PicoDet must never contend for it."""
        assert td.device() == "cpu"

    def test_cpu_is_pinned_before_predictor_construction(self, monkeypatch):
        """paddle.set_device('cpu') must happen before any predictor is built."""
        import sys
        import types

        calls: list[str] = []
        fake_paddle = types.ModuleType("paddle")
        fake_paddle.set_device = lambda d: calls.append(f"set_device:{d}")
        monkeypatch.setitem(sys.modules, "paddle", fake_paddle)

        fake_ocr = types.ModuleType("paddleocr")

        class _LD:
            def __init__(self, **kw):
                calls.append(f"construct:{kw.get('device')}")

            def predict(self, arr):
                return []

        fake_ocr.LayoutDetection = _LD
        monkeypatch.setitem(sys.modules, "paddleocr", fake_ocr)

        td.reset_backend()
        td._load_backend()
        assert calls == ["set_device:cpu", "construct:cpu"]

    def test_table_label_is_recognised_case_insensitively(self):
        for label in ("Table", "table", "TABLE", " Table "):
            assert td.is_table_label(label), label

    def test_non_table_labels_are_rejected(self):
        for label in ("Text", "Title", "Figure", "List-item", "Picture", ""):
            assert not td.is_table_label(label), label

    def test_only_table_regions_are_returned(self):
        td.set_backend(lambda _b: [
            _box("Text", 0.99), _box("Table", 0.80), _box("Figure", 0.95),
        ])
        boxes = td.detect_tables(b"png")
        assert [b.label for b in boxes] == ["Table"]

    def test_detections_below_threshold_are_dropped(self):
        td.set_backend(lambda _b: [_box("Table", 0.10)])
        assert td.detect_tables(b"png") == []

    def test_detection_at_exactly_the_threshold_is_kept(self):
        """>= threshold, not > threshold."""
        td.set_backend(lambda _b: [_box("Table", 0.25)])
        assert len(td.detect_tables(b"png")) == 1

    def test_threshold_change_flips_the_outcome(self, monkeypatch):
        td.set_backend(lambda _b: [_box("Table", 0.30)])
        assert td.detect_tables(b"png")           # default 0.25 -> detected
        monkeypatch.setenv("PICODET_SCORE_THRESHOLD", "0.5")
        assert not td.detect_tables(b"png")       # 0.5 -> not detected

    def test_integer_class_id_maps_to_publaynet_table(self):
        """Backends that report only class ids still resolve correctly."""
        td.set_backend(lambda _b: [
            {"cls_id": td.PUBLAYNET_TABLE_CLASS_ID, "score": 0.9, "coordinate": [0, 0, 5, 5]},
            {"cls_id": 0, "score": 0.9, "coordinate": [0, 0, 5, 5]},
        ])
        assert [b.label for b in td.detect_tables(b"png")] == ["table"]

    def test_boxes_are_returned_in_reading_order(self):
        td.set_backend(lambda _b: [
            _box("Table", 0.9, (0, 500, 10, 600)),
            _box("Table", 0.9, (0, 100, 10, 200)),
            _box("Table", 0.9, (0, 300, 10, 400)),
        ])
        assert [b.bbox[1] for b in td.detect_tables(b"png")] == [100, 300, 500]

    def test_unparseable_detection_is_skipped_not_fatal(self):
        td.set_backend(lambda _b: [None, "garbage", {"no": "fields"}, _box("Table", 0.9)])
        assert len(td.detect_tables(b"png")) == 1

    def test_backend_failure_raises_rather_than_reporting_no_table(self):
        """Silently reporting 'no table' would downgrade every page unnoticed."""
        def _boom(_b):
            raise RuntimeError("paddle segfault")

        td.set_backend(_boom)
        with pytest.raises(td.TableDetectorUnavailable):
            td.detect_tables(b"png")

    def test_missing_backend_raises_typed_error(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "paddle", None)
        td.reset_backend()
        with pytest.raises(td.TableDetectorUnavailable):
            td.detect_tables(b"png")

    def test_is_available_never_raises(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "paddle", None)
        td.reset_backend()
        assert td.is_available() is False


# ── 2. The DOTS validator ────────────────────────────────────────────────────


class TestValidatorRejects:
    def test_empty_output(self):
        r = dots.validate_dots_output("")
        assert not r.ok and r.rule == dots.RULE_EMPTY and r.action == "reject"

    def test_truncated_finish_reason(self):
        """Highest-value rule: the server reports this as fact."""
        r = dots.validate_dots_output(_layout(_table_el()), finish_reason="length")
        assert not r.ok and r.rule == dots.RULE_TRUNCATED

    def test_repetition_loop(self):
        """The documented dots.ocr failure mode (ellipses/underscores)."""
        body = "\n".join(["<tr><td>...</td></tr>"] * 400)
        r = dots.validate_dots_output(_layout(_table_el(f"<table>{body}</table>")))
        assert not r.ok and r.rule == dots.RULE_LINE_REPEAT

    def test_ngram_saturation(self):
        text = ("the quick brown fox jumps over the lazy dog again " * 400)
        r = dots.validate_dots_output(_layout(_text_el(text)))
        assert not r.ok and r.rule == dots.RULE_NGRAM_SATURATION

    def test_malformed_json(self):
        r = dots.validate_dots_output('{"category": "Table", "text": ')
        assert not r.ok and r.rule == dots.RULE_BAD_JSON

    def test_unexpected_schema(self):
        r = dots.validate_dots_output(json.dumps({"unexpected": "shape"}))
        assert not r.ok and r.rule == dots.RULE_BAD_SCHEMA

    def test_unbalanced_html(self):
        r = dots.validate_dots_output(_layout(_table_el("<table><tr><td>A</td></tr>")))
        assert not r.ok and r.rule == dots.RULE_HTML_UNBALANCED

    def test_control_characters(self):
        r = dots.validate_dots_output(_layout(_text_el("good text\x00\x01\x02 more")))
        assert not r.ok and r.rule == dots.RULE_CONTROL_CHARS

    def test_severe_truncation_against_native_text(self):
        r = dots.validate_dots_output(
            _layout(_text_el("tiny bit of text")), native_char_count=20000
        )
        assert not r.ok and r.rule == dots.RULE_SEVERE_TRUNCATION


class TestValidatorAccepts:
    """A false rejection permanently blocks a record from improving, so the
    validator must tolerate ordinary messy source documents."""

    def test_normal_table(self):
        assert dots.validate_dots_output(_layout(_table_el())).ok

    def test_mixed_layout(self):
        assert dots.validate_dots_output(_layout(_text_el(), _table_el(), _text_el())).ok

    def test_code_fenced_output(self):
        assert dots.validate_dots_output(f"```json\n{_layout(_table_el())}\n```").ok

    def test_legitimately_repetitive_table(self):
        """Real annexures repeat 'Nil' and '-' constantly; 20 rows is normal."""
        rows = "".join("<tr><td>Nil</td><td>-</td></tr>" for _ in range(20))
        assert dots.validate_dots_output(_layout(_table_el(f"<table>{rows}</table>"))).ok

    def test_repetition_just_under_the_limit(self):
        body = "\n".join(["<tr><td>Nil</td></tr>"] * 29)
        assert dots.validate_dots_output(_layout(_table_el(f"<table>{body}</table>"))).ok

    def test_nested_table_tags_balance(self):
        html = "<table><tr><td><table><tr><td>x</td></tr></table></td></tr></table>"
        assert dots.validate_dots_output(_layout(_table_el(html))).ok

    def test_does_not_judge_semantic_correctness(self):
        """Nonsense numbers and gibberish are a transcription question, not a
        validator question. The validator must stay out of it."""
        html = "<table><tr><td>99999999</td><td>qwrtpz</td></tr></table>"
        assert dots.validate_dots_output(_layout(_table_el(html))).ok

    def test_ragged_columns_are_accepted(self):
        html = "<table><tr><td>a</td><td>b</td><td>c</td></tr><tr><td>d</td></tr></table>"
        assert dots.validate_dots_output(_layout(_table_el(html))).ok

    def test_non_strict_mode_is_configurable(self, monkeypatch):
        monkeypatch.setenv("DOTS_MAX_LINE_REPEAT", "500")
        body = "\n".join(["<tr><td>x</td></tr>"] * 100)
        cfg = dots.DotsConfig.from_env()
        assert dots.validate_dots_output(
            _layout(_table_el(f"<table>{body}</table>")), cfg=cfg
        ).ok


class TestValidatorFallback:
    def test_detector_fired_but_dots_found_no_table_is_fallback_not_failure(self):
        """A PicoDet false positive must cost the page nothing."""
        r = dots.validate_dots_output(
            _layout(_text_el("Just prose, no table here.")), detector_found_table=True
        )
        assert not r.ok
        assert r.rule == dots.RULE_NO_TABLE_FOUND
        assert r.action == "fallback"   # NOT "reject"

    def test_no_table_check_is_skipped_when_detector_did_not_fire(self):
        assert dots.validate_dots_output(
            _layout(_text_el("prose")), detector_found_table=False
        ).ok


class TestRendering:
    def test_page_markers_are_applied(self):
        out = dots.render_elements([_text_el("line one\nline two")], 7)
        assert out.splitlines() == ["[p7] line one", "[p7] line two"]

    def test_table_gets_an_explicit_marker(self):
        out = dots.render_elements([_table_el()], 3)
        assert out.splitlines()[0].startswith("[p3] TABLE")

    def test_picture_without_text_is_skipped(self):
        out = dots.render_elements([{"category": "Picture", "bbox": [0, 0, 1, 1]}], 1)
        assert out == ""

    def test_html_to_markdown_conversion_is_opt_in(self):
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        out = dots.render_elements([_table_el(html)], 1, table_format="markdown")
        assert "| A | B |" in out and "| 1 | 2 |" in out

    def test_markdown_conversion_never_destroys_unparseable_html(self):
        out = dots._html_table_to_markdown("<table>no rows here</table>")
        assert out == "<table>no rows here</table>"


# ── 3. Page routing ──────────────────────────────────────────────────────────


class _FakePage:
    """Minimal PyMuPDF page stand-in."""

    def __init__(self, text="Plain prose line."):
        self._text = text

    def get_pixmap(self, **_kw):
        class _Pix:
            width = height = 100

            def tobytes(self, _fmt):
                return b"\x89PNG-fake"

        return _Pix()


def _install_dots_stub(monkeypatch, content, finish="stop"):
    """Point the DOTS client at a canned response, no network involved."""
    monkeypatch.setenv("DOTS_BASE_URL", "http://stub:8200")
    monkeypatch.setenv("DOTS_MODEL", "stub-model")
    client = dots.DotsClient(dots.DotsConfig.from_env())
    monkeypatch.setattr(client, "_post_chat", lambda _b64: (content, finish))
    monkeypatch.setattr(dots, "get_client", lambda cfg=None: client)
    return client


class TestPageRouting:
    def test_routing_is_off_by_default(self):
        """Deploying this code must change nothing until an operator opts in."""
        assert pte.dots_routing_enabled() is False

    def test_routing_switches_on_via_env(self, monkeypatch):
        monkeypatch.setenv("DOTS_ENABLED", "true")
        assert pte.dots_routing_enabled() is True

    def test_detected_table_is_routed_to_dots(self, monkeypatch):
        monkeypatch.setenv("DOTS_ENABLED", "true")
        td.set_backend(lambda _b: [_box("Table", 0.9)])
        _install_dots_stub(monkeypatch, _layout(_table_el()))

        payload = pte._extract_page_payload_dots(_FakePage(), 4, [], 0)
        assert payload["type"] == pte.DOTS_PAYLOAD_TYPE
        assert payload["page_num"] == 4
        assert "TABLE" in payload["content"]

    def test_undetected_page_takes_the_legacy_prose_path(self, monkeypatch):
        monkeypatch.setenv("DOTS_ENABLED", "true")
        td.set_backend(lambda _b: [])  # no table
        lines = [{"text": "hello", "yc": 1.0, "x0": 0, "x1": 5, "y0": 0, "y1": 2}]
        payload = pte._extract_page_payload_dots(_FakePage(), 1, lines, 5)
        assert payload["type"] == "prose"
        assert payload["content"] == pte._render_merged(lines)

    def test_legacy_table_strategies_are_bypassed_when_routing_is_on(self, monkeypatch):
        """PyMuPDF find_tables() must not run on a DOTS-owned page."""
        monkeypatch.setenv("DOTS_ENABLED", "true")
        td.set_backend(lambda _b: [_box("Table", 0.9)])
        _install_dots_stub(monkeypatch, _layout(_table_el()))

        called = {"find_tables": False}

        class _Page(_FakePage):
            def find_tables(self, **_kw):
                called["find_tables"] = True
                raise AssertionError("find_tables() must not run under DOTS routing")

        monkeypatch.setattr(pte, "_page_lines", lambda _p: [
            {"text": "x" * 50, "yc": 1.0, "x0": 0, "x1": 5, "y0": 0, "y1": 2}
        ])
        payload = pte._extract_page_payload(_Page(), 1, enable_ocr=False)
        assert payload["type"] == pte.DOTS_PAYLOAD_TYPE
        assert called["find_tables"] is False

    def test_dots_text_fully_replaces_the_page(self, monkeypatch):
        """No merging with any legacy table extraction."""
        monkeypatch.setenv("DOTS_ENABLED", "true")
        td.set_backend(lambda _b: [_box("Table", 0.9)])
        _install_dots_stub(monkeypatch, _layout(_table_el("<table><tr><td>DOTS</td></tr></table>")))
        lines = [{"text": "LEGACY TEXT", "yc": 1.0, "x0": 0, "x1": 5, "y0": 0, "y1": 2}]
        payload = pte._extract_page_payload_dots(_FakePage(), 1, lines, 11)
        assert "DOTS" in payload["content"]
        assert "LEGACY TEXT" not in payload["content"]

    def test_page_order_is_preserved_through_the_stitcher(self):
        payloads = [
            {"page_num": 1, "type": "prose", "content": "page one", "tables": []},
            {"page_num": 2, "type": pte.DOTS_PAYLOAD_TYPE, "content": "page two table", "tables": []},
            {"page_num": 3, "type": "prose", "content": "page three", "tables": []},
        ]
        assert pte._stitch_multipage_payloads(payloads) == [
            "page one", "page two table", "page three",
        ]

    def test_dots_payload_passes_through_the_stitcher_unmodified(self):
        content = "[p1] TABLE #p1t1\n[p1] <table><tr><td>A</td></tr></table>"
        out = pte._stitch_multipage_payloads(
            [{"page_num": 1, "type": pte.DOTS_PAYLOAD_TYPE, "content": content, "tables": []}]
        )
        assert out == [content]


class TestFailureIsHard:
    def test_dots_unavailable_propagates(self, monkeypatch):
        """DOTS down must never be silently downgraded to a worse extractor."""
        monkeypatch.setenv("DOTS_ENABLED", "true")
        td.set_backend(lambda _b: [_box("Table", 0.9)])
        monkeypatch.setenv("DOTS_BASE_URL", "http://stub:8200")
        monkeypatch.setenv("DOTS_MODEL", "stub")
        client = dots.DotsClient(dots.DotsConfig.from_env())

        def _down(_b64):
            raise dots.DotsUnavailable("connection refused")

        monkeypatch.setattr(client, "_post_chat", _down)
        monkeypatch.setattr(dots, "get_client", lambda cfg=None: client)

        with pytest.raises(dots.DotsUnavailable):
            pte._extract_page_payload_dots(_FakePage(), 1, [], 0)

    def test_invalid_dots_output_propagates(self, monkeypatch):
        monkeypatch.setenv("DOTS_ENABLED", "true")
        td.set_backend(lambda _b: [_box("Table", 0.9)])
        _install_dots_stub(monkeypatch, _layout(_table_el()), finish="length")
        with pytest.raises(dots.DotsInvalidOutput) as exc:
            pte._extract_page_payload_dots(_FakePage(), 1, [], 0)
        assert exc.value.rule == dots.RULE_TRUNCATED

    def test_detector_unavailable_propagates(self, monkeypatch):
        monkeypatch.setenv("DOTS_ENABLED", "true")

        def _boom(_b):
            raise RuntimeError("paddle missing")

        td.set_backend(_boom)
        with pytest.raises(td.TableDetectorUnavailable):
            pte._extract_page_payload_dots(_FakePage(), 1, [], 0)

    def test_false_positive_detection_degrades_to_legacy_text(self, monkeypatch):
        """RULE_NO_TABLE_FOUND is the one non-fatal rejection."""
        monkeypatch.setenv("DOTS_ENABLED", "true")
        td.set_backend(lambda _b: [_box("Table", 0.9)])
        _install_dots_stub(monkeypatch, _layout(_text_el("no table at all")))
        lines = [{"text": "legacy", "yc": 1.0, "x0": 0, "x1": 5, "y0": 0, "y1": 2}]
        payload = pte._extract_page_payload_dots(_FakePage(), 1, lines, 6)
        assert payload["type"] == "prose"
        assert payload["content"] == pte._render_merged(lines)

    def test_rs_wrapper_reraises_infrastructure_failures(self, monkeypatch):
        """RS must not stamp answer_source=unavailable because DOTS was down."""
        from src.scraping.rs import pipeline as rsp

        def _down(_data, **_kw):
            raise dots.DotsUnavailable("endpoint down")

        monkeypatch.setattr(pte, "extract_pdf_text", _down)
        with pytest.raises(dots.DotsUnavailable):
            rsp._extract_answer_fallback(b"%PDF-1.4")

    def test_rs_wrapper_still_swallows_corrupt_pdfs(self, monkeypatch):
        """A genuinely unopenable document is still 'unavailable'."""
        from src.scraping.rs import pipeline as rsp

        def _corrupt(_data, **_kw):
            raise RuntimeError("Corrupted or unreadable PDF stream")

        monkeypatch.setattr(pte, "extract_pdf_text", _corrupt)
        assert rsp._extract_answer_fallback(b"not-a-pdf") is None

    def test_rs_wrapper_reraises_missing_dependency(self, monkeypatch):
        """DependencyMissingError subclasses ImportError — it must not be eaten."""
        from src.scraping.rs import pipeline as rsp

        def _nodep(_data, **_kw):
            raise pte.DependencyMissingError("Tesseract unavailable")

        monkeypatch.setattr(pte, "extract_pdf_text", _nodep)
        with pytest.raises(pte.DependencyMissingError):
            rsp._extract_answer_fallback(b"%PDF-1.4")


class TestDotsClientTransport:
    def test_missing_base_url_is_a_clear_error(self, monkeypatch):
        monkeypatch.delenv("DOTS_BASE_URL", raising=False)
        client = dots.DotsClient(dots.DotsConfig.from_env())
        with pytest.raises(dots.DotsUnavailable, match="DOTS_BASE_URL"):
            client.list_models()

    def test_base_url_is_normalised_to_v1(self, monkeypatch):
        monkeypatch.setenv("DOTS_BASE_URL", "http://un003:8200/")
        assert dots.DotsConfig.from_env().base_url == "http://un003:8200/v1"

    def test_base_url_already_versioned_is_left_alone(self, monkeypatch):
        monkeypatch.setenv("DOTS_BASE_URL", "http://un003:8200/v1")
        assert dots.DotsConfig.from_env().base_url == "http://un003:8200/v1"

    def test_token_budget_is_capped_well_below_the_model_card_default(self):
        """24000 lets one repetition loop hold a decode slot for minutes."""
        assert dots.DEFAULT_MAX_OUTPUT_TOKENS == 8192

    def test_sampling_is_deterministic_by_default(self):
        cfg = dots.DotsConfig.from_env()
        assert cfg.temperature == 0.0
        assert cfg.repetition_penalty == 1.0  # would corrupt real repetitive tables

    def test_is_available_never_raises(self, monkeypatch):
        monkeypatch.setenv("DOTS_BASE_URL", "http://127.0.0.1:1")
        assert dots.DotsClient(dots.DotsConfig.from_env()).is_available() is False


# ── 4. Change detection ──────────────────────────────────────────────────────


def _record(**over):
    from src.models.qa_record import QARecord

    base = {
        "question_id": "rs-271-99",
        "question_text": "What is the status of coastal monitoring?",
        "answer_text": "The monitoring network is operational.",
        "scraped_at": "2026-01-01T00:00:00Z",
        "metadata": {"source": "rajya_sabha", "session": 271},
    }
    meta_over = over.pop("metadata", None)
    base.update(over)
    if meta_over:
        base["metadata"] = {**base["metadata"], **meta_over}
    return QARecord.model_validate(base)


class TestChangeDetection:
    def test_content_hash_is_question_text_only(self):
        """Pinning WHY content_hash must never drive change detection."""
        a = _record()
        b = _record(answer_text="A completely different answer entirely.")
        assert a.content_hash == b.content_hash

    def test_content_hash_does_not_participate_in_change_detection(self):
        from src.scripts.ingest_folder import qa_content_hash

        a = _record()
        b = _record(answer_text="A completely different answer entirely.")
        assert a.content_hash == b.content_hash      # blind to the change
        assert qa_content_hash(a) != qa_content_hash(b)  # authoritative hash is not

    def test_qa_content_hash_covers_answer_text(self):
        from src.scripts.ingest_folder import qa_content_hash

        assert qa_content_hash(_record()) != qa_content_hash(_record(answer_text="A different answer entirely."))

    def test_qa_content_hash_excludes_scraped_at(self):
        from src.scripts.ingest_folder import qa_content_hash

        assert qa_content_hash(_record()) == qa_content_hash(
            _record(scraped_at="2026-09-09T12:00:00Z")
        )

    def test_source_sha256_does_not_change_the_hash(self):
        """A re-stamped PDF with identical text is not a record change."""
        from src.scripts.ingest_folder import qa_content_hash

        assert qa_content_hash(_record()) == qa_content_hash(
            _record(metadata={"source_sha256": "a" * 64})
        )

    def test_extractor_version_does_not_change_the_hash(self):
        """An extractor bump must not mark the whole corpus changed."""
        from src.scripts.ingest_folder import qa_content_hash

        assert qa_content_hash(_record()) == qa_content_hash(
            _record(metadata={"extractor_version": "dots-1.5/picodet-x/dpi200"})
        )

    def test_both_new_fields_together_do_not_change_the_hash(self):
        from src.scripts.ingest_folder import qa_content_hash

        before = qa_content_hash(_record())
        after = qa_content_hash(_record(metadata={
            "source_sha256": "b" * 64,
            "extractor_version": "dots-1.5/picodet-x/dpi200",
        }))
        assert before == after

    def test_migration_does_not_rehash_the_corpus(self):
        """Decision: introducing the fields rewrites nothing."""
        from src.scripts.ingest_folder import qa_content_hash

        legacy = [_record(question_id=f"rs-271-{i}") for i in range(50)]
        before = {r.question_id: qa_content_hash(r) for r in legacy}
        migrated = [
            _record(question_id=f"rs-271-{i}", metadata={
                "source_sha256": f"{i:064x}",
                "extractor_version": "dots-1.5/picodet-x/dpi200",
            })
            for i in range(50)
        ]
        after = {r.question_id: qa_content_hash(r) for r in migrated}
        assert before == after

    def test_dots_rewriting_answer_text_does_change_the_hash(self):
        from src.scripts.ingest_folder import qa_content_hash

        old = _record(metadata={"source_sha256": "c" * 64})
        new = _record(
            answer_text="| State | Count |\n|---|---|\n| MP | 12 |",
            metadata={"source_sha256": "c" * 64},
        )
        assert qa_content_hash(old) != qa_content_hash(new)

    def test_changed_ids_contains_only_genuinely_changed_records(self):
        from src.scripts.ingest_folder import qa_content_hash

        old = {f"rs-271-{i}": _record(question_id=f"rs-271-{i}") for i in range(10)}
        seen = {qid: qa_content_hash(r) for qid, r in old.items()}

        new = []
        for i in range(10):
            # Every record gets fresh extraction provenance; only #3 and #7 changed.
            meta = {"source_sha256": f"{i:064x}", "extractor_version": "dots-2/picodet-y/dpi200"}
            if i in (3, 7):
                new.append(_record(question_id=f"rs-271-{i}",
                                   answer_text=f"Revised table content {i}.", metadata=meta))
            else:
                new.append(_record(question_id=f"rs-271-{i}", metadata=meta))

        changed = [r.question_id for r in new if seen.get(r.question_id) != qa_content_hash(r)]
        assert changed == ["rs-271-3", "rs-271-7"]


class TestSourceHashShortCircuit:
    def test_unchanged_source_and_version_skips_extraction(self):
        from src.scripts.ingest_folder import needs_reextraction

        rec = _record(metadata={"source_sha256": "d" * 64, "extractor_version": "v1"})
        assert needs_reextraction(rec, source_sha256="d" * 64, current_version="v1") is False

    def test_changed_source_triggers_reextraction(self):
        from src.scripts.ingest_folder import needs_reextraction

        rec = _record(metadata={"source_sha256": "d" * 64, "extractor_version": "v1"})
        assert needs_reextraction(rec, source_sha256="e" * 64, current_version="v1") is True

    def test_changed_extractor_version_triggers_reextraction(self):
        from src.scripts.ingest_folder import needs_reextraction

        rec = _record(metadata={"source_sha256": "d" * 64, "extractor_version": "v1"})
        assert needs_reextraction(rec, source_sha256="d" * 64, current_version="v2") is True

    def test_legacy_record_without_provenance_is_reextracted(self):
        from src.scripts.ingest_folder import needs_reextraction

        assert needs_reextraction(_record(), source_sha256="d" * 64, current_version="v1") is True

    def test_absent_record_is_reextracted(self):
        from src.scripts.ingest_folder import needs_reextraction

        assert needs_reextraction(None, source_sha256="d" * 64, current_version="v1") is True

    def test_source_sha256_is_computed_from_bytes(self):
        import hashlib

        from src.data.pdf_table_extract import source_sha256_of

        assert source_sha256_of(b"%PDF-1.4 abc") == hashlib.sha256(b"%PDF-1.4 abc").hexdigest()

    def test_different_bytes_give_different_hashes(self):
        from src.data.pdf_table_extract import source_sha256_of

        assert source_sha256_of(b"%PDF-a") != source_sha256_of(b"%PDF-b")


class TestExtractorVersion:
    def test_legacy_version_when_routing_is_off(self):
        assert pte.current_extractor_version().startswith("legacy/")

    def test_version_reports_dots_picodet_and_dpi_when_routing_is_on(self, monkeypatch):
        monkeypatch.setenv("DOTS_ENABLED", "true")
        monkeypatch.setenv("DOTS_MODEL", "dots-ocr")
        monkeypatch.setenv("DOTS_RENDER_DPI", "200")
        version = pte.current_extractor_version()
        assert "dots-" in version and "picodet-" in version and version.endswith("/dpi200")

    def test_threshold_change_changes_the_version(self, monkeypatch):
        monkeypatch.setenv("DOTS_ENABLED", "true")
        monkeypatch.setenv("DOTS_MODEL", "dots-ocr")
        monkeypatch.setenv("PICODET_SCORE_THRESHOLD", "0.25")
        v1 = pte.current_extractor_version()
        monkeypatch.setenv("PICODET_SCORE_THRESHOLD", "0.40")
        assert pte.current_extractor_version() != v1


class TestRunManifest:
    def test_manifest_exposes_the_four_id_buckets(self, tmp_path, monkeypatch):
        from src.scripts import ingest

        monkeypatch.setattr(ingest, "_data_path", lambda p: tmp_path / p)
        path = ingest.write_run_manifest("rs", {
            "added": ["a1", "a2"], "changed": ["c1"],
            "unchanged": ["u1", "u2", "u3"], "failed": ["f1"],
        })
        payload = json.loads(path.read_text())
        assert payload["added_ids"] == ["a1", "a2"]
        assert payload["changed_ids"] == ["c1"]
        assert payload["unchanged_ids"] == ["u1", "u2", "u3"]
        assert payload["failed_ids"] == ["f1"]

    def test_graphrag_pending_is_added_plus_changed_only(self, tmp_path, monkeypatch):
        """GraphRAG must process added+changed, never the whole corpus."""
        from src.scripts import ingest

        monkeypatch.setattr(ingest, "_data_path", lambda p: tmp_path / p)
        path = ingest.write_run_manifest("rs", {
            "added": ["a1"], "changed": ["c1"],
            "unchanged": [f"u{i}" for i in range(1000)], "failed": ["f1"],
        })
        payload = json.loads(path.read_text())
        assert payload["graphrag_pending_ids"] == ["a1", "c1"]
        assert len(payload["unchanged_ids"]) == 1000  # recorded, never processed

    def test_manifest_failure_never_breaks_ingest(self, monkeypatch):
        from src.scripts import ingest

        def _explode(_p):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(ingest, "_data_path", _explode)
        assert ingest.write_run_manifest("rs", {"added": ["a1"]}) is None
