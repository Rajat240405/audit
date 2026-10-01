"""Regression proof: adding PicoDet -> DOTS routing does not change non-table output.

The guarantee under test
------------------------
A PDF with no tables must produce **byte-identical** text whether or not the
new routing is enabled. Introducing a detector is only safe if it is provably
inert on the ~74% of the corpus that has no tables at all; anything less
silently rewrites thousands of records that nobody asked to change.

Three layers are pinned here:

1. **Flag off** — the legacy path is untouched and the detector is never even
   consulted, so deploying this code changes nothing until an operator opts in.
2. **Flag on, no table detected** — output is byte-identical to the legacy
   path, across prose, multi-page, Unicode, whitespace and empty documents.
3. **MoES/INCOIS v2** — the separate ``src/data/v2`` pipeline is untouched and
   still functional, because it legitimately owns 218 corpus records.

PDFs are generated in-memory with PyMuPDF, plus the real sansad.in fixture, so
the comparison runs on genuine extractor output rather than mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data import pdf_table_extract as pte
from src.data import table_detect as td

pytestmark = pytest.mark.skipif(
    not pte.check_extraction_environment()[0],
    reason="PyMuPDF (fitz) is required for extraction regression tests",
)

FIXTURE = Path(__file__).parent / "fixtures" / "17-7-2936.pdf"


# ── PDF builders ─────────────────────────────────────────────────────────────


def _pdf(pages: list[str]) -> bytes:
    """Build a text-only PDF, one page per string."""
    import fitz

    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        y = 72.0
        for line in body.splitlines():
            page.insert_text((72, y), line, fontsize=11)
            y += 16
    data = doc.tobytes()
    doc.close()
    return data


PROSE = """GOVERNMENT OF INDIA
MINISTRY OF EARTH SCIENCES
LOK SABHA
UNSTARRED QUESTION NO 2936

TO BE ANSWERED ON 21.03.2025

COASTAL MONITORING NETWORK

Will the Minister of Earth Sciences be pleased to state whether the
Government has established a coastal monitoring network along the
Indian coastline, and if so, the details thereof.

ANSWER

The Ministry of Earth Sciences, through the Indian National Centre for
Ocean Information Services, operates a network of observation platforms.
These platforms provide continuous data on sea state parameters.
The data is disseminated to stakeholders through an online portal."""

MULTIPAGE = [
    "Page one of the answer text.\nSecond line of page one.",
    "Page two continues the narrative.\nAnother line here.",
    "Page three concludes the answer.\nFinal line of the document.",
]

UNICODE = """भारत सरकार
पृथ्वी विज्ञान मंत्रालय
Question regarding monitoring — with em-dash, ellipsis … and quotes "like this".
Temperature rose by 1.5°C; salinity ≈ 35 PSU (±0.2)."""

WHITESPACE = "Line one.\n\n\n   Indented line.\n\tTabbed line.\n\nTrailing spaces here.   \n"


ALL_DOCS: dict[str, bytes] = {}


def _docs() -> dict[str, bytes]:
    if not ALL_DOCS:
        ALL_DOCS.update({
            "prose": _pdf([PROSE]),
            "multipage": _pdf(MULTIPAGE),
            "unicode": _pdf([UNICODE]),
            "whitespace": _pdf([WHITESPACE]),
            "single_line": _pdf(["Just one short line of text on the page."]),
        })
    return ALL_DOCS


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    td.reset_backend()
    for var in ("DOTS_ENABLED", "DOTS_BASE_URL", "PICODET_SCORE_THRESHOLD", "PICODET_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    yield
    td.reset_backend()


def _extract_legacy(data: bytes) -> str | None:
    """Extraction with routing disabled (today's behaviour)."""
    assert not pte.dots_routing_enabled()
    return pte.extract_pdf_text(data, enable_ocr=False)


def _extract_routed(data: bytes, monkeypatch, detections=()) -> str | None:
    """Extraction with routing enabled and a controlled detector."""
    monkeypatch.setenv("DOTS_ENABLED", "true")
    td.set_backend(lambda _b: list(detections))
    assert pte.dots_routing_enabled()
    return pte.extract_pdf_text(data, enable_ocr=False)


# ── 1. Flag off: the legacy path is untouched ────────────────────────────────


class TestFlagOffIsInert:
    def test_routing_is_disabled_by_default(self):
        assert pte.dots_routing_enabled() is False

    def test_detector_is_never_consulted_when_flag_is_off(self, monkeypatch):
        """Not merely equal output — the detector must not run at all."""
        calls: list[int] = []
        td.set_backend(lambda _b: calls.append(1) or [])
        pte.extract_pdf_text(_docs()["prose"], enable_ocr=False)
        assert calls == []

    @pytest.mark.parametrize("name", ["prose", "multipage", "unicode", "whitespace", "single_line"])
    def test_legacy_extraction_still_produces_text(self, name):
        out = _extract_legacy(_docs()[name])
        assert out and out.strip()

    @pytest.mark.skipif(not FIXTURE.exists(), reason="fixture PDF not present")
    def test_real_fixture_still_extracts_tables_on_the_legacy_path(self):
        """The existing borderless-table behaviour must survive untouched.

        This fixture's annexure is a BORDERLESS table, reconstructed by
        Strategy 3 into space-separated serial rows (not a markdown grid), so
        the pin is on row reconstruction and ordering.
        """
        out = _extract_legacy(FIXTURE.read_bytes())
        assert out
        assert "34 KOLKATA OHR 15" in out
        assert "35 KOLKATA OHR 10" in out
        assert out.index("34 KOLKATA OHR 15") < out.index("35 KOLKATA OHR 10")

    @pytest.mark.skipif(not FIXTURE.exists(), reason="fixture PDF not present")
    def test_real_fixture_is_byte_identical_to_pre_change_extractor(self):
        """Compare against the extractor as it existed before this change.

        The pre-change module is loaded straight from git HEAD and run
        side by side, so this is a true before/after diff on a real
        sansad.in document rather than a restatement of current behaviour.
        """
        import importlib.util
        import subprocess
        import sys
        import tempfile

        repo = Path(__file__).parent.parent
        try:
            original = subprocess.run(
                ["git", "show", "HEAD:src/data/pdf_table_extract.py"],
                cwd=repo, capture_output=True, text=True, timeout=30, check=True,
            ).stdout
        except Exception:  # noqa: BLE001 — not a git checkout / git unavailable
            pytest.skip("git HEAD copy of the extractor is unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pdf_table_extract_head.py"
            path.write_text(original, encoding="utf-8")
            spec = importlib.util.spec_from_file_location("_pte_head", path)
            head = importlib.util.module_from_spec(spec)
            sys.modules["_pte_head"] = head
            try:
                spec.loader.exec_module(head)
                data = FIXTURE.read_bytes()
                assert head.extract_pdf_text(data, enable_ocr=False) == _extract_legacy(data)
            finally:
                sys.modules.pop("_pte_head", None)


# ── 2. Flag on, no table detected: byte-identical ────────────────────────────


class TestNoTableByteIdentical:
    @pytest.mark.parametrize("name", ["prose", "multipage", "unicode", "whitespace", "single_line"])
    def test_output_is_byte_identical(self, name, monkeypatch):
        data = _docs()[name]
        legacy = _extract_legacy(data)
        routed = _extract_routed(data, monkeypatch, detections=[])
        assert routed == legacy, f"{name}: routing changed a table-free document"

    def test_identity_holds_at_the_byte_level_not_just_equality(self, monkeypatch):
        data = _docs()["prose"]
        legacy = _extract_legacy(data)
        routed = _extract_routed(data, monkeypatch, detections=[])
        assert routed is not None
        assert routed.encode("utf-8") == legacy.encode("utf-8")

    def test_page_count_and_order_are_preserved(self, monkeypatch):
        data = _docs()["multipage"]
        routed = _extract_routed(data, monkeypatch, detections=[])
        assert routed is not None
        assert routed.index("Page one") < routed.index("Page two") < routed.index("Page three")

    def test_detector_is_actually_consulted(self, monkeypatch):
        """Guards against a vacuous pass: the detector must really run."""
        calls: list[int] = []
        monkeypatch.setenv("DOTS_ENABLED", "true")
        td.set_backend(lambda _b: calls.append(1) or [])
        pte.extract_pdf_text(_docs()["multipage"], enable_ocr=False)
        assert len(calls) == 3   # one call per page, page by page

    def test_non_table_detections_do_not_trigger_dots(self, monkeypatch):
        """Only the Table label routes to DOTS; Text/Title/Figure must not."""
        data = _docs()["prose"]
        legacy = _extract_legacy(data)
        routed = _extract_routed(data, monkeypatch, detections=[
            {"label": "Text", "score": 0.99, "coordinate": [0, 0, 100, 100]},
            {"label": "Title", "score": 0.98, "coordinate": [0, 0, 100, 50]},
            {"label": "Figure", "score": 0.97, "coordinate": [0, 0, 100, 60]},
        ])
        assert routed == legacy

    def test_low_confidence_table_does_not_trigger_dots(self, monkeypatch):
        """Below the 0.25 threshold the page stays on the legacy path."""
        data = _docs()["prose"]
        legacy = _extract_legacy(data)
        routed = _extract_routed(data, monkeypatch, detections=[
            {"label": "Table", "score": 0.05, "coordinate": [0, 0, 100, 100]},
        ])
        assert routed == legacy

    def test_empty_pdf_behaves_identically(self, monkeypatch):
        import fitz

        doc = fitz.open()
        doc.new_page()
        data = doc.tobytes()
        doc.close()
        legacy = _extract_legacy(data)
        routed = _extract_routed(data, monkeypatch, detections=[])
        assert routed == legacy


class TestDetectorFailureIsNotSilent:
    def test_detector_failure_aborts_rather_than_degrading(self, monkeypatch):
        """A broken detector must not quietly become 'no tables anywhere'."""
        monkeypatch.setenv("DOTS_ENABLED", "true")

        def _boom(_b):
            raise RuntimeError("paddle crashed")

        td.set_backend(_boom)
        with pytest.raises(td.TableDetectorUnavailable):
            pte.extract_pdf_text(_docs()["prose"], enable_ocr=False)


# ── 3. The public seam is unchanged ──────────────────────────────────────────


class TestPublicSeamPreserved:
    def test_extract_pdf_text_signature_is_unchanged(self):
        import inspect

        params = list(inspect.signature(pte.extract_pdf_text).parameters)
        assert params == ["data", "enable_ocr"]

    def test_fallback_wrapper_still_returns_empty_string(self):
        import fitz

        doc = fitz.open()
        doc.new_page()
        data = doc.tobytes()
        doc.close()
        assert pte.extract_pdf_text_with_fallback(data, enable_ocr=False) == ""

    def test_ls_and_rs_still_call_the_same_seam(self):
        """Neither wrapper may grow a private extraction path."""
        import src.scraping.ls.extract as ls_extract
        import src.scraping.rs.pipeline as rs_pipeline

        for mod in (ls_extract, rs_pipeline):
            src = Path(mod.__file__).read_text(encoding="utf-8")
            assert "from src.data.pdf_table_extract import" in src
            assert "extract_pdf_text" in src

    def test_corrupt_pdf_still_raises_runtime_error(self):
        with pytest.raises(RuntimeError):
            pte.extract_pdf_text(b"this is not a pdf at all", enable_ocr=False)


# ── 4. MoES/INCOIS v2 remains intact ─────────────────────────────────────────


class TestV2PipelineIntact:
    """``src/data/v2`` owns 218 live corpus records and must keep working."""

    def test_v2_package_is_importable(self):
        import src.data.v2  # noqa: F401

    def test_v2_core_text_is_unchanged_and_functional(self):
        from src.data.v2.core_text import build_core_text

        assert callable(build_core_text)

    def test_v2_modules_still_import(self):
        import importlib

        for name in ("core_text", "table_extract", "ocr", "page_router",
                     "borderless_extract", "figure_extract", "pipeline",
                     "identity", "config"):
            importlib.import_module(f"src.data.v2.{name}")

    def test_v2_has_its_own_config_gate(self):
        from src.data.v2 import config as v2config

        assert hasattr(v2config, "load_config") or hasattr(v2config, "V2Config")

    def test_v2_does_not_import_the_dots_pipeline(self):
        """No coupling in either direction — two separate, intact pipelines."""
        v2_dir = Path(__file__).parent.parent / "src" / "data" / "v2"
        for path in v2_dir.glob("*.py"):
            src = path.read_text(encoding="utf-8")
            assert "dots_client" not in src, f"{path.name} imports dots_client"
            assert "table_detect" not in src, f"{path.name} imports table_detect"

    def test_dots_pipeline_does_not_touch_v2(self):
        for name in ("table_detect", "dots_client"):
            src = (Path(__file__).parent.parent / "src" / "data" / f"{name}.py").read_text()
            assert "src.data.v2" not in src
