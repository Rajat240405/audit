"""DOTS OCR client + deterministic output validator.

DOTS is the ONLY table extractor in this project. When PicoDet
(``src.data.table_detect``) flags a page, that page is rendered and sent here;
the returned text replaces the page's extraction entirely.

Transport
---------
DOTS is reached over an **external OpenAI-compatible vLLM endpoint**
(``DOTS_BASE_URL``) — never an in-process model and never a hard-coded runtime.
This mirrors the existing generation client (``src.generation.client`` /
``src.generation.vllm_discovery``): base URL from the environment, model name
optional with ``/v1/models`` discovery, TTL-cached health.

Failure policy
--------------
There is **no silent fallback to an inferior extractor**. Every failure is
typed and propagates:

* :class:`DotsUnavailable`   — endpoint down, timeout, transport error, bad HTTP.
* :class:`DotsInvalidOutput` — output rejected by the deterministic validator.

A caller that cannot proceed must fail the page, which fails the document,
which leaves the previous corpus row untouched. Silent corruption is never
preferable to a visible, diagnosable failure.

The validator
-------------
Deterministic, non-AI, and deliberately **conservative**: it rejects only
pathological output (empty, truncated, repetition-looped, structurally
malformed). It never judges semantic correctness — no numeric plausibility, no
dictionary checks, no column-uniformity rules, no similarity comparison against
the legacy extractor. Those would reject correct transcriptions of messy
source documents, and rejection permanently blocks a record from improving.

One check (:data:`RULE_NO_TABLE_FOUND`) is not a failure at all: when PicoDet
fired but DOTS reports no table, the page falls back to its legacy text. That
converts a detector false positive into a no-op instead of a loss.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, NamedTuple, Sequence

__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_PROMPT_MODE",
    "DEFAULT_RENDER_DPI",
    "DotsClient",
    "DotsConfig",
    "DotsInvalidOutput",
    "DotsUnavailable",
    "RULE_BAD_JSON",
    "RULE_BAD_SCHEMA",
    "RULE_CONTROL_CHARS",
    "RULE_EMPTY",
    "RULE_HTML_UNBALANCED",
    "RULE_LINE_REPEAT",
    "RULE_NGRAM_SATURATION",
    "RULE_NO_TABLE_FOUND",
    "RULE_OK",
    "RULE_SEVERE_TRUNCATION",
    "RULE_TRUNCATED",
    "ValidationResult",
    "dots_version",
    "enabled",
    "get_client",
    "parse_layout_json",
    "render_elements",
    "reset_client",
    "validate_dots_output",
]

log = logging.getLogger(__name__)

# ── defaults ─────────────────────────────────────────────────────────────────

#: Official dots.ocr guidance is DPI 200; higher ratios can destabilise parsing.
DEFAULT_RENDER_DPI = 200

#: dots.ocr performs best below this total pixel count. Oversize pages (A3 /
#: legal annexures) are downscaled client-side rather than failing server-side.
DEFAULT_MAX_PIXELS = 11_289_600

#: Deliberately far below the model card's ``max_new_tokens=24000``. An
#: unbounded cap lets one repetition loop hold a decode slot for minutes and
#: is the documented root cause of VLM-OCR throughput collapse.
DEFAULT_MAX_OUTPUT_TOKENS = 8192

DEFAULT_PROMPT_MODE = "prompt_layout_all_en"
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_HEALTH_TIMEOUT = 5.0
DEFAULT_DISCOVERY_TTL = 300.0

# Validator thresholds — all environment-tunable, all biased toward acceptance.
DEFAULT_MIN_CHARS = 16
DEFAULT_MAX_LINE_REPEAT = 30
# Reject only when >98% of 10-grams are duplicates. Healthy pages sit near
# 1.0; even a highly repetitive annexure rarely drops below 0.10.
DEFAULT_MIN_DISTINCT_NGRAM_RATIO = 0.02
DEFAULT_MIN_RATIO = 0.25

RULE_OK = "ok"
RULE_EMPTY = "empty_output"
RULE_TRUNCATED = "truncated_finish_reason"
RULE_LINE_REPEAT = "line_repetition_loop"
RULE_NGRAM_SATURATION = "ngram_saturation"
RULE_BAD_JSON = "malformed_json"
RULE_BAD_SCHEMA = "unexpected_schema"
RULE_HTML_UNBALANCED = "unbalanced_html"
RULE_CONTROL_CHARS = "control_characters"
RULE_SEVERE_TRUNCATION = "severe_truncation"
RULE_NO_TABLE_FOUND = "no_table_in_output"

#: The official dots.ocr full-layout prompt. Tables come back as HTML, formulas
#: as LaTeX, everything else as Markdown, sorted in human reading order.
LAYOUT_PROMPT_EN = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object."""

PROMPTS: dict[str, str] = {
    "prompt_layout_all_en": LAYOUT_PROMPT_EN,
    "prompt_ocr": "Extract the text content from this image.",
    "prompt_layout_only_en": (
        "Please output the layout information from the PDF image, including each "
        "layout element's bbox and its category. The output must be a single JSON object."
    ),
}

#: Marker prefixing a table block, matching ``src/data/v2/core_text.py`` so page
#: provenance reads identically across every source in the corpus.
TABLE_MARKER = "TABLE"


class DotsUnavailable(RuntimeError):
    """DOTS could not be reached or did not answer (transport/HTTP/timeout)."""


class DotsInvalidOutput(RuntimeError):
    """DOTS answered, but the deterministic validator rejected the output."""

    def __init__(self, message: str, *, rule: str = RULE_OK, detail: str = "") -> None:
        super().__init__(message)
        self.rule = rule
        self.detail = detail


# ── configuration ────────────────────────────────────────────────────────────


def _env_str(name: str, default: str = "") -> str:
    return (os.environ.get(name) or "").strip() or default


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r is not an integer; using %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def enabled() -> bool:
    """Master switch. Default **off** so adding DOTS cannot change behaviour
    until an operator deliberately turns it on."""
    return _env_bool("DOTS_ENABLED", False)


@dataclass(frozen=True)
class DotsConfig:
    """Runtime configuration, resolved entirely from the environment.

    Nothing here is baked into the container image: endpoint, model, DPI,
    token budget, concurrency and every validator threshold are tunable
    without a SIF rebuild.
    """

    base_url: str = ""
    model: str = ""
    api_key: str = ""
    prompt_mode: str = DEFAULT_PROMPT_MODE
    render_dpi: int = DEFAULT_RENDER_DPI
    max_pixels: int = DEFAULT_MAX_PIXELS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    health_timeout: float = DEFAULT_HEALTH_TIMEOUT
    discovery_ttl: float = DEFAULT_DISCOVERY_TTL
    max_retries: int = 2
    retry_backoff: float = 2.0
    table_format: str = "html"
    strict: bool = True
    min_chars: int = DEFAULT_MIN_CHARS
    max_line_repeat: int = DEFAULT_MAX_LINE_REPEAT
    min_distinct_ngram_ratio: float = DEFAULT_MIN_DISTINCT_NGRAM_RATIO
    min_ratio: float = DEFAULT_MIN_RATIO
    version: str = ""

    @classmethod
    def from_env(cls) -> "DotsConfig":
        base = _env_str("DOTS_BASE_URL")
        if base:
            base = base.rstrip("/")
            if not base.endswith("/v1"):
                base = f"{base}/v1"
        return cls(
            base_url=base,
            model=_env_str("DOTS_MODEL"),
            api_key=_env_str("DOTS_API_KEY"),
            prompt_mode=_env_str("DOTS_PROMPT_MODE", DEFAULT_PROMPT_MODE),
            render_dpi=_env_int("DOTS_RENDER_DPI", DEFAULT_RENDER_DPI),
            max_pixels=_env_int("DOTS_MAX_PIXELS", DEFAULT_MAX_PIXELS),
            max_output_tokens=_env_int("DOTS_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS),
            temperature=_env_float("DOTS_TEMPERATURE", 0.0),
            repetition_penalty=_env_float("DOTS_REPETITION_PENALTY", 1.0),
            request_timeout=_env_float("DOTS_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT),
            connect_timeout=_env_float("DOTS_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT),
            health_timeout=_env_float("DOTS_HEALTH_TIMEOUT", DEFAULT_HEALTH_TIMEOUT),
            discovery_ttl=_env_float("DOTS_DISCOVERY_TTL_SECONDS", DEFAULT_DISCOVERY_TTL),
            max_retries=_env_int("DOTS_MAX_RETRIES", 2),
            retry_backoff=_env_float("DOTS_RETRY_BACKOFF_SECONDS", 2.0),
            table_format=_env_str("DOTS_TABLE_FORMAT", "html").lower(),
            strict=_env_bool("DOTS_VALIDATOR_STRICT", True),
            min_chars=_env_int("DOTS_MIN_CHARS", DEFAULT_MIN_CHARS),
            max_line_repeat=_env_int("DOTS_MAX_LINE_REPEAT", DEFAULT_MAX_LINE_REPEAT),
            min_distinct_ngram_ratio=_env_float(
                "DOTS_MIN_DISTINCT_NGRAM_RATIO", DEFAULT_MIN_DISTINCT_NGRAM_RATIO
            ),
            min_ratio=_env_float("DOTS_MIN_RATIO", DEFAULT_MIN_RATIO),
            version=_env_str("DOTS_VERSION"),
        )

    @property
    def prompt(self) -> str:
        return PROMPTS.get(self.prompt_mode, LAYOUT_PROMPT_EN)

    @property
    def expects_json(self) -> bool:
        """Only the layout prompts promise a single JSON object."""
        return self.prompt_mode.startswith("prompt_layout")


def dots_version(cfg: DotsConfig | None = None) -> str:
    """Identifier for the DOTS half of ``extractor_version``."""
    cfg = cfg or DotsConfig.from_env()
    if cfg.version:
        return cfg.version
    model = cfg.model or "auto"
    return f"dots-{model}@{cfg.prompt_mode}"


# ── validator ────────────────────────────────────────────────────────────────


class ValidationResult(NamedTuple):
    """Outcome of validating one DOTS response.

    ``action`` is one of:

    * ``accept``   — use the DOTS text.
    * ``reject``   — hard failure; the page (and therefore the document) fails.
    * ``fallback`` — not a failure; discard the DOTS text and keep the legacy
      extraction for this page (detector false positive).
    """

    ok: bool
    rule: str
    detail: str = ""
    action: str = "accept"


_WS = re.compile(r"\s+")
_C0 = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """Remove a surrounding markdown code fence, if present."""
    match = _FENCE.match(text or "")
    return match.group(1) if match else (text or "")


def _max_consecutive_line_repeat(text: str) -> tuple[int, str]:
    """Longest run of identical consecutive non-blank lines (whitespace-normalised)."""
    best = 0
    best_line = ""
    current = 0
    previous: str | None = None
    for raw in (text or "").splitlines():
        norm = _WS.sub(" ", raw).strip()
        if not norm:
            previous = None
            current = 0
            continue
        if norm == previous:
            current += 1
        else:
            current = 1
            previous = norm
        if current > best:
            best = current
            best_line = norm
    return best, best_line


def _distinct_ngram_ratio(text: str, n: int = 10) -> tuple[float, str]:
    """Fraction of n-grams that are distinct, plus the most frequent n-gram.

    Near 1.0 for healthy prose or a real table; collapses toward 0 when the
    model is looping.

    Deliberately NOT "share taken by the top n-gram": for a loop whose
    repeating unit is k tokens long, the top n-gram can never exceed about
    1/k of the total, so a k=10 loop tops out near 0.10 and slips past any
    sane threshold. Distinctness has no such ceiling.
    """
    tokens = (text or "").split()
    total = len(tokens) - n + 1
    if total < n:
        return 1.0, ""
    counts: dict[tuple[str, ...], int] = {}
    for i in range(total):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    gram, _hits = max(counts.items(), key=lambda kv: kv[1])
    return len(counts) / total, " ".join(gram)


def _distinct_char_ngram_ratio(text: str, n: int = 40) -> tuple[float, str]:
    """Distinct-ratio over CHARACTER n-grams, for loops with no whitespace.

    Necessary because a looping HTML table is frequently emitted as one
    unbroken line with no spaces, e.g.
    ``<table><tr><td>...</td></tr><tr><td>...</td></tr>...``. That defeats
    the line check (one line) and the word check (``str.split`` yields a
    single token), so the most likely real-world shape of the documented
    dots.ocr repetition failure would otherwise pass validation.

    For a unit of length ``u`` repeated ``k`` times the ratio tends to
    ``1/k``, which is what lets the threshold be derived from the same
    "maximum acceptable repeats" knob the line check uses. Varied text sits
    near 1.0.
    """
    sample = (text or "")[:200_000]  # bound the cost on pathological output
    total = len(sample) - n + 1
    if total < n:
        return 1.0, ""
    counts: dict[str, int] = {}
    for i in range(total):
        gram = sample[i : i + n]
        counts[gram] = counts.get(gram, 0) + 1
    gram, _hits = max(counts.items(), key=lambda kv: kv[1])
    return len(counts) / total, gram


def _tag_count(text: str, tag: str) -> tuple[int, int]:
    opens = len(re.findall(rf"<{tag}\b", text, re.IGNORECASE))
    closes = len(re.findall(rf"</{tag}\s*>", text, re.IGNORECASE))
    return opens, closes


def validate_dots_output(
    text: str,
    *,
    finish_reason: str | None = None,
    native_char_count: int = 0,
    detector_found_table: bool = False,
    cfg: DotsConfig | None = None,
) -> ValidationResult:
    """Deterministically validate one DOTS page response.

    Conservative by construction: every threshold sits far outside the range
    legitimate parliamentary tables occupy, because a false rejection
    permanently blocks a record from improving while a false acceptance is
    caught by sampling.
    """
    cfg = cfg or DotsConfig.from_env()
    raw = text or ""
    body = strip_code_fence(raw).strip()

    # V1 — empty / trivially short.
    if len(body) < max(1, cfg.min_chars):
        return ValidationResult(
            False, RULE_EMPTY, f"{len(body)} chars < {cfg.min_chars}", "reject"
        )

    # V2 — the server told us it ran out of room. Highest-value check: this is
    # a reported fact, not an inference.
    if (finish_reason or "").lower() == "length":
        return ValidationResult(
            False, RULE_TRUNCATED, "finish_reason=length", "reject"
        )

    # V5/V6 run BEFORE the repetition and control-character checks, because the
    # layout prompt wraps everything in a JSON object: newlines arrive as the
    # two characters "\\n" and control bytes as "\\u0000". Scanning the raw
    # envelope would see a single line and zero control characters, so a 744x
    # repeated table row would sail straight through. The checks below run on
    # the DECODED element text, where the pathology is actually visible.
    elements: list[dict[str, Any]] | None = None
    if cfg.expects_json:
        # V5 — the layout prompt promises a single JSON object. A loop severe
        # enough to truncate the JSON is caught right here.
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, ValueError) as exc:
            return ValidationResult(
                False, RULE_BAD_JSON, f"json decode failed: {exc}", "reject"
            )
        # V6 — structural shape only. Never checks whether a category is right.
        elements = _coerce_elements(parsed)
        if elements is None:
            return ValidationResult(
                False,
                RULE_BAD_SCHEMA,
                f"expected a list of layout elements, got {type(parsed).__name__}",
                "reject",
            )

    # Decoded text for every content check from here on.
    haystack = body if elements is None else "\n".join(
        str(e.get("text") or "") for e in elements
    )

    # V8 — decode corruption, on decoded text.
    controls = len(_C0.findall(haystack))
    if "\x00" in haystack or (controls and controls / max(len(haystack), 1) > 0.01):
        return ValidationResult(
            False, RULE_CONTROL_CHARS, f"{controls} control chars", "reject"
        )

    # V3 — exact consecutive line repetition (the 744x / 355x loop shape).
    # Checked against both the decoded text and the raw envelope, so a loop in
    # the JSON structure itself (a layout box repeated 355x) is caught too.
    for scope, subject in (("text", haystack), ("envelope", body)):
        repeats, line = _max_consecutive_line_repeat(subject)
        if repeats > cfg.max_line_repeat:
            return ValidationResult(
                False,
                RULE_LINE_REPEAT,
                f"{scope}: line repeated {repeats}x "
                f"(> {cfg.max_line_repeat}): {line[:80]!r}",
                "reject",
            )

    # V4 — n-gram saturation, for loops V3 cannot see. Guarded by a length
    # floor so short legitimate output never trips it. Two passes:
    #   * word 10-grams  — whitespace-separated loops
    #   * char 40-grams  — loops with no whitespace at all (unbroken HTML)
    if len(haystack) > 2000:
        ratio, gram = _distinct_ngram_ratio(haystack)
        if ratio < cfg.min_distinct_ngram_ratio:
            return ValidationResult(
                False,
                RULE_NGRAM_SATURATION,
                f"only {ratio:.2%} of word 10-grams are distinct "
                f"(< {cfg.min_distinct_ngram_ratio:.2%}); top gram {gram[:80]!r}",
                "reject",
            )
        # Threshold derived from the same knob as V3: a unit repeated k times
        # drives this ratio to ~1/k, so 1/max_line_repeat keeps the two checks
        # consistent instead of introducing a second arbitrary constant.
        char_floor = 1.0 / max(cfg.max_line_repeat, 1)
        char_ratio, char_gram = _distinct_char_ngram_ratio(haystack)
        if char_ratio < char_floor:
            return ValidationResult(
                False,
                RULE_NGRAM_SATURATION,
                f"only {char_ratio:.2%} of char 40-grams are distinct "
                f"(< {char_floor:.2%}); repeated unit {char_gram[:60]!r}",
                "reject",
            )

    # V7 — HTML tag balance. Pure syntax, no structure judgement.
    for tag in ("table", "tr"):
        opens, closes = _tag_count(haystack, tag)
        if opens != closes:
            return ValidationResult(
                False,
                RULE_HTML_UNBALANCED,
                f"<{tag}> opens={opens} closes={closes}",
                "reject",
            )

    # V9 — consistency, NOT quality. PicoDet fired but DOTS saw no table:
    # treat as a detector false positive and keep the legacy text.
    if detector_found_table and elements is not None:
        has_table = any(
            str(e.get("category") or "").strip().lower() == "table" for e in elements
        )
        if not has_table:
            return ValidationResult(
                False,
                RULE_NO_TABLE_FOUND,
                "detector flagged a table; DOTS returned none",
                "fallback",
            )

    # V10 — severe truncation against the native text layer, when one exists.
    if native_char_count > 500:
        produced = len(haystack)
        if produced < cfg.min_ratio * native_char_count:
            return ValidationResult(
                False,
                RULE_SEVERE_TRUNCATION,
                f"{produced} chars < {cfg.min_ratio:g} x {native_char_count} native",
                "reject",
            )

    return ValidationResult(True, RULE_OK, "", "accept")


def _coerce_elements(parsed: Any) -> list[dict[str, Any]] | None:
    """Normalise a parsed layout payload into a list of element dicts."""
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        for key in ("elements", "layout", "results", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                items = value
                break
        else:
            # A single element object is still a valid shape.
            items = [parsed] if "category" in parsed or "bbox" in parsed else None
            if items is None:
                return None
    else:
        return None

    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        if "category" not in item:
            return None
        out.append(item)
    return out


def parse_layout_json(text: str) -> list[dict[str, Any]]:
    """Parse a DOTS layout response into elements. Raises on malformed input."""
    body = strip_code_fence(text or "").strip()
    parsed = json.loads(body)
    elements = _coerce_elements(parsed)
    if elements is None:
        raise ValueError("unexpected layout schema")
    return elements


# ── rendering ────────────────────────────────────────────────────────────────


def render_elements(
    elements: Sequence[dict[str, Any]],
    page_num: int,
    *,
    doc_id: str | None = None,
    table_format: str = "html",
) -> str:
    """Render DOTS layout elements into page-tagged corpus text.

    Follows the convention already in production via
    ``src/data/v2/core_text.build_core_text``: every line carries a ``[pN]``
    prefix and every table is announced by an explicit ``TABLE`` marker. Page
    provenance therefore survives into retrieval and evidence without any
    JSONL schema change.
    """
    out: list[str] = []
    table_index = 0
    for element in elements:
        category = str(element.get("category") or "").strip()
        text = str(element.get("text") or "").strip()
        if not text:
            continue  # 'Picture' legitimately omits text
        if category.lower() == "table":
            table_index += 1
            marker = f"{TABLE_MARKER} "
            if doc_id:
                marker += f"{doc_id}#p{page_num}t{table_index}"
            else:
                marker += f"#p{page_num}t{table_index}"
            out.append(f"[p{page_num}] {marker}".rstrip())
            if table_format == "markdown":
                text = _html_table_to_markdown(text)
        out.extend(f"[p{page_num}] {ln}" for ln in text.splitlines() if ln.strip())
    return "\n".join(out)


_TD = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TR = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


def _html_table_to_markdown(html: str) -> str:
    """Best-effort HTML -> Markdown pipe table.

    Optional: enabled by ``DOTS_TABLE_FORMAT=markdown`` for operators who
    prefer pipe tables (BM25 tokenises them more cleanly than HTML tags).
    Falls back to the original HTML when no rows can be recovered, so this can
    never destroy content.
    """
    rows: list[list[str]] = []
    for row_html in _TR.findall(html or ""):
        cells = [_WS.sub(" ", _TAG.sub("", c)).strip() for c in _TD.findall(row_html)]
        if cells:
            rows.append(cells)
    if not rows:
        return html
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines.extend("| " + " | ".join(r) + " |" for r in rows[1:])
    return "\n".join(lines)


# ── client ───────────────────────────────────────────────────────────────────


@dataclass
class _Discovery:
    model: str = ""
    at: float = 0.0


class DotsClient:
    """OpenAI-compatible client for a DOTS vLLM server."""

    def __init__(self, cfg: DotsConfig | None = None) -> None:
        self.cfg = cfg or DotsConfig.from_env()
        self._discovery = _Discovery()
        self._lock = threading.Lock()

    # -- plumbing ------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def _require_base_url(self) -> str:
        if not self.cfg.base_url:
            raise DotsUnavailable(
                "DOTS_BASE_URL is not set. The extraction app reaches DOTS over an "
                "external vLLM endpoint; set DOTS_BASE_URL (e.g. "
                "http://un003:8200/v1) or set DOTS_ENABLED=false."
            )
        return self.cfg.base_url

    def list_models(self) -> list[str]:
        """Model ids served by the endpoint. Raises :class:`DotsUnavailable`."""
        import requests

        url = f"{self._require_base_url()}/models"
        try:
            resp = requests.get(
                url,
                headers=self._headers(),
                timeout=(self.cfg.connect_timeout, self.cfg.health_timeout),
            )
        except Exception as exc:  # noqa: BLE001 - any transport problem
            raise DotsUnavailable(f"DOTS endpoint unreachable at {url}: {exc}") from exc
        if resp.status_code != 200:
            raise DotsUnavailable(
                f"DOTS /models returned HTTP {resp.status_code} at {url}"
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise DotsUnavailable(f"DOTS /models returned non-JSON: {exc}") from exc
        return [m.get("id", "") for m in (payload.get("data") or []) if m.get("id")]

    def resolve_model(self) -> str:
        """Pinned ``DOTS_MODEL``, else TTL-cached discovery from ``/v1/models``."""
        if self.cfg.model:
            return self.cfg.model
        now = time.monotonic()
        with self._lock:
            if self._discovery.model and (now - self._discovery.at) < self.cfg.discovery_ttl:
                return self._discovery.model
        models = self.list_models()
        if not models:
            raise DotsUnavailable(
                "DOTS endpoint reports no served models — the server is up but the "
                "model has not finished loading."
            )
        with self._lock:
            self._discovery = _Discovery(model=models[0], at=now)
        return models[0]

    def is_available(self) -> bool:
        """Health probe for preflight. Never raises."""
        try:
            self.resolve_model()
            return True
        except Exception:  # noqa: BLE001 - probe must never raise
            return False

    # -- inference -----------------------------------------------------------

    def _post_chat(self, image_b64: str) -> tuple[str, str]:
        """POST one page image, holding one global DOTS slot.

        This is the single chokepoint for DOTS inference, so the invariant
        "total in-flight DOTS requests <= DOTS_CONCURRENCY" holds no matter
        which pipeline initiated the request — tender, LS and RS cannot each
        open their own pool. The slot covers the whole call including retries,
        because a retry is still an in-flight request to the server.
        """
        from src.utils.concurrency import dots_slot

        with dots_slot():
            return self._post_chat_unlimited(image_b64)

    def _post_chat_unlimited(self, image_b64: str) -> tuple[str, str]:
        """The unthrottled POST. Call only via :meth:`_post_chat`."""
        import requests

        url = f"{self._require_base_url()}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.resolve_model(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": self.cfg.prompt},
                    ],
                }
            ],
            "max_tokens": self.cfg.max_output_tokens,
            "temperature": self.cfg.temperature,
        }
        if self.cfg.repetition_penalty and self.cfg.repetition_penalty != 1.0:
            payload["repetition_penalty"] = self.cfg.repetition_penalty

        last: Exception | None = None
        for attempt in range(max(1, self.cfg.max_retries + 1)):
            try:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=(self.cfg.connect_timeout, self.cfg.request_timeout),
                )
            except Exception as exc:  # noqa: BLE001 - transport only
                last = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff * (attempt + 1))
                    continue
                raise DotsUnavailable(f"DOTS request failed at {url}: {exc}") from exc

            if resp.status_code >= 500:
                last = RuntimeError(f"HTTP {resp.status_code}")
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff * (attempt + 1))
                    continue
            if resp.status_code != 200:
                raise DotsUnavailable(
                    f"DOTS returned HTTP {resp.status_code}: {resp.text[:300]}"
                )

            try:
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                content = (choice.get("message") or {}).get("content") or ""
                finish = choice.get("finish_reason") or ""
            except Exception as exc:  # noqa: BLE001
                raise DotsUnavailable(f"DOTS returned an unreadable body: {exc}") from exc
            return content, finish

        raise DotsUnavailable(f"DOTS request failed after retries: {last}")

    def page_image(self, page: Any) -> bytes:
        """Rasterise a PyMuPDF page to PNG, clamped to ``DOTS_MAX_PIXELS``."""
        dpi = self.cfg.render_dpi
        pixmap = page.get_pixmap(dpi=dpi)
        if self.cfg.max_pixels and (pixmap.width * pixmap.height) > self.cfg.max_pixels:
            # Downscale client-side rather than letting the server choke on an
            # oversize A3/legal annexure page.
            scale = (self.cfg.max_pixels / (pixmap.width * pixmap.height)) ** 0.5
            pixmap = page.get_pixmap(dpi=max(72, int(dpi * scale)))
        return pixmap.tobytes("png")

    def page_to_text(
        self,
        page: Any,
        page_num: int,
        *,
        doc_id: str | None = None,
        native_char_count: int = 0,
        detector_found_table: bool = True,
        image_bytes: bytes | None = None,
    ) -> str:
        """Extract one page via DOTS, validated.

        Raises :class:`DotsUnavailable` on transport failure and
        :class:`DotsInvalidOutput` on validator rejection — including the
        ``fallback`` outcome, which the caller distinguishes via ``.rule``.
        """
        png = self.page_image(page) if image_bytes is None else image_bytes
        content, finish = self._post_chat(base64.b64encode(png).decode("ascii"))

        result = validate_dots_output(
            content,
            finish_reason=finish,
            native_char_count=native_char_count,
            detector_found_table=detector_found_table,
            cfg=self.cfg,
        )
        if not result.ok and self.cfg.strict:
            raise DotsInvalidOutput(
                f"DOTS output rejected on page {page_num} [{result.rule}]: {result.detail}",
                rule=result.rule,
                detail=result.detail,
            )
        if not result.ok:
            log.warning(
                "DOTS validator (non-strict) flagged page %s [%s]: %s",
                page_num, result.rule, result.detail,
            )

        if self.cfg.expects_json:
            try:
                elements = parse_layout_json(content)
            except Exception as exc:  # noqa: BLE001 - already validated; be safe
                raise DotsInvalidOutput(
                    f"DOTS layout unparseable on page {page_num}: {exc}",
                    rule=RULE_BAD_JSON,
                    detail=str(exc),
                ) from exc
            return render_elements(
                elements, page_num, doc_id=doc_id, table_format=self.cfg.table_format
            )

        body = strip_code_fence(content).strip()
        return "\n".join(f"[p{page_num}] {ln}" for ln in body.splitlines() if ln.strip())


_client: DotsClient | None = None
_client_lock = threading.Lock()
_fallback_client: DotsClient | None = None
_fallback_lock = threading.Lock()

#: Prompt used for the ONE per-page retry after the normal prompt fails.
#: Already present in ``PROMPTS`` — a plain-text transcription prompt with no
#: layout JSON contract, so it degrades gracefully on the pages that defeat
#: the layout prompt (endless repetition, truncation, unparseable JSON).
FALLBACK_PROMPT_MODE = "prompt_ocr"


def get_fallback_client(cfg: DotsConfig | None = None) -> DotsClient:
    """Client for the single per-page fallback attempt (``prompt_ocr``).

    Same endpoint, same transport, same ``_post_chat`` — therefore the SAME
    process-wide DOTS semaphore. A fallback retry consumes a normal DOTS slot
    and can never bypass the global concurrency limit.

    Cached: one extra instance per process, not one per page. ``expects_json``
    is False for this prompt mode, so the validator automatically skips the
    JSON/layout rules and ``page_to_text`` returns the plain ``[pN] line``
    form that the rest of the pipeline already consumes.
    """
    global _fallback_client
    if cfg is not None:
        return DotsClient(replace(cfg, prompt_mode=FALLBACK_PROMPT_MODE))
    if _fallback_client is None:
        with _fallback_lock:
            if _fallback_client is None:
                # Built from the environment, not from get_client(): the
                # fallback must stay available even when the primary client
                # has been swapped out (tests, alternate configs).
                _fallback_client = DotsClient(
                    replace(DotsConfig.from_env(),
                            prompt_mode=FALLBACK_PROMPT_MODE))
    return _fallback_client


def get_client(cfg: DotsConfig | None = None) -> DotsClient:
    """Process-wide client (cheap; holds only config + a discovery cache)."""
    global _client
    if cfg is not None:
        return DotsClient(cfg)
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = DotsClient()
    return _client


def reset_client() -> None:
    """Drop the cached client so the next call re-reads the environment."""
    global _client
    with _client_lock:
        _client = None
    global _fallback_client
    _fallback_client = None
