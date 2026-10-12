"""Recognise provider context-overflow rejections (vLLM / OpenAI-compatible).

A request is a context overflow only when the provider answers HTTP 400 AND
its error body carries reliable context-limit evidence: the vLLM / OpenAI
context-limit wording, or the specific ``context_length_exceeded`` code. A bare
HTTP 400 is not enough, and neither is ``param=input_tokens`` on its own:
other validation errors can name that parameter too, and they must keep failing
loudly.

Privacy: the helpers read only the provider's own error fields (message,
type, param, code) and return at most a short truncated summary for logs.
They never return or log the request prompt or any retrieved document text.
"""

from __future__ import annotations

import json

import httpx

OVERFLOW_STATUS = 400

# Lower-case substrings that identify a context-limit rejection. Each one is
# reliable evidence on its own. Deliberately NOT included: the bare parameter
# name "input_tokens". vLLM reports its overflow with param "input_tokens", but
# so can unrelated validation errors, and a bare HTTP 400 is never enough.
# vLLM: "This model's maximum context length is N tokens ... your prompt
# contains at least M input tokens".
OVERFLOW_MARKERS: tuple[str, ...] = (
    "maximum context length",
    "prompt contains at least",
    "context_length_exceeded",
)

_LOG_CHARS = 200
_MATCH_CHARS = 4000


def find_http_status_error(exc: BaseException | None) -> httpx.HTTPStatusError | None:
    """Return *exc* if it is an HTTPStatusError, else the one it wraps via ``__cause__``.

    The non-streaming provider re-raises transport/HTTP failures as
    ``RuntimeError(...) from exc``; only the explicit ``__cause__`` link is
    followed, so an unrelated earlier exception is never picked up.
    """
    cur: BaseException | None = exc
    for _ in range(4):
        if cur is None:
            return None
        if isinstance(cur, httpx.HTTPStatusError):
            return cur
        cur = cur.__cause__
    return None


def error_text(response: httpx.Response) -> str:
    """The provider's error fields as one string (original case).

    Returns "" when the body is not available (e.g. an unread streamed
    response), so classification fails closed instead of raising.
    """
    try:
        body = response.text
    except Exception:  # noqa: BLE001 - unread stream / no body
        return ""
    if not body:
        return ""
    try:
        data = json.loads(body)
    except ValueError:
        return body[:_MATCH_CHARS]
    parts: list[str] = []
    err = data.get("error", data) if isinstance(data, dict) else None
    if isinstance(err, dict):
        for key in ("message", "type", "param", "code"):
            value = err.get(key)
            if value is not None:
                parts.append(str(value))
    elif isinstance(err, str):
        parts.append(err)
    if not parts:
        parts.append(body)
    return " ".join(parts)[:_MATCH_CHARS]


def is_context_overflow_error(exc: BaseException | None) -> bool:
    """True only for HTTP 400 whose provider error names a context/token limit."""
    err = find_http_status_error(exc)
    if err is None or err.response.status_code != OVERFLOW_STATUS:
        return False
    text = error_text(err.response).lower()
    return any(marker in text for marker in OVERFLOW_MARKERS)


def describe_http_error(response: httpx.Response) -> str:
    """Short, truncated, prompt-free summary of a provider error for logs."""
    summary = " ".join(error_text(response).split())[:_LOG_CHARS]
    return f"HTTP {response.status_code}" + (f": {summary}" if summary else "")
