"""Safe, user-facing descriptions of generation failures.

Two audiences, two strings:

* ``user_facing_error`` is what the browser shows. It uses a fixed vocabulary
  and never contains a provider response body, a traceback, an internal path,
  a prompt, or an exception message.
* ``log_line`` is for the server log. It holds the exception type and, for HTTP
  failures, the status plus the truncated provider error fields.

Both read only the exception type, the HTTP status and the provider's own error
fields (via ``context_overflow``). Neither returns or logs request prompts or
retrieved document text.
"""

from __future__ import annotations

import httpx

from src.generation.context_overflow import (
    describe_http_error,
    find_http_status_error,
    is_context_overflow_error,
)

_TOO_LARGE = "the input was too large for the model"


def _classify(exc: BaseException) -> tuple[bool, str]:
    """Return ``(context_too_large, reason)`` using a fixed set of reasons."""
    http = find_http_status_error(exc)
    if http is not None:
        if http.response.status_code == 413 or is_context_overflow_error(exc):
            return True, _TOO_LARGE
        return False, f"the model service returned an error (HTTP {http.response.status_code})"
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return False, "the model did not respond in time"
    if isinstance(exc, httpx.ConnectError):
        return False, "the model service could not be reached"
    if isinstance(exc, (httpx.TransportError, ConnectionError)):
        return False, "the connection to the model service was interrupted"
    return False, "an internal error occurred"


def user_facing_error(exc: BaseException, *, partial_output: bool, action: str = "answer") -> str:
    """Safe message for the browser.

    ``partial_output`` is True once text has already reached the client. The
    message then says the output stopped early, so partial text is not presented
    as complete. No retry is attempted in that case. ``action`` is "answer" for
    chat or "edit" for the draft editor.
    """
    too_large, reason = _classify(exc)
    if partial_output:
        return (
            f"Generation stopped before the {action} was complete ({reason}). "
            f"The text above may be incomplete. Please try again."
        )
    if action == "edit":
        if too_large:
            return "Edit failed: the draft was too large for the model. Please shorten it and try again."
        return f"Edit failed: {reason}. Please try again."
    if too_large:
        return (
            "Answer generation failed: the input was too large for the model. "
            "Please ask a shorter or more specific question."
        )
    return f"Answer generation failed: {reason}. Please try again."


def log_line(exc: BaseException) -> str:
    """Server-log summary: exception type plus safe HTTP detail. No prompts."""
    http = find_http_status_error(exc)
    if http is not None:
        return f"{type(exc).__name__}: {describe_http_error(http.response)}"
    return f"{type(exc).__name__}: {str(exc)[:200]}"
