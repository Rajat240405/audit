"""Error types shared by the Phase 2 signup flow.

These are **domain** errors. They carry a stable machine-readable ``code`` that
the HTTP layer maps to a status, so the service layer never knows about HTTP
and never leaks internals into a message.

Nothing here logs a secret: OTP codes, passwords and SMTP credentials are never
placed on an exception.
"""

from __future__ import annotations

__all__ = ["AuthError", "RateLimited", "OtpError", "DuplicateAccount"]

#: HTTP statuses for each code. Kept here so the router stays a thin adapter.
STATUS_BY_CODE = {
    "email_required": 400,
    "email_invalid": 400,
    "email_not_allowed": 403,
    "otp_invalid": 400,
    "otp_expired": 400,
    "otp_attempts_exceeded": 429,
    "otp_already_used": 400,
    "otp_not_found": 400,
    "otp_resend_cooldown": 429,
    "token_invalid": 400,
    "token_expired": 400,
    "password_invalid": 400,
    "account_exists": 409,
    "rate_limited": 429,
    "mailer_failed": 502,
    # Phase 3
    "invalid_credentials": 401,
    "not_authenticated": 401,
}


class AuthError(Exception):
    """Base class. ``message`` is safe to return to the client."""

    def __init__(self, message: str, *, code: str = "auth_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    @property
    def http_status(self) -> int:
        return STATUS_BY_CODE.get(self.code, 400)


class RateLimited(AuthError):
    """Too many requests. ``retry_after`` is seconds."""

    def __init__(
        self, message: str, *, retry_after: int, code: str = "rate_limited"
    ) -> None:
        super().__init__(message, code=code)
        self.retry_after = max(0, int(retry_after))


class OtpError(AuthError):
    """Any OTP verification failure — intentionally does not say which."""


class DuplicateAccount(AuthError):
    """An account already exists for this address."""

    def __init__(self, message: str = "an account already exists for this email") -> None:
        super().__init__(message, code="account_exists")
