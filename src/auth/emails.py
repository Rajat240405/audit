"""Email normalisation and the signup allowlist (Phase 2).

Normalisation rules — applied identically before **validation**, before
**hashing/lookup** and before **database uniqueness**, so a user can never
register two rows that differ only by case or stray whitespace:

1. strip surrounding whitespace;
2. lowercase the whole address (local part AND domain).

Lowercasing the local part is technically not RFC-5321-strict (the local part
is case-sensitive in theory) but it is the standard, predictable behaviour for
account identity and is what "normalise consistently" requires. It is applied
everywhere, so it is at least *consistent*.

No plus-address or dot folding is performed. In particular the single developer
Gmail exception is an **exact** match on the normalised address: `dev+alias@`
is NOT accepted. That is deliberate — the exception is meant to admit exactly
one address, not a family of them.

The allowlist is the real security boundary for signup:
  * `<anything>@incois.gov.in`
  * the single address named by `DEV_ADMIN_EMAIL` (env only, never hard-coded)
"""

from __future__ import annotations

import re

from src.auth.errors import AuthError

__all__ = [
    "ALLOWED_DOMAIN",
    "normalize_email",
    "validate_signup_email",
    "is_allowed_signup_email",
]

#: The one organisational domain allowed to self-register.
ALLOWED_DOMAIN = "incois.gov.in"

# Deliberately conservative: no quoted local parts, no IP literals, no
# address literals. Signup is restricted to a single organisational domain
# plus one developer address, so exotic-but-valid syntax is not a use case —
# and every accepted form is additionally gated by the allowlist below.
_LOCAL = r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
_DOMAIN_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_DOMAIN = rf"{_DOMAIN_LABEL}(?:\.{_DOMAIN_LABEL})+"
_EMAIL_RE = re.compile(rf"^{_LOCAL}@{_DOMAIN}$")

MAX_EMAIL_LENGTH = 254  # RFC 5321


def normalize_email(raw: str) -> str:
    """Canonical form: trimmed and fully lowercased.

    Raises:
        AuthError: if the input is not usable as an email address at all.
    """
    value = (raw or "").strip()
    if not value:
        raise AuthError("email is required", code="email_required")
    if len(value) > MAX_EMAIL_LENGTH:
        raise AuthError("email is too long", code="email_invalid")
    # Reject anything with whitespace or control characters inside: these are
    # classic header-injection and homoglyph vectors and never legitimate.
    if any(ch.isspace() or ord(ch) < 0x20 for ch in value):
        raise AuthError("email contains invalid characters", code="email_invalid")
    normalized = value.lower()
    if not _EMAIL_RE.match(value):
        raise AuthError("email address is not valid", code="email_invalid")
    return normalized


def is_allowed_signup_email(
    normalized_email: str, *, dev_admin_email: str | None
) -> bool:
    """True when the (already normalised) address may sign up."""
    if not normalized_email:
        return False
    if normalized_email.endswith("@" + ALLOWED_DOMAIN):
        return True
    if dev_admin_email:
        # Exact match only. No plus-address or dot folding, so `dev+x@` is
        # rejected: the variable is meant to admit exactly one mailbox.
        if normalized_email == normalize_email(dev_admin_email):
            return True
    return False


def validate_signup_email(raw: str, *, dev_admin_email: str | None) -> str:
    """Normalise and enforce the allowlist.

    Returns:
        The normalised address.

    Raises:
        AuthError: code ``email_invalid`` or ``email_not_allowed``.
    """
    normalized = normalize_email(raw)
    if not is_allowed_signup_email(normalized, dev_admin_email=dev_admin_email):
        raise AuthError(
            "this email address is not allowed to sign up",
            code="email_not_allowed",
        )
    return normalized
