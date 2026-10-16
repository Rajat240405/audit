"""Argon2id password hashing and backend password policy (Phase 2).

Hashing uses ``argon2-cffi`` (``PasswordHasher``), the maintained reference
implementation of Argon2 and the OWASP first recommendation. Argon2id is the
default variant of that hasher. Passwords are never logged, never returned and
never stored in any form other than the encoded Argon2 hash.

The policy is enforced **on the backend**; the frontend mirrors it only so the
user gets instant feedback. The backend copy is authoritative.
"""

from __future__ import annotations

import re

from argon2 import PasswordHasher, Type
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)

from src.auth.config import AuthConfig
from src.auth.errors import AuthError

__all__ = ["hash_password", "verify_password", "validate_password", "password_policy"]

# A deliberately small list. The point is to reject the very worst choices, not
# to replace a real breach-corpus check; length and the allowlist do the rest.
_COMMON_PASSWORDS = frozenset(
    {
        "password", "password1", "password123", "12345678", "123456789",
        "1234567890", "qwerty123456", "letmein123", "welcome123",
        "admin12345", "incois123", "incois12345", "changeme123",
        "iloveyou123", "monkey1234", "dragon1234", "sunshine123",
        "princess123", "football123", "passwordpassword",
    }
)


def _hasher(config: AuthConfig) -> PasswordHasher:
    return PasswordHasher(
        time_cost=config.argon2_time_cost,
        memory_cost=config.argon2_memory_cost,
        parallelism=config.argon2_parallelism,
        hash_len=32,
        salt_len=16,
        type=Type.ID,  # Argon2id — hybrid, side-channel resistant
    )


def hash_password(password: str, *, config: AuthConfig) -> str:
    """Return the encoded Argon2id hash. Never logs or stores the plaintext."""
    return _hasher(config).hash(password)


def verify_password(password: str, hashed: str, *, config: AuthConfig) -> bool:
    """Return True when ``password`` matches ``hashed``.

    Any failure (mismatch, malformed hash) returns False rather than raising, so
    the caller has a single code path. ``check_needs_rehash`` is deliberately
    NOT called here: rehashing on login requires a session, which is Phase 3.
    """
    try:
        return _hasher(config).verify(hashed, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def password_policy(config: AuthConfig) -> dict[str, object]:
    """The policy, as JSON — so the UI can render the real rules."""
    return {
        "min_length": config.password_min_length,
        "max_length": config.password_max_length,
        "require_lowercase": config.password_require_lowercase,
        "require_uppercase": config.password_require_uppercase,
        "require_digit": config.password_require_digit,
        "require_symbol": config.password_require_symbol,
        "rejects_email_substring": True,
        "rejects_common_passwords": True,
    }


def validate_password(
    password: str, *, config: AuthConfig, email_normalized: str = ""
) -> None:
    """Enforce the backend password policy.

    Raises:
        AuthError: code ``password_invalid``, with a message listing every
            unmet rule (all of them, so the user is not made to guess).
    """
    problems: list[str] = []

    if not password:
        raise AuthError("password is required", code="password_invalid")

    if len(password) < config.password_min_length:
        problems.append(
            f"at least {config.password_min_length} characters "
            f"(got {len(password)})"
        )
    if len(password) > config.password_max_length:
        problems.append(
            f"at most {config.password_max_length} characters "
            f"(got {len(password)})"
        )
    if config.password_require_lowercase and not re.search(r"[a-z]", password):
        problems.append("a lowercase letter")
    if config.password_require_uppercase and not re.search(r"[A-Z]", password):
        problems.append("an uppercase letter")
    if config.password_require_digit and not re.search(r"[0-9]", password):
        problems.append("a digit")
    if config.password_require_symbol and not re.search(r"[^A-Za-z0-9]", password):
        problems.append("a symbol")

    lowered = password.lower()

    if email_normalized:
        local_part = email_normalized.split("@", 1)[0]
        # Only reject meaningful substrings: a 3-character local part would
        # otherwise forbid almost every password.
        for fragment in {email_normalized, local_part}:
            if len(fragment) >= 4 and fragment in lowered:
                problems.append("must not contain your email address")
                break

    if lowered in _COMMON_PASSWORDS:
        problems.append("must not be a commonly used password")

    if problems:
        raise AuthError(
            "password must contain " + "; ".join(problems), code="password_invalid"
        )
