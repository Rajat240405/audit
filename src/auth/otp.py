"""One-time-passcode generation, keyed verification and hashing (Phase 2).

Security properties, all of which are covered by tests:

* **Cryptographically secure generation.** :func:`generate_code` uses
  :mod:`secrets` (a CSPRNG), never :mod:`random`.
* **No plaintext storage.** Only an HMAC digest is persisted
  (:func:`hash_code`), so a database leak does not hand an attacker usable
  codes.
* **Keyed construction.** The digest is keyed with a subkey derived from
  ``AUTH_SECRET`` (environment only), so an attacker who reads the database
  but not the environment cannot mount an offline guessing attack — and,
  because there are only 10**6 codes, that distinction is the whole defence.
* **Domain separation.** The HMAC message binds purpose, email, salt and code,
  and the key is derived with a purpose-specific label.
* **Constant-time comparison.** :func:`verify_code` uses
  ``hmac.compare_digest``.
* **Single use and expiry** are enforced by the service layer, which marks
  rows consumed rather than deleting them.

This module never logs a code.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

__all__ = ["generate_code", "hash_code", "verify_code", "new_salt"]


def new_salt() -> str:
    """A per-record random salt (hex). 128 bits of entropy."""
    return secrets.token_hex(16)


def generate_code(length: int = 6) -> str:
    """Return a zero-padded numeric code from a CSPRNG.

    ``secrets.randbelow`` is unbiased over the full 10**length space, including
    codes with leading zeros, which are then re-padded.
    """
    if length < 4:
        raise ValueError("OTP length must be at least 4")
    return f"{secrets.randbelow(10**length):0{length}d}"


def hash_code(*, key: bytes, email_normalized: str, purpose: str, salt: str, code: str) -> str:
    """HMAC-SHA256 verifier for one (email, purpose, code) triple.

    The message binds every field that scopes the code, so a verifier copied
    from another row (different salt) or another purpose will not validate.
    A NUL separator is used so no concatenation of fields can be ambiguous.
    """
    message = "\x00".join((purpose, email_normalized, salt, code)).encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def verify_code(
    *,
    key: bytes,
    email_normalized: str,
    purpose: str,
    salt: str,
    code: str,
    expected_hash: str,
) -> bool:
    """Constant-time comparison of a candidate code against the stored digest.

    A malformed candidate simply fails; it is never an exception, so the caller
    cannot be tricked into a different code path.
    """
    if not code or not expected_hash:
        return False
    candidate = hash_code(
        key=key,
        email_normalized=email_normalized,
        purpose=purpose,
        salt=salt,
        code=code,
    )
    return hmac.compare_digest(candidate, expected_hash)
