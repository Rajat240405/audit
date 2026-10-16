"""Short-lived, stateless signup tokens (Phase 2).

After an OTP is verified the client receives a token that authorises exactly one
action: setting the password for that address. It is:

* **stateless** — an HMAC-signed payload, so Phase 2 needs no extra table and
  no server-side session (sessions are Phase 3);
* **short-lived** — ``AUTH_SIGNUP_TOKEN_TTL_SECONDS``, default 15 minutes;
* **purpose-bound** — the ``purpose`` field is inside the signed payload;
* **replay-resistant in practice** — using it creates the account, and the
  account's unique email constraint makes a second use fail.

The token is signed with a subkey derived from ``AUTH_SECRET`` under a label
distinct from the OTP one, so an OTP verifier can never be presented as a token.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time

from src.auth.config import AuthConfig
from src.auth.errors import AuthError

__all__ = ["issue_signup_token", "read_signup_token"]

PURPOSE = "signup"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _sign(key: bytes, payload: str) -> str:
    return _b64url(hmac.new(key, payload.encode("ascii"), hashlib.sha256).digest())


def issue_signup_token(
    *, key: bytes, email_normalized: str, ttl_seconds: int, now: float | None = None
) -> tuple[str, int]:
    """Return ``(token, expires_at_epoch)``."""
    moment = time.time() if now is None else now
    expires_at = int(moment) + int(ttl_seconds)
    payload_obj = {
        "e": email_normalized,
        "p": PURPOSE,
        "exp": expires_at,
        "n": secrets.token_hex(8),
    }
    payload = _b64url(json.dumps(payload_obj, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return f"{payload}.{_sign(key, payload)}", expires_at


def read_signup_token(
    token: str, *, key: bytes, now: float | None = None
) -> str:
    """Validate a signup token and return the normalised email it authorises.

    Raises:
        AuthError: ``token_invalid`` for a malformed or forged token,
            ``token_expired`` for one past its TTL.
    """
    moment = time.time() if now is None else now
    if not token or "." not in token:
        raise AuthError("signup token is missing or malformed", code="token_invalid")
    payload, _, signature = token.rpartition(".")
    if not payload or not signature:
        raise AuthError("signup token is malformed", code="token_invalid")

    # Constant-time: compare the signature before trusting the payload at all.
    if not hmac.compare_digest(_sign(key, payload), signature):
        raise AuthError("signup token is not valid", code="token_invalid")

    try:
        data = json.loads(_b64url_decode(payload).decode("utf-8"))
    except Exception:  # noqa: BLE001 - any decoding fault is just a bad token
        raise AuthError("signup token is not valid", code="token_invalid") from None

    if not isinstance(data, dict) or data.get("p") != PURPOSE:
        raise AuthError("signup token is not valid", code="token_invalid")
    email = data.get("e")
    if not isinstance(email, str) or not email:
        raise AuthError("signup token is not valid", code="token_invalid")

    expires_at = data.get("exp")
    if not isinstance(expires_at, int) or expires_at <= moment:
        raise AuthError("signup token has expired", code="token_expired")

    return email


def signup_token_key(config: AuthConfig) -> bytes:
    """Convenience wrapper so the service does not touch key derivation."""
    return config.signup_token_key()
