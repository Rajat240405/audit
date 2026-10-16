"""Phase 2 — signup and email OTP verification.

Scope, stated plainly so nobody mistakes this for a finished auth system:

**In this phase**
* signup with an email allowlist (``@incois.gov.in`` plus one ``DEV_ADMIN_EMAIL``);
* cryptographically secure OTPs, stored only as keyed HMAC digests;
* OTP expiry, single use, attempt caps, resend cooldown and rate limits;
* Argon2id password hashing with a backend-enforced policy;
* account creation with the email marked verified.

**Explicitly NOT in this phase** (later phases)
* login / logout, sessions or cookies;
* password reset;
* authentication middleware or protection of existing application routes;
* roles and role enforcement;
* chat-history persistence;
* Saved Knowledge / ingestion authorisation changes.

The application is therefore **not** production-secured because signup works.
Every existing route remains unauthenticated until Phase 4.

Importing this package performs no I/O: no database connection, no SMTP
connection. Configuration is read from the environment on demand.
"""

from __future__ import annotations

from src.auth.config import AuthConfig, derive_key, load_auth_config
from src.auth.emails import ALLOWED_DOMAIN, is_allowed_signup_email, normalize_email
from src.auth.errors import AuthError, DuplicateAccount, OtpError, RateLimited
from src.auth.models import EmailOtp, User
from src.auth.passwords import hash_password, validate_password, verify_password

__all__ = [
    "ALLOWED_DOMAIN",
    "AuthConfig",
    "AuthError",
    "DuplicateAccount",
    "EmailOtp",
    "OtpError",
    "RateLimited",
    "User",
    "derive_key",
    "hash_password",
    "is_allowed_signup_email",
    "load_auth_config",
    "normalize_email",
    "validate_password",
    "verify_password",
]
