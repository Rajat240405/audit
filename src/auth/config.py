"""Phase 2 signup configuration — environment-driven, fails loudly.

Follows the same convention as ``src/db/config.py`` and
``src/graphrag/config.py``: every value comes from the environment, secrets
come **only** from the environment, required values have no default, and
``validate()`` raises rather than degrading silently.

Nothing in this module performs I/O.

Environment variables
---------------------
Required::

    AUTH_SECRET      High-entropy server secret used to key the OTP verifier
                     HMACs and the signup tokens. Generate with:
                       python -c "import secrets; print(secrets.token_urlsafe(48))"
                     Rotating it invalidates outstanding OTPs and signup tokens.

Email allowlist::

    DEV_ADMIN_EMAIL  Optional. Exactly ONE additional address (e.g. a Gmail)
                     permitted to sign up. Matched EXACTLY after normalisation
                     — no plus-addresses, no aliases. Never hard-code it.

OTP::

    AUTH_OTP_LENGTH                     default 6
    AUTH_OTP_TTL_SECONDS                default 600  (10 minutes)
    AUTH_OTP_MAX_ATTEMPTS               default 5
    AUTH_OTP_RESEND_COOLDOWN_SECONDS    default 60
    AUTH_OTP_REQUEST_LIMIT              default 5   (per window)
    AUTH_OTP_REQUEST_WINDOW_SECONDS     default 3600
    AUTH_SIGNUP_LIMIT                   default 5
    AUTH_SIGNUP_WINDOW_SECONDS          default 3600
    AUTH_SIGNUP_TOKEN_TTL_SECONDS       default 900

Passwords (Argon2id)::

    AUTH_PASSWORD_MIN_LENGTH            default 12
    AUTH_PASSWORD_MAX_LENGTH            default 256
    AUTH_PASSWORD_REQUIRE_LOWERCASE     default 1
    AUTH_PASSWORD_REQUIRE_UPPERCASE     default 1
    AUTH_PASSWORD_REQUIRE_DIGIT         default 1
    AUTH_PASSWORD_REQUIRE_SYMBOL        default 0
    AUTH_ARGON2_TIME_COST               default 3
    AUTH_ARGON2_MEMORY_COST             default 65536  (KiB)
    AUTH_ARGON2_PARALLELISM             default 4

Mail::

    SMTP_HOST, SMTP_PORT (587), SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM,
    SMTP_STARTTLS (1), SMTP_SSL (0), SMTP_TIMEOUT (10)
    AUTH_MAILER          auto | smtp | capturing | console   (default auto)
    AUTH_DEV_PRINT_OTP   default 0 — log the OTP locally so you can copy it
    AUTH_DEV_RETURN_OTP  default 0 — return the OTP in the API response

Both dev OTP switches are REFUSED when APP_ENV=production.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Mapping
from dataclasses import dataclass

from src.auth.errors import AuthError

__all__ = ["AuthConfig", "load_auth_config", "derive_key"]

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})

#: Minimum length accepted for AUTH_SECRET. Below this an offline-guessing
#: defence is meaningless, so we refuse rather than warn.
MIN_SECRET_LENGTH = 32

_PRODUCTION_ENVS = frozenset({"prod", "production"})


def derive_key(secret: str, label: str) -> bytes:
    """Derive a domain-separated subkey from the master secret.

    Using a distinct label per purpose means an OTP verifier can never be
    replayed as a signup token (or vice versa) even if one construction is
    ever weakened.
    """
    return hmac.new(secret.encode("utf-8"), label.encode("utf-8"), hashlib.sha256).digest()


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _int_env(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _clean(env.get(name))
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise AuthError(f"{name} must be an integer, got {raw!r}.", code="config_invalid") from exc
    if value < 0:
        raise AuthError(f"{name} must be >= 0, got {value}.", code="config_invalid")
    return value


def _bool_env(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _clean(env.get(name)).lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise AuthError(
        f"{name} must be a boolean (1/true/yes/on or 0/false/no/off), got {raw!r}.",
        code="config_invalid",
    )


@dataclass(frozen=True, repr=False)
class AuthConfig:
    """Validated Phase 2 settings.

    ``secret`` and ``smtp_password`` are excluded from ``repr`` and replaced by
    a fixed placeholder, so a stray log of the config cannot leak them.
    """

    secret: str
    dev_admin_email: str | None = None

    otp_length: int = 6
    otp_ttl_seconds: int = 600
    otp_max_attempts: int = 5
    otp_resend_cooldown_seconds: int = 60
    otp_request_limit: int = 5
    otp_request_window_seconds: int = 3600
    signup_limit: int = 5
    signup_window_seconds: int = 3600
    signup_token_ttl_seconds: int = 900

    # ── Phase 3: sessions, login and password reset ───────────────────────
    #: Absolute lifetime of a session. Never slid forward.
    session_ttl_seconds: int = 28800  # 8 hours
    #: Inactivity limit, measured from the last time the session was validated.
    session_idle_timeout_seconds: int = 7200  # 2 hours
    session_cookie_name: str = "auditpro_session"
    #: None = decide from APP_ENV (secure in production, not over local HTTP).
    session_cookie_secure: bool | None = None
    session_cookie_samesite: str = "lax"
    #: Failed/completed login attempts per (address, client) per window.
    login_limit: int = 10
    login_window_seconds: int = 900
    #: Password-reset code requests per (address, client) per window.
    password_reset_limit: int = 5
    password_reset_window_seconds: int = 3600

    password_min_length: int = 12
    password_max_length: int = 256
    password_require_lowercase: bool = True
    password_require_uppercase: bool = True
    password_require_digit: bool = True
    password_require_symbol: bool = False

    argon2_time_cost: int = 3
    argon2_memory_cost: int = 65536
    argon2_parallelism: int = 4

    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    smtp_starttls: bool = True
    smtp_ssl: bool = False
    smtp_timeout_seconds: int = 10

    mailer: str = "auto"
    app_env: str = "development"
    dev_print_otp: bool = False
    dev_return_otp: bool = False

    def __repr__(self) -> str:  # never leak secrets
        return (
            f"AuthConfig(secret='<redacted>', "
            f"dev_admin_email={self.dev_admin_email!r}, "
            f"otp_ttl_seconds={self.otp_ttl_seconds!r}, "
            f"otp_max_attempts={self.otp_max_attempts!r}, "
            f"otp_resend_cooldown_seconds={self.otp_resend_cooldown_seconds!r}, "
            f"password_min_length={self.password_min_length!r}, "
            f"smtp_host={self.smtp_host!r}, smtp_port={self.smtp_port!r}, "
            f"smtp_from={self.smtp_from!r}, "
            f"smtp_password={'<set>' if self.smtp_password else '<unset>'}, "
            f"mailer={self.mailer!r}, app_env={self.app_env!r}, "
            f"dev_print_otp={self.dev_print_otp!r}, "
            f"dev_return_otp={self.dev_return_otp!r}, "
            f"session_ttl_seconds={self.session_ttl_seconds!r}, "
            f"session_idle_timeout_seconds={self.session_idle_timeout_seconds!r}, "
            f"session_cookie_name={self.session_cookie_name!r}, "
            f"session_cookie_secure={self.cookie_secure!r})"
        )

    # ── derived keys ──────────────────────────────────────────────────────
    def otp_key(self) -> bytes:
        return derive_key(self.secret, "auditpro/otp-verifier/v1")

    def signup_token_key(self) -> bytes:
        return derive_key(self.secret, "auditpro/signup-token/v1")

    def session_key(self) -> bytes:
        return derive_key(self.secret, "auditpro/session-token/v1")

    @property
    def cookie_secure(self) -> bool:
        """Secure flag for the session cookie.

        Defaults to ``APP_ENV``: on in production, off for local HTTP so a
        developer does not need TLS to sign in. Settable explicitly for the
        (unusual) case of a TLS-terminating reverse proxy on plain HTTP.
        """
        if self.session_cookie_secure is not None:
            return self.session_cookie_secure
        return self.is_production

    @property
    def is_production(self) -> bool:
        return self.app_env.strip().lower() in _PRODUCTION_ENVS

    def validate(self) -> AuthConfig:
        if len(self.secret) < MIN_SECRET_LENGTH:
            raise AuthError(
                f"AUTH_SECRET must be at least {MIN_SECRET_LENGTH} characters "
                f"(got {len(self.secret)}). Generate one with: "
                f'python -c "import secrets; print(secrets.token_urlsafe(48))"',
                code="config_invalid",
            )
        if self.session_ttl_seconds <= 0:
            raise AuthError(
                "AUTH_SESSION_TTL_SECONDS must be positive.", code="config_invalid"
            )
        if self.session_idle_timeout_seconds <= 0:
            raise AuthError(
                "AUTH_SESSION_IDLE_TIMEOUT_SECONDS must be positive.",
                code="config_invalid",
            )
        if not self.session_cookie_name.strip():
            raise AuthError(
                "AUTH_SESSION_COOKIE_NAME must not be empty.", code="config_invalid"
            )
        if self.session_cookie_samesite.lower() not in {"lax", "strict", "none"}:
            raise AuthError(
                "AUTH_SESSION_COOKIE_SAMESITE must be one of lax|strict|none, got "
                f"{self.session_cookie_samesite!r}.",
                code="config_invalid",
            )
        if self.login_limit < 1:
            raise AuthError("AUTH_LOGIN_LIMIT must be at least 1.", code="config_invalid")
        if self.password_reset_limit < 1:
            raise AuthError(
                "AUTH_PASSWORD_RESET_LIMIT must be at least 1.", code="config_invalid"
            )
        if self.otp_length < 4:
            raise AuthError("AUTH_OTP_LENGTH must be at least 4.", code="config_invalid")
        if self.otp_ttl_seconds <= 0:
            raise AuthError("AUTH_OTP_TTL_SECONDS must be positive.", code="config_invalid")
        if self.otp_max_attempts < 1:
            raise AuthError("AUTH_OTP_MAX_ATTEMPTS must be at least 1.", code="config_invalid")
        if self.password_min_length < 8:
            raise AuthError(
                "AUTH_PASSWORD_MIN_LENGTH must be at least 8.", code="config_invalid"
            )
        if self.password_max_length > 1024:
            raise AuthError(
                "AUTH_PASSWORD_MAX_LENGTH must not exceed 1024 (Argon2 DoS guard).",
                code="config_invalid",
            )
        if self.mailer not in {"auto", "smtp", "capturing", "console"}:
            raise AuthError(
                f"AUTH_MAILER must be one of auto|smtp|capturing|console, "
                f"got {self.mailer!r}.",
                code="config_invalid",
            )
        if self.mailer == "smtp" and not self.smtp_host:
            raise AuthError(
                "AUTH_MAILER=smtp requires SMTP_HOST.", code="config_invalid"
            )
        if self.mailer == "smtp" and not self.smtp_from:
            raise AuthError(
                "AUTH_MAILER=smtp requires SMTP_FROM.", code="config_invalid"
            )
        if self.smtp_ssl and self.smtp_starttls:
            raise AuthError(
                "SMTP_SSL and SMTP_STARTTLS are mutually exclusive.", code="config_invalid"
            )
        # Refuse to switch OTP disclosure on in production, however it was set.
        if self.is_production and (self.dev_print_otp or self.dev_return_otp):
            raise AuthError(
                "AUTH_DEV_PRINT_OTP / AUTH_DEV_RETURN_OTP must not be enabled "
                "when APP_ENV=production.",
                code="config_invalid",
            )
        return self



def _tri_bool_env(env: Mapping[str, str], name: str) -> bool | None:
    """A boolean that is ``None`` when unset, so 'auto' is representable."""
    raw = _clean(env.get(name))
    if not raw:
        return None
    return _bool_env(env, name, False)

def load_auth_config(env: Mapping[str, str] | None = None) -> AuthConfig:
    """Build and validate :class:`AuthConfig` from ``env`` (default ``os.environ``)."""
    parts = os.environ if env is None else env

    secret = _clean(parts.get("AUTH_SECRET"))
    if not secret:
        raise AuthError(
            "AUTH_SECRET is required and is not set. Generate one with: "
            'python -c "import secrets; print(secrets.token_urlsafe(48))" '
            "and put it in .env (local) or .env.hpc (HPC) — never in tracked source.",
            code="config_invalid",
        )

    dev_admin = _clean(parts.get("DEV_ADMIN_EMAIL")) or None

    config = AuthConfig(
        secret=secret,
        dev_admin_email=dev_admin,
        otp_length=_int_env(parts, "AUTH_OTP_LENGTH", 6),
        otp_ttl_seconds=_int_env(parts, "AUTH_OTP_TTL_SECONDS", 600),
        otp_max_attempts=_int_env(parts, "AUTH_OTP_MAX_ATTEMPTS", 5),
        otp_resend_cooldown_seconds=_int_env(parts, "AUTH_OTP_RESEND_COOLDOWN_SECONDS", 60),
        otp_request_limit=_int_env(parts, "AUTH_OTP_REQUEST_LIMIT", 5),
        otp_request_window_seconds=_int_env(parts, "AUTH_OTP_REQUEST_WINDOW_SECONDS", 3600),
        signup_limit=_int_env(parts, "AUTH_SIGNUP_LIMIT", 5),
        signup_window_seconds=_int_env(parts, "AUTH_SIGNUP_WINDOW_SECONDS", 3600),
        signup_token_ttl_seconds=_int_env(parts, "AUTH_SIGNUP_TOKEN_TTL_SECONDS", 900),
        session_ttl_seconds=_int_env(parts, "AUTH_SESSION_TTL_SECONDS", 28800),
        session_idle_timeout_seconds=_int_env(
            parts, "AUTH_SESSION_IDLE_TIMEOUT_SECONDS", 7200
        ),
        session_cookie_name=_clean(parts.get("AUTH_SESSION_COOKIE_NAME"))
        or "auditpro_session",
        session_cookie_secure=_tri_bool_env(parts, "AUTH_SESSION_COOKIE_SECURE"),
        session_cookie_samesite=(
            _clean(parts.get("AUTH_SESSION_COOKIE_SAMESITE")).lower() or "lax"
        ),
        login_limit=_int_env(parts, "AUTH_LOGIN_LIMIT", 10),
        login_window_seconds=_int_env(parts, "AUTH_LOGIN_WINDOW_SECONDS", 900),
        password_reset_limit=_int_env(parts, "AUTH_PASSWORD_RESET_LIMIT", 5),
        password_reset_window_seconds=_int_env(
            parts, "AUTH_PASSWORD_RESET_WINDOW_SECONDS", 3600
        ),
        password_min_length=_int_env(parts, "AUTH_PASSWORD_MIN_LENGTH", 12),
        password_max_length=_int_env(parts, "AUTH_PASSWORD_MAX_LENGTH", 256),
        password_require_lowercase=_bool_env(parts, "AUTH_PASSWORD_REQUIRE_LOWERCASE", True),
        password_require_uppercase=_bool_env(parts, "AUTH_PASSWORD_REQUIRE_UPPERCASE", True),
        password_require_digit=_bool_env(parts, "AUTH_PASSWORD_REQUIRE_DIGIT", True),
        password_require_symbol=_bool_env(parts, "AUTH_PASSWORD_REQUIRE_SYMBOL", False),
        argon2_time_cost=_int_env(parts, "AUTH_ARGON2_TIME_COST", 3),
        argon2_memory_cost=_int_env(parts, "AUTH_ARGON2_MEMORY_COST", 65536),
        argon2_parallelism=_int_env(parts, "AUTH_ARGON2_PARALLELISM", 4),
        smtp_host=_clean(parts.get("SMTP_HOST")) or None,
        smtp_port=_int_env(parts, "SMTP_PORT", 587),
        smtp_username=_clean(parts.get("SMTP_USERNAME")) or None,
        smtp_password=_clean(parts.get("SMTP_PASSWORD")) or None,
        smtp_from=_clean(parts.get("SMTP_FROM")) or None,
        smtp_starttls=_bool_env(parts, "SMTP_STARTTLS", True),
        smtp_ssl=_bool_env(parts, "SMTP_SSL", False),
        smtp_timeout_seconds=_int_env(parts, "SMTP_TIMEOUT", 10),
        mailer=_clean(parts.get("AUTH_MAILER")).lower() or "auto",
        app_env=_clean(parts.get("APP_ENV")).lower() or "development",
        dev_print_otp=_bool_env(parts, "AUTH_DEV_PRINT_OTP", False),
        dev_return_otp=_bool_env(parts, "AUTH_DEV_RETURN_OTP", False),
    )
    return config.validate()
