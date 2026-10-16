"""Authentication orchestration — Phase 2 signup + Phase 3 login / reset.

The service is transport-agnostic: it takes a ``Session`` and raises domain
errors from :mod:`src.auth.errors`. The HTTP layer only maps those to statuses.

Phase 2 (top): start signup → verify the emailed code → set a password →
create the account. Phase 3 (bottom): sign in, sign out, resolve the session
cookie, and reset a forgotten password by email OTP.

What this module deliberately does **not** do — all later phases:
  * no authentication middleware and no protection of existing routes (Phase 4);
  * no roles, permissions or role enforcement (Phase 4);
  * no chat-history persistence (Phase 5).

Invariants enforced here
------------------------
* email allowlist (``@incois.gov.in`` plus the one ``DEV_ADMIN_EMAIL``);
* normalisation before validation, hashing and uniqueness;
* OTP expiry, single use, attempt cap, resend cooldown;
* OTP purpose scoping — a signup code cannot reset a password, or vice versa;
* request rate limits (in-process, see :mod:`src.auth.rate_limit`);
* duplicate accounts rejected, including under concurrency;
* at most one live OTP per (email, purpose);
* login failures are indistinguishable: unknown address, wrong password and a
  disabled account all raise the same ``invalid_credentials`` error.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.auth import otp as otp_mod
from src.auth import sessions as sessions_mod
from src.auth.config import AuthConfig
from src.auth.emails import normalize_email, validate_signup_email
from src.auth.errors import (
    AuthError,
    DuplicateAccount,
    OtpError,
    RateLimited,
)
from src.auth.mailer import Mailer, MailMessage
from src.auth.models import (
    OTP_PURPOSE_PASSWORD_RESET,
    OTP_PURPOSE_SIGNUP,
    EmailOtp,
    User,
    as_utc,
    new_id,
    utcnow,
)
from src.auth.passwords import hash_password, validate_password, verify_password
from src.auth.rate_limit import get_limiter

__all__ = [
    "SignupStartResult",
    "SignupVerifyResult",
    "SignupCompleteResult",
    "LoginResult",
    "LogoutResult",
    "PasswordResetStartResult",
    "PasswordResetCompleteResult",
    "start_signup",
    "verify_signup_otp",
    "resend_signup_otp",
    "complete_signup",
    "login",
    "logout",
    "current_user",
    "start_password_reset",
    "complete_password_reset",
]


class Clock(Protocol):
    """Injectable clock so tests can advance time without sleeping."""

    def __call__(self) -> float: ...


@dataclass(frozen=True)
class SignupStartResult:
    email: str
    expires_in_seconds: int
    resend_available_in_seconds: int
    #: Present ONLY when AUTH_DEV_RETURN_OTP is on outside production.
    dev_otp: str | None = None


@dataclass(frozen=True)
class SignupVerifyResult:
    email: str
    signup_token: str
    expires_in_seconds: int


@dataclass(frozen=True)
class SignupCompleteResult:
    user_id: str
    email: str
    email_verified: bool


# ── Phase 3 results ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LoginResult:
    user_id: str
    email: str
    #: Raw session token — returned exactly once, to be written to the cookie.
    session_token: str
    expires_in_seconds: int


@dataclass(frozen=True)
class LogoutResult:
    #: False when there was no live session (already logged out, or the cookie
    #: was stale). Logging out is idempotent, so this is never an error.
    revoked: bool


@dataclass(frozen=True)
class PasswordResetStartResult:
    email: str
    expires_in_seconds: int
    resend_available_in_seconds: int
    dev_otp: str | None = None


@dataclass(frozen=True)
class PasswordResetCompleteResult:
    email: str
    #: Every live session for the account was revoked (including the caller's).
    sessions_revoked: int


def _pending_otp(
    session: Session, email: str, purpose: str = OTP_PURPOSE_SIGNUP
) -> EmailOtp | None:
    """The single live, unconsumed, unexpired OTP for (email, purpose)."""
    now = utcnow()
    row = (
        session.execute(
            select(EmailOtp)
            .where(
                EmailOtp.email_normalized == email,
                EmailOtp.purpose == purpose,
                EmailOtp.consumed_at.is_(None),
                EmailOtp.expires_at > now,
            )
            .order_by(EmailOtp.created_at.desc())
        )
        .scalars()
        .first()
    )
    return row


def _supersede_previous(
    session: Session, email: str, purpose: str = OTP_PURPOSE_SIGNUP
) -> None:
    """Expire any earlier live challenge so only one code is ever valid."""
    now = utcnow()
    for row in (
        session.execute(
            select(EmailOtp).where(
                EmailOtp.email_normalized == email,
                EmailOtp.purpose == purpose,
                EmailOtp.consumed_at.is_(None),
            )
        )
        .scalars()
        .all()
    ):
        # Expire rather than delete: keeps the audit trail, and `expires_at`
        # in the past is unambiguous.
        row.expires_at = now - timedelta(seconds=1)


def _existing_user(session: Session, email: str) -> User | None:
    return session.execute(
        select(User).where(User.email_normalized == email)
    ).scalar_one_or_none()


def _mail_body(code: str, ttl_minutes: int, *, action: str) -> str:
    return (
        f"Your INCOIS Audit Pro {action} code is: {code}\n\n"
        f"It expires in {ttl_minutes} minutes and can be used once.\n"
        f"If you did not request this, you can ignore this message.\n"
    )


def _rate_limit_start(email: str, client_key: str, config: AuthConfig) -> None:
    limiter = get_limiter()
    limiter.check(
        f"signup-start:email:{email}",
        limit=config.signup_limit,
        window_seconds=config.signup_window_seconds,
    )
    if client_key:
        limiter.check(
            f"signup-start:client:{client_key}",
            limit=config.signup_limit,
            window_seconds=config.signup_window_seconds,
        )


def _rate_limit_otp_request(email: str, client_key: str, config: AuthConfig) -> None:
    limiter = get_limiter()
    limiter.check(
        f"otp-request:email:{email}",
        limit=config.otp_request_limit,
        window_seconds=config.otp_request_window_seconds,
    )
    if client_key:
        limiter.check(
            f"otp-request:client:{client_key}",
            limit=config.otp_request_limit,
            window_seconds=config.otp_request_window_seconds,
        )


def _rate_limit_verify(email: str, client_key: str, config: AuthConfig) -> None:
    limiter = get_limiter()
    limiter.check(
        f"otp-verify:email:{email}",
        limit=config.otp_max_attempts * 3,
        window_seconds=config.otp_request_window_seconds,
    )
    if client_key:
        limiter.check(
            f"otp-verify:client:{client_key}",
            limit=config.otp_max_attempts * 3,
            window_seconds=config.otp_request_window_seconds,
        )


def _issue(
    session: Session,
    *,
    email: str,
    config: AuthConfig,
    mailer: Mailer,
    resend_count: int = 0,
    purpose: str = OTP_PURPOSE_SIGNUP,
) -> SignupStartResult:
    """Create one OTP challenge, supersede older ones, and mail it.

    ``resend_count`` is carried forward from the superseded challenge, so the
    row records how many codes this address has actually consumed instead of
    restarting at zero on every resend.

    ``purpose`` scopes the HMAC and the row, so a signup code is meaningless to
    the password-reset endpoint and vice versa.
    """
    _supersede_previous(session, email, purpose)

    code = otp_mod.generate_code(config.otp_length)
    salt = otp_mod.new_salt()
    code_hash = otp_mod.hash_code(
        key=config.otp_key(),
        email_normalized=email,
        purpose=purpose,
        salt=salt,
        code=code,
    )

    now = utcnow()
    row = EmailOtp(
        id=new_id(),
        email_normalized=email,
        purpose=purpose,
        code_hash=code_hash,
        salt=salt,
        expires_at=now + timedelta(seconds=config.otp_ttl_seconds),
        attempts=0,
        max_attempts=config.otp_max_attempts,
        consumed_at=None,
        created_at=now,
        last_sent_at=now,
        resend_count=resend_count,
    )
    session.add(row)

    ttl_minutes = max(1, config.otp_ttl_seconds // 60)
    is_signup = purpose == OTP_PURPOSE_SIGNUP
    try:
        mailer.send(
            MailMessage(
                to=email,
                subject=(
                    "Verify your email — INCOIS Audit Pro"
                    if is_signup
                    else "Reset your password — INCOIS Audit Pro"
                ),
                body=_mail_body(
                    code,
                    ttl_minutes,
                    action="verification" if is_signup else "password reset",
                ),
            )
        )
    except Exception:
        # Nothing was delivered, so nothing may be stored. Without this the row
        # stays pending on the session and a later commit would silently persist
        # a code the user never received.
        session.rollback()
        raise

    # Commit the challenge BEFORE returning so the row exists for /verify even
    # if the caller's transaction context differs.
    session.commit()

    return SignupStartResult(
        email=email,
        expires_in_seconds=config.otp_ttl_seconds,
        resend_available_in_seconds=config.otp_resend_cooldown_seconds,
        dev_otp=(code if config.dev_return_otp and not config.is_production else None),
    )


def start_signup(
    session: Session,
    *,
    email: str,
    config: AuthConfig,
    mailer: Mailer,
    client_key: str = "",
) -> SignupStartResult:
    """Begin signup: validate the address and send a verification code.

    Raises:
        AuthError: ``email_*`` for an unusable or disallowed address,
            ``account_exists`` if that address already has an account,
            ``rate_limited`` if too many requests.
    """
    normalized = validate_signup_email(email, dev_admin_email=config.dev_admin_email)
    _rate_limit_start(normalized, client_key, config)

    if _existing_user(session, normalized) is not None:
        raise DuplicateAccount()

    return _issue(session, email=normalized, config=config, mailer=mailer, resend_count=0)


def resend_signup_otp(
    session: Session,
    *,
    email: str,
    config: AuthConfig,
    mailer: Mailer,
    client_key: str = "",
) -> SignupStartResult:
    """Send a fresh code, honouring the resend cooldown."""
    normalized = validate_signup_email(email, dev_admin_email=config.dev_admin_email)
    _rate_limit_otp_request(normalized, client_key, config)

    if _existing_user(session, normalized) is not None:
        raise DuplicateAccount()

    carry = 0
    pending = _pending_otp(session, normalized)
    if pending is not None:
        elapsed = (utcnow() - as_utc(pending.last_sent_at)).total_seconds()
        wait = config.otp_resend_cooldown_seconds - elapsed
        if wait > 0:
            raise RateLimited(
                "please wait before requesting another code",
                retry_after=int(wait) + 1,
                code="otp_resend_cooldown",
            )
        pending.resend_count += 1
        carry = pending.resend_count

    return _issue(
        session, email=normalized, config=config, mailer=mailer, resend_count=carry
    )


def _verify_otp(
    session: Session,
    *,
    email: str,
    code: str,
    purpose: str,
    config: AuthConfig,
) -> None:
    """Consume one OTP, or raise :class:`OtpError`.

    Shared by signup verification and password reset so the two flows cannot
    drift apart: expiry, the attempt cap and single use behave identically, and
    the ``purpose`` scoping means a code from one flow is rejected by the other.
    """
    row = _pending_otp(session, email, purpose)
    if row is None:
        # Deliberately identical wording for "no code", "expired" and "used":
        # the response must not tell an attacker which.
        raise OtpError("the code is invalid or has expired", code="otp_invalid")

    if row.attempts >= row.max_attempts:
        row.consumed_at = utcnow()
        session.commit()
        raise OtpError("too many incorrect attempts", code="otp_attempts_exceeded")

    ok = otp_mod.verify_code(
        key=config.otp_key(),
        email_normalized=email,
        purpose=purpose,
        salt=row.salt,
        code=(code or "").strip(),
        expected_hash=row.code_hash,
    )

    if not ok:
        row.attempts += 1
        remaining = row.max_attempts - row.attempts
        session.commit()
        if remaining <= 0:
            row.consumed_at = utcnow()
            session.commit()
            raise OtpError("too many incorrect attempts", code="otp_attempts_exceeded")
        raise OtpError(
            f"the code is incorrect — {remaining} attempt(s) remaining",
            code="otp_invalid",
        )

    # Single use: mark consumed before the caller acts on the result.
    row.consumed_at = utcnow()
    session.commit()


def verify_signup_otp(
    session: Session,
    *,
    email: str,
    code: str,
    config: AuthConfig,
    client_key: str = "",
    token_ttl_seconds: int | None = None,
) -> SignupVerifyResult:
    """Verify a code and return a short-lived signup token.

    The token authorises setting the password. No account is created yet and no
    session is issued.

    Raises:
        OtpError: for a wrong, expired, already-used or over-attempted code.
    """
    normalized = validate_signup_email(email, dev_admin_email=config.dev_admin_email)
    _rate_limit_verify(normalized, client_key, config)
    _verify_otp(
        session,
        email=normalized,
        code=code,
        purpose=OTP_PURPOSE_SIGNUP,
        config=config,
    )

    from src.auth import tokens as tokens_mod  # local import: keeps cycles away

    ttl = config.signup_token_ttl_seconds if token_ttl_seconds is None else token_ttl_seconds
    token, _ = tokens_mod.issue_signup_token(
        key=config.signup_token_key(),
        email_normalized=normalized,
        ttl_seconds=ttl,
    )
    return SignupVerifyResult(email=normalized, signup_token=token, expires_in_seconds=ttl)


def complete_signup(
    session: Session,
    *,
    signup_token: str,
    password: str,
    config: AuthConfig,
) -> SignupCompleteResult:
    """Validate the password and create the verified account.

    Raises:
        AuthError: ``token_*`` for a bad/expired token, ``password_invalid``
            for a policy failure, ``account_exists`` on a duplicate.
    """
    from src.auth import tokens as tokens_mod  # local import: keeps cycles away

    email = tokens_mod.read_signup_token(
        signup_token, key=config.signup_token_key()
    )

    # Re-check the allowlist: the token is signed by us, but validating again
    # keeps the invariant local and independent of how the token was issued.
    normalized = validate_signup_email(email, dev_admin_email=config.dev_admin_email)

    validate_password(password, config=config, email_normalized=normalized)

    if _existing_user(session, normalized) is not None:
        raise DuplicateAccount()

    user = User(
        id=new_id(),
        email=normalized,
        email_normalized=normalized,
        password_hash=hash_password(password, config=config),
        email_verified=True,
        is_active=True,
    )
    session.add(user)
    try:
        session.commit()
    except IntegrityError:
        # Concurrent signup for the same address: another request won the race.
        session.rollback()
        raise DuplicateAccount() from None

    return SignupCompleteResult(
        user_id=user.id, email=user.email, email_verified=user.email_verified
    )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — login, logout, sessions and password reset
#
# Signup (above) is unchanged. Everything below is additive.
# ═══════════════════════════════════════════════════════════════════════════


def _rate_limit_login(email: str, client_key: str, config: AuthConfig) -> None:
    limiter = get_limiter()
    limiter.check(
        f"login:email:{email}",
        limit=config.login_limit,
        window_seconds=config.login_window_seconds,
    )
    if client_key:
        limiter.check(
            f"login:client:{client_key}",
            limit=config.login_limit,
            window_seconds=config.login_window_seconds,
        )


def _rate_limit_password_reset(email: str, client_key: str, config: AuthConfig) -> None:
    limiter = get_limiter()
    limiter.check(
        f"password-reset:email:{email}",
        limit=config.password_reset_limit,
        window_seconds=config.password_reset_window_seconds,
    )
    if client_key:
        limiter.check(
            f"password-reset:client:{client_key}",
            limit=config.password_reset_limit,
            window_seconds=config.password_reset_window_seconds,
        )


#: Built once per process and reused. Argon2 verification cost is dominated by
#: the parameters embedded in the hash, so any real hash gives a comparable
#: baseline — that is all this is for.
_DUMMY_PASSWORD_HASH: str | None = None


def _dummy_password_check(password: str, config: AuthConfig) -> None:
    """Burn roughly one Argon2 verification, discarding the result.

    Without this, "no such account" would return noticeably faster than "wrong
    password", because the Argon2 verification is the expensive part. Measuring
    that difference turns the login form into an account-enumeration oracle.
    """
    global _DUMMY_PASSWORD_HASH
    if _DUMMY_PASSWORD_HASH is None:
        _DUMMY_PASSWORD_HASH = hash_password("auditpro-timing-equaliser", config=config)
    verify_password(password, _DUMMY_PASSWORD_HASH, config=config)


def login(
    session: Session,
    *,
    email: str,
    password: str,
    config: AuthConfig,
    client_key: str = "",
    user_agent: str | None = None,
) -> LoginResult:
    """Verify credentials and open a server-side session.

    Every failure — unknown address, wrong password, inactive or unverified
    account — raises the *same* ``invalid_credentials`` error with the *same*
    wording, so the endpoint cannot be used to discover which addresses have
    accounts.

    Returns the raw session token once; the caller writes it to the cookie.
    """
    # Normalise only — do NOT enforce the signup allowlist here.
    # The allowlist gates account *creation* (start/resend/verify/complete
    # signup); it is not an authentication check. Applying it at login locked
    # out every existing user whose address is not currently allowlisted
    # (e.g. after the domain policy or DEV_ADMIN_EMAIL changes), returning 403
    # email_not_allowed instead of verifying their password.
    # `normalize_email` still rejects empty/malformed input (400 email_invalid)
    # and still canonicalises case/whitespace so the lookup, the rate-limit key
    # and the stored address agree.
    normalized = normalize_email(email)
    _rate_limit_login(normalized, client_key, config)

    user = _existing_user(session, normalized)

    if user is None or not user.is_active or not user.email_verified:
        # Equalise the response time before rejecting (see _dummy_password_check).
        _dummy_password_check(password, config)
        raise AuthError(
            "the email or password is incorrect", code="invalid_credentials"
        )

    if not verify_password(password, user.password_hash, config=config):
        raise AuthError("the email or password is incorrect", code="invalid_credentials")

    token, _row = sessions_mod.create_session(
        session, user_id=user.id, config=config, user_agent=user_agent
    )
    return LoginResult(
        user_id=user.id,
        email=user.email,
        session_token=token,
        expires_in_seconds=config.session_ttl_seconds,
    )


def logout(
    session: Session,
    *,
    session_token: str | None,
    config: AuthConfig,
) -> LogoutResult:
    """Revoke the session identified by the cookie token.

    Idempotent: logging out twice, or with a stale cookie, reports
    ``revoked=False`` rather than failing.
    """
    revoked = sessions_mod.revoke_session(session, token=session_token, config=config)
    return LogoutResult(revoked=revoked)


def current_user(
    session: Session,
    *,
    session_token: str | None,
    config: AuthConfig,
) -> User | None:
    """The account behind a live session cookie, or ``None``.

    Returns ``None`` for: no cookie, unknown token, revoked, expired,
    idle-timed-out, and an account that no longer exists or is inactive. Callers
    cannot distinguish these, which is the point.
    """
    row = sessions_mod.resolve_session(session, token=session_token, config=config)
    if row is None:
        return None
    user = session.get(User, row.user_id)
    if user is None or not user.is_active:
        return None
    return user


def start_password_reset(
    session: Session,
    *,
    email: str,
    config: AuthConfig,
    mailer: Mailer,
    client_key: str = "",
) -> PasswordResetStartResult | None:
    """Email a reset code, or return ``None`` if there is no such account.

    Returning ``None`` is deliberate: the caller replies with the *same*
    "if that address exists, a code is on its way" message either way, so the
    endpoint cannot be used to test who has an account. It also means no code
    is generated or mailed for an unknown address.
    """
    normalized = validate_signup_email(email, dev_admin_email=config.dev_admin_email)
    _rate_limit_password_reset(normalized, client_key, config)

    if _existing_user(session, normalized) is None:
        return None

    issued = _issue(
        session,
        email=normalized,
        config=config,
        mailer=mailer,
        resend_count=0,
        purpose=OTP_PURPOSE_PASSWORD_RESET,
    )
    return PasswordResetStartResult(
        email=issued.email,
        expires_in_seconds=issued.expires_in_seconds,
        resend_available_in_seconds=issued.resend_available_in_seconds,
        dev_otp=issued.dev_otp,
    )


def complete_password_reset(
    session: Session,
    *,
    email: str,
    code: str,
    new_password: str,
    config: AuthConfig,
    client_key: str = "",
) -> PasswordResetCompleteResult:
    """Verify the reset code, set a new password, and sign every device out.

    Revoking all sessions matters: whoever reset the password may have done so
    precisely because the old one leaked.
    """
    normalized = validate_signup_email(email, dev_admin_email=config.dev_admin_email)
    _rate_limit_password_reset(normalized, client_key, config)
    _verify_otp(
        session,
        email=normalized,
        code=code,
        purpose=OTP_PURPOSE_PASSWORD_RESET,
        config=config,
    )

    user = _existing_user(session, normalized)
    if user is None:
        raise OtpError("the code is invalid or has expired", code="otp_invalid")

    # Server-side policy check — the client-side one is only a convenience.
    validate_password(new_password, config=config, email_normalized=normalized)

    user.password_hash = hash_password(new_password, config=config)
    revoked = sessions_mod.revoke_all_sessions_for_user(session, user_id=user.id)
    session.commit()

    return PasswordResetCompleteResult(email=normalized, sessions_revoked=revoked)
