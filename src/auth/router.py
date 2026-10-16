"""Auth HTTP surface (Phase 2 signup + Phase 3 login/logout/password reset).

Mounted by ``src/retrieval/frontend/server.py`` under ``/api/auth``. It adds
routes only — it does **not** register middleware and does **not** protect any
existing route. Phase 3 issues a session cookie, but nothing consumes it yet:
every pre-existing endpoint stays unauthenticated until Phase 4.

Nothing here logs an OTP, a password, a session token or an SMTP credential.
Request bodies are never echoed.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.auth.config import AuthConfig, load_auth_config
from src.auth.errors import AuthError, RateLimited
from src.auth.mailer import get_mailer
from src.auth.passwords import password_policy
from src.auth.service import (
    complete_password_reset,
    complete_signup,
    current_user,
    login,
    logout,
    resend_signup_otp,
    start_password_reset,
    start_signup,
    verify_signup_otp,
)
from src.db.session import get_db

router = APIRouter(prefix="/api/auth", tags=["auth"])


class SignupStartRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)


class SignupVerifyRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    otp: str = Field(min_length=4, max_length=16)


class SignupCompleteRequest(BaseModel):
    signup_token: str = Field(min_length=8, max_length=2048)
    password: str = Field(min_length=1, max_length=1024)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=1024)


class PasswordForgotRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)


class PasswordResetRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    otp: str = Field(min_length=4, max_length=16)
    new_password: str = Field(min_length=1, max_length=1024)


def _config():
    """Load config per request. Raises AuthError -> 500 (loud by design)."""
    try:
        return load_auth_config()
    except AuthError as exc:
        raise HTTPException(status_code=500, detail=_detail(exc)) from exc


def _detail(exc: AuthError) -> dict:
    return {"code": exc.code, "message": exc.message}


def _fail(exc: AuthError) -> HTTPException:
    headers = None
    if isinstance(exc, RateLimited):
        headers = {"Retry-After": str(exc.retry_after)}
    return HTTPException(status_code=exc.http_status, detail=_detail(exc), headers=headers)


def _client_key(request: Request) -> str:
    """Best-effort client identifier for rate limiting."""
    return request.client.host if request.client else ""


def _set_session_cookie(response: Response, token: str, config: AuthConfig) -> None:
    """Issue the session cookie.

    HttpOnly stops JavaScript from reading it (so an XSS bug cannot steal the
    session); SameSite=Lax stops it riding on cross-site POSTs; Secure is on in
    production and off for local HTTP so a developer does not need TLS.
    """
    response.set_cookie(
        key=config.session_cookie_name,
        value=token,
        max_age=config.session_ttl_seconds,
        httponly=True,
        secure=config.cookie_secure,
        samesite=config.session_cookie_samesite,
        path="/",
    )


def _clear_session_cookie(response: Response, config: AuthConfig) -> None:
    """Expire the cookie. Called on every logout, and on a failed /me lookup."""
    response.delete_cookie(
        key=config.session_cookie_name,
        path="/",
        httponly=True,
        secure=config.cookie_secure,
        samesite=config.session_cookie_samesite,
    )


def _read_session_token(request: Request, config: AuthConfig) -> str | None:
    return request.cookies.get(config.session_cookie_name)


@router.post("/signup/start")
def signup_start(payload: SignupStartRequest, request: Request, db: Session = Depends(get_db)):
    """Validate the address and email a verification code."""
    config = _config()
    try:
        result = start_signup(
            db,
            email=payload.email,
            config=config,
            mailer=get_mailer(config),
            client_key=_client_key(request),
        )
    except AuthError as exc:
        raise _fail(exc) from exc
    return {
        "status": "code_sent",
        "email": result.email,
        "expires_in_seconds": result.expires_in_seconds,
        "resend_available_in_seconds": result.resend_available_in_seconds,
        **({"dev_otp": result.dev_otp} if result.dev_otp else {}),
    }


@router.post("/signup/resend")
def signup_resend(payload: SignupStartRequest, request: Request, db: Session = Depends(get_db)):
    """Send a fresh code, subject to the resend cooldown."""
    config = _config()
    try:
        result = resend_signup_otp(
            db,
            email=payload.email,
            config=config,
            mailer=get_mailer(config),
            client_key=_client_key(request),
        )
    except AuthError as exc:
        raise _fail(exc) from exc
    return {
        "status": "code_sent",
        "email": result.email,
        "expires_in_seconds": result.expires_in_seconds,
        "resend_available_in_seconds": result.resend_available_in_seconds,
        **({"dev_otp": result.dev_otp} if result.dev_otp else {}),
    }


@router.post("/signup/verify")
def signup_verify(payload: SignupVerifyRequest, request: Request, db: Session = Depends(get_db)):
    """Verify the code; returns a short-lived token to set the password."""
    config = _config()
    try:
        result = verify_signup_otp(
            db,
            email=payload.email,
            code=payload.otp,
            config=config,
            client_key=_client_key(request),
        )
    except AuthError as exc:
        raise _fail(exc) from exc
    return {
        "status": "verified",
        "email": result.email,
        "signup_token": result.signup_token,
        "expires_in_seconds": result.expires_in_seconds,
    }


@router.post("/signup/complete")
def signup_complete(payload: SignupCompleteRequest, db: Session = Depends(get_db)):
    """Set the password and create the verified account."""
    config = _config()
    try:
        result = complete_signup(
            db,
            signup_token=payload.signup_token,
            password=payload.password,
            config=config,
        )
    except AuthError as exc:
        raise _fail(exc) from exc
    return {
        "status": "created",
        "user_id": result.user_id,
        "email": result.email,
        "email_verified": result.email_verified,
    }


@router.get("/password-policy")
def auth_password_policy():
    """The authoritative password rules, for the UI to render."""
    return password_policy(_config())


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — login, logout, session lookup and password reset
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/login")
def auth_login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """Verify credentials and start a session. Sets the session cookie."""
    config = _config()
    try:
        result = login(
            db,
            email=payload.email,
            password=payload.password,
            config=config,
            client_key=_client_key(request),
            user_agent=request.headers.get("user-agent"),
        )
    except AuthError as exc:
        raise _fail(exc) from exc

    _set_session_cookie(response, result.session_token, config)
    return {
        "status": "signed_in",
        "user_id": result.user_id,
        "email": result.email,
        "expires_in_seconds": result.expires_in_seconds,
    }


@router.post("/logout")
def auth_logout(request: Request, response: Response, db: Session = Depends(get_db)):
    """Revoke the session and expire the cookie. Always succeeds."""
    config = _config()
    result = logout(
        db, session_token=_read_session_token(request, config), config=config
    )
    # Clear the cookie whether or not a session was found, so a stale cookie
    # never survives the user's intent to sign out.
    _clear_session_cookie(response, config)
    return {"status": "signed_out", "revoked": result.revoked}


@router.get("/me")
def auth_me(request: Request, response: Response, db: Session = Depends(get_db)):
    """The signed-in account, or 401 with code ``not_authenticated``.

    This reports who you are; it does not gate anything. No existing route is
    protected until Phase 4.
    """
    config = _config()
    user = current_user(
        db, session_token=_read_session_token(request, config), config=config
    )
    if user is None:
        _clear_session_cookie(response, config)
        raise HTTPException(
            status_code=401,
            detail=_detail(AuthError("you are not signed in", code="not_authenticated")),
        )
    return {
        "authenticated": True,
        "user": {
            "id": user.id,
            "email": user.email,
            "email_verified": user.email_verified,
            "is_active": user.is_active,
        },
    }


@router.post("/password/forgot")
def auth_password_forgot(
    payload: PasswordForgotRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Email a reset code.

    The response is identical whether or not the address has an account, so
    this cannot be used to discover who is registered. When there is no account
    no code is generated and no mail is sent.
    """
    config = _config()
    try:
        result = start_password_reset(
            db,
            email=payload.email,
            config=config,
            mailer=get_mailer(config),
            client_key=_client_key(request),
        )
    except AuthError as exc:
        raise _fail(exc) from exc

    # Same shape either way — the caller cannot tell the two apart.
    return {
        "status": "code_sent",
        "email": result.email if result else payload.email.strip().lower(),
        "expires_in_seconds": (
            result.expires_in_seconds if result else config.otp_ttl_seconds
        ),
        "resend_available_in_seconds": (
            result.resend_available_in_seconds
            if result
            else config.otp_resend_cooldown_seconds
        ),
        **({"dev_otp": result.dev_otp} if result and result.dev_otp else {}),
    }


@router.post("/password/reset")
def auth_password_reset(
    payload: PasswordResetRequest, db: Session = Depends(get_db)
):
    """Verify the reset code, set a new password, and sign out every device."""
    config = _config()
    try:
        result = complete_password_reset(
            db,
            email=payload.email,
            code=payload.otp,
            new_password=payload.new_password,
            config=config,
        )
    except AuthError as exc:
        raise _fail(exc) from exc
    return {
        "status": "password_updated",
        "email": result.email,
        "sessions_revoked": result.sessions_revoked,
    }
