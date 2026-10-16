"""Phase 3 — login, server-side sessions, logout and password reset.

Runs against SQLite in-memory with mail captured. Deliberately does **not**
test that routes are protected — nothing is protected until Phase 4, and
`test_auth_phase_boundary` below asserts that outright so nobody reads these
tests as a security guarantee.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import select

from src.auth import models as am
from src.auth import sessions as sess_mod
from src.auth.errors import AuthError, OtpError
from src.auth.mailer import CapturingMailer
from src.auth.models import OTP_PURPOSE_PASSWORD_RESET, OTP_PURPOSE_SIGNUP
from src.auth.passwords import verify_password
from src.auth.service import (
    complete_password_reset,
    current_user,
    login,
    logout,
    start_password_reset,
)

from _auth_helpers import (  # noqa: F401  (fixtures must be imported)
    _reset_auth_state,
    code_from_mail,
    code_from_message,
    good_password,
    make_config,
    register_user,
    sqlite_session,
)

ALLOWED = "boss@incois.gov.in"
NEW_PASSWORD = "Tide-Quiet-77-K9"


def _signed_in(sqlite_session, config, mailer, email=ALLOWED, password=None):
    """Register an account and return (user_id, session_token)."""
    register_user(sqlite_session, config, mailer, email=email, password=password)
    result = login(
        sqlite_session,
        email=email,
        password=password or good_password(),
        config=config,
    )
    return result.user_id, result.session_token


# ═══════════════════════════════════════════════════════════════════════════
# login
# ═══════════════════════════════════════════════════════════════════════════


def test_login_with_correct_credentials_opens_a_session(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    register_user(sqlite_session, config, mailer)

    result = login(
        sqlite_session, email=ALLOWED, password=good_password(), config=config
    )

    assert result.email == ALLOWED
    assert result.expires_in_seconds == config.session_ttl_seconds
    assert len(result.session_token) >= 40  # opaque, high entropy


def test_login_wrong_password_is_rejected(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    register_user(sqlite_session, config, mailer)

    with pytest.raises(AuthError) as exc:
        login(sqlite_session, email=ALLOWED, password="Wrong-Pass-1a", config=config)

    assert exc.value.code == "invalid_credentials"
    assert exc.value.http_status == 401


def test_an_unknown_address_is_rejected_identically(sqlite_session):
    """Unknown address and wrong password must be indistinguishable."""
    config = make_config()
    mailer = CapturingMailer()
    register_user(sqlite_session, config, mailer)

    with pytest.raises(AuthError) as known:
        login(sqlite_session, email=ALLOWED, password="Wrong-Pass-1a", config=config)
    with pytest.raises(AuthError) as unknown:
        login(
            sqlite_session,
            email="nobody@incois.gov.in",
            password=good_password(),
            config=config,
        )

    assert known.value.code == unknown.value.code == "invalid_credentials"
    assert known.value.message == unknown.value.message
    assert known.value.http_status == unknown.value.http_status == 401


def test_login_does_not_apply_the_signup_allowlist(sqlite_session):
    """Login must not answer with the *signup* allowlist error.

    The allowlist is a policy about who may CREATE an account
    (start/resend/verify/complete signup). It is not an authentication check:
    applying it here meant any existing user whose address is not currently
    allowlisted got 403 ``email_not_allowed`` instead of having their password
    verified, which is both a lock-out and a needless disclosure of signup
    policy.

    An unknown address — allowlisted or not — gets the same generic
    ``invalid_credentials`` (401) as a wrong password, so login cannot be used
    to enumerate accounts or to probe the allowlist.
    """
    config = make_config()
    with pytest.raises(AuthError) as exc:
        login(sqlite_session, email="someone@gmail.com", password="x", config=config)
    assert exc.value.code == "invalid_credentials"
    assert exc.value.http_status == 401


def test_login_is_indistinguishable_for_allowlisted_and_unallowlisted_unknowns(
    sqlite_session,
):
    """Both an on-list and an off-list unknown address give the same error."""
    config = make_config()
    codes = set()
    for address in ("ghost@incois.gov.in", "someone@gmail.com"):
        with pytest.raises(AuthError) as exc:
            login(sqlite_session, email=address, password="x", config=config)
        codes.add(exc.value.code)
    assert codes == {"invalid_credentials"}


def test_login_is_refused_for_a_deactivated_account(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    register_user(sqlite_session, config, mailer)

    user = sqlite_session.scalars(
        select(am.User).where(am.User.email_normalized == ALLOWED)
    ).one()
    user.is_active = False
    sqlite_session.commit()

    with pytest.raises(AuthError) as exc:
        login(sqlite_session, email=ALLOWED, password=good_password(), config=config)
    assert exc.value.code == "invalid_credentials"


def test_login_rate_limit_engages(sqlite_session):
    config = make_config(AUTH_LOGIN_LIMIT=2)
    mailer = CapturingMailer()
    register_user(sqlite_session, config, mailer)

    for _ in range(2):
        with pytest.raises(AuthError) as exc:
            login(sqlite_session, email=ALLOWED, password="nope", config=config)
        assert exc.value.code == "invalid_credentials"

    # Third attempt is refused as a rate limit, not as bad credentials.
    with pytest.raises(AuthError) as exc:
        login(sqlite_session, email=ALLOWED, password="nope", config=config)
    assert exc.value.code == "rate_limited"


# ═══════════════════════════════════════════════════════════════════════════
# how the session is stored
# ═══════════════════════════════════════════════════════════════════════════


def test_only_the_token_digest_is_stored(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    _user_id, token = _signed_in(sqlite_session, config, mailer)

    row = sqlite_session.scalars(select(am.Session)).one()
    assert token not in row.token_hash
    assert row.token_hash == sess_mod.hash_session_token(
        token, key=config.session_key()
    )
    # Nothing anywhere in the row contains the raw token.
    assert not any(token in str(getattr(row, c.name)) for c in row.__table__.columns)


def test_each_login_creates_its_own_session(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    register_user(sqlite_session, config, mailer)

    a = login(sqlite_session, email=ALLOWED, password=good_password(), config=config)
    b = login(sqlite_session, email=ALLOWED, password=good_password(), config=config)

    assert a.session_token != b.session_token
    assert len(sqlite_session.scalars(select(am.Session)).all()) == 2


def test_a_valid_token_resolves_to_its_user(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    user_id, token = _signed_in(sqlite_session, config, mailer)

    user = current_user(sqlite_session, session_token=token, config=config)
    assert user is not None
    assert user.id == user_id
    assert user.email == ALLOWED


def test_an_unknown_token_resolves_to_nothing(sqlite_session):
    config = make_config()
    assert current_user(sqlite_session, session_token="garbage", config=config) is None
    assert current_user(sqlite_session, session_token=None, config=config) is None


# ═══════════════════════════════════════════════════════════════════════════
# expiry and idle timeout
# ═══════════════════════════════════════════════════════════════════════════


def test_a_session_past_its_absolute_expiry_is_dead(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    _user_id, token = _signed_in(sqlite_session, config, mailer)

    row = sqlite_session.scalars(select(am.Session)).one()
    row.expires_at = am.utcnow() - timedelta(seconds=1)
    sqlite_session.commit()

    assert current_user(sqlite_session, session_token=token, config=config) is None


def test_an_idle_session_is_dead(sqlite_session):
    config = make_config(AUTH_SESSION_IDLE_TIMEOUT_SECONDS=60)
    mailer = CapturingMailer()
    _user_id, token = _signed_in(sqlite_session, config, mailer)

    row = sqlite_session.scalars(select(am.Session)).one()
    row.last_seen_at = am.utcnow() - timedelta(seconds=120)
    sqlite_session.commit()

    assert current_user(sqlite_session, session_token=token, config=config) is None


def test_using_a_session_refreshes_last_seen(sqlite_session):
    config = make_config()  # default idle timeout is 2h
    mailer = CapturingMailer()
    _user_id, token = _signed_in(sqlite_session, config, mailer)

    row = sqlite_session.scalars(select(am.Session)).one()
    # Stale, but inside the idle window, so the session is still alive.
    row.last_seen_at = am.utcnow() - timedelta(minutes=30)
    sqlite_session.commit()

    assert current_user(sqlite_session, session_token=token, config=config) is not None
    refreshed = sqlite_session.scalars(select(am.Session)).one()
    assert (am.utcnow() - am.as_utc(refreshed.last_seen_at)).total_seconds() < 60


def test_expiry_is_enforced_without_sleeping(sqlite_session):
    """`resolve_session(now=...)` proves the deadline, not the clock."""
    config = make_config(AUTH_SESSION_TTL_SECONDS=600)
    mailer = CapturingMailer()
    _user_id, token = _signed_in(sqlite_session, config, mailer)

    later = am.utcnow() + timedelta(seconds=601)
    assert (
        sess_mod.resolve_session(
            sqlite_session, token=token, config=config, now=later
        )
        is None
    )


# ═══════════════════════════════════════════════════════════════════════════
# logout
# ═══════════════════════════════════════════════════════════════════════════


def test_logout_revokes_the_session(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    _user_id, token = _signed_in(sqlite_session, config, mailer)

    result = logout(sqlite_session, session_token=token, config=config)

    assert result.revoked is True
    assert current_user(sqlite_session, session_token=token, config=config) is None


def test_logout_is_idempotent(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    _user_id, token = _signed_in(sqlite_session, config, mailer)

    assert logout(sqlite_session, session_token=token, config=config).revoked is True
    assert logout(sqlite_session, session_token=token, config=config).revoked is False
    assert logout(sqlite_session, session_token=None, config=config).revoked is False


def test_logout_leaves_other_sessions_alone(sqlite_session):
    """Signing out on one device must not sign out the others."""
    config = make_config()
    mailer = CapturingMailer()
    register_user(sqlite_session, config, mailer)
    a = login(sqlite_session, email=ALLOWED, password=good_password(), config=config)
    b = login(sqlite_session, email=ALLOWED, password=good_password(), config=config)

    logout(sqlite_session, session_token=a.session_token, config=config)

    assert current_user(sqlite_session, session_token=a.session_token, config=config) is None
    assert (
        current_user(sqlite_session, session_token=b.session_token, config=config)
        is not None
    )


# ═══════════════════════════════════════════════════════════════════════════
# password reset
# ═══════════════════════════════════════════════════════════════════════════


def test_a_reset_code_is_emailed_to_a_known_account(sqlite_session):
    config = make_config(AUTH_OTP_RESEND_COOLDOWN_SECONDS=0)
    mailer = CapturingMailer()
    register_user(sqlite_session, config, mailer)

    result = start_password_reset(
        sqlite_session, email=ALLOWED, config=config, mailer=mailer
    )

    assert result is not None
    assert result.email == ALLOWED
    reset_mail = mailer.last
    assert "reset your password" in reset_mail.subject.lower()
    assert code_from_message(reset_mail)


def test_an_unknown_address_gets_no_code_and_no_error(sqlite_session):
    """No account -> None, so the endpoint cannot reveal who is registered."""
    config = make_config()
    mailer = CapturingMailer()

    assert (
        start_password_reset(
            sqlite_session, email="nobody@incois.gov.in", config=config, mailer=mailer
        )
        is None
    )
    # No account was registered in this test, so nothing was sent or stored.
    assert mailer.sent == []
    assert sqlite_session.scalars(select(am.EmailOtp)).all() == []


def test_a_reset_code_uses_the_reset_purpose(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    register_user(sqlite_session, config, mailer)

    start_password_reset(sqlite_session, email=ALLOWED, config=config, mailer=mailer)

    # register_user() leaves its consumed signup row behind, so select by purpose.
    rows = sqlite_session.scalars(
        select(am.EmailOtp).where(am.EmailOtp.purpose == OTP_PURPOSE_PASSWORD_RESET)
    ).all()
    assert len(rows) == 1
    assert rows[0].email_normalized == ALLOWED


def test_a_signup_code_cannot_reset_a_password(sqlite_session):
    """Purpose scoping: the two flows must not accept each other's codes."""
    config = make_config()
    mailer = CapturingMailer()
    # A live *signup* code for an address that has no account yet.
    from src.auth.service import start_signup

    start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    signup_code = code_from_mail(mailer)
    register_user(sqlite_session, config, CapturingMailer())

    with pytest.raises(OtpError):
        complete_password_reset(
            sqlite_session,
            email=ALLOWED,
            code=signup_code,
            new_password=NEW_PASSWORD,
            config=config,
        )


def test_reset_changes_the_password(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    register_user(sqlite_session, config, mailer)
    start_password_reset(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    code = code_from_message(mailer.last)

    result = complete_password_reset(
        sqlite_session,
        email=ALLOWED,
        code=code,
        new_password=NEW_PASSWORD,
        config=config,
    )
    assert result.email == ALLOWED

    user = sqlite_session.scalars(
        select(am.User).where(am.User.email_normalized == ALLOWED)
    ).one()
    assert verify_password(NEW_PASSWORD, user.password_hash, config=config) is True
    assert verify_password(good_password(), user.password_hash, config=config) is False

    # And signing in with the new password works.
    assert (
        login(sqlite_session, email=ALLOWED, password=NEW_PASSWORD, config=config).email
        == ALLOWED
    )


def test_a_reset_code_is_single_use(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    register_user(sqlite_session, config, mailer)
    start_password_reset(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    code = code_from_message(mailer.last)

    complete_password_reset(
        sqlite_session,
        email=ALLOWED,
        code=code,
        new_password=NEW_PASSWORD,
        config=config,
    )
    with pytest.raises(OtpError):
        complete_password_reset(
            sqlite_session,
            email=ALLOWED,
            code=code,
            new_password="Another-Pass-8c",
            config=config,
        )


def test_reset_enforces_the_password_policy(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    register_user(sqlite_session, config, mailer)
    start_password_reset(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    code = code_from_message(mailer.last)

    with pytest.raises(AuthError) as exc:
        complete_password_reset(
            sqlite_session,
            email=ALLOWED,
            code=code,
            new_password="weak",
            config=config,
        )
    assert exc.value.code == "password_invalid"


def test_reset_revokes_every_session(sqlite_session):
    """The strongest reason to reset is a leaked password — kill all devices."""
    config = make_config()
    mailer = CapturingMailer()
    register_user(sqlite_session, config, mailer)
    first = login(sqlite_session, email=ALLOWED, password=good_password(), config=config)
    second = login(sqlite_session, email=ALLOWED, password=good_password(), config=config)

    start_password_reset(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    code = code_from_message(mailer.last)
    result = complete_password_reset(
        sqlite_session,
        email=ALLOWED,
        code=code,
        new_password=NEW_PASSWORD,
        config=config,
    )

    assert result.sessions_revoked == 2
    assert current_user(sqlite_session, session_token=first.session_token, config=config) is None
    assert current_user(sqlite_session, session_token=second.session_token, config=config) is None


# ═══════════════════════════════════════════════════════════════════════════
# phase boundary — say plainly what is NOT true yet
# ═══════════════════════════════════════════════════════════════════════════


def test_auth_phase_boundary():
    """Phase 3 signs you IN. It does not lock anything DOWN.

    Existing application routes stay unauthenticated until Phase 4; this test
    exists so nobody mistakes a passing suite for a protected application.
    """
    import src.auth as auth_pkg

    # No role/permission surface anywhere in the auth package.
    role_names = [n for n in dir(auth_pkg) if "role" in n.lower()]
    assert role_names == [], f"role concepts are Phase 4: {role_names}"

    # The User model has no role column.
    assert "role" not in {c.name for c in am.User.__table__.columns}

    # No role/permission table in the ORM metadata.
    assert {"roles", "permissions", "user_roles"}.isdisjoint(
        set(am.User.metadata.tables)
    )

    # Password reset reuses the OTP table — there is no second code table.
    assert OTP_PURPOSE_PASSWORD_RESET != OTP_PURPOSE_SIGNUP
    assert not hasattr(am, "PasswordReset")
