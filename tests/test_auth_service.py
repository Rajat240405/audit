"""Phase 2 — signup service behaviour.

These tests exercise the real service functions against a real SQL database
(SQLite in-memory), with mail captured instead of sent. They prove the
*service logic* is correct.

They do NOT prove the PostgreSQL schema or the Alembic migration is correct —
see ``test_auth_database_integration.py`` for that, which requires a real
throwaway PostgreSQL database and is skipped by default.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from _auth_helpers import (  # noqa: F401  (the autouse fixtures must be imported)
    DEV_EMAIL,
    _reset_auth_state,
    code_from_mail,
    code_from_message,
    good_password,
    make_config,
    sqlite_session,
    temporary_session,
)
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from src.auth import models as am
from src.auth.errors import AuthError, DuplicateAccount, OtpError, RateLimited
from src.auth.mailer import CapturingMailer
from src.auth.passwords import verify_password
from src.auth.service import (
    complete_signup,
    resend_signup_otp,
    start_signup,
    verify_signup_otp,
)

ALLOWED = "boss@incois.gov.in"

# ── starting signup ──────────────────────────────────────────────────────────


def test_start_signup_emails_a_code_to_an_allowed_address(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()

    result = start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)

    assert result.email == ALLOWED
    assert len(mailer.sent) == 1
    assert mailer.last.to == ALLOWED
    assert code_from_mail(mailer)


def test_start_signup_normalises_the_address(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()

    result = start_signup(
        sqlite_session, email="  Boss.User@INCOIS.GOV.IN  ", config=config, mailer=mailer
    )

    assert result.email == "boss.user@incois.gov.in"


@pytest.mark.parametrize(
    "email", ["someone@gmail.com", "someone@example.com", "boss@incois.gov.in.evil.com"]
)
def test_start_signup_rejects_disallowed_addresses(sqlite_session, email):
    config = make_config()
    mailer = CapturingMailer()

    with pytest.raises(AuthError) as exc:
        start_signup(sqlite_session, email=email, config=config, mailer=mailer)

    assert exc.value.code == "email_not_allowed"
    assert mailer.sent == []
    assert sqlite_session.scalars(select(am.EmailOtp)).all() == []


def test_start_signup_accepts_exactly_the_dev_email(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()

    result = start_signup(sqlite_session, email=DEV_EMAIL, config=config, mailer=mailer)

    assert result.email == DEV_EMAIL
    assert len(mailer.sent) == 1


def test_the_plaintext_code_is_never_persisted(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    code = code_from_mail(mailer)

    row = sqlite_session.scalars(select(am.EmailOtp)).one()
    stored = {str(getattr(row, c.name)) for c in row.__table__.columns}
    assert code not in row.code_hash
    assert not any(code in value for value in stored), "plaintext code found in the row"


def test_start_signup_for_an_existing_account_is_a_duplicate(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    token = verify_signup_otp(
        sqlite_session,
        email=ALLOWED,
        code=code_from_mail(mailer),
        config=config,
    ).signup_token
    complete_signup(
        sqlite_session, signup_token=token, password=good_password(), config=config
    )

    with pytest.raises(DuplicateAccount) as exc:
        start_signup(sqlite_session, email="BOSS@incois.gov.in", config=config, mailer=mailer)

    assert exc.value.code == "account_exists"
    assert exc.value.http_status == 409


# ── verifying the code ───────────────────────────────────────────────────────


def test_a_correct_code_is_accepted(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)

    result = verify_signup_otp(
        sqlite_session, email=ALLOWED, code=code_from_mail(mailer), config=config
    )

    assert result.email == ALLOWED
    assert result.signup_token


def test_a_wrong_code_is_rejected(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    wrong = "000000" if code_from_mail(mailer) != "000000" else "111111"

    with pytest.raises(OtpError) as exc:
        verify_signup_otp(
            sqlite_session, email=ALLOWED, code=wrong, config=config
        )

    assert exc.value.code == "otp_invalid"


def test_failed_attempts_are_counted_and_reported(sqlite_session):
    config = make_config(AUTH_OTP_MAX_ATTEMPTS=5)
    mailer = CapturingMailer()
    start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)

    for remaining in (4, 3, 2):
        with pytest.raises(OtpError) as exc:
            verify_signup_otp(sqlite_session, email=ALLOWED, code="000000", config=config)
        assert f"{remaining} attempt(s) remaining" in exc.value.message

    row = sqlite_session.scalars(select(am.EmailOtp)).one()
    assert row.attempts == 3


def test_exhausting_the_attempts_locks_the_code(sqlite_session):
    config = make_config(AUTH_OTP_MAX_ATTEMPTS=2)
    mailer = CapturingMailer()
    start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    code = code_from_mail(mailer)

    with pytest.raises(OtpError):
        verify_signup_otp(sqlite_session, email=ALLOWED, code="000000", config=config)
    with pytest.raises(OtpError) as exc:
        verify_signup_otp(sqlite_session, email=ALLOWED, code="000000", config=config)
    assert exc.value.code == "otp_attempts_exceeded"

    # Even the *correct* code is now refused: the challenge is spent.
    with pytest.raises(OtpError):
        verify_signup_otp(sqlite_session, email=ALLOWED, code=code, config=config)


def test_an_expired_code_is_rejected(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    code = code_from_mail(mailer)

    row = sqlite_session.scalars(select(am.EmailOtp)).one()
    row.expires_at = am.utcnow() - timedelta(seconds=1)
    sqlite_session.commit()

    with pytest.raises(OtpError) as exc:
        verify_signup_otp(sqlite_session, email=ALLOWED, code=code, config=config)
    assert exc.value.code == "otp_invalid"


def test_a_code_is_single_use(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    code = code_from_mail(mailer)

    verify_signup_otp(sqlite_session, email=ALLOWED, code=code, config=config)

    with pytest.raises(OtpError) as exc:
        verify_signup_otp(sqlite_session, email=ALLOWED, code=code, config=config)
    assert exc.value.code == "otp_invalid"


def test_a_code_is_scoped_to_its_own_address(sqlite_session):
    """A code issued to one address must not verify a different address."""
    config = make_config()
    mailer = CapturingMailer()
    start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    start_signup(sqlite_session, email="other@incois.gov.in", config=config, mailer=mailer)

    codes = {m.to: code_from_message(m) for m in mailer.sent}
    assert codes[ALLOWED] != codes["other@incois.gov.in"]
    assert len(sqlite_session.scalars(select(am.EmailOtp)).all()) == 2

    with pytest.raises(OtpError):
        verify_signup_otp(
            sqlite_session,
            email=ALLOWED,
            code=codes["other@incois.gov.in"],
            config=config,
        )

    # The code actually issued to this address still works.
    assert verify_signup_otp(
        sqlite_session, email=ALLOWED, code=codes[ALLOWED], config=config
    ).signup_token


# ── resending ────────────────────────────────────────────────────────────────


def test_resending_within_the_cooldown_is_rate_limited(sqlite_session):
    config = make_config(AUTH_OTP_RESEND_COOLDOWN_SECONDS=600)
    mailer = CapturingMailer()
    start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)

    with pytest.raises(RateLimited) as exc:
        resend_signup_otp(sqlite_session, email=ALLOWED, config=config, mailer=mailer)

    assert exc.value.code == "otp_resend_cooldown"
    assert exc.value.http_status == 429
    assert exc.value.retry_after > 0
    assert len(mailer.sent) == 1


def test_resending_after_the_cooldown_supersedes_the_old_code(sqlite_session):
    config = make_config(AUTH_OTP_RESEND_COOLDOWN_SECONDS=0)
    mailer = CapturingMailer()
    start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    first = code_from_mail(mailer)

    resend_signup_otp(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    second = code_from_mail(mailer)

    assert first != second
    # The old code no longer works.
    with pytest.raises(OtpError):
        verify_signup_otp(sqlite_session, email=ALLOWED, code=first, config=config)
    # The new one does.
    assert verify_signup_otp(
        sqlite_session, email=ALLOWED, code=second, config=config
    ).signup_token


def test_the_resend_counter_is_carried_forward(sqlite_session):
    config = make_config(AUTH_OTP_RESEND_COOLDOWN_SECONDS=0)
    mailer = CapturingMailer()
    start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    assert sqlite_session.scalars(select(am.EmailOtp)).one().resend_count == 0

    resend_signup_otp(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    resend_signup_otp(sqlite_session, email=ALLOWED, config=config, mailer=mailer)

    latest = (
        sqlite_session.scalars(
            select(am.EmailOtp).order_by(am.EmailOtp.created_at.desc())
        )
        .first()
    )
    assert latest.resend_count == 2


def test_superseded_codes_are_expired_not_deleted(sqlite_session):
    """Only one code is ever valid, but the audit trail survives."""
    config = make_config(AUTH_OTP_RESEND_COOLDOWN_SECONDS=0)
    mailer = CapturingMailer()
    start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    resend_signup_otp(sqlite_session, email=ALLOWED, config=config, mailer=mailer)

    assert len(sqlite_session.scalars(select(am.EmailOtp)).all()) == 2
    live = sqlite_session.scalars(
        select(am.EmailOtp).where(am.EmailOtp.consumed_at.is_(None))
    ).all()
    still_valid = [
        r for r in live if am.as_utc(r.expires_at) > am.utcnow()
    ]
    assert len(still_valid) == 1


def test_a_mailer_failure_rolls_the_challenge_back(sqlite_session):
    """If the code could not be delivered, nothing must be stored."""
    config = make_config()
    mailer = CapturingMailer()
    mailer.fail_next = True

    with pytest.raises(AuthError) as exc:
        start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)

    assert exc.value.code == "mailer_failed"
    assert exc.value.http_status == 502
    assert sqlite_session.scalars(select(am.EmailOtp)).all() == []


# ── completing signup ────────────────────────────────────────────────────────


def _verified_token(session, config, mailer, email=ALLOWED) -> str:
    start_signup(session, email=email, config=config, mailer=mailer)
    return verify_signup_otp(
        session, email=email, code=code_from_mail(mailer), config=config
    ).signup_token


def test_completing_signup_creates_a_verified_account(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    token = _verified_token(sqlite_session, config, mailer)

    result = complete_signup(
        sqlite_session, signup_token=token, password=good_password(), config=config
    )

    user = sqlite_session.scalars(select(am.User)).one()
    assert result.user_id == user.id
    assert user.email == ALLOWED
    assert user.email_normalized == ALLOWED
    assert user.email_verified is True
    assert user.is_active is True


def test_the_password_is_stored_as_argon2id_and_verifies(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    token = _verified_token(sqlite_session, config, mailer)
    password = good_password()

    complete_signup(
        sqlite_session, signup_token=token, password=password, config=config
    )

    user = sqlite_session.scalars(select(am.User)).one()
    assert user.password_hash.startswith("$argon2id$")
    assert password not in user.password_hash
    assert verify_password(password, user.password_hash, config=config) is True
    assert verify_password("wrong-password", user.password_hash, config=config) is False


@pytest.mark.parametrize(
    "password",
    [
        "short1A",  # too short
        "alllowercase123",  # no uppercase
        "ALLUPPERCASE123",  # no lowercase
        "NoDigitsInHereAtAll",  # no digit
    ],
)
def test_completing_signup_rejects_weak_passwords(sqlite_session, password):
    config = make_config()
    mailer = CapturingMailer()
    token = _verified_token(sqlite_session, config, mailer)

    with pytest.raises(AuthError) as exc:
        complete_signup(
            sqlite_session, signup_token=token, password=password, config=config
        )

    assert exc.value.code == "password_invalid"
    assert sqlite_session.scalars(select(am.User)).all() == []


def test_completing_signup_rejects_a_password_containing_the_email(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    token = _verified_token(sqlite_session, config, mailer)

    with pytest.raises(AuthError) as exc:
        complete_signup(
            sqlite_session,
            signup_token=token,
            password="Boss.User-Pass123",
            config=config,
        )

    assert exc.value.code == "password_invalid"
    assert "email address" in exc.value.message


def test_completing_signup_rejects_a_tampered_token(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    token = _verified_token(sqlite_session, config, mailer)
    payload, signature = token.split(".")
    tampered = payload[:-2] + ("aa" if not payload.endswith("aa") else "bb") + "." + signature

    with pytest.raises(AuthError) as exc:
        complete_signup(
            sqlite_session, signup_token=tampered, password=good_password(), config=config
        )

    assert exc.value.code in {"token_invalid", "token_expired"}


def test_completing_signup_rejects_a_token_signed_with_another_secret(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    token = _verified_token(sqlite_session, config, mailer)
    other = make_config(AUTH_SECRET="z" * 48)

    with pytest.raises(AuthError) as exc:
        complete_signup(
            sqlite_session, signup_token=token, password=good_password(), config=other
        )

    assert exc.value.code == "token_invalid"


def test_completing_signup_rejects_an_expired_token(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    start_signup(sqlite_session, email=ALLOWED, config=config, mailer=mailer)
    token = verify_signup_otp(
        sqlite_session,
        email=ALLOWED,
        code=code_from_mail(mailer),
        config=config,
        token_ttl_seconds=-1,
    ).signup_token

    with pytest.raises(AuthError) as exc:
        complete_signup(
            sqlite_session, signup_token=token, password=good_password(), config=config
        )

    assert exc.value.code == "token_expired"


def test_the_same_token_cannot_create_two_accounts(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()
    token = _verified_token(sqlite_session, config, mailer)

    complete_signup(sqlite_session, signup_token=token, password=good_password(), config=config)

    with pytest.raises(DuplicateAccount) as exc:
        complete_signup(
            sqlite_session, signup_token=token, password=good_password(), config=config
        )

    assert exc.value.code == "account_exists"
    assert len(sqlite_session.scalars(select(am.User)).all()) == 1


def test_email_normalized_is_unique_across_case_variants(sqlite_session):
    """The duplicate guard the database enforces, not just the service."""
    sqlite_session.add(
        am.User(
            id=am.new_id(),
            email="boss@incois.gov.in",
            email_normalized="boss@incois.gov.in",
            password_hash="x",
        )
    )
    sqlite_session.commit()

    sqlite_session.add(
        am.User(
            id=am.new_id(),
            email="BOSS@INCOIS.GOV.IN",
            email_normalized="boss@incois.gov.in",
            password_hash="y",
        )
    )
    with pytest.raises(IntegrityError):
        sqlite_session.commit()
    sqlite_session.rollback()


def test_the_expected_indexes_exist():
    with temporary_session() as session:
        bind = session.get_bind()
        users_idx = {i["name"] for i in inspect(bind).get_indexes("users")}
        otp_idx = {i["name"] for i in inspect(bind).get_indexes("email_otp")}

    assert "ix_users_email_normalized" in users_idx
    assert "ix_email_otp_email_purpose" in otp_idx


def test_full_signup_flow_end_to_end(sqlite_session):
    config = make_config()
    mailer = CapturingMailer()

    start = start_signup(sqlite_session, email="Boss.User@INCOIS.GOV.IN", config=config, mailer=mailer)
    assert start.email == "boss.user@incois.gov.in"
    assert start.expires_in_seconds == 600

    code = code_from_mail(mailer)
    verify = verify_signup_otp(
        sqlite_session, email="boss.user@incois.gov.in", code=code, config=config
    )
    assert verify.expires_in_seconds == 900

    done = complete_signup(
        sqlite_session, signup_token=verify.signup_token, password=good_password(), config=config
    )
    assert done.email_verified is True

    user = sqlite_session.scalars(select(am.User)).one()
    assert user.email_verified is True
    assert verify_password(good_password(), user.password_hash, config=config) is True
