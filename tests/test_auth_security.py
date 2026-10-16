"""Phase 2 — the security primitives: OTPs, password hashing, signup tokens,
configuration, rate limiting and mail delivery.

Everything here is pure and offline. Nothing connects to a database or an SMTP
server.
"""

from __future__ import annotations

import smtplib
from pathlib import Path

import pytest
from _auth_helpers import (  # noqa: F401  (autouse fixture must be imported)
    TEST_SECRET,
    _reset_auth_state,
    base_env,
    good_password,
    make_config,
)

from src.auth import otp as otp_mod
from src.auth import passwords as passwords_mod
from src.auth import tokens as tokens_mod
from src.auth.config import derive_key, load_auth_config
from src.auth.errors import AuthError, RateLimited
from src.auth.mailer import (
    CapturingMailer,
    ConsoleMailer,
    MailMessage,
    SmtpMailer,
    get_mailer,
)
from src.auth.otp import generate_code, hash_code, new_salt, verify_code
from src.auth.passwords import (
    hash_password,
    password_policy,
    validate_password,
    verify_password,
)
from src.auth.rate_limit import RateLimiter, get_limiter
from src.auth.tokens import issue_signup_token, read_signup_token

KEY = b"\x01" * 32


# ── OTP generation ───────────────────────────────────────────────────────────


def test_the_code_generator_uses_the_cryptographic_rng():
    """`random` would be predictable; `secrets` is the CSPRNG."""
    source = Path(otp_mod.__file__).read_text(encoding="utf-8")
    assert "secrets." in source
    assert "random." not in source, "OTP must not use the `random` module"


def test_generated_codes_have_the_requested_length_and_are_digits():
    for length in (4, 6, 8):
        code = generate_code(length)
        assert len(code) == length
        assert code.isdigit()


def test_generated_codes_vary():
    codes = {generate_code() for _ in range(200)}
    assert len(codes) > 50, "the generator looks predictable"


def test_salt_is_unique_and_hex_encoded():
    salts = {new_salt() for _ in range(50)}
    assert len(salts) == 50
    assert all(len(s) == 32 and set(s) <= set("0123456789abcdef") for s in salts)


# ── OTP hashing (keyed, so an offline guess is not possible without the key) ──


def test_hashing_is_deterministic_for_the_same_inputs():
    kwargs = dict(
        key=KEY,
        email_normalized="a@incois.gov.in",
        purpose="signup",
        salt="deadbeef",
        code="123456",
    )
    assert hash_code(**kwargs) == hash_code(**kwargs)


def test_the_same_code_hashes_differently_per_salt():
    common = dict(key=KEY, email_normalized="a@incois.gov.in", purpose="signup", code="123456")
    assert hash_code(salt="salt-one", **common) != hash_code(salt="salt-two", **common)


def test_the_hash_is_keyed_not_a_bare_digest():
    """A different key must produce a different verifier."""
    import hashlib

    common = dict(email_normalized="a@incois.gov.in", purpose="signup", salt="s", code="123456")
    keyed = hash_code(key=KEY, **common)
    other = hash_code(key=b"\x02" * 32, **common)
    plain = hashlib.sha256(b"123456").hexdigest()
    assert keyed != other
    assert keyed != plain


def test_the_code_cannot_be_recovered_from_the_hash():
    code = "424242"
    digest = hash_code(
        key=KEY, email_normalized="a@incois.gov.in", purpose="signup", salt="s", code=code
    )
    assert len(digest) == 64
    assert code not in digest


def test_verification_accepts_the_right_code():
    digest = hash_code(
        key=KEY, email_normalized="a@incois.gov.in", purpose="signup", salt="s", code="123456"
    )
    assert (
        verify_code(
            key=KEY,
            email_normalized="a@incois.gov.in",
            purpose="signup",
            salt="s",
            code="123456",
            expected_hash=digest,
        )
        is True
    )


@pytest.mark.parametrize(
    "override",
    [
        {"code": "999999"},
        {"email_normalized": "b@incois.gov.in"},
        {"purpose": "password_reset"},
        {"salt": "different"},
        {"key": b"\x09" * 32},
    ],
)
def test_verification_rejects_every_mismatch(override):
    kwargs = dict(
        key=KEY,
        email_normalized="a@incois.gov.in",
        purpose="signup",
        salt="s",
        code="123456",
    )
    digest = hash_code(**kwargs)
    kwargs.update(override)
    assert verify_code(expected_hash=digest, **kwargs) is False


@pytest.mark.parametrize("code", ["", "abcdef", "12345", "1234567", " 123456 "])
def test_verification_returns_false_not_an_exception_for_bad_input(code):
    digest = hash_code(
        key=KEY, email_normalized="a@incois.gov.in", purpose="signup", salt="s", code="123456"
    )
    assert (
        verify_code(
            key=KEY,
            email_normalized="a@incois.gov.in",
            purpose="signup",
            salt="s",
            code=code,
            expected_hash=digest,
        )
        is False
    )


def test_verification_survives_a_malformed_stored_hash():
    assert (
        verify_code(
            key=KEY,
            email_normalized="a@incois.gov.in",
            purpose="signup",
            salt="s",
            code="123456",
            expected_hash="not-hex-at-all",
        )
        is False
    )


# ── password hashing ─────────────────────────────────────────────────────────


def test_argon2id_is_the_algorithm_in_use():
    assert "argon2" in Path(passwords_mod.__file__).read_text(encoding="utf-8")
    hashed = hash_password(good_password(), config=make_config())
    assert hashed.startswith("$argon2id$")


def test_a_password_round_trips():
    config = make_config()
    hashed = hash_password(good_password(), config=config)
    assert verify_password(good_password(), hashed, config=config) is True
    assert verify_password("something-else", hashed, config=config) is False


def test_each_hash_has_its_own_salt():
    config = make_config()
    assert hash_password("same-password", config=config) != hash_password(
        "same-password", config=config
    )


def test_the_plaintext_password_is_never_in_the_hash():
    config = make_config()
    password = good_password()
    assert password not in hash_password(password, config=config)


@pytest.mark.parametrize("bad", ["", "$argon2id$garbage", "not-a-hash-at-all"])
def test_a_malformed_stored_hash_is_false_not_an_exception(bad):
    assert verify_password(good_password(), bad, config=make_config()) is False


# ── password policy ──────────────────────────────────────────────────────────


def test_a_compliant_password_is_accepted():
    validate_password(good_password(), config=make_config())


@pytest.mark.parametrize(
    "password, expected",
    [
        ("Short1A", "at least 12 characters"),
        ("alllowercase123", "an uppercase letter"),
        ("ALLUPPERCASE123", "a lowercase letter"),
        ("NoDigitsHereAtAll", "a digit"),
    ],
)
def test_each_missing_rule_is_reported(password, expected):
    with pytest.raises(AuthError) as exc:
        validate_password(password, config=make_config())
    assert exc.value.code == "password_invalid"
    assert expected in exc.value.message


def test_all_unmet_rules_are_reported_at_once():
    with pytest.raises(AuthError) as exc:
        validate_password("ABC", config=make_config())
    message = exc.value.message
    # short, and missing lowercase + digit — all in one message.
    assert "at least 12 characters" in message
    assert "a lowercase letter" in message
    assert "a digit" in message


def test_an_over_long_password_is_rejected():
    with pytest.raises(AuthError) as exc:
        validate_password("Aa1" + "x" * 300, config=make_config())
    assert "at most 256 characters" in exc.value.message


def test_a_password_containing_the_address_is_rejected():
    with pytest.raises(AuthError) as exc:
        validate_password(
            "MyBoss.User1pass",
            config=make_config(),
            email_normalized="boss.user@incois.gov.in",
        )
    assert "email address" in exc.value.message


def test_a_commonly_used_password_is_rejected():
    # Relax the other rules so the common-password rule is tested in isolation;
    # none of the blacklisted strings satisfy the default policy on their own.
    config = make_config(
        AUTH_PASSWORD_MIN_LENGTH="8",
        AUTH_PASSWORD_REQUIRE_UPPERCASE="0",
        AUTH_PASSWORD_REQUIRE_LOWERCASE="0",
        AUTH_PASSWORD_REQUIRE_DIGIT="0",
    )
    with pytest.raises(AuthError) as exc:
        validate_password("password123", config=config)
    assert "commonly used" in exc.value.message


def test_the_symbol_rule_can_be_enabled():
    config = make_config(AUTH_PASSWORD_REQUIRE_SYMBOL="1")
    with pytest.raises(AuthError) as exc:
        validate_password("K7QuietTide92", config=config)
    assert "a symbol" in exc.value.message
    validate_password("K7QuietTide92!", config=config)


def test_the_policy_is_exposed_for_the_ui():
    policy = password_policy(make_config())
    assert policy["min_length"] == 12
    assert policy["require_uppercase"] is True
    assert policy["require_digit"] is True
    assert policy["rejects_common_passwords"] is True


# ── signup tokens ────────────────────────────────────────────────────────────


def test_a_token_round_trips():
    token, _ = issue_signup_token(
        key=KEY, email_normalized="a@incois.gov.in", ttl_seconds=900
    )
    assert read_signup_token(token, key=KEY) == "a@incois.gov.in"


def test_a_tampered_payload_is_rejected():
    token, _ = issue_signup_token(
        key=KEY, email_normalized="a@incois.gov.in", ttl_seconds=900
    )
    payload, signature = token.split(".")
    forged = tokens_mod._b64url(b'{"email_normalized":"boss@incois.gov.in","purpose":"signup","expires_at":99999999999}')

    with pytest.raises(AuthError) as exc:
        read_signup_token(f"{forged}.{signature}", key=KEY)
    assert exc.value.code == "token_invalid"


def test_a_tampered_signature_is_rejected():
    token, _ = issue_signup_token(
        key=KEY, email_normalized="a@incois.gov.in", ttl_seconds=900
    )
    payload, signature = token.split(".")
    bad = ("a" if not signature.startswith("a") else "b") + signature[1:]

    with pytest.raises(AuthError) as exc:
        read_signup_token(f"{payload}.{bad}", key=KEY)
    assert exc.value.code == "token_invalid"


def test_a_token_from_another_secret_is_rejected():
    token, _ = issue_signup_token(
        key=KEY, email_normalized="a@incois.gov.in", ttl_seconds=900
    )
    with pytest.raises(AuthError) as exc:
        read_signup_token(token, key=b"\x7f" * 32)
    assert exc.value.code == "token_invalid"


def test_an_expired_token_is_rejected():
    token, _ = issue_signup_token(
        key=KEY, email_normalized="a@incois.gov.in", ttl_seconds=-1
    )
    with pytest.raises(AuthError) as exc:
        read_signup_token(token, key=KEY)
    assert exc.value.code == "token_expired"


@pytest.mark.parametrize("garbage", ["", "abc", "a.b", "...."])
def test_garbage_tokens_are_rejected(garbage):
    with pytest.raises(AuthError) as exc:
        read_signup_token(garbage, key=KEY)
    assert exc.value.code == "token_invalid"


# ── configuration ────────────────────────────────────────────────────────────


def test_a_missing_secret_is_refused():
    env = base_env()
    env.pop("AUTH_SECRET")
    with pytest.raises(AuthError):
        load_auth_config(env)


def test_a_short_secret_is_refused():
    with pytest.raises(AuthError):
        make_config(AUTH_SECRET="too-short")


def test_the_secret_is_never_rendered():
    config = make_config(SMTP_PASSWORD="hunter2")
    rendered = repr(config)
    assert TEST_SECRET not in rendered
    assert "hunter2" not in rendered
    assert "<set>" in rendered


def test_dev_otp_switches_are_refused_in_production():
    for switch in ("AUTH_DEV_PRINT_OTP", "AUTH_DEV_RETURN_OTP"):
        with pytest.raises(AuthError):
            make_config(APP_ENV="production", **{switch: "1"})


def test_derive_key_is_deterministic_and_domain_separated():
    assert derive_key(TEST_SECRET, "label-a") == derive_key(TEST_SECRET, "label-a")
    assert derive_key(TEST_SECRET, "label-a") != derive_key(TEST_SECRET, "label-b")
    assert len(derive_key(TEST_SECRET, "label-a")) == 32


def test_the_otp_and_token_keys_are_different():
    config = make_config()
    assert config.otp_key() != config.signup_token_key()


def test_production_defaults_are_strong():
    """Argon2 is deliberately slowed in tests; the defaults must not be."""
    config = load_auth_config({"AUTH_SECRET": TEST_SECRET, "APP_ENV": "production"})
    assert config.argon2_time_cost >= 3
    assert config.argon2_memory_cost >= 65536
    assert config.argon2_parallelism >= 1
    assert config.otp_ttl_seconds == 600
    assert config.otp_length == 6
    assert config.otp_max_attempts == 5


def test_an_invalid_integer_setting_is_rejected():
    env = base_env()
    env["AUTH_OTP_TTL_SECONDS"] = "not-a-number"
    with pytest.raises(AuthError):
        load_auth_config(env)


# ── rate limiting ────────────────────────────────────────────────────────────


def _fake_clock():
    now = [0.0]
    return now, (lambda: now[0])


def test_requests_are_allowed_up_to_the_limit():
    _, clock = _fake_clock()
    limiter = RateLimiter(clock=clock)
    for _ in range(3):
        limiter.check("key", limit=3, window_seconds=60)


def test_the_limit_is_enforced_and_carries_a_retry_after():
    _, clock = _fake_clock()
    limiter = RateLimiter(clock=clock)
    for _ in range(3):
        limiter.check("key", limit=3, window_seconds=60)

    with pytest.raises(RateLimited) as exc:
        limiter.check("key", limit=3, window_seconds=60)
    assert exc.value.retry_after > 0


def test_the_window_slides():
    now, clock = _fake_clock()
    limiter = RateLimiter(clock=clock)
    for _ in range(3):
        limiter.check("key", limit=3, window_seconds=60)
    with pytest.raises(RateLimited):
        limiter.check("key", limit=3, window_seconds=60)

    now[0] = 61.0  # the window has passed
    limiter.check("key", limit=3, window_seconds=60)


def test_limits_are_keyed_independently():
    _, clock = _fake_clock()
    limiter = RateLimiter(clock=clock)
    for _ in range(2):
        limiter.check("alice", limit=2, window_seconds=60)
    limiter.check("bob", limit=2, window_seconds=60)


def test_the_shared_limiter_is_reachable_and_resettable():
    limiter = get_limiter()
    limiter.check("shared", limit=1, window_seconds=60)
    assert limiter.hits("shared") == 1


# ── mail delivery ────────────────────────────────────────────────────────────


def test_the_capturing_mailer_records_instead_of_sending():
    mailer = CapturingMailer()
    mailer.send(MailMessage(to="a@incois.gov.in", subject="s", body="body"))
    assert len(mailer.sent) == 1
    assert mailer.last.to == "a@incois.gov.in"
    assert mailer.failures == 0


def test_the_capturing_mailer_can_simulate_a_delivery_failure():
    mailer = CapturingMailer()
    mailer.fail_next = True
    with pytest.raises(AuthError) as exc:
        mailer.send(MailMessage(to="a@incois.gov.in", subject="s", body="body"))
    assert exc.value.code == "mailer_failed"
    assert mailer.sent == []

    mailer.send(MailMessage(to="a@incois.gov.in", subject="s", body="body"))
    assert len(mailer.sent) == 1


def test_auto_mailer_captures_when_smtp_is_unconfigured():
    assert isinstance(get_mailer(make_config()), CapturingMailer)


def test_auto_mailer_uses_smtp_when_it_is_configured():
    mailer = get_mailer(
        make_config(SMTP_HOST="mail.example.in", SMTP_FROM="noreply@incois.gov.in")
    )
    assert isinstance(mailer, SmtpMailer)


def test_the_console_mailer_refuses_to_print_codes_in_production():
    with pytest.raises(AuthError):
        ConsoleMailer(reveal_body=True, is_production=True)


def test_smtp_errors_are_scrubbed(monkeypatch):
    """A driver exception must not reach the client — it can hold credentials."""

    class Boom:
        def __init__(self, *args, **kwargs):
            raise OSError("authentication failed for user hunter2")

    monkeypatch.setattr(smtplib, "SMTP", Boom)
    config = make_config(
        SMTP_HOST="mail.example.in",
        SMTP_FROM="noreply@incois.gov.in",
        SMTP_STARTTLS="0",
        SMTP_PASSWORD="hunter2",
    )

    with pytest.raises(AuthError) as exc:
        SmtpMailer(config).send(
            MailMessage(to="a@incois.gov.in", subject="s", body="your code is 123456")
        )

    assert exc.value.code == "mailer_failed"
    assert "hunter2" not in exc.value.message
    assert "123456" not in exc.value.message
    assert "OSError" in exc.value.message


def test_an_unconfigured_smtp_mailer_fails_cleanly():
    with pytest.raises(AuthError) as exc:
        SmtpMailer(make_config()).send(
            MailMessage(to="a@incois.gov.in", subject="s", body="b")
        )
    assert exc.value.code == "mailer_failed"
