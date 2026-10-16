"""Phase 2 — email normalisation and the signup allowlist.

The allowlist is the real security boundary for signup, so these tests are
deliberately exhaustive about the ways an address could sneak past it:
lookalike domains, subdomains, plus-addressing and case tricks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.auth.emails import (
    ALLOWED_DOMAIN,
    is_allowed_signup_email,
    normalize_email,
    validate_signup_email,
)
from src.auth.errors import AuthError

SRC_AUTH = Path(__file__).resolve().parents[1] / "src" / "auth"

# ── normalisation ────────────────────────────────────────────────────────────


def test_strips_surrounding_whitespace():
    assert normalize_email("  a@b.com  ") == "a@b.com"


def test_lowercases_both_local_part_and_domain():
    assert normalize_email("Boss.User@INCOIS.GOV.IN") == "boss.user@incois.gov.in"


def test_normalisation_is_idempotent():
    once = normalize_email("  Boss.User@INCOIS.GOV.IN ")
    assert normalize_email(once) == once


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_empty_input_is_rejected(raw):
    with pytest.raises(AuthError) as exc:
        normalize_email(raw)
    assert exc.value.code == "email_required"


def test_over_long_address_is_rejected():
    # 255 characters — one over the RFC 5321 limit of 254.
    with pytest.raises(AuthError) as exc:
        normalize_email("a" * 243 + "@incois.gov.in")
    assert exc.value.code == "email_invalid"


@pytest.mark.parametrize(
    "raw",
    [
        "a b@incois.gov.in",  # internal space
        "a@incois .gov.in",  # space in the domain
        "a@incois.gov.in\tbcc:evil@x.com",  # tab + header injection
        "a@b.com\x00",  # NUL
        "a\x01b@incois.gov.in",  # control character inside the local part
        "a@incois.gov.in\x02",  # trailing control character
    ],
)
def test_whitespace_and_control_characters_are_rejected(raw):
    with pytest.raises(AuthError) as exc:
        normalize_email(raw)
    assert exc.value.code == "email_invalid"


def test_a_trailing_newline_is_trimmed_not_rejected():
    """`.strip()` trims surrounding whitespace — including a pasted newline."""
    assert normalize_email("boss@incois.gov.in\n") == "boss@incois.gov.in"
    assert normalize_email("\tboss@incois.gov.in  ") == "boss@incois.gov.in"


@pytest.mark.parametrize(
    "raw",
    [
        "not-an-email",
        "@incois.gov.in",
        "boss@",
        "boss@incois",  # no TLD — the regex demands a dotted domain
        "boss@@incois.gov.in",
        "boss@incois..gov.in",
        "boss@incois.gov.in.",
        "boss @incois.gov.in",
    ],
)
def test_malformed_addresses_are_rejected(raw):
    with pytest.raises(AuthError) as exc:
        normalize_email(raw)
    assert exc.value.code == "email_invalid"


# ── the allowlist ────────────────────────────────────────────────────────────


def test_allowed_domain_constant_is_the_org_domain():
    assert ALLOWED_DOMAIN == "incois.gov.in"


@pytest.mark.parametrize(
    "email",
    [
        "boss@incois.gov.in",
        "boss.user@incois.gov.in",
        "boss+tag@incois.gov.in",
        "BOSS@INCOIS.GOV.IN",  # normalised before the check
    ],
)
def test_org_domain_addresses_are_allowed(email):
    assert (
        is_allowed_signup_email(
            normalize_email(email), dev_admin_email="dev.admin@gmail.com"
        )
        is True
    )


@pytest.mark.parametrize(
    "email",
    [
        "boss@incois.gov.in.evil.com",  # suffix lookalike
        "boss@evil-incois.gov.in",  # prefixed lookalike
        "boss@notincois.gov.in",  # extra characters before the domain
        "boss@mail.incois.gov.in",  # subdomain is NOT covered
    ],
)
def test_lookalike_and_subdomains_are_rejected(email):
    # Rejected at the allowlist stage, so these must normalise cleanly first.
    normalized = normalize_email(email)
    assert (
        is_allowed_signup_email(normalized, dev_admin_email="dev.admin@gmail.com")
        is False
    )


@pytest.mark.parametrize(
    "email",
    [
        "someone@gmail.com",  # an unrelated Gmail
        "someone@googlemail.com",
        "someone@outlook.com",
        "someone@yahoo.co.in",
    ],
)
def test_unrelated_addresses_are_rejected(email):
    assert (
        is_allowed_signup_email(
            normalize_email(email), dev_admin_email="dev.admin@gmail.com"
        )
        is False
    )


# ── the single developer exception ───────────────────────────────────────────


def test_exact_dev_address_is_allowed():
    assert (
        is_allowed_signup_email("dev.admin@gmail.com", dev_admin_email=DEV) is True
    ) if (DEV := "dev.admin@gmail.com") else False


def test_dev_address_matches_case_insensitively_after_normalisation():
    # The env value may be typed with any case; both sides are normalised.
    assert (
        is_allowed_signup_email(
            normalize_email("DEV.ADMIN@GMAIL.COM"), dev_admin_email="Dev.Admin@Gmail.com"
        )
        is True
    )


@pytest.mark.parametrize(
    "email",
    [
        "dev.admin+alias@gmail.com",  # plus-addressing
        "devadmin@gmail.com",  # dots folded away
        "dev.admin2@gmail.com",  # different local part
        "other.person@gmail.com",  # a different Gmail entirely
    ],
)
def test_only_the_exact_dev_address_is_allowed(email):
    """The exception admits exactly one mailbox — not a family of them."""
    assert (
        is_allowed_signup_email(
            normalize_email(email), dev_admin_email="dev.admin@gmail.com"
        )
        is False
    )


def test_no_gmail_is_allowed_when_dev_email_is_unset():
    for candidate in ("dev.admin@gmail.com", "anyone@gmail.com"):
        assert is_allowed_signup_email(candidate, dev_admin_email=None) is False
        assert is_allowed_signup_email(candidate, dev_admin_email="") is False


def test_dev_email_is_never_hard_coded_in_the_source():
    """The developer address must come from the environment, only.

    A hard-coded address would silently widen the allowlist for anyone reading
    the source, and would not be revocable by configuration.
    """
    offenders = []
    for path in sorted(SRC_AUTH.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        if "gmail.com" in lowered or "googlemail.com" in lowered:
            offenders.append(path.name)
    assert offenders == [], f"developer address hard-coded in: {offenders}"


# ── validate_signup_email (the function the service actually calls) ───────────


def test_validate_returns_the_normalised_address():
    assert (
        validate_signup_email(
            "  Boss.User@INCOIS.GOV.IN  ", dev_admin_email="dev.admin@gmail.com"
        )
        == "boss.user@incois.gov.in"
    )


def test_validate_rejects_disallowed_with_403():
    with pytest.raises(AuthError) as exc:
        validate_signup_email("someone@gmail.com", dev_admin_email="dev.admin@gmail.com")
    assert exc.value.code == "email_not_allowed"
    assert exc.value.http_status == 403


def test_validate_rejects_malformed_with_400():
    with pytest.raises(AuthError) as exc:
        validate_signup_email("nonsense", dev_admin_email="dev.admin@gmail.com")
    assert exc.value.code == "email_invalid"
    assert exc.value.http_status == 400


def test_validation_and_uniqueness_use_the_same_normalisation():
    """A case variant must map to the identity used for uniqueness checks.

    If validation lowercased but the DB did not, `Boss@` and `boss@` would be
    two different accounts.
    """
    a = validate_signup_email("Boss@incois.gov.in", dev_admin_email=None)
    b = validate_signup_email("boss@INCOIS.gov.in", dev_admin_email=None)
    assert a == b == "boss@incois.gov.in"
