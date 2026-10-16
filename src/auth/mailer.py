"""Mail delivery for signup OTPs (Phase 2).

Email delivery sits behind a small interface so that:

* automated tests simulate delivery without any network (:class:`CapturingMailer`);
* local development without SMTP still works (and can, **only if you opt in**,
  print the code so you can copy it from the log);
* production uses real SMTP with credentials taken from the environment.

Rules enforced here:

* SMTP settings and credentials come only from the environment.
* SMTP credentials are never logged; exception messages are scrubbed so a
  driver error cannot echo them.
* The OTP body is never printed or returned unless an explicit dev switch is on
  **and** ``APP_ENV`` is not production.
"""

from __future__ import annotations

import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Protocol

from src.auth.config import AuthConfig
from src.auth.errors import AuthError

__all__ = [
    "MailMessage",
    "Mailer",
    "SmtpMailer",
    "CapturingMailer",
    "ConsoleMailer",
    "get_mailer",
]


@dataclass(frozen=True)
class MailMessage:
    """One outbound email. ``body`` may contain an OTP — treat it as sensitive."""

    to: str
    subject: str
    body: str


class Mailer(Protocol):
    """Anything that can deliver a :class:`MailMessage`."""

    def send(self, message: MailMessage) -> None:
        """Deliver the message. Raises on failure."""
        ...


class CapturingMailer:
    """In-memory mailer for tests. Records everything, sends nothing."""

    def __init__(self) -> None:
        self.sent: list[MailMessage] = []
        self.failures: int = 0
        self.fail_next: bool = False

    def send(self, message: MailMessage) -> None:
        if self.fail_next:
            self.fail_next = False
            self.failures += 1
            raise AuthError("simulated SMTP failure", code="mailer_failed")
        self.sent.append(message)

    @property
    def last(self) -> MailMessage | None:
        return self.sent[-1] if self.sent else None


class ConsoleMailer:
    """Prints to the server log instead of sending — for local development.

    ``reveal_body`` must be False in production; :func:`get_mailer` enforces
    that, and the check is repeated here so a misconfigured caller cannot leak
    a code by constructing this class directly.
    """

    def __init__(self, *, reveal_body: bool = False, is_production: bool = False) -> None:
        if reveal_body and is_production:
            raise AuthError(
                "refusing to reveal OTP bodies when APP_ENV=production",
                code="config_invalid",
            )
        self.reveal_body = reveal_body
        self.sent: list[MailMessage] = []

    def send(self, message: MailMessage) -> None:
        self.sent.append(message)
        if self.reveal_body:
            print(f"[auth][mailer:console] to={message.to} subject={message.subject}")
            print(f"[auth][mailer:console] body:\n{message.body}")
        else:
            # Never print the body: it contains the OTP.
            print(
                f"[auth][mailer:console] to={message.to} subject={message.subject} "
                f"(body withheld — set AUTH_DEV_PRINT_OTP=1 outside production to see it)"
            )


class SmtpMailer:
    """Real SMTP delivery. Credentials come from config, never from source."""

    def __init__(self, config: AuthConfig) -> None:
        self._config = config

    def send(self, message: MailMessage) -> None:
        cfg = self._config
        if not cfg.smtp_host or not cfg.smtp_from:
            raise AuthError("SMTP is not configured", code="mailer_failed")

        email = EmailMessage()
        email["Subject"] = message.subject
        email["From"] = cfg.smtp_from
        email["To"] = message.to
        email.set_content(message.body)

        try:
            if cfg.smtp_ssl:
                server: smtplib.SMTP = smtplib.SMTP_SSL(
                    cfg.smtp_host, cfg.smtp_port, timeout=cfg.smtp_timeout_seconds
                )
            else:
                server = smtplib.SMTP(
                    cfg.smtp_host, cfg.smtp_port, timeout=cfg.smtp_timeout_seconds
                )
            with server:
                server.ehlo()
                if cfg.smtp_starttls and not cfg.smtp_ssl:
                    server.starttls(context=ssl.create_default_context())
                    server.ehlo()
                if cfg.smtp_username:
                    server.login(cfg.smtp_username, cfg.smtp_password or "")
                server.send_message(email)
        except Exception as exc:  # noqa: BLE001
            # `exc` may contain the username or a driver detail. Report only the
            # class name plus a stable, actionable message.
            raise AuthError(
                f"could not send the verification email "
                f"({type(exc).__name__}). Check the SMTP_* settings.",
                code="mailer_failed",
            ) from None


def get_mailer(config: AuthConfig) -> Mailer:
    """Resolve the mailer from configuration. Pure — no global state.

    ``auto`` → smtp when ``SMTP_HOST`` is set, else console (when
    ``AUTH_DEV_PRINT_OTP`` is on and this is not production), else capturing.
    The capturing fallback means an unconfigured deployment never crashes on
    signup and, more importantly, never silently emails a real code anywhere.
    """
    choice = config.mailer
    if choice == "auto":
        if config.smtp_host:
            choice = "smtp"
        elif config.dev_print_otp and not config.is_production:
            choice = "console"
        else:
            choice = "capturing"

    if choice == "smtp":
        return SmtpMailer(config)
    if choice == "console":
        return ConsoleMailer(
            reveal_body=config.dev_print_otp and not config.is_production,
            is_production=config.is_production,
        )
    return CapturingMailer()
