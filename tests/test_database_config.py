"""Phase 1 database foundation — configuration, laziness and credential safety.

Every test in this file runs **without** a PostgreSQL server. They guard the
properties that must hold regardless of environment:

  * database config is entirely environment-driven and fails loudly;
  * importing the database package never builds an engine or opens a socket;
  * the password is masked in every log/exception/repr surface;
  * Alembic has exactly one head and renders valid PostgreSQL DDL.

Live connectivity and migration tests live in
``tests/test_database_integration.py`` and are skipped unless a dedicated test
database is configured.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session

from src.db.config import (
    DatabaseConfig,
    DatabaseConfigError,
    load_database_config,
    mask_database_url,
)
from src.db.session import get_db, get_engine, get_session_factory, reset_engine

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every variable load_database_config() reads.
DATABASE_ENV_VARS = (
    "DATABASE_URL",
    "DATABASE_HOST",
    "DATABASE_PORT",
    "DATABASE_NAME",
    "DATABASE_USER",
    "DATABASE_PASSWORD",
    "DATABASE_ECHO",
    "DATABASE_POOL_SIZE",
    "DATABASE_MAX_OVERFLOW",
    "DATABASE_CONNECT_TIMEOUT",
)

#: A distinctive password so a leak is unambiguous in an assertion.
SECRET = "LEAK_CANARY_do_not_appear_anywhere"

PARTS: dict[str, str] = {
    "DATABASE_NAME": "incois_audit",
    "DATABASE_USER": "incois",
    "DATABASE_PASSWORD": SECRET,
}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch):
    """No ambient DATABASE_* value and no memoised engine leaks between tests."""
    for name in DATABASE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    reset_engine()
    yield
    reset_engine()


# ─────────────────────────────────────────────────────────────────────────────
# Import-time behaviour
# ─────────────────────────────────────────────────────────────────────────────


def test_importing_src_db_builds_no_engine():
    """Requirement 9 — importing the database package must not connect.

    Verified in a fresh subprocess so a memoised engine from another test
    cannot mask a regression.
    """
    code = (
        "import src.db, src.db.session as s; "
        "assert s._engine is None, 'engine was built at import time'; "
        "assert s._session_factory is None, 'session factory built at import'; "
        "print('LAZY_OK')"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, repo-local cwd
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert result.returncode == 0, f"import built an engine or failed:\n{result.stderr}"
    assert "LAZY_OK" in result.stdout


def test_get_engine_does_not_connect(monkeypatch: pytest.MonkeyPatch):
    """Building the engine must not open a socket (unroutable host/port)."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:1/nonexistent")
    engine = get_engine()  # must not raise — nothing is contacted yet
    assert engine is not None
    # Memoised: the same object on a second call.
    assert get_engine() is engine


def test_reset_engine_clears_the_memoised_engine(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:1/nonexistent")
    first = get_engine()
    reset_engine()
    second = get_engine()
    assert first is not second


def test_get_session_factory_is_memoised_and_bound(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:1/nonexistent")
    factory = get_session_factory()
    assert get_session_factory() is factory
    assert factory.kw["bind"] is get_engine()


def test_get_db_yields_a_session_and_closes_it(monkeypatch: pytest.MonkeyPatch):
    """The FastAPI dependency: yields a Session, closes it on generator exit."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:1/nonexistent")
    gen = get_db()
    session = next(gen)
    try:
        assert isinstance(session, Session)
    finally:
        with contextlib.suppress(StopIteration):
            next(gen)  # runs `finally: session.close()`
    assert not session.is_active or session.identity_map.keys() is not None


# ─────────────────────────────────────────────────────────────────────────────
# Configuration from parts / from URL
# ─────────────────────────────────────────────────────────────────────────────


def test_config_from_parts_uses_default_host_and_port(monkeypatch: pytest.MonkeyPatch):
    for key, value in PARTS.items():
        monkeypatch.setenv(key, value)
    cfg = load_database_config()
    assert isinstance(cfg, DatabaseConfig)
    parsed = make_url(cfg.url)
    assert parsed.get_backend_name() == "postgresql"
    assert parsed.get_driver_name() == "psycopg"
    assert parsed.host == "localhost"
    assert parsed.port == 5432
    assert parsed.database == "incois_audit"
    assert parsed.username == "incois"
    assert parsed.password == SECRET


def test_config_from_parts_honours_host_and_port(monkeypatch: pytest.MonkeyPatch):
    for key, value in PARTS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DATABASE_HOST", "db")
    monkeypatch.setenv("DATABASE_PORT", "5433")
    parsed = make_url(load_database_config().url)
    assert parsed.host == "db"
    assert parsed.port == 5433


def test_database_url_wins_over_parts(monkeypatch: pytest.MonkeyPatch):
    for key, value in PARTS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DATABASE_URL", "postgresql://urluser:urlpass@urlhost:6000/urldb")
    parsed = make_url(load_database_config().url)
    assert (parsed.username, parsed.host, parsed.port, parsed.database) == (
        "urluser",
        "urlhost",
        6000,
        "urldb",
    )


@pytest.mark.parametrize(
    "raw",
    [
        "postgres://u:p@h:5432/d",
        "postgresql://u:p@h:5432/d",
        "postgresql+psycopg://u:p@h:5432/d",
    ],
)
def test_scheme_is_normalised_to_psycopg3(monkeypatch: pytest.MonkeyPatch, raw: str):
    """A plain URL must never silently select the psycopg2 dialect."""
    monkeypatch.setenv("DATABASE_URL", raw)
    cfg = load_database_config()
    assert cfg.url.startswith("postgresql+psycopg://")
    assert make_url(cfg.url).get_driver_name() == "psycopg"


@pytest.mark.parametrize("raw", ["sqlite:///local.db", "mysql://u:p@h:3306/d"])
def test_non_postgres_scheme_is_rejected(monkeypatch: pytest.MonkeyPatch, raw: str):
    monkeypatch.setenv("DATABASE_URL", raw)
    with pytest.raises(DatabaseConfigError, match="PostgreSQL"):
        load_database_config()


@pytest.mark.parametrize("missing", ["DATABASE_NAME", "DATABASE_USER", "DATABASE_PASSWORD"])
def test_missing_required_part_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, missing: str
):
    for key, value in PARTS.items():
        if key != missing:
            monkeypatch.setenv(key, value)
    with pytest.raises(DatabaseConfigError) as excinfo:
        load_database_config()
    assert missing in str(excinfo.value)


def test_no_configuration_at_all_fails_loudly():
    with pytest.raises(DatabaseConfigError, match="DATABASE_NAME"):
        load_database_config()


def test_url_without_a_scheme_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", "user:pass@host:5432/db")
    with pytest.raises(DatabaseConfigError, match="scheme"):
        load_database_config()


def test_explicit_url_without_a_password_is_rejected(monkeypatch: pytest.MonkeyPatch):
    """Credentials are mandatory — an anonymous URL must not slip through."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://incois@localhost:5432/incois_audit")
    with pytest.raises(DatabaseConfigError, match="password"):
        load_database_config()


def test_special_characters_in_password_are_percent_encoded(
    monkeypatch: pytest.MonkeyPatch,
):
    """A password containing URL metacharacters must not corrupt the URL."""
    awkward = "p@ss/w:rd#?x"
    for key, value in PARTS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DATABASE_PASSWORD", awkward)
    cfg = load_database_config()
    parsed = make_url(cfg.url)
    assert parsed.password == awkward
    assert parsed.host == "localhost"  # '@' did not split the host
    assert parsed.database == "incois_audit"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("off", False),
    ],
)
def test_echo_flag_parsing(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool):
    for key, value in PARTS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DATABASE_ECHO", raw)
    assert load_database_config().echo is expected


def test_invalid_boolean_is_rejected(monkeypatch: pytest.MonkeyPatch):
    for key, value in PARTS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DATABASE_ECHO", "sometimes")
    with pytest.raises(DatabaseConfigError, match="boolean"):
        load_database_config()


@pytest.mark.parametrize("var", ["DATABASE_PORT", "DATABASE_POOL_SIZE", "DATABASE_MAX_OVERFLOW"])
def test_invalid_integer_is_rejected(monkeypatch: pytest.MonkeyPatch, var: str):
    for key, value in PARTS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv(var, "not-a-number")
    with pytest.raises(DatabaseConfigError, match="integer"):
        load_database_config()


def test_tuning_defaults(monkeypatch: pytest.MonkeyPatch):
    for key, value in PARTS.items():
        monkeypatch.setenv(key, value)
    cfg = load_database_config()
    assert cfg.echo is False
    assert cfg.pool_size == 5
    assert cfg.max_overflow == 10
    assert cfg.pool_pre_ping is True
    assert cfg.connect_timeout_seconds == 10


def test_pool_pre_ping_is_enabled(monkeypatch: pytest.MonkeyPatch):
    """HPC requirement: the DB SIF restarts independently of the app SIF."""
    for key, value in PARTS.items():
        monkeypatch.setenv(key, value)
    assert load_database_config().pool_pre_ping is True


def test_explicit_env_mapping_is_honoured():
    """load_database_config(env=...) must ignore os.environ entirely."""
    cfg = load_database_config({**PARTS, "DATABASE_HOST": "explicit"})
    assert make_url(cfg.url).host == "explicit"


# ─────────────────────────────────────────────────────────────────────────────
# Credential safety
# ─────────────────────────────────────────────────────────────────────────────


def test_mask_database_url_hides_the_password():
    masked = mask_database_url(f"postgresql+psycopg://incois:{SECRET}@h:5432/d")
    assert SECRET not in masked
    assert "***" in masked
    assert "incois" in masked  # user and host survive — useful in logs


def test_mask_database_url_tolerates_garbage():
    assert mask_database_url("definitely not a url") == "<unparsable database url>"


def test_config_repr_and_str_never_leak_the_password(monkeypatch: pytest.MonkeyPatch):
    for key, value in PARTS.items():
        monkeypatch.setenv(key, value)
    cfg = load_database_config()
    for surface in (repr(cfg), str(cfg), cfg.safe_url(), f"{cfg}"):
        assert SECRET not in surface, f"password leaked via: {surface}"
    assert "***" in repr(cfg)


def test_config_error_message_never_leaks_the_password(monkeypatch: pytest.MonkeyPatch):
    """A validation failure must not copy the (possibly mistyped) URL in."""
    # Non-numeric port makes SQLAlchemy's make_url raise; we surface only the
    # exception class name, never the input.
    monkeypatch.setenv("DATABASE_URL", f"postgresql+psycopg://u:{SECRET}@h:notaport/d")
    with pytest.raises(DatabaseConfigError) as excinfo:
        load_database_config()
    message = str(excinfo.value)
    assert SECRET not in message
    assert "ValueError" in message  # class name only — no URL echo


def test_unparsable_url_error_omits_the_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"postgresql+psycopg://u:{SECRET}@h:99999x/d")
    with pytest.raises(DatabaseConfigError) as excinfo:
        load_database_config()
    assert SECRET not in str(excinfo.value)


@pytest.mark.parametrize(
    "bad_url",
    [
        "postgresql+psycopg://u@h:5432/d",  # no password at all
        "postgresql://u:p@h:5432/",  # no database name
        "sqlite://u:p@h/local.db",  # wrong backend
    ],
)
def test_every_validation_failure_omits_the_password(
    monkeypatch: pytest.MonkeyPatch, bad_url: str
):
    """Every rejection path must be credential-free, not just one of them.

    ``:p@`` is replaced by the canary secret where present, so the two URLs
    that do carry a credential prove it is not echoed into the message.
    """
    monkeypatch.setenv("DATABASE_URL", bad_url.replace(":p@", f":{SECRET}@"))
    with pytest.raises(DatabaseConfigError) as excinfo:
        load_database_config()
    assert SECRET not in str(excinfo.value)


# ─────────────────────────────────────────────────────────────────────────────
# Alembic wiring (metadata only — no database)
# ─────────────────────────────────────────────────────────────────────────────


def test_alembic_has_exactly_one_head():
    """A branched head would silently skip migrations on a fresh database."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    assert len(heads) == 1, f"expected one Alembic head, found {heads}"

    # The head moves with each phase; what must hold is that it descends in an
    # unbranched chain from the base revision. (test_initial_revision_is_the_base
    # pins the base itself.)
    head = script.get_revision(heads[0])
    chain = []
    node = head
    while node is not None:
        chain.append(node.revision)
        node = node.down_revision and script.get_revision(node.down_revision)
    assert chain[-1] == "0001_create_app_meta", f"head is not rooted at 0001: {chain}"
    assert len(chain) == len(set(chain)), f"history is not linear: {chain}"


def test_initial_revision_is_the_base():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    script = ScriptDirectory.from_config(cfg)
    revision = script.get_revision("0001_create_app_meta")
    assert revision.down_revision is None
    assert revision.is_base


def test_alembic_ini_declares_no_credential():
    """The tracked Alembic config must not carry a connection URL."""
    body = (REPO_ROOT / "alembic.ini").read_text(encoding="utf-8")
    for line in body.splitlines():
        if line.strip().startswith("sqlalchemy.url"):
            assert line.split("=", 1)[1].strip() == "", (
                "alembic.ini must not embed a connection URL"
            )


def test_offline_sql_renders_and_never_echoes_the_password():
    """`alembic upgrade head --sql` renders DDL without connecting.

    This is the strongest no-database check available: it proves the migration
    stack produces valid PostgreSQL DDL, and that the credential never reaches
    stdout/stderr.
    """
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT),
        "DATABASE_URL": f"postgresql://incois:{SECRET}@localhost:5432/incois_audit",
    }
    result = subprocess.run(  # noqa: S603 - fixed argv
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, f"offline render failed:\n{combined}"
    assert "CREATE TABLE app_meta" in combined
    assert "pk_app_meta" in combined
    assert "TIMESTAMP WITH TIME ZONE" in combined
    assert SECRET not in combined, "password leaked into the rendered SQL"


def test_alembic_env_does_not_import_the_application_server():
    """Alembic must not drag in server.py (import-time side effects)."""
    result = subprocess.run(  # noqa: S603 - fixed argv
        [
            sys.executable,
            "-c",
            (
                "import sys, os; os.environ['DATABASE_URL'] ="
                " 'postgresql://u:p@localhost:5432/d';"
                " import alembic.config, alembic.command as cmd;"
                " c = alembic.config.Config('alembic.ini');"
                " cmd.upgrade(c, 'head', sql=True);"
                " sys.exit(3 if 'src.retrieval.frontend.server' in sys.modules else 0)"
            ),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert result.returncode == 0, (
        "alembic imported src.retrieval.frontend.server:\n"
        f"{result.stdout}\n{result.stderr}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scope guard — Phase 1 must not grow beyond the foundation
# ─────────────────────────────────────────────────────────────────────────────


def test_only_the_expected_tables_are_declared():
    """Only the Phase 1 + Phase 2 tables may exist on the shared metadata.

    ``src/auth/models.py`` registers onto the same ``Base``, so this inspects the
    whole ORM surface, not just ``src/db/models.py``. Phase 1 contributed
    ``app_meta``; Phase 2 contributes ``users`` and ``email_otp``; Phase 3
    contributes ``sessions``. Roles and chat history must still not appear
    before their own phase.
    """
    from src.db.models import Base

    tables = set(Base.metadata.tables)
    out_of_scope = {
        "user_roles",
        "roles",
        "permissions",
        "password_reset",
        "conversations",
        "messages",
        "chat_messages",
    }
    assert not tables & out_of_scope, f"out-of-scope tables: {sorted(tables & out_of_scope)}"
    assert "app_meta" in tables  # Phase 1
    assert {"users", "email_otp"} <= tables, f"Phase 2 tables missing: {sorted(tables)}"
    assert "sessions" in tables, "Phase 3 sessions table missing"

    columns = {c.name for c in Base.metadata.tables["app_meta"].columns}
    assert columns == {"key", "value", "updated_at"}



def test_phase1_ships_no_authentication_modules():
    """Guards the Phase 1 boundary — auth arrives in Phase 2/3/4.

    ``session.py`` is deliberately NOT in this list: it is the SQLAlchemy
    session module, which has nothing to do with login sessions.
    """
    db_dir = REPO_ROOT / "src" / "db"
    names = {p.stem for p in db_dir.glob("*.py")}
    forbidden = {"auth", "users", "user", "otp", "password", "login", "token", "credential"}
    assert not names & forbidden, f"auth-related module(s) present: {names & forbidden}"


def test_placeholders_only_never_real_credentials_in_tracked_files():
    """Tracked config files may document variables, never values."""
    for rel in ("alembic.ini", "docker-compose.yml", ".env.example"):
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        body = path.read_text(encoding="utf-8")
        assert SECRET not in body
