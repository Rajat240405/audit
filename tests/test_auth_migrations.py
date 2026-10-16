"""Phase 2 — migration hygiene.

These checks run **without** a database: Alembic's ``--sql`` mode emits the DDL
offline, so the chain, the schema and the safety of the revision can be
verified in CI with no PostgreSQL server present.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = REPO_ROOT / "alembic.ini"
VERSIONS = REPO_ROOT / "alembic" / "versions"

#: A never-connected URL. `alembic upgrade --sql` does not open a connection.
DUMMY_URL = "postgresql+psycopg://u:p@127.0.0.1:1/does_not_exist"


def _alembic(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["DATABASE_URL"] = DUMMY_URL
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


@pytest.fixture(scope="module")
def offline_sql() -> str:
    result = _alembic("upgrade", "head", "--sql")
    assert result.returncode == 0, result.stderr
    return result.stdout


# ── the revision chain ───────────────────────────────────────────────────────


def test_there_is_exactly_one_alembic_head():
    """A branched head would silently skip migrations on a fresh database.

    The head moves with each phase; what must hold is that there is exactly one
    and that it descends from the base revision.
    """
    result = _alembic("heads")
    assert result.returncode == 0, result.stderr
    heads = [ln for ln in result.stdout.splitlines() if "(head)" in ln]
    assert len(heads) == 1, f"expected a single head, got: {heads}"

    history = _alembic("history")
    lines = [ln for ln in history.stdout.splitlines() if "->" in ln]
    # "<base> -> 0001", "0001 -> 0002", ... one line per revision.
    revision_files = [
        f for f in VERSIONS.glob("*.py") if not f.name.startswith("_")
    ]
    assert len(lines) == len(revision_files), (
        f"history has {len(lines)} revisions but {len(revision_files)} migration "
        f"files exist — a revision is missing or orphaned"
    )
    assert "<base> -> 0001_create_app_meta" in history.stdout


def test_the_phase_2_revision_depends_on_the_phase_1_revision():
    source = (VERSIONS / "0002_signup_email_otp.py").read_text(encoding="utf-8")
    assert 'down_revision: str | None = "0001_create_app_meta"' in source


def test_the_history_is_a_single_linear_chain():
    result = _alembic("history")
    assert result.returncode == 0, result.stderr
    assert "<base> -> 0001_create_app_meta" in result.stdout
    assert "0001_create_app_meta -> 0002_signup_email_otp" in result.stdout
    assert "0002_signup_email_otp -> 0003_auth_sessions_reset" in result.stdout
    # Linear == exactly one line names <base>, and no revision has two children.
    bases = [ln for ln in result.stdout.splitlines() if ln.strip().startswith("<base>")]
    assert len(bases) == 1, f"expected one base, got {bases}"


def test_migrations_live_only_in_the_central_versions_directory():
    """No duplicate migration files scattered under docs/ or elsewhere."""
    strays = [
        str(p.relative_to(REPO_ROOT))
        for p in REPO_ROOT.rglob("*.py")
        if p.parent != VERSIONS
        and "versions" in p.parts
        and p.name[0].isdigit()
        and "_" in p.name
    ]
    assert strays == [], f"migration files outside alembic/versions: {strays}"


# ── what the migration actually creates ──────────────────────────────────────


def test_both_signup_tables_are_created(offline_sql):
    assert "CREATE TABLE users" in offline_sql
    assert "CREATE TABLE email_otp" in offline_sql


def test_the_phase_1_table_is_still_created_first(offline_sql):
    assert offline_sql.index("CREATE TABLE app_meta") < offline_sql.index(
        "CREATE TABLE users"
    )


def test_email_normalized_is_unique(offline_sql):
    assert "uq_users_email_normalized" in offline_sql
    assert "UNIQUE (email_normalized)" in offline_sql


def test_the_expected_indexes_are_created(offline_sql):
    assert "ix_users_email_normalized" in offline_sql
    assert "ix_email_otp_email_purpose" in offline_sql


def test_the_otp_table_never_stores_a_plain_code(offline_sql):
    """Only a digest and a salt — there is no column that could hold the code."""
    block = offline_sql.split("CREATE TABLE email_otp")[1].split(");")[0]
    assert "code_hash" in block
    assert "salt" in block
    assert "code VARCHAR" not in block


def test_no_out_of_scope_tables_are_created(offline_sql):
    """Phase 2 is signup only — sessions, roles and chat history come later."""
    # `sessions` became in-scope in Phase 3; everything below is still later.
    lowered = offline_sql.lower()
    for table in (
        "user_roles",
        "roles",
        "password_reset",
        "conversations",
        "messages",
    ):
        assert f"create table {table}" not in lowered, f"{table} is out of scope"


def test_the_migration_is_reversible():
    result = _alembic("downgrade", "0002_signup_email_otp:0001_create_app_meta", "--sql")
    assert result.returncode == 0, result.stderr
    assert "DROP TABLE email_otp" in result.stdout
    assert "DROP TABLE users" in result.stdout
    assert "DROP TABLE app_meta" not in result.stdout  # Phase 1 stays
