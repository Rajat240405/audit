# PostgreSQL setup (Phase 1 foundation)

Phase 1 delivers the **foundation only**: environment-driven configuration, a
lazy SQLAlchemy engine, and Alembic migrations with a single `app_meta` table.

There is **no authentication, no user table, no OTP, no password reset and no
chat-history persistence** in this phase, and the FastAPI application does not
yet connect to the database. Those arrive in Phases 2â€“5.

Read `../.env.example` alongside this file for the full variable list.

---

## 1. What was added

| Path | Purpose |
|---|---|
| `src/db/config.py` | `DatabaseConfig` + `load_database_config()` â€” environment-driven, fails loudly |
| `src/db/session.py` | Lazy engine (`get_engine`), `get_session_factory()`, `get_db()` dependency |
| `src/db/models.py` | Declarative `Base` + the single `app_meta` table |
| `src/db/__init__.py` | Public exports |
| `alembic.ini`, `alembic/` | Migration configuration and revisions |
| `alembic/versions/0001_create_app_meta.py` | The only migration |
| `tests/test_database_config.py` | Runs with **no** database (50 tests) |
| `tests/test_database_integration.py` | Live DB + migrations (skipped unless configured) |
| `docker-compose.yml` | Opt-in `db` service behind the `db` profile |

Modified: `requirements.txt`, `pyproject.toml`, `Dockerfile.base`,
`docker-compose.yml`, `.env.example`, `.gitignore`.

**Nothing** was changed in `src/retrieval/frontend/server.py`, any frontend
file, `start_hpc.sh` or `stop_hpc.sh`, and no pinned ML dependency was touched.

---

## 2. Local development â€” Windows PC + Docker

### 2.1 First-time setup

1. Copy the environment template and set a **local-only** password:

   ```bat
   copy .env.example .env
   ```

   Then edit `.env` and fill in:

   ```
   DATABASE_NAME=incois_audit
   DATABASE_USER=incois
   DATABASE_PASSWORD=<a local-only password â€” never reuse the HPC one>
   DATABASE_HOST_PORT=5433
   ```

2. Start the database (opt-in profile):

   ```bat
   docker compose --profile db up -d
   ```

3. Confirm it is healthy:

   ```bat
   docker compose --profile db ps
   docker compose exec db pg_isready -U incois -d incois_audit
   ```

### 2.2 The application service is unchanged

```bat
docker compose up -d          :: starts the app ONLY (as before)
docker compose --profile db up -d   :: app + database
```

The `db` service sits behind a Compose **profile**, so the default behaviour of
`docker compose up -d` is byte-for-byte what it was before Phase 1. The
application service was not given any `DATABASE_*` variable, because nothing in
the app reads them yet (Phase 4/5).

### 2.3 Data persistence

Data lives in the **named volume** `pgdata`, not in a host bind mount.
Bind-mounting a PostgreSQL data directory into a Windows path is a known source
of permission and locking failures, so the volume keeps it on Docker's own
Linux filesystem.

| Command | Effect on data |
|---|---|
| `docker compose --profile db down` | **Kept** |
| `docker compose --profile db stop` / `start` | **Kept** |
| `docker compose --profile db down -v` | **DESTROYED** (removes the `pgdata` volume) |

### 2.4 Port choice

Host port **5433** maps to container port **5432**, so a PostgreSQL you may have
installed natively on 5432 is not disturbed. Override with
`DATABASE_HOST_PORT=5432` in `.env` if you prefer.

> **Windows port note:** the same excluded-range problem documented for Ollama in
> `.env.example` applies here. If the port refuses to bind while nothing appears
> to be listening, check
> `netsh interface ipv4 show excludedportrange protocol=tcp` and pick a free port.

### 2.5 Running migrations

Always from the **repository root**:

```bat
set DATABASE_URL=postgresql://incois:<your-password>@localhost:5433/incois_audit
alembic upgrade head
alembic current
alembic heads          :: must print exactly ONE head
```

Migrations are **never** run automatically at application start. This is
deliberate: it keeps serving containers immutable and stops a bad migration from
bricking a running service.

---

## 3. Test databases

Integration tests need a **separate** database and will refuse to run against
anything else:

```bat
docker compose exec db createdb -U incois incois_audit_test
```

```bat
set DATABASE_URL_TEST=postgresql://incois:<your-password>@localhost:5433/incois_audit_test
python -m pytest tests\test_database_config.py tests\test_database_integration.py -v
```

Two guards, both enforced in `tests/test_database_integration.py`:

1. Tests only run when **`DATABASE_URL_TEST`** is set â€” never `DATABASE_URL`.
2. The database **name must contain `test`**. If it does not, every destructive
   test is skipped. A typo therefore cannot run `alembic downgrade base` against
   real data.

> The migration tests are genuinely destructive (they run `downgrade base`).
> Only ever point `DATABASE_URL_TEST` at a throwaway database.

---

## 4. Verifying the foundation without a database

```bat
python -m pytest tests\test_database_config.py -v
```

50 tests covering: environment parsing, loud failure on a missing variable,
URL precedence, scheme normalisation, rejection of non-PostgreSQL backends,
password masking in `repr`/`str`/log/exception surfaces, percent-encoding of
special characters in passwords, lazy engine creation (verified in a subprocess),
a single Alembic head, and offline DDL rendering with no credential in the
output.

---

## 5. HPC deployment (Phase 8 â€” **documentation only, not implemented**)

Nothing in this phase changes or adds HPC scripts. `start_hpc.sh` and
`stop_hpc.sh` are untouched. This section records the intended approach so it is
reviewed before implementation.

### 5.1 Principles

The codebase already follows "PC vs HPC is an environment difference, never a
code difference" (see `APP_PROVIDERS`). The database follows the same rule:
**identical variable names, different values.**

| | Local (Windows) | HPC |
|---|---|---|
| Server | `postgres:16-alpine` in Docker | Separate PostgreSQL SIF **or** organisation-managed service |
| Data | Named volume `pgdata` | Bind-mounted HPC directory (e.g. `storage/postgres/data`) |
| Address | `localhost:5433` (host) | `127.0.0.1:5432` (shared host network namespace) |
| Credentials | `.env` (gitignored) | `.env.hpc` (gitignored, `chmod 600`) |
| Variables | `DATABASE_*` | `DATABASE_*` â€” same names |

### 5.2 Hard constraints

1. **The database must never live inside the application SIF.**
2. **Never rely on container-local storage.** `start_hpc.sh` runs the app with
   `--writable-tmpfs`, so any write to a path that is not bind-mounted is lost
   when the container exits. A database data directory therefore **must** be
   bind-mounted.
3. **Never pass `--net`, `--contain` or `--containall`** to either container.
   Singularity CE creates no network namespace by default, which is exactly why
   the app reaches the database over `127.0.0.1`. Those flags break that.
4. **The database must listen on `127.0.0.1` only.** The containers share the
   host network namespace, so binding `0.0.0.0` would expose the database to the
   whole cluster network.
5. **Do not pass the application's `--env-file` to the database container** â€”
   the app's configuration must not leak into it.
6. Create the data directory **host-side before first start** so it carries your
   uid/gid. Singularity has no `--user` / `--group-add`.
7. Run the server in the **foreground** under `nohup`. A daemonising entrypoint
   forks, and Singularity loses the child, leaving an unsupervisable PID.

### 5.3 Template

`start_neo4j_hpc.sh` in the repository root is a complete, working example of
every rule above. Model `start_postgres_hpc.sh` / `stop_postgres_hpc.sh` on it:
separate SIF, host-side directories, real bind mounts, no `--writable-tmpfs`,
loopback-only binding, password read from `.env.hpc` and never echoed or placed
on a command line, PID file, port-conflict check, disk-headroom warning,
readiness wait, and a cron-safe `PATH` export with a `module load singularity`
fallback.

### 5.4 Open questions to resolve before Phase 8

1. **Does cluster policy permit a user-run database daemon?** If an
   organisation-managed service is mandated instead, no startup script is needed
   and the work reduces to supplying a URL and credentials.
2. **Does the filesystem holding `storage/` support a PostgreSQL data
   directory?** Lustre in particular is often unsuitable (locking, `fsync`).
   Confirm with your cluster administrator â€” this is the largest unresolved
   assumption.
3. **How is the Docker image converted to a `.sif`?** No `singularity build`
   command exists anywhere in this repository (only `singularity exec`), so this
   step is currently undocumented.
4. **Does the managed service require TLS (`sslmode=require`)?** If so, extend
   `src/db/config.py` to pass `sslmode` through `connect_args`.
5. **Who owns credential rotation?**

---

## 6. Operational notes

- **One Uvicorn worker only.** `ACTIVE_CONFIG`, `llm_client`, `generator` and
  the generation-run registry are process globals. Adding PostgreSQL does **not**
  make the application multi-worker safe. Keep `--workers 1`.
- **`pool_pre_ping` is on** (and cannot be disabled via configuration). On HPC
  the database restarts independently of the application, and without pre-ping
  the pool keeps handing out dead connections.
- **`connect_timeout` defaults to 10 s**, so an unreachable database fails fast
  instead of hanging a request.
- **Single worker + shared globals** means one connection pool for the whole
  process. `DATABASE_POOL_SIZE=5` / `DATABASE_MAX_OVERFLOW=10` are ample.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `DATABASE_NAME is required and is not set` | `.env` is missing or you are not in the repository root. |
| `database URL must use the psycopg 3 driver` | You passed `postgresql+psycopg2://`. Use `postgresql://` or `postgresql+psycopg://`. |
| `alembic: database configuration error` | The environment is incomplete. `alembic` reads the same `DATABASE_*` variables as the app. |
| `password authentication failed` | `DATABASE_PASSWORD` does not match `POSTGRES_PASSWORD` used when the volume was first created. Either fix the value or `docker compose --profile db down -v` (destroys local data). |
| Port 5433 already in use | Set `DATABASE_HOST_PORT` in `.env` to a free port. |
| `More than one head` from `alembic heads` | A branched revision. Fix before deploying; `tests/test_database_config.py` fails on this. |
| Integration tests all skip | `DATABASE_URL_TEST` is unset, or the database name lacks `test`. This is the safety guard working. |

