# Phase 6: Dashboard + Monitoring for Paper Trading and Full Lightweight Containerisation - Research

**Researched:** 2026-02-19
**Domain:** Web dashboard, operational monitoring, Docker containerisation
**Confidence:** MEDIUM-HIGH

## Summary

Phase 6 adds two independent capabilities to the completed HYDRA v1 system: (1) a lightweight monitoring dashboard that surfaces paper trading metrics already collected by existing modules (FillJournal, DriftObserver, AgentLoop, SlippageReconciler, RiskGate), and (2) Docker Compose containerisation of the full HYDRA + IB Gateway stack for reproducible deployment.

The dashboard domain is well-understood. HYDRA already has all the data sources -- FillJournal (SQLite WAL), ExperimentJournal (SQLite), MLflow registry, structlog events, and the daily cycle runner. The dashboard is a read-only layer that queries these existing stores and renders them. The recommended stack is FastAPI + Jinja2 + htmx (CDN) + SSE for auto-refresh. This avoids npm, avoids JavaScript frameworks, and keeps the entire dashboard in Python -- consistent with the project's CLI-first, Python-only philosophy.

The containerisation domain has one well-documented community solution: `gnzsnz/ib-gateway-docker` which packages IB Gateway + IBC + Xvfb into a headless Docker image. HYDRA itself is a standard Python application that runs with `uv`. Docker Compose ties the two containers together on a shared network, with SQLite databases on named volumes.

**Primary recommendation:** Use FastAPI + Jinja2 + htmx (CDN) for the dashboard with SSE for live updates. Use `gnzsnz/ib-gateway-docker` for IB Gateway containerisation. Docker Compose orchestrates both services. No Kubernetes, no JavaScript build tools, no SPA frameworks.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | >=0.129.0 | Dashboard HTTP server + API endpoints | Standard Python async web framework; already conceptually compatible with HYDRA's async execution layer |
| Jinja2 | >=3.1.6 | Server-side HTML templating | FastAPI's built-in template support; no npm needed |
| uvicorn | >=0.41.0 | ASGI server to run FastAPI | Standard production server for FastAPI |
| sse-starlette | >=2.0 | Server-Sent Events for live dashboard updates | Production-ready SSE for Starlette/FastAPI; auto-disconnect detection |
| htmx | 2.0.8 (CDN) | Frontend interactivity without JavaScript | Loaded via `<script>` from unpkg CDN; zero build tooling |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| gnzsnz/ib-gateway-docker | stable (10.37.1o) | Headless IB Gateway Docker image | Container deployment; IBC handles login automation |
| Docker Compose | v2+ | Multi-container orchestration | Tying HYDRA + IB Gateway containers together |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| FastAPI + Jinja2 + htmx | Streamlit / Dash / Gradio | Heavier dependencies, less control, non-standard for operational dashboards; Streamlit is data-science-focused not ops-focused |
| FastAPI + Jinja2 + htmx | React/Vue SPA | Requires npm, node, JavaScript build tools -- violates project constraint of no additional ecosystems |
| SSE (sse-starlette) | WebSockets | More complex; SSE is simpler for one-way server-to-client data push (dashboard is read-only) |
| gnzsnz/ib-gateway-docker | waytrade/ib-gateway | gnzsnz is more actively maintained (2025 releases), better documented, supports both live and paper modes |
| Docker Compose | Kubernetes | Explicitly out of scope per project constraints |

**Installation:**
```bash
uv add fastapi uvicorn jinja2 sse-starlette
# htmx loaded from CDN: https://unpkg.com/htmx.org@2.0.8/dist/htmx.min.js
# Docker images pulled at runtime
```

## Architecture Patterns

### Recommended Project Structure
```
src/hydra/
    dashboard/
        __init__.py
        app.py           # FastAPI app factory, mount routes
        routes/
            __init__.py
            pages.py     # HTML page routes (Jinja2 templates)
            api.py       # JSON API endpoints for data
            sse.py       # SSE streaming endpoints
        templates/
            base.html    # Layout: header, nav, htmx CDN script
            index.html   # Overview dashboard
            fills.html   # Fill journal + slippage
            agent.html   # Agent loop status + experiment journal
            drift.html   # Drift monitoring
            system.html  # Broker connection, risk gates, config
        static/
            style.css    # Minimal custom CSS (use system defaults + htmx)
docker/
    Dockerfile           # HYDRA application container
    docker-compose.yml   # HYDRA + IB Gateway orchestration
    .env.example         # Template for secrets
```

### Pattern 1: Read-Only Dashboard over Existing Data Sources
**What:** The dashboard reads from FillJournal (SQLite), ExperimentJournal (SQLite), MLflow registry, and structlog JSON logs. It does NOT write to these stores. The daily cycle runner and agent loop continue to be the only writers.
**When to use:** Always -- the dashboard is a monitoring tool, not a control surface. Control stays in the CLI.
**Rationale:** SQLite WAL mode supports concurrent readers with one writer. Since the dashboard only reads, there is zero contention with the runner process. Both processes share the same SQLite files via volume mounts or filesystem paths.

```python
# Dashboard reads from same FillJournal that runner writes to
from hydra.execution.fill_journal import FillJournal

fj = FillJournal("/data/fill_journal.db")  # read-only consumer
fills = fj.get_fills(symbol="ZO", limit=50)
```

### Pattern 2: SSE for Auto-Refreshing Dashboard Panels
**What:** Instead of polling or full page reloads, use Server-Sent Events to push updates to the browser. The FastAPI SSE endpoint yields new data as JSON. htmx `hx-ext="sse"` on the client listens and swaps HTML fragments.
**When to use:** For the overview panel (latest cycle result), fill count, agent status. Polling interval: 60 seconds is fine for a daily-cycle system.

```python
# SSE endpoint example
from sse_starlette.sse import EventSourceResponse

async def cycle_status_stream(request: Request):
    async def event_generator():
        while True:
            data = get_latest_cycle_summary()  # reads from FillJournal/logs
            yield {"event": "cycle_update", "data": json.dumps(data)}
            await asyncio.sleep(60)
    return EventSourceResponse(event_generator())
```

```html
<!-- htmx SSE listener in template -->
<div hx-ext="sse" sse-connect="/api/sse/cycle-status" sse-swap="cycle_update">
    Loading...
</div>
```

### Pattern 3: Health Check Endpoint for Docker
**What:** A `/health` endpoint that checks: (a) FastAPI is responsive, (b) SQLite databases are accessible, (c) broker connection status is available.
**When to use:** Docker Compose `healthcheck` for the HYDRA container.

```python
@app.get("/health")
async def health():
    checks = {
        "api": "ok",
        "fill_journal": check_sqlite("/data/fill_journal.db"),
        "experiment_journal": check_sqlite("/data/experiment_journal.db"),
    }
    healthy = all(v == "ok" for v in checks.values())
    return JSONResponse(
        content=checks,
        status_code=200 if healthy else 503,
    )
```

### Pattern 4: Docker Compose with Named Volumes
**What:** Two services: `ib-gateway` (from gnzsnz image) and `hydra` (custom Dockerfile). Shared network for API communication. Named volumes for SQLite databases and MLflow artifacts.
**When to use:** Production deployment on any Docker-capable host.

```yaml
# docker-compose.yml skeleton
services:
  ib-gateway:
    image: ghcr.io/gnzsnz/ib-gateway:stable
    environment:
      TWS_USERID: ${TWS_USERID}
      TWS_PASSWORD: ${TWS_PASSWORD}
      TRADING_MODE: paper
    ports:
      - "127.0.0.1:4002:4004"  # paper API
      - "127.0.0.1:5900:5900"  # VNC (optional)
    restart: always

  hydra:
    build: .
    depends_on:
      ib-gateway:
        condition: service_started
    environment:
      IB_GATEWAY_HOST: ib-gateway
      IB_GATEWAY_PORT: 4002
    volumes:
      - hydra-data:/data
      - mlflow-data:/mlflow
    ports:
      - "127.0.0.1:8080:8080"  # dashboard
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"]
      interval: 30s
      timeout: 5s
      retries: 3
    restart: always

volumes:
  hydra-data:   # fill_journal.db, experiment_journal.db, agent_state.json
  mlflow-data:  # MLflow tracking artifacts
```

### Pattern 5: Structured Log Sink for Dashboard
**What:** Configure structlog to write JSON logs to a file (with rotation) in addition to console output. The dashboard reads the JSON log file for recent events display.
**When to use:** Always in container mode. The dashboard's "Recent Events" panel reads the last N lines of the JSON log.

```python
import logging
from logging.handlers import RotatingFileHandler
import structlog

def configure_logging(log_path: str = "/data/logs/hydra.jsonl"):
    handler = RotatingFileHandler(log_path, maxBytes=10_000_000, backBytes=3)
    handler.setLevel(logging.INFO)

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
```

### Anti-Patterns to Avoid
- **Dashboard as control surface:** Do NOT add "start/stop trading", "change parameters", or "promote model" buttons to the dashboard. The CLI remains the only control interface. Dashboard is read-only monitoring.
- **Shared SQLite writers:** Do NOT have the dashboard write to FillJournal or ExperimentJournal. SQLite supports one writer at a time. The runner is the sole writer.
- **Heavy frontend frameworks:** Do NOT introduce React, Vue, Angular, or any npm-based toolchain. htmx from CDN is the maximum client-side complexity.
- **Cross-container SQLite sharing:** Do NOT mount the same SQLite file into multiple containers. The dashboard and runner must run in the SAME container (or use separate read replicas). SQLite WAL does not work reliably across Docker container boundaries with shared volumes in all configurations.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Server-Sent Events | Custom async generator with raw HTTP | sse-starlette | Handles client disconnect, reconnection, keep-alive, and event formatting per W3C spec |
| IB Gateway automation | Custom login scripts + Xvfb | gnzsnz/ib-gateway-docker | Handles IBC, 2FA, Xvfb, reconnection, timezone, VNC -- years of community testing |
| HTML templating | f-strings or string concatenation | Jinja2 via FastAPI | Auto-escaping, template inheritance, macro system |
| Log rotation | Custom file management | logging.handlers.RotatingFileHandler | Thread-safe, configurable size + backup count, standard library |
| Live HTML updates | Custom JavaScript polling loops | htmx `hx-trigger="every 60s"` or SSE extension | 14KB library handles DOM swapping, history, error states |

**Key insight:** The dashboard is a thin read layer over data that already exists. Every metric it displays (fills, slippage, drift, agent results, risk events) is already stored by Phase 1-5 modules. The engineering challenge is wiring, not computation.

## Common Pitfalls

### Pitfall 1: SQLite Locking in Multi-Container Deployments
**What goes wrong:** Two containers (dashboard + runner) both open the same SQLite file via a shared Docker volume. Under concurrent access, WAL mode can produce `database is locked` errors or corrupt the WAL index.
**Why it happens:** SQLite relies on POSIX file locking. Docker named volumes on Linux use overlayfs which supports POSIX locks, but edge cases exist with NFS-backed volumes or cross-platform mounts. More critically, the WAL's shared memory (-shm file) must be accessible to all processes.
**How to avoid:** Run the dashboard and runner in the SAME container as separate processes (e.g., the runner as the main process, the dashboard as a background uvicorn). OR use the dashboard as read-only with `PRAGMA query_only=ON` to ensure it never accidentally writes.
**Warning signs:** Intermittent `OperationalError: database is locked` in logs.

### Pitfall 2: IB Gateway Credential Management
**What goes wrong:** IB credentials end up in docker-compose.yml, committed to git, or visible in `docker inspect`.
**Why it happens:** The gnzsnz image requires TWS_USERID and TWS_PASSWORD environment variables.
**How to avoid:** Use a `.env` file (git-ignored) referenced via `env_file:` in docker-compose.yml. Never put credentials in the compose file directly. Add `.env` to `.gitignore` and provide `.env.example` with placeholder values.
**Warning signs:** Credentials visible in `docker-compose config` output or in version control.

### Pitfall 3: Dashboard Becomes Scope Creep
**What goes wrong:** The "simple monitoring dashboard" grows into a full trading UI with order entry, parameter tuning, model management, and alerting.
**Why it happens:** Once a web interface exists, every feature request gravitates toward it.
**How to avoid:** Enforce read-only dashboard. No POST/PUT/DELETE endpoints that modify system state. All control remains through the Typer CLI. The dashboard has exactly one job: visualize existing data.
**Warning signs:** Feature requests for "add a button to..." in the dashboard.

### Pitfall 4: IB Gateway Port Mapping Confusion
**What goes wrong:** The gnzsnz image maps internal port 4003/4004 to external 4001/4002 via socat. HYDRA tries to connect to port 4002 but the container exposes it on a different host port.
**Why it happens:** Multiple layers of port translation (container internal -> socat -> host mapping).
**How to avoid:** When HYDRA runs in the same Docker network as IB Gateway, connect to hostname `ib-gateway` on internal port 4003 (live) or 4004 (paper). When HYDRA runs on the host, connect to `127.0.0.1:4002` (mapped from container 4004). Document the port mapping clearly.
**Warning signs:** Connection refused or "not a paper account" errors.

### Pitfall 5: Forgetting structlog Configuration
**What goes wrong:** Logs are only visible in the console (dev mode), not written to files. The dashboard's "recent events" panel is empty in production.
**Why it happens:** structlog defaults to console output. File output requires explicit configuration with stdlib integration.
**How to avoid:** Create a `configure_logging()` function called at application startup that sets up both console (dev) and JSON file (production) outputs. Use `RotatingFileHandler` to prevent unbounded log growth.
**Warning signs:** Dashboard shows "No recent events" despite the runner logging extensively.

## Code Examples

### Example 1: FastAPI Dashboard App Factory

```python
# src/hydra/dashboard/app.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

def create_app(data_dir: str = "/data") -> FastAPI:
    app = FastAPI(title="HYDRA Dashboard", docs_url=None, redoc_url=None)

    templates_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"

    templates = Jinja2Templates(directory=str(templates_dir))
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Store data_dir in app state for route access
    app.state.data_dir = data_dir
    app.state.templates = templates

    from hydra.dashboard.routes import pages, api, sse
    app.include_router(pages.router)
    app.include_router(api.router, prefix="/api")
    app.include_router(sse.router, prefix="/api/sse")

    return app
```

### Example 2: Dashboard Overview Page Route

```python
# src/hydra/dashboard/routes/pages.py
from fastapi import APIRouter, Request
from hydra.execution.fill_journal import FillJournal
from hydra.cli.state import get_state

router = APIRouter()

@router.get("/")
async def index(request: Request):
    app = request.app
    fj = FillJournal(f"{app.state.data_dir}/fill_journal.db")
    recent_fills = fj.get_fills(limit=10)
    fill_count = fj.count()
    fj.close()

    agent_state = get_state()

    return app.state.templates.TemplateResponse("index.html", {
        "request": request,
        "fill_count": fill_count,
        "recent_fills": recent_fills,
        "agent_state": agent_state.value,
    })
```

### Example 3: htmx Auto-Refresh Pattern (No SSE)

```html
<!-- Simpler alternative to SSE: htmx polling -->
<div hx-get="/api/fills/summary" hx-trigger="every 60s" hx-swap="innerHTML">
    <!-- Initial content rendered by Jinja2 -->
    <p>Fills: {{ fill_count }} | Agent: {{ agent_state }}</p>
</div>
```

### Example 4: Dockerfile for HYDRA

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install dependencies
RUN uv sync --frozen

# Data directory for SQLite DBs and logs
VOLUME /data

EXPOSE 8080

# Run both the trading runner and dashboard
CMD ["uv", "run", "uvicorn", "hydra.dashboard.app:create_app", \
     "--host", "0.0.0.0", "--port", "8080", "--factory"]
```

### Example 5: Docker Compose Full Configuration

```yaml
version: "3.8"

services:
  ib-gateway:
    image: ghcr.io/gnzsnz/ib-gateway:stable
    restart: always
    env_file: .env
    environment:
      TRADING_MODE: ${TRADING_MODE:-paper}
    ports:
      - "127.0.0.1:4002:4004"
      - "127.0.0.1:5900:5900"

  hydra:
    build:
      context: .
      dockerfile: docker/Dockerfile
    restart: always
    depends_on:
      - ib-gateway
    env_file: .env
    environment:
      IB_GATEWAY_HOST: ib-gateway
      IB_GATEWAY_PORT: 4004
      HYDRA_DATA_DIR: /data
    volumes:
      - hydra-data:/data
    ports:
      - "127.0.0.1:8080:8080"
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"]
      interval: 30s
      timeout: 5s
      retries: 3

volumes:
  hydra-data:
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Grafana + Prometheus for monitoring | FastAPI + htmx for lightweight self-hosted dashboards | 2024-2025 | Eliminates heavy monitoring stack for single-app systems |
| ib_insync for IB connectivity | ib_async 2.1.0 (community fork) | 2024 | Already adopted in Phase 5; containerisation uses same library |
| Custom IB login automation | gnzsnz/ib-gateway-docker with IBC | 2023-2025 | Community-maintained, handles 2FA, Xvfb, reconnection |
| Multi-process supervisord in containers | Single-process containers with Docker Compose | 2024-2025 | Docker best practice: one concern per container |
| Custom JavaScript dashboards | htmx 2.0 (no build step) | 2024 | Zero npm, zero bundling; interactivity via HTML attributes |

**Deprecated/outdated:**
- **ib_insync**: Unmaintained; use ib_async instead (already done in Phase 5)
- **waytrade/ib-gateway-docker**: Less actively maintained than gnzsnz version
- **htmx 1.x**: htmx 2.0 has breaking changes from 1.x (attribute naming); use 2.0.8+

## Open Questions

1. **Single container vs two containers for HYDRA?**
   - What we know: SQLite works best in single-container mode. The runner and dashboard both need access to the same SQLite files.
   - What's unclear: Whether to run runner + dashboard in one container (simpler SQLite access, but violates "one process per container") or in two containers with a shared named volume (Docker best practice, but SQLite locking risk).
   - Recommendation: **Run both in one container.** Use uvicorn for the dashboard and the APScheduler runner as a background task within the same Python process. This is the safest approach for SQLite WAL mode and avoids cross-container file locking issues entirely. The "one process per container" rule is a guideline, not a law -- and it yields to data integrity concerns.

2. **Dashboard authentication**
   - What we know: This is a personal quant tool (per PROJECT.md: "Consumer-facing features -- this is a personal quant tool"). The dashboard binds to localhost only (`127.0.0.1:8080`).
   - What's unclear: Whether any authentication is needed if always accessed via SSH tunnel or localhost.
   - Recommendation: **No authentication for v1.** Bind to localhost only. If remote access is needed later, use SSH tunneling (same approach as IB Gateway). Add basic auth as a future enhancement only if actually needed.

3. **How to run dashboard + runner in same process**
   - What we know: The runner uses APScheduler AsyncIOScheduler. FastAPI/uvicorn also runs on asyncio.
   - What's unclear: Exact integration pattern.
   - Recommendation: Start the PaperTradingRunner as a FastAPI startup event handler (`@app.on_event("startup")`). Both share the same asyncio event loop. The runner's APScheduler fires the daily cycle while uvicorn serves the dashboard. Alternatively, use FastAPI `lifespan` context manager (newer pattern).

4. **Alerting beyond dashboard display**
   - What we know: The PAPER_TRADING_PLAN.md defines daily and weekly checks. Currently these are manual CLI commands.
   - What's unclear: Whether to add email/Slack alerting in Phase 6 or defer.
   - Recommendation: **Defer alerting to a future phase.** Phase 6 focuses on making data visible via the dashboard. The dashboard replaces the manual `hydra status` / `hydra fill-report` workflow. Push notifications are a separate concern.

## Sources

### Primary (HIGH confidence)
- HYDRA codebase: `src/hydra/execution/runner.py`, `fill_journal.py`, `reconciler.py`, `broker.py`, `risk_gate.py` -- examined for data sources the dashboard will expose
- HYDRA codebase: `src/hydra/cli/app.py`, `formatters.py` -- examined for existing data rendering patterns to replicate in dashboard
- HYDRA codebase: `src/hydra/agent/loop.py`, `sandbox/observer.py` -- examined for agent status and drift data
- HYDRA codebase: `src/hydra/sandbox/journal.py` -- ExperimentJournal data model
- gnzsnz/ib-gateway-docker: https://github.com/gnzsnz/ib-gateway-docker -- Docker image, compose templates, environment variables
- FastAPI official docs: https://fastapi.tiangolo.com/advanced/templates/ -- Jinja2 template integration
- sse-starlette: https://pypi.org/project/sse-starlette/ -- SSE implementation for FastAPI
- htmx: https://htmx.org/ -- Frontend interactivity library, CDN distribution
- structlog: https://www.structlog.org/en/stable/logging-best-practices.html -- JSON logging configuration

### Secondary (MEDIUM confidence)
- Docker Compose health checks: https://last9.io/blog/docker-compose-health-checks/ -- healthcheck configuration patterns
- SQLite in Docker: https://oneuptime.com/blog/post/2026-02-08-how-to-run-sqlite-in-docker-when-and-how/view -- WAL mode container guidance
- FastAPI + htmx patterns: https://testdriven.io/blog/fastapi-htmx/ -- integration patterns

### Tertiary (LOW confidence)
- htmx 2.0.8 version: Inferred from unpkg listing; exact latest version may differ at plan time
- FastAPI 0.129.0: From PyPI search results; pin to whatever is current at build time
- uvicorn 0.41.0: From PyPI search results; same caveat

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - FastAPI, Jinja2, htmx, Docker Compose are all mature, well-documented technologies. No novel integration patterns required.
- Architecture: MEDIUM-HIGH - The read-only dashboard over existing SQLite stores is straightforward. The single-container decision for runner+dashboard is a pragmatic tradeoff backed by SQLite WAL constraints.
- Pitfalls: HIGH - SQLite locking, credential management, port mapping confusion are well-documented issues with clear mitigations.
- Containerisation: MEDIUM - gnzsnz/ib-gateway-docker is community-maintained and well-documented, but IB Gateway automation inherently has edge cases (2FA, session timeouts, API changes) that require operational vigilance.

**Research date:** 2026-02-19
**Valid until:** 2026-03-19 (30 days -- all technologies involved are stable and slow-moving)
