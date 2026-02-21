FROM python:3.11-slim AS base

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies only (project not yet available)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ ./src/

# Install the project itself (deps already cached)
RUN uv sync --frozen --no-dev

# Create data directory for SQLite DBs and logs
RUN mkdir -p /data/logs

VOLUME /data

EXPOSE 8080

# Default: run dashboard via uvicorn (runner starts as lifespan event)
CMD ["uv", "run", "uvicorn", "hydra.dashboard.app:create_app", \
     "--host", "0.0.0.0", "--port", "8080", "--factory"]
