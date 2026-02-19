FROM python:3.11-slim AS base

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies (no dev group)
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/

# Create data directory for SQLite DBs and logs
RUN mkdir -p /data/logs

VOLUME /data

EXPOSE 8080

# Default: run dashboard via uvicorn (runner starts as lifespan event)
CMD ["uv", "run", "uvicorn", "hydra.dashboard.app:create_app", \
     "--host", "0.0.0.0", "--port", "8080", "--factory"]
