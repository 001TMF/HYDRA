"""FastAPI dashboard application factory.

Creates the HYDRA monitoring dashboard with route registration,
Jinja2 template rendering, and static file serving.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from hydra.dashboard.routes import api, pages, sse


def create_app(data_dir: str = "~/.hydra") -> FastAPI:
    """Create and configure the HYDRA Dashboard FastAPI application.

    Parameters
    ----------
    data_dir : str
        Path to the HYDRA data directory containing SQLite databases.
        Defaults to ``~/.hydra``. Tilde is expanded.

    Returns
    -------
    FastAPI
        Configured application instance.
    """
    app = FastAPI(title="HYDRA Dashboard", docs_url=None, redoc_url=None)

    app.state.data_dir = Path(data_dir).expanduser()
    app.state.templates = Jinja2Templates(
        directory=str(Path(__file__).parent / "templates")
    )

    app.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).parent / "static")),
        name="static",
    )

    app.include_router(pages.router)
    app.include_router(api.router, prefix="/api")
    app.include_router(sse.router, prefix="/api/sse")

    return app
