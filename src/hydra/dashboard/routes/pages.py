"""HTML page routes for the HYDRA dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/")
async def index(request: Request):
    """Render the overview page."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "base.html", {"request": request, "page_title": "Overview"}
    )
