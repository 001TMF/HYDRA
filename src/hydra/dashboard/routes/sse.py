"""SSE streaming endpoint for live dashboard updates."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

router = APIRouter()


@router.get("/cycle-status")
async def cycle_status(request: Request):
    """Stream agent cycle status updates every 60 seconds."""

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            from hydra.cli.state import get_state

            data = json.dumps(
                {
                    "agent_state": get_state().value,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            yield {"event": "cycle_update", "data": data}
            await asyncio.sleep(60)

    return EventSourceResponse(event_generator())
