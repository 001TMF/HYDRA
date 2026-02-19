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
    data_dir = request.app.state.data_dir

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            from hydra.cli.state import get_state

            fill_count = 0
            fill_db = data_dir / "fill_journal.db"
            if fill_db.exists():
                fj = None
                try:
                    from hydra.execution.fill_journal import FillJournal

                    fj = FillJournal(fill_db)
                    fill_count = fj.count()
                except Exception:
                    pass
                finally:
                    if fj is not None:
                        fj.close()

            data = json.dumps(
                {
                    "agent_state": get_state().value,
                    "fill_count": fill_count,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            yield {"event": "cycle_update", "data": data}
            await asyncio.sleep(60)

    return EventSourceResponse(event_generator())
