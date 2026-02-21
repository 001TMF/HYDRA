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

            experiment_count = 0
            exp_db = data_dir / "experiment_journal.db"
            if exp_db.exists():
                ej = None
                try:
                    from hydra.sandbox.journal import ExperimentJournal

                    ej = ExperimentJournal(exp_db)
                    experiment_count = ej.count()
                except Exception:
                    pass
                finally:
                    if ej is not None:
                        ej.close()

            champion_version = None
            try:
                from hydra.sandbox.registry import ModelRegistry

                reg = ModelRegistry()
                info = reg.get_champion_info()
                champion_version = info.get("version")
            except Exception:
                pass

            data = json.dumps(
                {
                    "agent_state": get_state().value,
                    "fill_count": fill_count,
                    "experiment_count": experiment_count,
                    "champion_version": champion_version,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            yield {"event": "cycle_update", "data": data}
            await asyncio.sleep(60)

    return EventSourceResponse(event_generator())
