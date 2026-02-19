"""Agent loop state management via JSON file.

The autonomous agent loop checks this state file on each iteration.
``PAUSED`` is the safe default -- if the state file doesn't exist,
the agent loop will not execute.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class AgentState(Enum):
    """Agent loop states."""

    RUNNING = "running"
    PAUSED = "paused"


STATE_FILE: Path = Path.home() / ".hydra" / "agent_state.json"


def get_state() -> AgentState:
    """Read the current agent loop state.

    Returns ``AgentState.PAUSED`` if the state file does not exist
    (safe default -- agent won't run without explicit activation).
    """
    if not STATE_FILE.exists():
        return AgentState.PAUSED
    try:
        data = json.loads(STATE_FILE.read_text())
        return AgentState(data["state"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return AgentState.PAUSED


def set_state(state: AgentState) -> None:
    """Write the agent loop state to disk.

    Creates the parent directory if it doesn't exist.
    """
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state.value,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    STATE_FILE.write_text(json.dumps(payload, indent=2))
