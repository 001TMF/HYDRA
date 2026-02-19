"""Autonomy level gating for the HYDRA agent loop (AGNT-06).

Four-level autonomy system that gates which actions the agent is allowed
to perform. Used as a guard at the start of each agent loop step to
prevent unsafe operations at lower autonomy levels.

Levels:
    LOCKDOWN (0)   -- All actions blocked. Emergency stop.
    SUPERVISED (1) -- Observe, diagnose, hypothesize, rollback only.
    SEMI_AUTO (2)  -- Adds experiment capability.
    AUTONOMOUS (3) -- Full autonomy including model promotion.
"""

from __future__ import annotations

from enum import IntEnum


class AutonomyLevel(IntEnum):
    """Agent autonomy levels, ordered from most restricted to fully autonomous."""

    LOCKDOWN = 0
    SUPERVISED = 1
    SEMI_AUTO = 2
    AUTONOMOUS = 3


class PermissionDeniedError(Exception):
    """Raised when an action is blocked by the current autonomy level."""

    def __init__(self, action: str, level: AutonomyLevel) -> None:
        self.action = action
        self.level = level
        super().__init__(
            f"Action '{action}' is not permitted at autonomy level "
            f"{level.name} ({level.value})"
        )


# Mapping from action name to the minimum autonomy level required.
PERMISSIONS: dict[str, AutonomyLevel] = {
    "observe": AutonomyLevel.SUPERVISED,
    "diagnose": AutonomyLevel.SUPERVISED,
    "hypothesize": AutonomyLevel.SUPERVISED,
    "experiment": AutonomyLevel.SEMI_AUTO,
    "promote": AutonomyLevel.AUTONOMOUS,
    "rollback": AutonomyLevel.SUPERVISED,
}


def check_permission(action: str, level: AutonomyLevel) -> bool:
    """Return True if *action* is allowed at the given autonomy *level*.

    Unknown actions return False (never raise KeyError).
    """
    required = PERMISSIONS.get(action)
    if required is None:
        return False
    return level >= required


def require_permission(action: str, level: AutonomyLevel) -> None:
    """Raise :class:`PermissionDeniedError` if *action* is not allowed.

    Intended as a guard at the start of agent loop steps::

        require_permission("experiment", current_level)
        # ... proceed with experiment ...
    """
    if not check_permission(action, level):
        raise PermissionDeniedError(action, level)


def get_allowed_actions(level: AutonomyLevel) -> list[str]:
    """Return all actions permitted at the given autonomy *level*.

    Returns a sorted list for deterministic output.
    """
    return sorted(
        action for action, required in PERMISSIONS.items() if level >= required
    )
