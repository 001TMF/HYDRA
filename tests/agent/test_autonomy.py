"""Tests for autonomy level gating system (AGNT-06).

Verifies that AutonomyLevel enum, permission checking, and action gating
work correctly across all autonomy levels.
"""

import pytest

from hydra.agent.autonomy import (
    AutonomyLevel,
    PermissionDeniedError,
    check_permission,
    get_allowed_actions,
    require_permission,
)


# ---------------------------------------------------------------------------
# AutonomyLevel enum
# ---------------------------------------------------------------------------

class TestAutonomyLevel:
    def test_levels_are_ordered(self):
        assert AutonomyLevel.LOCKDOWN < AutonomyLevel.SUPERVISED
        assert AutonomyLevel.SUPERVISED < AutonomyLevel.SEMI_AUTO
        assert AutonomyLevel.SEMI_AUTO < AutonomyLevel.AUTONOMOUS

    def test_level_values(self):
        assert AutonomyLevel.LOCKDOWN == 0
        assert AutonomyLevel.SUPERVISED == 1
        assert AutonomyLevel.SEMI_AUTO == 2
        assert AutonomyLevel.AUTONOMOUS == 3


# ---------------------------------------------------------------------------
# LOCKDOWN blocks everything
# ---------------------------------------------------------------------------

class TestLockdown:
    @pytest.mark.parametrize("action", [
        "observe", "diagnose", "hypothesize", "experiment", "promote", "rollback",
    ])
    def test_lockdown_blocks_all_actions(self, action: str):
        assert check_permission(action, AutonomyLevel.LOCKDOWN) is False

    @pytest.mark.parametrize("action", [
        "observe", "diagnose", "hypothesize", "experiment", "promote", "rollback",
    ])
    def test_lockdown_require_raises(self, action: str):
        with pytest.raises(PermissionDeniedError):
            require_permission(action, AutonomyLevel.LOCKDOWN)

    def test_lockdown_allowed_actions_empty(self):
        assert get_allowed_actions(AutonomyLevel.LOCKDOWN) == []


# ---------------------------------------------------------------------------
# SUPERVISED allows observe/diagnose/hypothesize/rollback, blocks experiment/promote
# ---------------------------------------------------------------------------

class TestSupervised:
    @pytest.mark.parametrize("action", [
        "observe", "diagnose", "hypothesize", "rollback",
    ])
    def test_supervised_allows(self, action: str):
        assert check_permission(action, AutonomyLevel.SUPERVISED) is True

    @pytest.mark.parametrize("action", ["experiment", "promote"])
    def test_supervised_blocks(self, action: str):
        assert check_permission(action, AutonomyLevel.SUPERVISED) is False

    @pytest.mark.parametrize("action", ["experiment", "promote"])
    def test_supervised_require_raises(self, action: str):
        with pytest.raises(PermissionDeniedError):
            require_permission(action, AutonomyLevel.SUPERVISED)

    def test_supervised_allowed_actions(self):
        allowed = get_allowed_actions(AutonomyLevel.SUPERVISED)
        assert set(allowed) == {"observe", "diagnose", "hypothesize", "rollback"}


# ---------------------------------------------------------------------------
# SEMI_AUTO allows experiment, blocks promote
# ---------------------------------------------------------------------------

class TestSemiAuto:
    @pytest.mark.parametrize("action", [
        "observe", "diagnose", "hypothesize", "experiment", "rollback",
    ])
    def test_semi_auto_allows(self, action: str):
        assert check_permission(action, AutonomyLevel.SEMI_AUTO) is True

    def test_semi_auto_blocks_promote(self):
        assert check_permission("promote", AutonomyLevel.SEMI_AUTO) is False

    def test_semi_auto_require_promote_raises(self):
        with pytest.raises(PermissionDeniedError):
            require_permission("promote", AutonomyLevel.SEMI_AUTO)

    def test_semi_auto_allowed_actions(self):
        allowed = get_allowed_actions(AutonomyLevel.SEMI_AUTO)
        assert set(allowed) == {
            "observe", "diagnose", "hypothesize", "experiment", "rollback",
        }


# ---------------------------------------------------------------------------
# AUTONOMOUS allows everything
# ---------------------------------------------------------------------------

class TestAutonomous:
    @pytest.mark.parametrize("action", [
        "observe", "diagnose", "hypothesize", "experiment", "promote", "rollback",
    ])
    def test_autonomous_allows_all(self, action: str):
        assert check_permission(action, AutonomyLevel.AUTONOMOUS) is True

    @pytest.mark.parametrize("action", [
        "observe", "diagnose", "hypothesize", "experiment", "promote", "rollback",
    ])
    def test_autonomous_require_does_not_raise(self, action: str):
        require_permission(action, AutonomyLevel.AUTONOMOUS)  # should not raise

    def test_autonomous_allowed_actions(self):
        allowed = get_allowed_actions(AutonomyLevel.AUTONOMOUS)
        assert set(allowed) == {
            "observe", "diagnose", "hypothesize", "experiment", "promote", "rollback",
        }


# ---------------------------------------------------------------------------
# PermissionDeniedError message quality
# ---------------------------------------------------------------------------

class TestPermissionDeniedError:
    def test_error_includes_action_name(self):
        with pytest.raises(PermissionDeniedError, match="experiment"):
            require_permission("experiment", AutonomyLevel.SUPERVISED)

    def test_error_includes_level_name(self):
        with pytest.raises(PermissionDeniedError, match="SUPERVISED"):
            require_permission("experiment", AutonomyLevel.SUPERVISED)

    def test_is_exception_subclass(self):
        assert issubclass(PermissionDeniedError, Exception)


# ---------------------------------------------------------------------------
# Unknown action handling
# ---------------------------------------------------------------------------

class TestUnknownAction:
    def test_unknown_action_returns_false(self):
        assert check_permission("unknown_action", AutonomyLevel.AUTONOMOUS) is False

    def test_unknown_action_require_raises(self):
        with pytest.raises(PermissionDeniedError):
            require_permission("unknown_action", AutonomyLevel.AUTONOMOUS)

    def test_unknown_action_does_not_raise_key_error(self):
        """Unknown action should return False, not raise KeyError."""
        try:
            result = check_permission("nonexistent", AutonomyLevel.AUTONOMOUS)
            assert result is False
        except KeyError:
            pytest.fail("check_permission raised KeyError for unknown action")
