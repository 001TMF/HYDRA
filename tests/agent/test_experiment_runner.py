"""Tests for experiment runner with subprocess isolation and timeout.

Verifies:
- Successful run produces ExperimentResult(success=True)
- Timeout produces ExperimentResult(success=False, error_message about timeout)
- Crash produces ExperimentResult(success=False) with stderr
- Invalid JSON output produces ExperimentResult(success=False) with parse error
- Config merging with direct values
- Config merging with expression resolution (e.g., "current * 0.5")
- Temp file cleanup on success and failure
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from hydra.agent.experiment_runner import (
    ExperimentError,
    ExperimentResult,
    ExperimentRunner,
)
from hydra.agent.types import Hypothesis, MutationType


@pytest.fixture
def runner() -> ExperimentRunner:
    """Create a default ExperimentRunner."""
    return ExperimentRunner(timeout_seconds=30)


@pytest.fixture
def sample_hypothesis() -> Hypothesis:
    """Create a sample hypothesis for testing."""
    return Hypothesis(
        mutation_type=MutationType.HYPERPARAMETER,
        description="Reduce learning rate by half",
        config_diff={"lr": 0.05},
        expected_impact="Better convergence",
        testable_prediction="Fitness improves by 5%",
        source="playbook",
    )


@pytest.fixture
def base_config() -> dict:
    """A minimal base config for testing."""
    return {"lr": 0.1, "num_leaves": 31, "n_estimators": 100}


class TestExperimentRunnerRun:
    """Test the run() method with the real _train_candidate stub."""

    def test_successful_run(
        self, runner: ExperimentRunner, sample_hypothesis: Hypothesis, base_config: dict
    ) -> None:
        """Running with the stub should return ExperimentResult(success=True)."""
        result = runner.run(sample_hypothesis, base_config)

        assert isinstance(result, ExperimentResult)
        assert result.success is True
        assert result.fitness_score is not None
        assert isinstance(result.metrics, dict)
        assert isinstance(result.config_used, dict)
        assert result.duration_seconds >= 0
        assert result.error_message is None

    def test_timeout_handling(
        self, sample_hypothesis: Hypothesis, base_config: dict
    ) -> None:
        """Timeout should produce ExperimentResult(success=False) with timeout message."""
        runner = ExperimentRunner(timeout_seconds=30)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["python", "-m", "hydra.agent._train_candidate", "/tmp/cfg.json"],
                timeout=30,
            )
            result = runner.run(sample_hypothesis, base_config)

        assert result.success is False
        assert "timed out" in result.error_message.lower()
        assert "30" in result.error_message

    def test_crash_handling(
        self, runner: ExperimentRunner, sample_hypothesis: Hypothesis, base_config: dict
    ) -> None:
        """Subprocess crash should produce ExperimentResult(success=False) with stderr."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["python", "-m", "hydra.agent._train_candidate", "/tmp/cfg.json"],
                output="",
                stderr="Segmentation fault",
            )
            result = runner.run(sample_hypothesis, base_config)

        assert result.success is False
        assert "Segmentation fault" in result.stderr

    def test_invalid_json_output(
        self, runner: ExperimentRunner, sample_hypothesis: Hypothesis, base_config: dict
    ) -> None:
        """Non-JSON stdout should produce ExperimentResult(success=False)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["python"],
                returncode=0,
                stdout="This is not JSON",
                stderr="",
            )
            result = runner.run(sample_hypothesis, base_config)

        assert result.success is False
        assert result.error_message is not None
        assert "parse" in result.error_message.lower() or "json" in result.error_message.lower()

    def test_temp_file_cleanup_on_success(
        self, runner: ExperimentRunner, sample_hypothesis: Hypothesis, base_config: dict
    ) -> None:
        """Temp config file should be deleted after a successful run."""
        created_files: list[str] = []
        original_named = tempfile.NamedTemporaryFile

        # Track temp file creation
        with patch("hydra.agent.experiment_runner.tempfile") as mock_tempfile:
            real_tmp = original_named(
                mode="w", suffix=".json", delete=False
            )
            created_files.append(real_tmp.name)
            mock_tempfile.NamedTemporaryFile.return_value = real_tmp

            result = runner.run(sample_hypothesis, base_config)

        # File should have been cleaned up
        for f in created_files:
            assert not os.path.exists(f), f"Temp file {f} was not cleaned up"

    def test_temp_file_cleanup_on_failure(
        self, runner: ExperimentRunner, sample_hypothesis: Hypothesis, base_config: dict
    ) -> None:
        """Temp config file should be deleted even when subprocess fails."""
        created_files: list[str] = []
        original_named = tempfile.NamedTemporaryFile

        with patch("hydra.agent.experiment_runner.tempfile") as mock_tempfile:
            real_tmp = original_named(
                mode="w", suffix=".json", delete=False
            )
            created_files.append(real_tmp.name)
            mock_tempfile.NamedTemporaryFile.return_value = real_tmp

            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired(
                    cmd=["python"], timeout=30
                )
                result = runner.run(sample_hypothesis, base_config)

        for f in created_files:
            assert not os.path.exists(f), f"Temp file {f} was not cleaned up after failure"


class TestMergeConfig:
    """Test config merging with direct values and expressions."""

    def test_direct_value_override(self, runner: ExperimentRunner) -> None:
        """Direct value in diff replaces base value."""
        base = {"lr": 0.1}
        diff = {"lr": 0.05}
        merged = runner._merge_config(base, diff)
        assert merged["lr"] == 0.05

    def test_expression_multiply(self, runner: ExperimentRunner) -> None:
        """Expression 'current * 0.5' resolves using base value."""
        base = {"lr": 0.1}
        diff = {"lr": "current * 0.5"}
        merged = runner._merge_config(base, diff)
        assert abs(merged["lr"] - 0.05) < 1e-10

    def test_nested_merge(self, runner: ExperimentRunner) -> None:
        """Nested dicts are merged recursively."""
        base = {"model": {"lr": 0.1, "layers": 3}}
        diff = {"model": {"lr": 0.05}}
        merged = runner._merge_config(base, diff)
        assert merged["model"]["lr"] == 0.05
        assert merged["model"]["layers"] == 3

    def test_new_key_added(self, runner: ExperimentRunner) -> None:
        """Keys in diff not in base are added."""
        base = {"lr": 0.1}
        diff = {"dropout": 0.3}
        merged = runner._merge_config(base, diff)
        assert merged["lr"] == 0.1
        assert merged["dropout"] == 0.3

    def test_unresolvable_expression_passthrough(self, runner: ExperimentRunner) -> None:
        """Unresolvable expressions fall back to raw string."""
        base = {"mode": "train"}
        diff = {"mode": "current * 2"}  # 'train' * 2 doesn't make sense
        merged = runner._merge_config(base, diff)
        assert merged["mode"] == "current * 2"

    def test_expression_with_addition(self, runner: ExperimentRunner) -> None:
        """Expression 'current + 10' resolves using base value."""
        base = {"n_estimators": 100}
        diff = {"n_estimators": "current + 50"}
        merged = runner._merge_config(base, diff)
        assert merged["n_estimators"] == 150

    def test_expression_with_division(self, runner: ExperimentRunner) -> None:
        """Expression 'current / 2' resolves using base value."""
        base = {"num_leaves": 31}
        diff = {"num_leaves": "current / 2"}
        merged = runner._merge_config(base, diff)
        assert abs(merged["num_leaves"] - 15.5) < 1e-10


class TestExperimentResult:
    """Test ExperimentResult dataclass."""

    def test_default_values(self) -> None:
        """Default ExperimentResult should be constructable."""
        result = ExperimentResult(
            success=True,
            fitness_score=0.75,
            metrics={"sharpe": 1.2},
            config_used={"lr": 0.1},
            duration_seconds=5.0,
            error_message=None,
            stdout="{}",
            stderr="",
        )
        assert result.success is True
        assert result.fitness_score == 0.75


class TestExperimentError:
    """Test ExperimentError exception."""

    def test_is_exception(self) -> None:
        """ExperimentError should be an Exception subclass."""
        assert issubclass(ExperimentError, Exception)

    def test_message(self) -> None:
        """ExperimentError should carry a message."""
        err = ExperimentError("training timed out")
        assert str(err) == "training timed out"
