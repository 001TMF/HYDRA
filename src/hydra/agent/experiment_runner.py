"""Subprocess-isolated candidate training with configurable timeout.

The ``ExperimentRunner`` executes candidate model training in a separate
process to prevent memory leaks, crashes, or runaway computation from
affecting the main agent loop.  It communicates via a temporary JSON config
file and reads structured JSON results from subprocess stdout.

Architecture::

    Agent Loop  --->  ExperimentRunner.run(hypothesis, base_config)
                          |
                          v
                      subprocess.run(
                          python -m hydra.agent._train_candidate <config.json>,
                          timeout=N
                      )
                          |
                          v
                      Parse stdout JSON -> ExperimentResult

Exports:
    - ``ExperimentRunner``: Main class for running experiments.
    - ``ExperimentResult``: Structured outcome of an experiment.
    - ``ExperimentError``: Exception for experiment failures.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import structlog

from hydra.agent.types import Hypothesis

logger = structlog.get_logger()

# Pattern for expressions like "current * 0.5", "current + 10", "current / 2"
_EXPRESSION_RE = re.compile(
    r"^current\s*([+\-*/])\s*([0-9.]+)$"
)


class ExperimentError(Exception):
    """Raised on timeout, crash, or invalid output from subprocess."""


@dataclass
class ExperimentResult:
    """Structured outcome of a candidate training experiment.

    Attributes
    ----------
    success : bool
        Whether the experiment completed without errors.
    fitness_score : float | None
        Composite fitness score from evaluation, or None on failure.
    metrics : dict
        All metrics from evaluation.
    config_used : dict
        The merged config that was used for training.
    duration_seconds : float
        Wall-clock time of the experiment.
    error_message : str | None
        Error description on failure, None on success.
    stdout : str
        Raw stdout from the subprocess.
    stderr : str
        Raw stderr from the subprocess.
    """

    success: bool
    fitness_score: float | None
    metrics: dict
    config_used: dict
    duration_seconds: float
    error_message: str | None
    stdout: str
    stderr: str


class ExperimentRunner:
    """Execute candidate training in an isolated subprocess.

    Parameters
    ----------
    timeout_seconds : int
        Maximum wall-clock seconds for a training run.
    python_executable : str | None
        Path to the Python interpreter. Defaults to ``sys.executable``.
    """

    def __init__(
        self,
        timeout_seconds: int = 300,
        python_executable: str | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable or sys.executable

    def run(self, hypothesis: Hypothesis, base_config: dict) -> ExperimentResult:
        """Train a candidate model in a subprocess.

        Parameters
        ----------
        hypothesis : Hypothesis
            The mutation hypothesis to test.
        base_config : dict
            Champion model configuration to apply the diff on top of.

        Returns
        -------
        ExperimentResult
            Structured result including success/failure, metrics, timing.
        """
        merged_config = self._merge_config(base_config, hypothesis.config_diff)
        start = time.monotonic()

        # Write config to temp file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        config_path = tmp.name
        try:
            json.dump(merged_config, tmp)
            tmp.close()

            completed = subprocess.run(
                [
                    self.python_executable,
                    "-m",
                    "hydra.agent._train_candidate",
                    config_path,
                ],
                timeout=self.timeout_seconds,
                capture_output=True,
                text=True,
            )
            elapsed = time.monotonic() - start

            # Parse stdout JSON
            try:
                output = json.loads(completed.stdout)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "experiment_output_parse_failed",
                    stdout=completed.stdout[:500],
                    error=str(exc),
                )
                return ExperimentResult(
                    success=False,
                    fitness_score=None,
                    metrics={},
                    config_used=merged_config,
                    duration_seconds=elapsed,
                    error_message=f"Failed to parse subprocess output as JSON: {exc}",
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )

            return ExperimentResult(
                success=output.get("success", False),
                fitness_score=output.get("fitness_score"),
                metrics=output.get("metrics", {}),
                config_used=output.get("config_used", merged_config),
                duration_seconds=elapsed,
                error_message=output.get("error"),
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            msg = f"Training timed out after {self.timeout_seconds} seconds"
            logger.warning("experiment_timeout", timeout=self.timeout_seconds)
            return ExperimentResult(
                success=False,
                fitness_score=None,
                metrics={},
                config_used=merged_config,
                duration_seconds=elapsed,
                error_message=msg,
                stdout="",
                stderr="",
            )

        except subprocess.CalledProcessError as exc:
            elapsed = time.monotonic() - start
            logger.warning(
                "experiment_crash",
                returncode=exc.returncode,
                stderr=(exc.stderr or "")[:500],
            )
            return ExperimentResult(
                success=False,
                fitness_score=None,
                metrics={},
                config_used=merged_config,
                duration_seconds=elapsed,
                error_message=f"Subprocess exited with code {exc.returncode}",
                stdout=exc.output or "",
                stderr=exc.stderr or "",
            )

        finally:
            # Always clean up temp file
            try:
                os.unlink(config_path)
            except OSError:
                pass

    def _merge_config(self, base: dict, diff: dict) -> dict:
        """Deep-merge diff into base, resolving expressions.

        For numeric expressions like ``"current * 0.5"``, the ``current``
        token is replaced with the base value and the expression is
        evaluated.  Falls back to the raw diff value if the expression
        cannot be resolved (e.g., base value is non-numeric).

        Parameters
        ----------
        base : dict
            Base configuration.
        diff : dict
            Configuration diff to apply.

        Returns
        -------
        dict
            Merged configuration.
        """
        result = copy.deepcopy(base)

        for key, value in diff.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = self._merge_config(result[key], value)
            elif isinstance(value, str):
                result[key] = self._resolve_expression(value, result.get(key))
            else:
                result[key] = value

        return result

    def _resolve_expression(self, expr: str, current_value: object) -> object:
        """Resolve a config expression like 'current * 0.5'.

        Parameters
        ----------
        expr : str
            The expression string.
        current_value : object
            The current base value for ``current`` token.

        Returns
        -------
        object
            Resolved numeric value, or the raw expression string if
            resolution is not possible.
        """
        match = _EXPRESSION_RE.match(expr.strip())
        if match is None:
            return expr

        operator = match.group(1)
        operand = float(match.group(2))

        if not isinstance(current_value, (int, float)):
            return expr

        current = float(current_value)

        if operator == "*":
            return current * operand
        elif operator == "+":
            return current + operand
        elif operator == "-":
            return current - operand
        elif operator == "/":
            if operand == 0:
                return expr
            return current / operand

        return expr
