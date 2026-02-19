"""Subprocess entry point for candidate model training.

This module is invoked by ``ExperimentRunner.run()`` as::

    subprocess.run([python, "-m", "hydra.agent._train_candidate", config_path])

It reads a JSON config from the path in argv[1], runs (stub) training,
and prints a JSON result to stdout. The runner parses this stdout JSON.

**This is a STUB.** Full training integration wiring to
``BaselineModel`` + ``MarketReplayEngine`` is deferred to Plan 04-05
(agent loop assembly). The stub exists so subprocess calls have a real
target during testing.
"""

from __future__ import annotations

import json
import sys
import time


def main() -> None:
    """Read config, run stub training, print JSON result to stdout."""
    if len(sys.argv) < 2:
        print(
            json.dumps({"success": False, "error": "No config path provided"}),
            flush=True,
        )
        sys.exit(1)

    config_path = sys.argv[1]

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(
            json.dumps({"success": False, "error": f"Config load failed: {exc}"}),
            flush=True,
        )
        sys.exit(1)

    # --- STUB: Simulate training ---
    start = time.monotonic()
    # In production, this will:
    #   1. Build BaselineModel from config
    #   2. Train on designated data window
    #   3. Evaluate via MarketReplayEngine
    #   4. Return real fitness and metrics
    elapsed = time.monotonic() - start

    result = {
        "success": True,
        "fitness_score": 0.65,
        "metrics": {
            "sharpe_ratio": 0.8,
            "max_drawdown": -0.12,
            "win_rate": 0.52,
        },
        "config_used": config,
        "duration_seconds": round(elapsed, 4),
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
