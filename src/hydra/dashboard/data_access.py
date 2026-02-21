"""Centralized data access for HYDRA dashboard routes.

Provides safe helpers that return None when data sources are unavailable,
enabling graceful degradation across all dashboard pages.
"""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class DashboardData:
    """Centralized data access for dashboard routes."""

    def __init__(self, data_dir: Path, runner=None):
        self.data_dir = data_dir
        self.runner = runner

    # --- SQLite-based (always available if DB exists) ---

    def get_fill_journal(self):
        """Open FillJournal, return instance or None."""
        fill_db = self.data_dir / "fill_journal.db"
        if not fill_db.exists():
            return None
        try:
            from hydra.execution.fill_journal import FillJournal
            return FillJournal(fill_db)
        except Exception:
            return None

    def get_experiment_journal(self):
        """Open ExperimentJournal, return instance or None."""
        exp_db = self.data_dir / "experiment_journal.db"
        if not exp_db.exists():
            return None
        try:
            from hydra.sandbox.journal import ExperimentJournal
            return ExperimentJournal(exp_db)
        except Exception:
            return None

    def get_feature_store(self):
        """Open FeatureStore, return instance or None."""
        fs_db = self.data_dir / "feature_store.db"
        if not fs_db.exists():
            return None
        try:
            from hydra.data.store.feature_store import FeatureStore
            return FeatureStore(fs_db)
        except Exception:
            return None

    # --- MLflow (independent, no runner needed) ---

    def get_model_registry(self):
        """Get ModelRegistry instance or None."""
        try:
            from hydra.sandbox.registry import ModelRegistry
            return ModelRegistry()
        except Exception:
            return None

    # --- Runner-dependent (in-memory, graceful None) ---

    def get_circuit_breakers(self):
        """Get CircuitBreakerManager from runner or None."""
        if self.runner is None:
            return None
        try:
            risk_gate = getattr(self.runner, '_risk_gate', None)
            if risk_gate is not None:
                return getattr(risk_gate, '_breakers', None)
            return None
        except Exception:
            return None

    def get_baseline_model(self):
        """Get BaselineModel from runner or None."""
        if self.runner is None:
            return None
        try:
            return getattr(self.runner, '_model', None)
        except Exception:
            return None

    def get_drift_observer(self):
        """Get DriftObserver from runner or None."""
        if self.runner is None:
            return None
        try:
            agent_loop = getattr(self.runner, '_agent_loop', None)
            if agent_loop:
                return getattr(agent_loop, '_observer', None)
            return None
        except Exception:
            return None

    def get_agent_loop(self):
        """Get AgentLoop from runner or None."""
        if self.runner is None:
            return None
        try:
            return getattr(self.runner, '_agent_loop', None)
        except Exception:
            return None

    # --- Convenience queries ---

    def get_experiment_summary(self) -> dict:
        """Get experiment stats: total, promoted, rejected, rate."""
        ej = self.get_experiment_journal()
        if ej is None:
            return {"total": 0, "promoted": 0, "rejected": 0, "rate": 0.0}
        try:
            total = ej.count()
            all_exps = ej.query(limit=10000)
            promoted = sum(1 for e in all_exps if e.promotion_decision == "promoted")
            rejected = sum(1 for e in all_exps if e.promotion_decision == "rejected")
            rate = (promoted / total * 100) if total > 0 else 0.0
            return {"total": total, "promoted": promoted, "rejected": rejected, "rate": round(rate, 1)}
        finally:
            ej.close()

    def get_mutation_breakdown(self) -> dict:
        """Get count per mutation_type."""
        ej = self.get_experiment_journal()
        if ej is None:
            return {}
        try:
            all_exps = ej.query(limit=10000)
            breakdown = {}
            for e in all_exps:
                mt = e.mutation_type or "unknown"
                breakdown[mt] = breakdown.get(mt, 0) + 1
            return breakdown
        finally:
            ej.close()

    def get_feature_store_stats(self) -> dict:
        """Get feature store stats: count, markets, features."""
        fs = self.get_feature_store()
        if fs is None:
            return {"count": 0, "markets": [], "features": []}
        try:
            rows = fs.conn.execute("SELECT COUNT(*) FROM features").fetchone()
            count = rows[0] if rows else 0
            markets = [r[0] for r in fs.conn.execute("SELECT DISTINCT market FROM features ORDER BY market").fetchall()]
            features = [r[0] for r in fs.conn.execute("SELECT DISTINCT feature_name FROM features ORDER BY feature_name").fetchall()]
            return {"count": count, "markets": markets, "features": features}
        finally:
            fs.close()

    def get_latest_features(self, market: str | None = None, limit: int = 50) -> list[dict]:
        """Get latest feature values with quality flags."""
        fs = self.get_feature_store()
        if fs is None:
            return []
        try:
            if market:
                rows = fs.conn.execute(
                    """SELECT feature_name, value, quality, MAX(available_at) as last_updated
                       FROM features WHERE market = ?
                       GROUP BY feature_name ORDER BY last_updated DESC LIMIT ?""",
                    (market, limit)
                ).fetchall()
            else:
                rows = fs.conn.execute(
                    """SELECT feature_name, value, quality, MAX(available_at) as last_updated
                       FROM features
                       GROUP BY feature_name ORDER BY last_updated DESC LIMIT ?""",
                    (limit,)
                ).fetchall()
            return [{"feature_name": r[0], "value": r[1], "quality": r[2], "last_updated": r[3]} for r in rows]
        finally:
            fs.close()

    def get_component_health(self) -> list[dict]:
        """Get health status for all components."""
        components = []

        # Fill Journal
        fj = self.get_fill_journal()
        if fj is not None:
            try:
                count = fj.count()
                components.append({"name": "Fill Journal", "status": "ok", "details": f"{count} fills"})
            except Exception:
                components.append({"name": "Fill Journal", "status": "error", "details": "Query failed"})
            finally:
                fj.close()
        else:
            components.append({"name": "Fill Journal", "status": "unavailable", "details": "DB not found"})

        # Experiment Journal
        ej = self.get_experiment_journal()
        if ej is not None:
            try:
                count = ej.count()
                components.append({"name": "Experiment Journal", "status": "ok", "details": f"{count} experiments"})
            except Exception:
                components.append({"name": "Experiment Journal", "status": "error", "details": "Query failed"})
            finally:
                ej.close()
        else:
            components.append({"name": "Experiment Journal", "status": "unavailable", "details": "DB not found"})

        # Feature Store
        fs = self.get_feature_store()
        if fs is not None:
            try:
                count = fs.conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
                components.append({"name": "Feature Store", "status": "ok", "details": f"{count} features"})
            except Exception:
                components.append({"name": "Feature Store", "status": "error", "details": "Query failed"})
            finally:
                fs.close()
        else:
            components.append({"name": "Feature Store", "status": "unavailable", "details": "DB not found"})

        # MLflow
        try:
            reg = self.get_model_registry()
            if reg is not None:
                versions = reg.list_versions()
                components.append({"name": "MLflow", "status": "ok", "details": f"{len(versions)} model versions"})
            else:
                components.append({"name": "MLflow", "status": "unavailable", "details": "Cannot connect"})
        except Exception:
            components.append({"name": "MLflow", "status": "unavailable", "details": "Cannot connect"})

        # IB Broker
        if self.runner is not None:
            broker = getattr(self.runner, '_broker', None)
            if broker and getattr(broker, 'connected', False):
                components.append({"name": "IB Broker", "status": "ok", "details": "Connected"})
            else:
                components.append({"name": "IB Broker", "status": "unavailable", "details": "Not connected"})
        else:
            components.append({"name": "IB Broker", "status": "unavailable", "details": "Runner not active"})

        return components

    def get_circuit_breaker_states(self) -> list[dict]:
        """Get circuit breaker states with thresholds."""
        breakers = self.get_circuit_breakers()
        if breakers is None:
            # Return default info without live state
            from hydra.risk.circuit_breakers import CircuitBreakerManager
            defaults = CircuitBreakerManager.DEFAULT_CONFIG
            return [
                {"name": name, "state": "unknown", "threshold": thresh, "color": "muted"}
                for name, thresh in defaults.items()
            ]
        result = []
        for name, breaker in breakers.breakers.items():
            state = breaker.state.value
            if state == "active":
                color = "ok"
            elif state == "triggered":
                color = "alert"
            else:
                color = "warn"
            result.append({
                "name": name,
                "state": state.upper(),
                "threshold": breaker.threshold,
                "color": color,
            })
        return result

    def get_champion_fitness(self) -> dict | None:
        """Get fitness breakdown from latest promoted experiment results."""
        ej = self.get_experiment_journal()
        if ej is None:
            return None
        try:
            promoted = ej.query(outcome="promoted", limit=1)
            if not promoted:
                return None
            results = promoted[0].results
            if not isinstance(results, dict):
                return None
            return results
        finally:
            ej.close()

    def get_disk_usage(self) -> list[dict]:
        """Get SQLite file sizes."""
        files = [
            ("fill_journal.db", "Fill Journal"),
            ("experiment_journal.db", "Experiment Journal"),
            ("feature_store.db", "Feature Store"),
        ]
        usage = []
        for fname, label in files:
            fpath = self.data_dir / fname
            if fpath.exists():
                size_bytes = fpath.stat().st_size
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes / (1024*1024):.1f} MB"
                usage.append({"name": label, "file": fname, "size": size_str})
            else:
                usage.append({"name": label, "file": fname, "size": "Not created"})
        return usage
