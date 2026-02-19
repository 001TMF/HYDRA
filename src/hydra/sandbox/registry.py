"""MLflow model registry wrapper with champion/candidate/archived lifecycle.

Enforces HYDRA's model lifecycle conventions using MLflow aliases:
- champion: The current production model
- archived: The previous champion (rollback target)
- candidate: Any newly logged model version (implicit, no alias needed)

Uses explicit logging (not autolog) per research recommendation.
Local file backend by default -- no MLflow server required.
"""

from __future__ import annotations

import pathlib

import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


class ModelRegistry:
    """Thin wrapper over MLflow enforcing HYDRA lifecycle conventions.

    Parameters
    ----------
    tracking_uri : str | None
        MLflow tracking URI. Defaults to ``file://{project_root}/mlruns``
        using an absolute path to prevent the relative-path pitfall.
    model_name : str
        Registered model name in MLflow. Defaults to ``"hydra-baseline"``.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        model_name: str = "hydra-baseline",
    ) -> None:
        if tracking_uri is None:
            project_root = pathlib.Path(__file__).resolve().parents[3]
            tracking_uri = f"file://{project_root}/mlruns"

        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.model_name = model_name

        # Ensure registered model exists
        try:
            self.client.get_registered_model(model_name)
        except MlflowException:
            self.client.create_registered_model(model_name)

    def log_candidate(
        self,
        model,
        metrics: dict,
        config: dict,
        tags: dict | None = None,
        run_name: str | None = None,
    ) -> tuple[str, int]:
        """Log a trained model as a new candidate version.

        Parameters
        ----------
        model
            A BaselineModel instance. Its ``.model`` attribute (the
            underlying LGBMClassifier) is logged via ``mlflow.lightgbm``.
        metrics : dict
            Evaluation metrics to log (e.g. Sharpe, accuracy).
        config : dict
            Full config snapshot (hyperparams, features, etc.).
        tags : dict | None
            Optional tags for the run.
        run_name : str | None
            Optional human-readable run name.

        Returns
        -------
        tuple[str, int]
            ``(run_id, model_version_number)``
        """
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(config)
            mlflow.log_metrics(metrics)

            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, v)

            mlflow.lightgbm.log_model(
                model.model,
                artifact_path="model",
                registered_model_name=self.model_name,
            )

            run_id = run.info.run_id

        # Get the latest version number for this registered model
        versions = self.client.search_model_versions(
            f"name='{self.model_name}'"
        )
        latest_version = max(int(v.version) for v in versions)

        return run_id, latest_version

    def promote_to_champion(self, version: int) -> None:
        """Promote a model version to champion, archiving current champion.

        Parameters
        ----------
        version : int
            The model version number to promote.
        """
        # Archive current champion if one exists
        try:
            current = self.client.get_model_version_by_alias(
                self.model_name, "champion"
            )
            self.client.set_registered_model_alias(
                self.model_name, "archived", int(current.version)
            )
        except MlflowException:
            pass  # No current champion -- nothing to archive

        # Set new champion
        self.client.set_registered_model_alias(
            self.model_name, "champion", version
        )

    def rollback(self) -> int:
        """Rollback: swap champion alias to the previously archived version.

        Returns
        -------
        int
            The version number of the restored champion.

        Raises
        ------
        ValueError
            If no archived model exists to rollback to.
        """
        try:
            archived = self.client.get_model_version_by_alias(
                self.model_name, "archived"
            )
        except MlflowException:
            raise ValueError("No archived model to rollback to")

        self.promote_to_champion(int(archived.version))
        return int(archived.version)

    def load_champion(self):
        """Load the current champion model.

        Returns
        -------
        LightGBM Booster or sklearn-compatible model
            The loaded model, ready for prediction.

        Raises
        ------
        ValueError
            If no champion model alias is set.
        """
        try:
            self.client.get_model_version_by_alias(
                self.model_name, "champion"
            )
        except MlflowException:
            raise ValueError("No champion model set")

        return mlflow.lightgbm.load_model(
            f"models:/{self.model_name}@champion"
        )

    def get_champion_info(self) -> dict:
        """Return metadata about the current champion model.

        Returns
        -------
        dict
            Keys: ``version``, ``run_id``, ``metrics``, ``tags``,
            ``created_at``.

        Raises
        ------
        ValueError
            If no champion model alias is set.
        """
        try:
            champion = self.client.get_model_version_by_alias(
                self.model_name, "champion"
            )
        except MlflowException:
            raise ValueError("No champion model set")

        run = self.client.get_run(champion.run_id)

        return {
            "version": int(champion.version),
            "run_id": champion.run_id,
            "metrics": dict(run.data.metrics),
            "tags": {
                k: v
                for k, v in run.data.tags.items()
                if not k.startswith("mlflow.")
            },
            "created_at": champion.creation_timestamp,
        }

    def list_versions(self) -> list[dict]:
        """List all model versions for the registered model.

        Returns
        -------
        list[dict]
            Each dict has keys: ``version``, ``run_id``, ``aliases``,
            ``created_at``.
        """
        versions = self.client.search_model_versions(
            f"name='{self.model_name}'"
        )
        return [
            {
                "version": int(v.version),
                "run_id": v.run_id,
                "aliases": list(v.aliases) if v.aliases else [],
                "created_at": v.creation_timestamp,
            }
            for v in versions
        ]
