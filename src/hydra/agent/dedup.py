"""Semantic deduplication for hypothesis proposals (AGNT-09 layer 1).

Prevents the agent from re-proposing hypotheses that are semantically
identical (or near-identical) to recently tried experiments. Uses a local
sentence-transformers model (``all-MiniLM-L6-v2``, 22M params, 384-dim
embeddings, runs on CPU in <10ms) for embedding and cosine similarity.

The deduplicator maintains an in-memory list of recent hypothesis
embeddings.  On each ``is_duplicate()`` call, it computes cosine
similarity against all stored embeddings and rejects if the maximum
similarity exceeds the configured threshold (default 0.85).

Integration:
    - ``register()`` adds a hypothesis to memory after it's been tested.
    - ``load_from_journal()`` pre-loads recent hypotheses from the
      persistent ``ExperimentJournal`` on startup.
    - ``clear()`` resets memory (useful between test runs).

Exports:
    - ``HypothesisDeduplicator``: Main dedup class.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()


class HypothesisDeduplicator:
    """Reject hypotheses semantically similar to recently tried ones.

    Parameters
    ----------
    similarity_threshold : float
        Maximum cosine similarity before a hypothesis is flagged as
        a duplicate. Default 0.85.
    model_name : str
        Name of the sentence-transformers model. Default
        ``all-MiniLM-L6-v2`` (22M params, 384-dim, CPU-friendly).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.threshold = similarity_threshold
        self.model = SentenceTransformer(model_name)
        self._recent_embeddings: list[np.ndarray] = []
        self._recent_descriptions: list[str] = []

    def is_duplicate(self, description: str) -> tuple[bool, float]:
        """Check if a hypothesis description duplicates a recent one.

        Parameters
        ----------
        description : str
            The hypothesis description to check.

        Returns
        -------
        tuple[bool, float]
            ``(is_dup, max_similarity_score)``. If no recent embeddings,
            returns ``(False, 0.0)``.
        """
        if not self._recent_embeddings:
            return (False, 0.0)

        new_embedding = self.model.encode([description])[0]
        recent_matrix = np.stack(self._recent_embeddings)

        # Cosine similarity: dot(a, b) / (||a|| * ||b||)
        dot_products = np.dot(recent_matrix, new_embedding)
        norms = np.linalg.norm(recent_matrix, axis=1) * np.linalg.norm(new_embedding)
        similarities = dot_products / norms

        max_sim = float(np.max(similarities))
        is_dup = max_sim > self.threshold

        if is_dup:
            best_idx = int(np.argmax(similarities))
            logger.info(
                "hypothesis_duplicate_detected",
                similarity=round(max_sim, 4),
                matched=self._recent_descriptions[best_idx][:80],
            )

        return (is_dup, max_sim)

    def register(self, description: str) -> None:
        """Add a hypothesis embedding to recent memory.

        Parameters
        ----------
        description : str
            The hypothesis description to register.
        """
        embedding = self.model.encode([description])[0]
        self._recent_embeddings.append(embedding)
        self._recent_descriptions.append(description)

    def load_from_journal(
        self, journal: object, days: int = 30
    ) -> None:
        """Pre-load recent hypotheses from the experiment journal.

        Queries the journal for experiments in the last ``days`` days
        and registers each hypothesis description into dedup memory.

        Parameters
        ----------
        journal : ExperimentJournal
            The experiment journal to query. Typed as ``object`` to
            avoid circular import; expects ``.query(date_from=...)``
            returning records with ``.hypothesis`` attribute.
        days : int
            How many days back to load. Default 30.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat()
        recent = journal.query(date_from=cutoff)
        for record in recent:
            self.register(record.hypothesis)
        logger.info(
            "dedup_loaded_from_journal",
            count=len(recent),
            days=days,
        )

    def clear(self) -> None:
        """Reset in-memory recent embeddings."""
        self._recent_embeddings.clear()
        self._recent_descriptions.clear()
