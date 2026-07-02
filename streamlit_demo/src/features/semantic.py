"""
semantic.py — Semantic similarity feature using precomputed embeddings.

ARCHITECTURE:
  - Embeddings are precomputed OFFLINE (precompute_embeddings.py)
  - This module loads them ONCE at import time into memory
  - Lookups are O(1) dict lookups during the ranking step
  - Zero network calls, zero model inference at runtime
  - Fully deterministic and reproducible

STAGE 3 COMPLIANCE:
  - No network access required
  - No GPU required
  - Memory: ~150MB for embeddings dict
  - Lookup time: <1ms per candidate (dict lookup + dot product)

Usage:
    from features.semantic import SemanticScorer
    scorer = SemanticScorer()  # loads once
    score = scorer.similarity(candidate_id)  # fast lookup
"""

import os
import sys
import numpy as np
from typing import Optional

# Default artifact paths — relative to project root
_DEFAULT_ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))),
    "artifacts"
)

_JD_EMB_PATH  = os.path.join(_DEFAULT_ARTIFACTS, "jd_embedding.npy")
_IDS_PATH     = os.path.join(_DEFAULT_ARTIFACTS, "candidate_ids.npy")
_EMB_PATH     = os.path.join(_DEFAULT_ARTIFACTS, "candidate_embeddings.npy")


class SemanticScorer:
    """
    Loads precomputed embeddings once and provides fast per-candidate
    cosine similarity scores against the JD embedding.

    Embeddings are L2-normalized at precompute time, so cosine similarity
    reduces to a dot product — fast and exact.
    """

    def __init__(
        self,
        jd_emb_path: str = _JD_EMB_PATH,
        ids_path: str = _IDS_PATH,
        emb_path: str = _EMB_PATH,
    ):
        if not all(os.path.exists(p) for p in [jd_emb_path, ids_path, emb_path]):
            raise FileNotFoundError(
                "Precomputed embedding files not found. "
                "Run scripts/precompute_embeddings.py first.\n"
                f"  Expected: {jd_emb_path}\n"
                f"  Expected: {ids_path}\n"
                f"  Expected: {emb_path}"
            )

        import time
        t0 = time.time()

        jd_emb    = np.load(jd_emb_path)   # shape (1, 384)
        ids       = np.load(ids_path, allow_pickle=True)
        embeddings = np.load(emb_path)      # shape (N, 384), float32

        # Pre-compute all cosine similarities at load time
        # dot product = cosine because embeddings are L2-normalized
        all_sims = (embeddings @ jd_emb[0]).astype(np.float32)  # shape (N,)

        # Build O(1) lookup dict
        self._sims = dict(zip(ids.tolist(), all_sims.tolist()))

        elapsed = time.time() - t0
        print(
            f"[SemanticScorer] Loaded {len(self._sims):,} embeddings in {elapsed:.1f}s",
            file=sys.stderr
        )

    def similarity(self, candidate_id: str) -> float:
        """
        Returns cosine similarity between candidate and JD.
        Range: [-1, 1], typically [0.0, 0.6] in practice.
        Returns 0.0 if candidate not found (defensive default).
        """
        return self._sims.get(candidate_id, 0.0)

    def is_available(self) -> bool:
        return len(self._sims) > 0


# Module-level singleton — loaded once when scorer.py first imports this
# module. Subsequent calls reuse the same object with no re-loading.
_SCORER: Optional[SemanticScorer] = None
_AVAILABLE: Optional[bool] = None


def get_semantic_scorer() -> Optional[SemanticScorer]:
    """
    Returns the module-level SemanticScorer singleton, or None if
    embedding files are not yet available. Callers must handle None
    gracefully — the pipeline must work with or without embeddings.
    """
    global _SCORER, _AVAILABLE
    if _AVAILABLE is None:
        if all(os.path.exists(p) for p in [_JD_EMB_PATH, _IDS_PATH, _EMB_PATH]):
            try:
                _SCORER = SemanticScorer()
                _AVAILABLE = True
            except Exception as e:
                print(f"[SemanticScorer] Failed to load: {e}", file=sys.stderr)
                _AVAILABLE = False
        else:
            _AVAILABLE = False
            print(
                "[SemanticScorer] Embedding files not found — "
                "running without semantic signal.",
                file=sys.stderr
            )
    return _SCORER


def get_similarity(candidate_id: str) -> float:
    """
    Convenience function. Returns 0.0 if embeddings unavailable.
    This is the function scorer.py calls — it degrades gracefully.
    """
    scorer = get_semantic_scorer()
    if scorer is None:
        return 0.0
    return scorer.similarity(candidate_id)
