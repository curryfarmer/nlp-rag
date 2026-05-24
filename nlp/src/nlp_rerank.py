"""Phase 3 cross-encoder reranker.

Wraps `cross-encoder/ms-marco-MiniLM-L6-v2` (22M params, ~90MB). Lazy-loaded
so `import nlp_rerank` works on boxes without `sentence-transformers`. Env-flag
gated so the regex-only path stays the default until rerank is opted in.

API:
    ranker = get_reranker()           # cached singleton or None
    best = ranker.rank(q, candidates) # str

Disable with NLP_USE_RERANK=0. Override model with NLP_RERANK_MODEL=<hf-id>.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class Reranker:
    def __init__(self, model_id: str):
        # local import — sentence-transformers is heavy + optional
        from sentence_transformers import CrossEncoder
        self.model_id = model_id
        self.model = CrossEncoder(model_id, max_length=256)

    def rank(self, question: str, candidates: list[str]) -> str:
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]
        pairs = [(question, c) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        best = max(range(len(candidates)), key=lambda i: float(scores[i]))
        return candidates[best]

    def rank_with_scores(self, question: str, candidates: list[str]
                         ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        pairs = [(question, c) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        out = [(c, float(s)) for c, s in zip(candidates, scores)]
        out.sort(key=lambda t: -t[1])
        return out


_SINGLETON: Optional[Reranker] = None
_DISABLED: bool = False


def get_reranker() -> Optional[Reranker]:
    """Cached singleton. Returns None when disabled or load fails."""
    global _SINGLETON, _DISABLED
    if _DISABLED:
        return None
    if _SINGLETON is not None:
        return _SINGLETON
    if os.getenv("NLP_USE_RERANK", "1") == "0":
        _DISABLED = True
        return None
    model_id = os.getenv("NLP_RERANK_MODEL", _DEFAULT_MODEL)
    try:
        _SINGLETON = Reranker(model_id)
        logger.info(f"Reranker loaded: {model_id}")
        return _SINGLETON
    except Exception as e:
        logger.warning(f"Reranker load failed ({type(e).__name__}: {e}); regex-only fallback.")
        _DISABLED = True
        return None
