"""Document-clustered bootstrap. Questions cluster on documents (429 q / 263 docs);
naive binomial intervals are too tight. Percentile method, docs resampled with replacement."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ocr_eval_ext.metrics import FieldOutcome


def _by_doc(outcomes: list[FieldOutcome]) -> dict[str, list[FieldOutcome]]:
    d: dict[str, list[FieldOutcome]] = defaultdict(list)
    for o in outcomes:
        d[o.doc].append(o)
    return d


def _acc(outs: list[FieldOutcome]) -> float:
    return sum(1 for o in outs if o.status == "correct") / len(outs) if outs else 0.0


def _check_params(iters: int, alpha: float) -> None:
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters!r}")


def cluster_bootstrap_ci(outcomes, *, iters=2000, seed=0, alpha=0.05) -> tuple[float, float]:
    _check_params(iters, alpha)
    docs = sorted(_by_doc(outcomes).items())
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(iters):
        idx = rng.integers(0, len(docs), len(docs))
        sample = [o for i in idx for o in docs[i][1]]
        stats.append(_acc(sample))
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def paired_delta_ci(a, b, *, iters=2000, seed=0, alpha=0.05) -> tuple[float, float]:
    _check_params(iters, alpha)
    da, db = _by_doc(a), _by_doc(b)
    docs = sorted(set(da) | set(db))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(iters):
        idx = rng.integers(0, len(docs), len(docs))
        sa = [o for i in idx for o in da.get(docs[i], [])]
        sb = [o for i in idx for o in db.get(docs[i], [])]
        stats.append(_acc(sa) - _acc(sb))
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def separable(delta_ci: tuple[float, float]) -> bool:
    lo, hi = delta_ci
    return lo > 0 or hi < 0
