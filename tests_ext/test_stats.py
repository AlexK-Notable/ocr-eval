import math

from ocr_eval_ext.metrics import FieldOutcome
from ocr_eval_ext.stats import cluster_bootstrap_ci, paired_delta_ci, separable


def mk(doc, status, qid="q", key="k", gold=True):
    return FieldOutcome(qid, key, doc, gold, status)


def test_ci_is_deterministic_and_ordered():
    outs = [mk(f"d{i}", "correct" if i % 2 else "incorrect") for i in range(40)]
    lo1, hi1 = cluster_bootstrap_ci(outs, seed=7)
    lo2, hi2 = cluster_bootstrap_ci(outs, seed=7)
    assert (lo1, hi1) == (lo2, hi2) and lo1 < 0.5 < hi1


def test_clustered_wider_than_degenerate_clusters():
    # 10 docs x 10 perfectly correlated fields vs 100 independent docs - same 50% accuracy
    correlated = [mk(f"d{i//10}", "correct" if (i // 10) % 2 else "incorrect") for i in range(100)]
    independent = [mk(f"d{i}", "correct" if i % 2 else "incorrect") for i in range(100)]
    wc = cluster_bootstrap_ci(correlated, seed=1)
    wi = cluster_bootstrap_ci(independent, seed=1)
    assert (wc[1] - wc[0]) > (wi[1] - wi[0])   # correlation must widen the interval


def test_paired_delta_and_separability():
    a = [mk(f"d{i}", "correct") for i in range(30)]
    b = [mk(f"d{i}", "correct" if i < 15 else "incorrect") for i in range(30)]
    ci = paired_delta_ci(a, b, seed=3)
    assert separable(ci) and ci[0] > 0
    ci_same = paired_delta_ci(a, a, seed=3)
    assert not separable(ci_same)


def test_paired_delta_ci_partial_doc_overlap():
    # a has docs d0..d29, b has docs d10..d39 — only a partial overlap. _by_doc-based
    # pairing must resample the UNION of docs, treating docs missing from one side as
    # empty (via da.get(doc, [])), not crash or silently restrict to the intersection.
    a = [mk(f"d{i}", "correct" if i % 2 else "incorrect") for i in range(30)]
    b = [mk(f"d{i}", "correct" if i % 3 else "incorrect") for i in range(10, 40)]
    ci = paired_delta_ci(a, b, seed=11)
    lo, hi = ci
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo <= hi
    ci_again = paired_delta_ci(a, b, seed=11)
    assert ci == ci_again
