import math

import numpy as np
import pytest

from ocr_eval_ext.metrics import FieldOutcome
from ocr_eval_ext.stats import _acc, _by_doc, cluster_bootstrap_ci, paired_delta_ci, separable


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


def test_paired_ci_stays_narrow_despite_wild_doc_heterogeneity():
    # [I1] 30 docs, 20 fields each. Per-doc accuracy for `a` alternates wildly between
    # 0.1 (even docs) and 0.9 (odd docs). `b` mirrors `a` field-for-field, except that
    # in the 15 even (low-accuracy) docs exactly one extra field that `a` has correct
    # is wrong for `b` - a small, doc-local, tightly-correlated a-vs-b gap.
    #
    # A correctly PAIRED bootstrap draws the SAME doc for both arms every resample, so
    # that bounded per-doc gap is all that surfaces regardless of which docs get
    # oversampled -> narrow CI. A pairing bug that resamples the arms independently
    # lets the marginal 0.1-vs-0.9 heterogeneity leak into the delta instead (verified
    # empirically off-test: unpaired width on this exact fixture is ~0.43, over 25x
    # wider than the <0.10 bound asserted here).
    a, b = [], []
    n_fields = 20
    for i in range(30):
        base_correct = 2 if i % 2 == 0 else 18  # 0.1 vs 0.9 per-doc accuracy, alternating
        perturb = i % 2 == 0                    # the 15 low-accuracy docs get the extra b-flip
        for j in range(n_fields):
            qid = f"q{i}_{j}"
            a_correct = j < base_correct
            a.append(mk(f"d{i}", "correct" if a_correct else "incorrect", qid=qid))
            # a has field 0 correct whenever base_correct >= 2; b flips exactly that
            # field wrong on the 15 perturbed docs, matching a everywhere else.
            b_correct = a_correct and not (perturb and j == 0)
            b.append(mk(f"d{i}", "correct" if b_correct else "incorrect", qid=qid))

    lo, hi = paired_delta_ci(a, b, seed=1)
    assert (hi - lo) < 0.10


def test_paired_delta_ci_partial_overlap_pins_union_not_intersection_pairing():
    # [I2] a has docs d0..d29, b has docs d10..d39 - only a partial overlap (d10..d29
    # shared; d0..d9 exclusive to a; d30..d39 exclusive to b). `_by_doc`-based pairing
    # must resample the UNION of docs (missing side treated as an empty pool via
    # `.get(doc, [])`), not silently degrade to the shared-only intersection.
    #
    # A finite/deterministic result alone doesn't distinguish union from intersection
    # pairing (both produce one) - so this recomputes the intersection-only interval
    # independently, inline, using the same seed and iteration count, and asserts the
    # shipped result differs from it. (An earlier version of this test only checked
    # finiteness and determinism, which an intersection-only implementation also
    # satisfies - that version passed the exact mutant it was meant to exclude.)
    a = [mk(f"d{i}", "correct" if i % 2 else "incorrect") for i in range(30)]
    b = [mk(f"d{i}", "correct" if i % 3 else "incorrect") for i in range(10, 40)]

    ci = paired_delta_ci(a, b, seed=11)
    lo, hi = ci
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo <= hi
    ci_again = paired_delta_ci(a, b, seed=11)
    assert ci == ci_again

    da, db = _by_doc(a), _by_doc(b)
    shared_docs = sorted(set(da) & set(db))
    rng = np.random.default_rng(11)
    stats = []
    for _ in range(2000):
        idx = rng.integers(0, len(shared_docs), len(shared_docs))
        sa = [o for i in idx for o in da[shared_docs[i]]]
        sb = [o for i in idx for o in db[shared_docs[i]]]
        stats.append(_acc(sa) - _acc(sb))
    intersection_ci = (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))
    assert ci != intersection_ci


def test_cluster_bootstrap_ci_counts_errors_as_wrong():
    # [I3] Every doc has an identical composition: 1 correct, 1 incorrect, 2 error
    # fields. Because every doc contributes the same ratio, EVERY resample - no matter
    # which docs get drawn or how many times - yields the exact same acc_over_all, so
    # the CI collapses to a single point: 0.25 (1 correct / 4 total) only if errors
    # count as wrong. A mutant computing acc_over_answered (1 correct / 2 answered,
    # dropping errors from the denominator) would collapse to 0.5 instead.
    outs = []
    for i in range(20):
        outs += [
            mk(f"d{i}", "correct", qid=f"q{i}a", key="a"),
            mk(f"d{i}", "incorrect", qid=f"q{i}b", key="b"),
            mk(f"d{i}", "error", qid=f"q{i}c", key="c"),
            mk(f"d{i}", "error", qid=f"q{i}d", key="d"),
        ]
    lo, hi = cluster_bootstrap_ci(outs, seed=5)
    assert lo == pytest.approx(0.25) and hi == pytest.approx(0.25)


def test_ci_width_increases_as_alpha_shrinks():
    # [I4] A tighter confidence level (smaller alpha) must always widen the interval;
    # kills mutants that hardcode percentiles or ignore `alpha`.
    outs = [mk(f"d{i}", "correct" if i % 2 else "incorrect") for i in range(40)]
    lo05, hi05 = cluster_bootstrap_ci(outs, seed=9, alpha=0.05)
    lo20, hi20 = cluster_bootstrap_ci(outs, seed=9, alpha=0.20)
    assert (hi05 - lo05) > (hi20 - lo20)
    # [L-c] The width comparison alone passes a single-tail mutant (one that ignores alpha on
    # just the lo OR just the hi percentile, while still moving the other enough to widen the
    # overall interval) — pin BOTH tails moving in the expected direction independently.
    assert lo05 < lo20
    assert hi05 > hi20


def test_paired_delta_ci_width_increases_as_alpha_shrinks():
    # [L-c] Same two-sided-alpha assertion as above, for `paired_delta_ci` — a distinct
    # percentile call site from `cluster_bootstrap_ci`'s, so this is needed to kill a
    # paired-function-specific single-tail/hardcoded-percentile mutant that the
    # `cluster_bootstrap_ci` test above cannot reach.
    a = [mk(f"d{i}", "correct" if i % 2 else "incorrect") for i in range(40)]
    b = [mk(f"d{i}", "correct" if i % 3 else "incorrect") for i in range(40)]
    lo05, hi05 = paired_delta_ci(a, b, seed=9, alpha=0.05)
    lo20, hi20 = paired_delta_ci(a, b, seed=9, alpha=0.20)
    assert (hi05 - lo05) > (hi20 - lo20)
    assert lo05 < lo20
    assert hi05 > hi20


def test_invalid_alpha_raises():
    outs = [mk("d0", "correct")]
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(outs, alpha=0)
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(outs, alpha=1)
    with pytest.raises(ValueError):
        paired_delta_ci(outs, outs, alpha=-0.1)


def test_invalid_iters_raises():
    outs = [mk("d0", "correct")]
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(outs, iters=0)
    with pytest.raises(ValueError):
        paired_delta_ci(outs, outs, iters=-5)
