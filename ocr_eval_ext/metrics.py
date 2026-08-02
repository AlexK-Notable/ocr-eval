"""Boolean- and null-restricted metrics over upstream score-cache records.
Rule: accuracy-over-all (errors incorrect) is the ranking key. Every block carries n."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class FieldOutcome:
    qid: str
    key: str
    doc: str
    gold: object
    status: Literal["correct", "incorrect", "error"]


@dataclass
class MetricBlock:
    n: int
    n_answered: int
    acc_over_all: float
    acc_over_answered: float
    error_rate: float


def _block(outcomes: list[FieldOutcome]) -> MetricBlock:
    n = len(outcomes)
    answered = [o for o in outcomes if o.status != "error"]
    correct = sum(1 for o in outcomes if o.status == "correct")
    return MetricBlock(
        n=n,
        n_answered=len(answered),
        acc_over_all=correct / n if n else 0.0,
        acc_over_answered=correct / len(answered) if answered else 0.0,
        error_rate=(n - len(answered)) / n if n else 0.0,
    )


def field_outcomes(records: dict[str, dict], fields: list[tuple]) -> list[FieldOutcome]:
    out = []
    for qid, key, gold, doc in fields:
        rec = records.get(qid)
        ans = rec.get("answer") if rec else None
        if not rec or "error" in rec or "answer" not in rec or not isinstance(ans, dict):
            # No scorable answer at all (D7): qid absent, upstream reported an error (even
            # when answer is ALSO present — MINOR-1: error wins), the answer key is missing,
            # or the answer isn't a dict at all — None, a string, a list, an int (I3). An
            # empty dict IS scorable: every key is simply absent, which the D7 null-gold rule
            # below already treats as "incorrect", not "error".
            status = "error"
        elif gold is None:
            # Null gold: correct ONLY on key-present explicit None. Upstream's
            # deep_equal(None, None) would score key-absent as correct, which rewards
            # extractor collapse on the hallucination metric (D7).
            status = "correct" if key in ans and ans[key] is None else "incorrect"
        else:
            status = "correct" if rec.get("field_matches", {}).get(key) else "incorrect"
        out.append(FieldOutcome(qid, key, doc, gold, status))
    return out


def checkbox_metrics(outcomes: list[FieldOutcome]) -> dict:
    for o in outcomes:
        if o.gold is not True and o.gold is not False:
            # I6 defence-in-depth: Task 3's boolean_fields should never emit a non-boolean
            # gold, but if it (or a future caller) ever does, fail loud rather than silently
            # mis-bucketing it into the confusion matrix.
            raise ValueError(
                f"checkbox_metrics: outcome {o.qid}/{o.key} has non-boolean gold {o.gold!r}"
            )
    checked = [o for o in outcomes if o.gold is True]
    unchecked = [o for o in outcomes if o.gold is False]
    conf = {"tt": 0, "tf": 0, "ft": 0, "ff": 0, "err": 0}
    for o in outcomes:
        if o.status == "error":
            conf["err"] += 1
        elif o.gold is True:
            conf["tt" if o.status == "correct" else "tf"] += 1
        else:
            conf["ff" if o.status == "correct" else "ft"] += 1
    return {"overall": _block(outcomes), "confusion": conf,
            "polarity": {"checked": _block(checked), "unchecked": _block(unchecked)}}


def null_metrics(outcomes: list[FieldOutcome]) -> dict:
    """Blank-field (null-gold) metrics.

    `overall.acc_over_all` (correct-nulls / all null fields) is the HEADLINE blank-field
    number — it is fail-safe against both failure modes at once: a collapsed extractor
    (answer: None) scores "error" under D7, never "correct", and an extractor that invents
    a value scores "incorrect". Either failure mode drives acc_over_all down.

    `hallucination_rate` = incorrect / n_answered is narrower: the propensity to invent a
    value GIVEN that the extractor answered at all (0.0 when n_answered == 0 — a full
    collapse — which is a "not applicable" clamp, not a "did not hallucinate" score).
    hallucination_rate is MEANINGLESS read alone (C1): a collapsed extractor and a
    well-behaved one can both report hallucination_rate == 0.0. Consumers must read this
    dict whole — hallucination_rate alongside overall.n_answered and overall.error_rate —
    never the rate in isolation.
    """
    block = _block(outcomes)
    hall = sum(1 for o in outcomes if o.status == "incorrect")   # scored wrong on a null gold
    rate = hall / block.n_answered if block.n_answered else 0.0
    return {"overall": block, "hallucination_rate": rate}


def baseline_rows(fields: list[tuple]) -> dict:
    golds = [g for _, _, g, _ in fields]
    t = sum(1 for g in golds if g is True)
    f = len(golds) - t
    n = len(golds)
    always_true = t / n if n else 0.0
    always_false = f / n if n else 0.0        # computed from counts, not 1-x (float exactness)
    return {"always_true": always_true, "always_false": always_false,
            "majority": max(always_true, always_false),
            "class_balance": {"true": t, "false": f}}
