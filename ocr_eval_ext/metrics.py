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
        if not rec or "answer" not in rec or rec.get("answer") is None:
            status = "error"                       # no scorable answer at all (D7)
        elif gold is None:
            # Null gold: correct ONLY on key-present explicit None. Upstream's
            # deep_equal(None, None) would score key-absent/None-answer as correct,
            # which rewards extractor collapse on the hallucination metric (D7).
            ans = rec["answer"]
            status = "correct" if (isinstance(ans, dict) and key in ans and ans[key] is None) \
                else "incorrect"
        else:
            status = "correct" if rec.get("field_matches", {}).get(key) else "incorrect"
        out.append(FieldOutcome(qid, key, doc, gold, status))
    return out


def checkbox_metrics(outcomes: list[FieldOutcome]) -> dict:
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
    hall = sum(1 for o in outcomes if o.status == "incorrect")   # scored wrong on a null gold
    n = len(outcomes)
    return {"overall": _block(outcomes), "hallucination_rate": hall / n if n else 0.0}


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
