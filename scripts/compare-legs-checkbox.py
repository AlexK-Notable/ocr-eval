"""Compare two vlm-chat legs on the 258 boolean checkbox fields, paired, with McNemar.

WHY THIS IS A SCRIPT. Every checkbox comparison in docs/results-*.md was computed ad hoc and then
quoted from prose. That is how "not statistically significant" turns into "equivalent" three
documents later. This re-derives cb-acc, the checked/unchecked split, and the exact paired test from
the cache rows on demand, using the repo's OWN metric functions — `preconditions.boolean_fields`
picks the fields and `metrics.field_outcomes`/`checkbox_metrics` decide correct/incorrect/error — so
it cannot drift from what `report.md` renders.

Errors count as INCORRECT (accuracy-over-all), this repo's ranking key.

Usage:
    uv run python scripts/compare-legs-checkbox.py <parser_key_a> <parser_key_b> [run-dir]
    # parser keys are directory-style: vlm__<registry id>__<condition hash>

Example:
    uv run python scripts/compare-legs-checkbox.py \\
        vlm__claude-sonnet-4.6@bedrock__0373123f3b15 \\
        vlm__claude-haiku-4.5@bedrock__4caf39e9ac1f
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import binomtest

from ocr_eval_ext.metrics import checkbox_metrics, field_outcomes
from ocr_eval_ext.preconditions import CHECKBOX_TAGS, boolean_fields, items_with_tags
from realdoc_bench.evaluate.runs import RunLayout


def load_leg(layout: RunLayout, parser_key: str, qids: set[str]) -> dict[str, dict]:
    """Cache rows for one parser key, keyed by qid. Reads only the qids under test."""
    records: dict[str, dict] = {}
    for qid in qids:
        path = layout.cache_path(qid, parser_key)
        if path.exists():
            records[qid] = json.loads(path.read_text())
    return records


def describe(label: str, metrics: dict) -> None:
    o = metrics["overall"]
    p = metrics["polarity"]
    print(f"{label}")
    print(f"    cb-acc (errors wrong) : {o.acc_over_all:6.1%}  ({round(o.acc_over_all * o.n)}/{o.n})")
    print(f"    over answered only    : {o.acc_over_answered:6.1%}  (n_answered={o.n_answered}, "
          f"error_rate={o.error_rate:.1%})")
    print(f"    checked / unchecked   : {p['checked'].acc_over_all:6.1%} (n={p['checked'].n})"
          f" / {p['unchecked'].acc_over_all:6.1%} (n={p['unchecked'].n})")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    key_a, key_b = sys.argv[1], sys.argv[2]
    layout = RunLayout.at(Path(sys.argv[3] if len(sys.argv) > 3 else "runs/stage1"))

    bank = json.loads(layout.bank_path.read_text())["items"]
    fields = boolean_fields(items_with_tags(bank, CHECKBOX_TAGS))
    qids = {qid for qid, _, _, _ in fields}
    print(f"{len(fields)} boolean checkbox fields over {len(qids)} cells "
          f"({sum(1 for _, _, g, _ in fields if g)} true / "
          f"{sum(1 for _, _, g, _ in fields if not g)} false)\n")

    out_a = field_outcomes(load_leg(layout, key_a, qids), fields)
    out_b = field_outcomes(load_leg(layout, key_b, qids), fields)
    describe(key_a, checkbox_metrics(out_a))
    print()
    describe(key_b, checkbox_metrics(out_b))

    # Paired, field by field, in the order boolean_fields emits — same field in both legs.
    a_only = b_only = both = neither = 0
    for oa, ob in zip(out_a, out_b, strict=True):
        assert (oa.qid, oa.key) == (ob.qid, ob.key), "field order diverged between legs"
        ca, cb = oa.status == "correct", ob.status == "correct"
        both += ca and cb
        a_only += ca and not cb
        b_only += cb and not ca
        neither += not ca and not cb

    print(f"\npaired: both {both}, A-only {a_only}, B-only {b_only}, neither {neither}")
    discordant = a_only + b_only
    if not discordant:
        print("no discordant pairs — the legs agree on every field; no test to run")
        return 0
    p = binomtest(a_only, discordant, 0.5).pvalue
    print(f"McNemar exact on {discordant} discordant pairs: p={p:.4f}")
    print(f"agreement: {(both + neither) / len(fields):.1%}")
    # The phrasing rule this repo learned the hard way: never "equivalent".
    if p >= 0.05:
        print(f"\nNOT DETECTED at n={len(fields)}. Say \"no difference detected at this sample "
              f"size\", never \"equivalent\" — an n=300 A/B here reported p=1.000 and the same "
              f"comparison separated at p<1e-8 over 4,068 cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
