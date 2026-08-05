#!/usr/bin/env python
"""Does the extractor answer from the TRANSCRIPT, or from memory of the benchmark?

Why this exists: `gemini-3.6-flash` post-dates RealDoc-Bench's public availability
(CONTAMINATION_CUTOFF = 2026-05-24), so it may have seen the bank in training. A contaminated
extractor is not merely a labelling problem — it inflates EVERY transcriber row and compresses the
differences between them, because it can score a cell correctly even when the transcript it was
given contains nothing useful. That makes the whole Section B comparison less discriminating while
looking better.

The test: feed the extractor a transcript with NO answer in it and see whether it still produces the
gold value. An honest extractor returns null / a wrong guess; a memorizing one recalls the answer.

Three ablations per sampled item, all against the real pinned extractor:
  * empty      — an empty markdown body. Nothing to read at all.
  * decoy      — a plausible but unrelated page. Tests whether it invents rather than abstains.
  * shuffled   — the REAL transcript with its lines shuffled. Content present, structure destroyed;
                 a reader still succeeds here, so this separates "needs the layout" from "recalls".

A high correct-rate on `empty`/`decoy` is the alarm. Compare against the same items' real-transcript
rate to size the effect.

    uv run python scripts/extractor_memorization_probe.py --n 40
    uv run python scripts/extractor_memorization_probe.py --n 40 --parser docstrange_sync
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from realdoc_bench.evaluate.score import (
    DEFAULT_MODEL,
    _ensure_template,
    gemini_extract,
    score_typed,
)

DECOY = """## Page 1

**SHIPPING MANIFEST**

Carrier: Northwind Freight Co.
Origin: Portland, OR 97204
Destination: Boise, ID 83702
Pallets: 12
Gross weight: 8,410 lb
Seal: 4471902
Notes: Refrigerated unit, maintain 34-38F in transit.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("runs/stage1"))
    ap.add_argument("--parser", default="docstrange_sync",
                    help="which transcriber's real transcripts to use for the control arm")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0, help="item sample seed (reproducibility)")
    args = ap.parse_args()

    bank = json.loads((args.run_dir / "qa_bank.json").read_text())["items"]
    parses = args.run_dir / "parses" / args.parser
    # Only items whose transcript exists, so the control arm is a fair comparison.
    usable = [it for it in bank if (parses / f"{it['source_file']}.md").exists()]
    random.Random(args.seed).shuffle(usable)
    sample = usable[:args.n]
    print(f"extractor: {DEFAULT_MODEL}   items: {len(sample)}   control transcripts: {args.parser}\n")

    arms = {"real": None, "shuffled": None, "empty": "", "decoy": DECOY}
    score: dict[str, int] = dict.fromkeys(arms, 0)
    n_scored = 0

    for it in sample:
        _ensure_template(it)
        real = (parses / f"{it['source_file']}.md").read_text()
        lines = real.splitlines()
        random.Random(args.seed).shuffle(lines)
        bodies = {"real": real, "shuffled": "\n".join(lines), "empty": "", "decoy": DECOY}
        n_scored += 1
        for arm, md in bodies.items():
            try:
                ans = gemini_extract(it["question"], it["template"], md)
            except Exception as e:
                print(f"  ! {arm} {it['question_id']}: {type(e).__name__}: {str(e)[:120]}")
                continue
            _, ok = score_typed(ans or {}, it["gold_dict"], it["str_keys"])
            score[arm] += bool(ok)

    print(f"{'arm':10s} {'correct':>9s} {'rate':>7s}   interpretation")
    for arm in ("real", "shuffled", "empty", "decoy"):
        r = score[arm] / max(1, n_scored)
        note = {
            "real": "control — reading the actual transcript",
            "shuffled": "content kept, layout destroyed",
            "empty": "NOTHING to read: any correctness here is recall or luck",
            "decoy": "unrelated page: correctness here is recall or invention",
        }[arm]
        print(f"{arm:10s} {score[arm]:9d} {100*r:6.1f}%   {note}")

    blind = max(score["empty"], score["decoy"]) / max(1, n_scored)
    print(f"\nblind-arm ceiling: {100*blind:.1f}%")
    print("NB a nonzero blind rate is EXPECTED, not proof of contamination — many gold values are"
          "\n   guessable priors (a null blank field, a boolean that is usually true). Read it"
          "\n   against this same extractor's rate on OTHER arms, and against another extractor's"
          "\n   blind rate on the same items, before concluding anything.")


if __name__ == "__main__":
    main()
