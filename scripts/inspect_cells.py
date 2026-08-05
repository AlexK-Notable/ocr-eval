#!/usr/bin/env python
"""Dump direct-QA cache rows for eyeball verification.

Exists because a summary line (`{'ok': 20, ...}`) proves calls RETURNED, not that they returned
correct answers — a distinction that bit this project once already. Run it after any smoke pass,
before committing spend to a full bank.

    uv run python scripts/inspect_cells.py runs/stage1 --parser-substr haiku --limit 10
    uv run python scripts/inspect_cells.py runs/stage1 --mismatches-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--parser-substr", default="", help="only rows whose parser key contains this")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--mismatches-only", action="store_true")
    args = ap.parse_args()

    # Cache rows store `qid`, not the question/gold themselves — join the bank to make a mismatch
    # auditable rather than just countable.
    bank = json.loads((args.run_dir / "qa_bank.json").read_text())
    by_qid = {it["question_id"]: it for it in bank["items"]}

    rows = sorted((args.run_dir / "eval" / "cache").glob("*.json"))
    recs = []
    for p in rows:
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            print(f"!! unparseable: {p.name}")
            continue
        if args.parser_substr and args.parser_substr not in r.get("parser", ""):
            continue
        recs.append((p, r))

    matched = sum(1 for _, r in recs if r.get("match") is True)
    errs = [r.get("error_class") for _, r in recs if r.get("error_class") not in (None, "none")]
    print(f"rows: {len(recs)}  match=True: {matched}  errors: {len(errs)} {sorted(set(errs))}")

    shown = 0
    for p, r in recs:
        if args.mismatches_only and r.get("match") is True:
            continue
        if shown >= args.limit:
            break
        shown += 1
        u = r.get("usage") or {}
        item = by_qid.get(r.get("qid"), {})
        print(f"\n--- {p.name}  [{r.get('source_file')}]")
        print(f"  q      : {str(item.get('question'))[:150]}")
        print(f"  gold   : {item.get('gold_dict')}")
        print(f"  answer : {r.get('answer')}")
        print(f"  match  : {r.get('match')}   field_matches: {r.get('field_matches')}")
        print(f"  err    : {r.get('error_class')}  provider: {r.get('resolved_provider')}")
        print(f"  tokens : in={u.get('prompt_tokens')} out={u.get('completion_tokens')}"
              f"  note={u.get('sampling_note')}")


if __name__ == "__main__":
    main()
