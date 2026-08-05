"""Measure real Haiku 4.5 token usage on a stratified RealDocBench sample.

Ground truth, not estimates: real 150dpi renders through direct.py's own `_render_page`,
the real STAGE1_CONDITION prompt via `direct_prompt`, and `usage.inputTokens` /
`usage.outputTokens` straight off the Bedrock Converse response.

Written as a repo script (not /tmp) so the numbers behind any cost claim can be re-derived.
Usage: uv run python scripts/measure_haiku_cost.py [run-dir]
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

import boto3
from botocore.config import Config

from ocr_eval_ext.direct import (
    STAGE1_CONDITION,
    SYSTEM,
    _extract_json,
    _render_page,
    direct_prompt,
)
from realdoc_bench.evaluate.runs import RunLayout
from realdoc_bench.evaluate.score import _ensure_template

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = "us-east-1"


def main(run_dir: str = "runs/cost-probe") -> None:
    client = boto3.client(
        "bedrock-runtime", region_name=REGION,
        config=Config(retries={"max_attempts": 3, "mode": "standard"}, read_timeout=300),
    )
    layout = RunLayout.at(run_dir)
    bank = json.loads(layout.bank_path.read_text())
    stems = {p.stem for p in layout.docs_dir.glob("*.pdf")}
    items = [i for i in bank["items"] if i["source_file"] in stems]
    png_cache = layout.root / "docs_png"

    rows: list[dict] = []
    for n, it in enumerate(items, 1):
        _ensure_template(it)
        try:
            png = _render_page(layout, it["source_file"], STAGE1_CONDITION, png_cache)
        except Exception as e:
            print(f"[{n}/{len(items)}] render fail {it['source_file']}: {str(e)[:60]}", flush=True)
            continue
        prompt = direct_prompt(it["question"], it["template"])
        try:
            resp = client.converse(
                modelId=MODEL_ID,
                messages=[{"role": "user", "content": [
                    {"image": {"format": "png", "source": {"bytes": png}}},
                    {"text": prompt}]}],
                system=[{"text": SYSTEM}],
                inferenceConfig={"temperature": 0.0, "maxTokens": 1024},
            )
        except Exception as e:
            print(f"[{n}/{len(items)}] api fail {it['question_id']}: {str(e)[:70]}", flush=True)
            continue
        usage = resp["usage"]
        text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
        answer = _extract_json(text.strip())
        rows.append({
            "domain": it.get("domain", "?"), "qid": it["question_id"],
            "source_file": it["source_file"], "n_fields": len(it.get("gold_dict") or {}),
            "in": usage["inputTokens"], "out": usage["outputTokens"],
            "parsed": answer is not None, "stop": resp.get("stopReason", ""),
            "png_bytes": len(png),
        })
        print(f"[{n}/{len(items)}] {it['question_id']:28} in={usage['inputTokens']:5} "
              f"out={usage['outputTokens']:4} parsed={answer is not None}", flush=True)

    out_path = layout.root / "usage.json"
    out_path.write_text(json.dumps(rows, indent=2))
    report(rows, out_path)


def report(rows: list[dict], out_path: Path) -> None:
    if not rows:
        print("no rows measured")
        return
    ins = [r["in"] for r in rows]
    outs = [r["out"] for r in rows]

    def pct(v: list[int], q: float) -> int:
        return sorted(v)[min(len(v) - 1, int(len(v) * q))]

    print(f"\n=== {len(rows)} real Haiku 4.5 calls (stratified sample) ===")
    print(f"input  : min {min(ins)}  p50 {int(statistics.median(ins))}  "
          f"mean {statistics.mean(ins):.0f}  p95 {pct(ins, .95)}  max {max(ins)}")
    print(f"output : min {min(outs)}  p50 {int(statistics.median(outs))}  "
          f"mean {statistics.mean(outs):.0f}  p95 {pct(outs, .95)}  max {max(outs)}")
    print(f"parsed : {sum(r['parsed'] for r in rows)}/{len(rows)}   "
          f"stopReasons: {dict(collections.Counter(r['stop'] for r in rows))}")
    print("\nper-domain mean input tokens:")
    by: dict[str, list[int]] = collections.defaultdict(list)
    for r in rows:
        by[r["domain"]].append(r["in"])
    for dom, vals in sorted(by.items()):
        print(f"  {dom:20} n={len(vals):3}  mean {statistics.mean(vals):7.0f}  "
              f"min {min(vals):5}  max {max(vals):5}")
    print(f"\nraw rows -> {out_path}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
