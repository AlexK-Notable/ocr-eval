#!/usr/bin/env bash
# Full re-score of all three transcriber legs under the D11-rev3 extractor (gemini-3.6-flash).
# Kept in-repo per the no-/tmp rule; safe to re-run — completed cells are cache hits and cost
# nothing, so an interrupted run resumes for free.
#
# TWO THINGS THIS SCRIPT LEARNED THE HARD WAY, both on the 3.6-flash run:
#
#  1. `ocr-eval score` EXITS 2 when any cell errors — that is its fail-visible contract, not a
#     crash. A bare `set -e` therefore treats a handful of rate-limited cells as fatal and silently
#     abandons the remaining legs (observed: 15 429s on leg 1 killed legs 2 and 3, and the run
#     looked "still going" for 25 minutes because rows are written incrementally). Each leg's exit
#     status is captured and reported instead of aborting the loop.
#  2. `gemini-3.6-flash` has much tighter quota than the flash-lite tiers and is ~12x slower per
#     leg (10 min vs 0.8 min). 16 workers (the CLI default) sustains 429s against it, so WORKERS is
#     lowered here and left overridable.
#
# Re-running after a throttled pass costs nothing for cells that already landed and re-attempts only
# the failures: `_worker` treats a row with an "error" key as re-attemptable, while a row with an
# "answer" key is a cache hit.
set -uo pipefail          # deliberately NOT -e: see note 1 above
cd /home/alexk/repos/ocr-eval
set -a; . ~/.config/ocr-eval/secrets.env; set +a

WORKERS="${WORKERS:-6}"
PARSERS=(
  docstrange_sync
  qwen3-vl-8b_openrouter-transcriber__01f4c303710b
  qwen3-vl-32b_openrouter-transcriber__01f4c303710b
)

rc_any=0
for p in "${PARSERS[@]}"; do
  echo "=== $p (workers=$WORKERS) ==="
  .venv/bin/ocr-eval score --run-dir runs/stage1 --registry configs/registry.yaml \
      -p "$p" --workers "$WORKERS" ${EXTRA:-}
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!! $p exited $rc (2 = some cells errored; rows still written, re-run to re-attempt)"
    rc_any=$rc
  fi
done

echo "=== all legs attempted; worst exit status: $rc_any ==="
exit "$rc_any"
