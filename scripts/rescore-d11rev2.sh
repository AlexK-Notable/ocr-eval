#!/usr/bin/env bash
# Full re-score of all three transcriber legs under the D11-rev2 extractor
# (gemini-3.6-flash). Kept in-repo per the no-/tmp rule; safe to re-run — completed cells are
# cache hits and cost nothing, so an interrupted run resumes for free.
set -euo pipefail
cd /home/alexk/repos/ocr-eval
set -a; . ~/.config/ocr-eval/secrets.env; set +a
for p in docstrange_sync \
         qwen3-vl-8b_openrouter-transcriber__01f4c303710b \
         qwen3-vl-32b_openrouter-transcriber__01f4c303710b; do
  echo "=== $p ==="
  .venv/bin/ocr-eval score --run-dir runs/stage1 --registry configs/registry.yaml -p "$p"
done
