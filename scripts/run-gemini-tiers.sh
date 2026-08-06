#!/usr/bin/env bash
# Full-bank vlm-chat runs for the gemini-native candidates, in the user-specified tier order:
# flash-lite -> flash -> pro (cheapest and fastest first, so a systemic problem surfaces on the
# cheap models rather than on $7/leg pro runs).
#
# 2026-08-06: the three 2.5 models were dropped from the registry (imminent deprecation), so this
# script now covers THREE models, not six. Leaving them listed would abort each tier on an unknown
# model id.
#
# Deliberately NOT `set -e`: `ocr-eval direct` exits 2 when any cell errors — that is its
# fail-visible contract, not a crash — and a bare `set -e` would silently abandon every remaining
# tier the first time one cell rate-limits. Learned on the 3.6-flash extractor re-score, where 15
# throttled cells killed two of three legs while appearing to still run.
#
# Safe to re-run: cells that already landed are cache hits and cost nothing, so an interrupted or
# throttled pass resumes for free and re-attempts only what is missing.
#
#   bash scripts/run-gemini-tiers.sh                 # all three tiers
#   TIERS=lite bash scripts/run-gemini-tiers.sh      # one tier
#   WORKERS=2 bash scripts/run-gemini-tiers.sh       # throttle harder
set -uo pipefail
cd /home/alexk/repos/ocr-eval
set -a; . ~/.config/ocr-eval/secrets.env; set +a

REG=configs/registry-gemini.yaml
WORKERS="${WORKERS:-4}"        # Gemini quota is per-model-per-minute; 4 held clean on the smoke
TIERS="${TIERS:-lite flash pro}"

# Per-leg spend ceilings, sized to fit a $25 TOTAL authorization that also has to cover the two
# Mistral legs (~$6.50: $4.65 of per-page OCR, which is exact, plus ~$1.85 of token-billed extractor
# calls). That leaves ~$18.50 for Gemini against a $13.89 estimate.
#
# Sizing these at the estimator's stated +-2x error bar would sum to $29 and blow the ceiling on
# Gemini alone, so they are deliberately tighter than 2x. A leg that trips its cap stops and can be
# resumed for free (landed cells are cache hits), which is the right failure mode — unlike
# discovering the overspend afterwards.
#
# `run_direct` raises the moment realized spend crosses the cap, and REFUSES to run a model whose
# provider returns no token counts rather than silently proceeding uncapped.
declare -A TIER_CAP=(
  [gemini-3.5-flash-lite@google-native]=2.00     # est 1.22
  [gemini-3.6-flash@google-native]=7.00          # est 5.29
  [gemini-3.1-pro-preview@google-native]=9.50    # est 7.38
)                                                # sum 18.50 + Mistral ~6.50 = ~25.00

declare -A TIER_MODELS=(
  [lite]="gemini-3.5-flash-lite@google-native"
  [flash]="gemini-3.6-flash@google-native"
  [pro]="gemini-3.1-pro-preview@google-native"
)

rc_any=0
for tier in $TIERS; do
  for model in ${TIER_MODELS[$tier]}; do
    cap="${TIER_CAP[$model]}"
    echo "=== [$tier] $model (workers=$WORKERS, max-spend=\$$cap) ==="
    .venv/bin/ocr-eval direct --run-dir runs/stage1 --registry "$REG" \
        -m "$model" --workers "$WORKERS" --max-spend "$cap"
    rc=$?
    [ "$rc" -ne 0 ] && { echo "!! $model exited $rc (2 = some cells errored; rerun to re-attempt)"; rc_any=$rc; }
  done
done
echo "=== all requested tiers attempted; worst exit status: $rc_any ==="
exit "$rc_any"
