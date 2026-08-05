#!/usr/bin/env bash
# Full-bank vlm-chat runs for the six gemini-native candidates, in the user-specified tier order:
# flash-lite -> flash -> pro (cheapest and fastest first, so a systemic problem surfaces on the
# cheap models rather than on $7/leg pro runs).
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

declare -A TIER_MODELS=(
  [lite]="gemini-2.5-flash-lite@google-native gemini-3.5-flash-lite@google-native"
  [flash]="gemini-2.5-flash@google-native gemini-3.6-flash@google-native"
  [pro]="gemini-2.5-pro@google-native gemini-3.1-pro-preview@google-native"
)

rc_any=0
for tier in $TIERS; do
  for model in ${TIER_MODELS[$tier]}; do
    echo "=== [$tier] $model (workers=$WORKERS) ==="
    .venv/bin/ocr-eval direct --run-dir runs/stage1 --registry "$REG" \
        -m "$model" --workers "$WORKERS"
    rc=$?
    [ "$rc" -ne 0 ] && { echo "!! $model exited $rc (2 = some cells errored; rerun to re-attempt)"; rc_any=$rc; }
  done
done
echo "=== all requested tiers attempted; worst exit status: $rc_any ==="
exit "$rc_any"
