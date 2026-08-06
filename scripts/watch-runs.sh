#!/usr/bin/env bash
# Poll the in-flight legs until all are done, then print a final tally.
# Backgrounded so the agent turn never blocks on a long sleep (a `sleep 580` monitor once had to be
# interrupted to stop a run, leaving the user no way in).
set -uo pipefail
cd /home/alexk/repos/ocr-eval
C=runs/stage1/eval/cache
while true; do
  running=0
  pgrep -f "ocr-eval score"  >/dev/null && running=1
  pgrep -f "run-gemini-tiers" >/dev/null && running=1
  printf '%s  36flash=%s/1356  pro=%s/1356  m41=%s/1356  m40=%s/1356\n' \
    "$(date +%H:%M:%S)" \
    "$(ls $C | grep -c '3.6-flash@google-native.*872144e4aecb')" \
    "$(ls $C | grep -c '3.1-pro-preview@google-native.*872144e4aecb')" \
    "$(ls $C | grep -c '__mistral_ocr_4\.json')" \
    "$(ls $C | grep -c '__mistral_ocr_4_0\.json')"
  [ "$running" -eq 0 ] && { echo "ALL LEGS DONE"; break; }
  sleep 60
done
