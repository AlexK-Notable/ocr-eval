#!/usr/bin/env bash
# Ask AWS's PUBLIC price list what it charges for Anthropic models on Bedrock.
#
# WHY THIS EXISTS. configs/registry-bedrock.yaml carries per-token rates for the two Anthropic
# entries that came from ONE source: the rendered pricing page, read by a human on 2026-08-11.
# This repo's convention for a committed rate is two independent sources agreeing exactly. This
# script is the attempt at the second source, kept as a script rather than a note so the answer can
# be re-derived rather than remembered.
#
# WHAT IT FOUND ON 2026-08-11 (publication 20260811161754, us-east-1): 1,014 rate rows, of which
# FIVE mention Anthropic — Claude Instant, Claude 2.0, Claude 2.1, Claude 3 Haiku, Claude 3 Sonnet.
# No Claude 4.x or 5 row of any kind. So the second source did not exist yet. If this script ever
# prints a "Claude 4.5 Haiku" or "Claude Sonnet 4.6" row, that IS the second source: put it in the
# registry comment next to the page figure and say whether the two agree.
#
# No credentials are involved. pricing.us-east-1.amazonaws.com serves the bulk price list
# unauthenticated, which is why this works where `aws pricing get-products` does not —
# `pricing:GetProducts` is IAM-denied to this role, and that denial is what left the rate
# unverifiable in the first place.
#
# Deliberately no `set -e`: a fetch failure should print what failed and still let the rest run,
# per the exit-code discipline in CLAUDE.md.
set -uo pipefail

BASE="https://pricing.us-east-1.amazonaws.com"
REGION="${1:-us-east-1}"
INDEX="$BASE/offers/v1.0/aws/AmazonBedrock/current/region_index.json"

url=$(curl -sS --max-time 60 "$INDEX" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('# publication:', d.get('publicationDate'), file=sys.stderr)
r = d['regions'].get('$REGION')
if not r:
    sys.exit('no such region in the Bedrock price list: $REGION')
print(r['currentVersionUrl'])
") || { echo "FAILED to read the region index" >&2; exit 1; }

# The index advertises the JSON offer file; the CSV at the same path is the same data an order of
# magnitude smaller (457 KB vs ~5 MB) and is what this script parses. Feeding the JSON to
# csv.DictReader does not fail loudly — it yields tens of thousands of junk rows — so the swap is
# explicit rather than incidental.
csv_url="${url%index.json}index.csv"
echo "# region:      $REGION" >&2
echo "# offer file:  $csv_url" >&2

curl -sS --max-time 300 "$BASE/${csv_url#/}" | python3 -c "
import csv, io, re, sys

# The bulk CSV carries 5 metadata lines before the real header row.
lines = sys.stdin.read().splitlines()
rows = list(csv.DictReader(lines[5:]))
# Scoped to the three columns that name a model, not every column: a whole-row substring match
# also catches unrelated rows through opaque fields like usageType.
pat = re.compile(r'claude|anthropic', re.I)
hits = [r for r in rows
        if any(pat.search(str(r.get(c) or '')) for c in ('Provider', 'Model', 'PriceDescription'))]

print(f'{len(rows)} rate rows total, {len(hits)} mentioning Anthropic/Claude')
seen = set()
for r in hits:
    d = r['PriceDescription']
    if d in seen:
        continue
    seen.add(d)
    print(f\"  {r['PricePerUnit']:>14} per {r['Unit']:<12} | {d}\")

# Fail visibly if the modern models are still missing, so a silent empty result cannot read as
# 'checked, and the rate is confirmed'.
modern = [d for d in seen if re.search(r'4\.5 Haiku|Sonnet 4|Sonnet 5|Opus 4|Opus 5', d)]
print()
if modern:
    print('CHANGED: the price list now carries current-generation Claude rows (above).')
    print('Cross-check them against configs/registry-bedrock.yaml and record whether they agree.')
else:
    print('UNCHANGED from 2026-08-11: legacy Claude ids only, no current-generation rows.')
    print('The registry rates remain single-source.')
"
