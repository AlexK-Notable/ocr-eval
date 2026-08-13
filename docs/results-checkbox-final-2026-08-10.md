# Checkbox state — final results, 15 legs (2026-08-10, extended 2026-08-11)

Supersedes `results-checkbox-accuracy-cost-2026-08-06.md` as the current numbers table. That file
is retained: its "Correction: why Mistral OCR 4 loses checkboxes" section is the diagnostic
write-up and is **not** duplicated here.

Every leg: **1,356 cells**, pinned dataset revision `906170ab201d7b8238a32a9115fc66b4b72e0710`.
**Checkbox denominator: 258 boolean fields — 165 `true`, 93 `false`. Majority-class baseline
64.0%** (always answer `true`). Any leg below that is worse than a constant.

`cb-acc` is accuracy-over-**all** (an errored cell counts wrong) — this repo's ranking key.
`strict` is the share of cells where *every* field was right, over all 1,356 cells, so it moves
independently. All figures below were recomputed from the cache rows on 2026-08-10, not carried
forward from an earlier report. **Leg 15 was added on 2026-08-11** and leg 8's cost was measured the
same day; everything else is unchanged. Re-derive any comparison here with
`scripts/compare-legs-checkbox.py <parser_key_a> <parser_key_b>`, which uses the repo's own metric
functions so it cannot drift from `report.md`.

---

## The table

| # | Leg | Shape | cb-acc | ✓chk | ✓unchk | Strict | Err | Total cost | Per call | Cost basis |
|---|---|---|---|---|---|---|---|---|---|---|
| 15 | **claude-sonnet-4.6** (Bedrock) | vlm-chat | **98.1%** | 100.0% | 94.6% | 86.4% | 1.5% ⁵ | $8.692 ⁴ | $0.00641 | token rates, single source |
| 1 | gemini-3.6-flash | vlm-chat | **97.7%** | 100.0% | 93.5% | 92.0% | 0.0% | $8.060 | $0.00594 | token rates, incl. thinking |
| 2 | gemini-3.5-flash-lite | vlm-chat | 97.3% | 98.8% | 94.6% | 86.2% | 0.0% | $0.666 | $0.00049 | token rates |
| 3 | docstrange@nanonets | transcriber | **93.8%** | 95.8% | 90.3% | 84.4% | 0.0% | $5.810 | $0.00428 | per-page, exact |
| 4 | **mistral-ocr-4-0** | transcriber | **90.7%** | 91.5% | 89.2% | 82.2% | 0.0% | $2.324 | $0.00171 | per-page, exact |
| 5 | qwen3-vl-8b (OpenRouter) | vlm-chat | 90.7% | 95.8% | 81.7% | 75.1% | 0.0% | $0.390 | $0.00029 | provider cost field |
| 6 | qwen3-vl-32b | transcriber | 90.3% | 93.3% | 84.9% | 79.4% | 0.0% | unknown | unknown | §Cost gaps |
| 7 | **mistral-ocr-4-0 + annotation** | transcriber³ | **89.9%** | 87.9% | **93.5%** | 81.7% | 0.0% | $2.905 | $0.00214 | per-page, rate unverified |
| 8 | claude-haiku-4.5 (Bedrock) | vlm-chat | 88.8% | 96.4% | 75.3% | 71.7% | 0.0% | $2.871 ⁴ | $0.00212 | token rates, single source |
| 9 | gemini-3.1-pro-preview | vlm-chat | 88.8% | 89.7% | 87.1% | 89.0% | **10.1%** | $9.509 | $0.00701 | token rates, incl. thinking |
| 10 | qwen3-vl-8b-instruct (ollama) | vlm-chat | 87.2% | 93.3% | 76.3% | 70.3% | 0.0% | $0 (local) | $0 | no API billing |
| 11 | **mistral-ocr-4-1** | transcriber | 85.7% | 86.1% | 84.9% | 80.7% | 0.0% | $2.324 | $0.00171 | per-page, exact |
| 12 | qwen3-vl-8b | transcriber | 81.8% | 77.6% | 89.2% | 72.7% | 0.0% | unknown | unknown | §Cost gaps |
| 13 | mistral-small-4 Doc QnA *(vendor default)* | vlm-chat | 80.6% | 88.5% | 66.7% | 68.7% | 0.0% | $0.309 | $0.00023 | token rates |
| 14 | mistral-small-4 Doc QnA *(temp 0.0)* | vlm-chat | 79.5% | 87.3% | 65.6% | 67.3% | 0.0% | $0.309 | $0.00023 | token rates |

³ Leg 7 is **three stages** (Mistral OCR → Mistral vision LLM → gemini extractor), not two. See
§Leg 7 below — it is not a like-for-like against leg 4.

⁴ **Priced 2026-08-11, from one source.** Leg 8 read `unknown` when this table was published, and
leg 15 was measured that day. See §Addendum — the rates are in the registry now, but each has a
single source, and each figure covers the 1,336 of 1,356 cells that recorded usage.

⁵ Leg 15's 1.5% error rate is **20 cells rejected for payload size and 7 unparseable responses, and
it costs it nothing here**: the `Err` column is bank-wide, while `cb-acc` is over the 258 boolean
fields — none of which live in an errored cell (`n_answered=258`, error_rate 0.0%). Leg 8 loses the
*same* 20 cells for the same reason. See §Addendum.

**Row numbers are stable identifiers assigned when a leg was added, not ranks.** Leg 15 was measured
on 2026-08-11 and sorts first; legs 1–14 keep the numbers other sections reference (§Leg 7).

Rows 13 and 14 are the **same model under two sampling conditions**, kept as separate rows under
distinct condition hashes (`859a42584aea` vendor-default, `a37c368fa220` temperature 0.0).

**Transcriber legs understate cost.** Add ~**$0.00068/cell** (~$0.92/leg) for the per-cell
`gemini-3.6-flash` extractor call, which is not recorded per row and so cannot be attributed
exactly.

---

## Cost gaps — read before quoting any per-call figure

**Two** legs report `unknown`, **not `$0`** — a printed `$0.0000` would read as "free".
(Three when this was published; leg 8 was priced on 2026-08-11, see §Addendum.)

* **qwen3-vl-8b / qwen3-vl-32b transcribers** — registry carries per-token rates, but the parse
  records persisted `cost_usd: 0.0` (upstream writes `result.cost_estimate_usd or 0.0`, converting
  unknown into a literal zero) and token counts were never stored. Real spend ≈$0.70/leg.
  **Not recoverable from disk.** These two rows rendered **`$0.0000`** in `report.md` until
  2026-08-11: the guard that catches a coerced zero required the entry to be *unpriced*, and these
  are priced, so they fell through to a printed zero. They now render `n/a (unrecorded)` — the
  distinction the guard was written for, finally applied to the case that actually shipped.
* **qwen3-vl-8b-instruct (ollama)** — a genuine `$0`: `local: true`, own hardware. A real zero,
  unlike the two above.

**Leg 7's rate is unverified.** Priced at the annotations rate ($0.005/page) rather than plain OCR
($0.004), because Mistral lists "Document AI" higher. The response's `usage_info` is *byte-identical*
with and without `document_annotation_format`, so the API never confirms which rate applied. Priced
high deliberately — an underpriced parser defeats `--max-spend`. True cost may be $2.324.

---

## What the numbers say

### Gemini flash: tied on checkboxes, 12× apart on price

McNemar over the 258 fields: 252 vs 251 correct, **p=1.0**, 96.5% agreement. The 97.7% vs 97.3%
gap is noise. **3.5-flash-lite costs 12× less** ($0.666 vs $8.060), so on checkbox state it is the
better buy. The gap is that wide because 3.6-flash bills **672,517 hidden thinking tokens** — 11×
its visible output — while flash-lite emits none. They *do* separate on `strict` (92.0% vs 86.2%) — a different, harder claim.

### Mistral OCR: 4-0 beats 4-1, and the alias points at the weaker pin

234 vs 221 correct, discordant 27/14, **p=0.060** — not significant at α=0.05; the honest statement
is "not detected at n=258". Two things qualify it further:

* **Excluding the 3 documents affected by 4-1's repetition defect moves p from 0.060 to 0.108**
  (231 vs 220, n=255). Three documents carry a noticeable share of the result.
* `mistral-ocr-latest`, `mistral-ocr-4` and `mistral-ocr-4-1` are **one mutually-aliased model**;
  `mistral-ocr-4-0` has no aliases. So the alias resolves to the apparently weaker pin, and anyone
  trusting it silently gets 4-1.

### Leg 7 — annotations recover marks case-by-case but net to zero

`document_annotation` with an explicit `{label, checked}` schema. The mechanism is real: on
`finance_74`, where plain OCR emits bare `Methane (CH4)` with no glyph, the annotation on the same
call returns `{"label":"Methane (CH4)","checked":true}` — the engine holds selection state that the
markdown serialization drops. A hand-picked probe on the two worst pages scored **6/6 against gold,
including 4 `false` values**, on fields plain OCR got wrong.

**At full scale it nets to nothing: 232 vs 234, discordant 9/11, p=0.82.** It fixes 11 fields and
breaks 9.

> **Method warning, paid for here.** The 6/6 probe selected pages *on the outcome* (chosen because
> plain OCR failed there), so it could only ever show fixes, never regressions. It demonstrated a
> mechanism and was mistaken for an effect size. A hand-picked probe is never an effect-size
> estimate. Cost of learning this: **$3.83**.

The genuinely interesting result is the **polarity inversion**: leg 7 has the best unchecked-box
accuracy of any Mistral leg — **93.5%**, beating even DocStrange's 90.3% — while dropping from
91.5% to 87.9% on checked boxes. Asking Mistral for structured checkbox state makes it better at
confirming a box is empty and worse at confirming it is filled. A single accuracy figure hides
this completely.

### Polarity separates legs that share a headline number

Legs 4 and 5 tie at 90.7% with opposite profiles: **mistral-ocr-4-0** balanced (91.5%/89.2%),
**qwen3-vl-8b direct** skewed (95.8%/81.7%). **claude-haiku-4.5** is the most skewed
(96.4%/75.3%) — it reads boxes as checked. **Doc QnA is the worst on unchecked boxes** (66.7%). For
a checkbox benchmark the asymmetry is usually the more useful number.

### Doc QnA: the temperature fix changed nothing

Leg 14 ran at `temperature: 0.0` — inherited from a condition ratified for Qwen/Bedrock, never
checked against Mistral, whose `/models` endpoint reports `default_model_temperature: 0.3` for
`mistral-small-2603`. Leg 13 re-ran at fully vendor-default sampling (all params omitted, no token
cap). **80.6% vs 79.5%, discordant 8/5, p=0.58** — not detected. The condition was genuinely wrong
and worth $0.31 to eliminate; it explains none of the gap.

### The pro leg's error rate is why it ranks low

`gemini-3.1-pro-preview` has a **10.1% error rate over the 258 checkbox fields** (26 of 258; 47 of
1,356 rows bank-wide: 43 parse errors, 3 API errors, 1 empty). Errors count wrong under
accuracy-over-all, which is why its 88.8% `cb-acc` sits *below* its 89.0% `strict`. **Only the 3
API errors are plausibly transient** — a `parse_error` is a malformed response that re-running
re-bills with no reason to expect a different result.

---

## Mistral configuration: every lever tested, none helped

| Lever | Result |
|---|---|
| `temperature` on `/v1/ocr` | **Not exposed.** No sampling parameter exists on the endpoint. |
| Doc QnA sampling | Vendor default vs temp 0: **p=0.58**, no effect (legs 13/14). |
| Re-running loss pages | **Deterministic.** Byte-identical glyph counts on re-probe; `finance_74` reproduced to the exact character. |
| `include_blocks: true` | 45 blocks, 16 KB of layout geometry, **zero checkbox glyphs**. Markdown byte-identical. |
| `table_format: html` | **Harmful.** Moves table content out of `markdown` into a separate `tables` field (4,608 → 289 chars); only `markdown` reaches the extractor. |
| `document_annotation` + schema | Recovers marks case-by-case, **nets to zero** (leg 7, p=0.82). |
| Vendor guidance on checkbox glyphs | **None exists.** Four Mistral pages checked (OCR basic, annotations, Document QnA, launch post) — none mentions checkboxes, checkmarks, radio buttons or selection state. |

**Conclusion: Mistral's checkbox ceiling on this corpus is ~90–91% regardless of configuration**,
and the shortfall is undocumented model behaviour with no available knob. DocStrange stays ahead at
93.8%; both Gemini flash models are far ahead at ~97.5%.

---

## Correction: thinking tokens were missing from every published cost (2026-08-10)

Cost was computed from `prompt_tokens + completion_tokens`. A reasoning model bills its internal
reasoning as **output**, and Gemini reports that in `usageMetadata.thoughtsTokenCount` — *outside*
`candidatesTokenCount`, which is what this repo maps onto `completion_tokens`. So every
thinking-model figure published before this date understated a real bill:

| leg | thinking tokens | was | **is** | understated |
|---|---|---|---|---|
| gemini-3.6-flash | 672,517 | $3.016 | **$8.060** | +167% |
| gemini-3.1-pro-preview | 447,654 | $4.137 | **$9.509** | +130% |
| gemini-3.5-flash-lite | **0** | $0.666 | $0.666 | — |

`flash-lite` emits no thinking tokens on any of its 1,356 rows, so its figure was always right —
which is exactly why the bug survived a side-by-side against a thinking model.

**Proof it is real, not a rate assumption:** `promptTokenCount + candidatesTokenCount +
thoughtsTokenCount == totalTokenCount` **exactly**, on all 4,065 Gemini rows. Google counts them;
we did not read them.

**It also weakened `--max-spend`**, which shares the same formula: on a thinking model the guard
was enforcing a ceiling ~2.7× higher than the operator asked for.

Fixed in `direct.billable_output_tokens`, used by both cost paths. It reads the preserved native
payload rather than a new stored field, so already-written rows now cost correctly with no cache
rewrite and no condition-hash change. `direct.unaccounted_tokens` is the generalizable guard —
`totalTokenCount` minus what we count must be 0 — with a positive-control test proving it detects
a hypothetical unread field.

Same family as the omitted `candidatesTokenCount` that killed a full-bank run ~300 paid cells in:
**a billable quantity the provider reports and we did not read.** Assume every new transport has
one until its usage payload has been enumerated field by field.

---

## Addendum 2026-08-11: leg 15 added, leg 8 priced, and Bedrock's real image ceiling

### Leg 15 — claude-sonnet-4.6 is the nominal best leg and changes nothing

**98.1% (253/258)**, the highest checkbox accuracy measured here, 100.0% on checked boxes and 94.6%
on unchecked. It is **not separable from either Gemini flash model**:

| comparison | correct | discordant | p (McNemar exact) | agreement |
|---|---|---|---|---|
| vs gemini-3.6-flash | 253 vs 252 | 4 / 3 | **1.000** | 97.3% |
| vs gemini-3.5-flash-lite | 253 vs 251 | 5 / 3 | **0.727** | 96.9% |
| vs claude-haiku-4.5 (same transport) | 253 vs 229 | **25 / 1** | **8.0e-7** | 89.9% |

So the top of this table is a **three-way tie at ~97.5–98%** that no longer separates, and
`gemini-3.5-flash-lite` remains the buy at **1/13th of leg 15's cost** ($0.666 vs $8.692). The
Haiku comparison is the one that resolves: same transport, same region, same provider, and Sonnet
4.6 is ahead by **9.3 points** at 3.0× the price.

**One thing here is genuinely new, and it is not the accuracy.** Legs 1 and 2 both post-date
`CONTAMINATION_CUTOFF` and are labelled ceiling anchors for that reason. Leg 15 is
**pre-cutoff (2025-11-14)** — so a clean-baseline leg now matches the contaminated ceiling. That is
the strongest evidence in this table that the ~97–98% figures are not artefacts of training-set
overlap.

Two caveats travel with it. It ran at condition `0373123f3b15` (temp 0.0, max_tokens 12288) while
leg 8 is `4caf39e9ac1f` (temp 0.0, **max_tokens 1024**) — checked before comparing: leg 8's cap was
never binding (all 1,336 of its rows with usage ended `end_turn`, zero max-token stops), so that
difference is a hash, not a measurement artefact. Against the Gemini legs it is the usual
cross-regime comparison (they run at vendor sampling, temp 1.0 / top_p 0.95).

**What the 20-cell smoke got right and wrong**, since it authorised the spend: it projected **$8.01**
against an actual **$8.6917** — **8.5% low**, because the smoke's 20 cells (bank order, all
`finance`) averaged 30.2 output tokens against the corpus's 56.9, while input was only 2.9% low. It
still beat the dry-run's ±2x heuristic ($11.63, 34% high). Read a bank-order smoke as a proof that
the pipeline works and an input-token measurement — not as an output-length estimate.

### Bedrock's "5 MB image" limit — already known, now bounded exactly

**This was documented in `docs/api.md` on 2026-08-04 and I rediscovered it.** Recording that
plainly, because rediscovering a written-down fact is the failure the repo's notes exist to prevent,
and because it made me wrong twice today: I attributed leg 8's 20 errors to throttling and
recommended `--workers 4` to reduce them. They are not throttling. `--workers` is irrelevant to them.

What is new is the boundary, measured over two complete banks. Both Bedrock legs lose the **same 20
of 1,356 cells** across the same 10 documents with
`ValidationException: image exceeds 5 MB maximum: 7057648 bytes > 5242880 bytes`, and the data rules
out the obvious reading of that message on its own: the **largest failing render is 3,933,782 raw
bytes — under 5 MB**. The cap applies after base64 inflation. Every failing cell clears it only once
encoded (3,933,782 → 5,245,044, over by 2,164 bytes); the largest succeeding render encodes to
5,035,960, under. **So the real ceiling on a PNG is 3,932,160 raw bytes** (5 MiB × ¾), and at
`render.dpi: 150` those 10 documents are unreachable through Converse. Lowering dpi is a different
condition, not a fix.

**It costs the checkbox figures nothing:** none of the 258 boolean fields live in those cells, which
is why both Bedrock legs report `error_rate` 0.0% on `cb-acc` while carrying a 1.5% bank-wide error
rate. Confirm that again before assuming it holds for any other metric.

### Leg 8 is priced

**Leg 8 now costs $2.8706.** `claude-haiku-4.5@bedrock` carried `unknown` because Bedrock Converse
returns no cost field and `pricing:GetProducts` is IAM-denied to this role. Its token counts were on
disk the whole time, so pricing it needed a rate and nothing else:

| | tokens | rate (per Mtok) | |
|---|---|---|---|
| input | 2,252,308 | $1.10 | $2.4775 |
| output | 71,464 | $5.50 | $0.3931 |
| | | **total** | **$2.8706** |

The rate is the "Claude 4.5 Haiku" row of the Anthropic on-demand table on
`aws.amazon.com/bedrock/pricing/`, read 2026-08-11, and is now in
`configs/registry-bedrock.yaml`, so `report.md` computes this figure itself and `--max-spend` works
against Anthropic-on-Bedrock entries for the first time.

**Three things this figure is not.**

1. **It is not two-source.** This repo's convention for a committed rate is two independent sources
   agreeing exactly, as the Gemini rows have. That was attempted and failed. AWS's *public* bulk
   Price List (no auth, unlike the denied `GetProducts`) carries 1,014 us-east-1 rate rows at
   publication `2026-08-11T16:17:54Z` and **five** Anthropic rows, all legacy — Claude Instant, 2.0,
   2.1, 3 Haiku, 3 Sonnet. No Claude 4.x or 5 row exists in it, input or output. The pricing page's
   own table also does not survive automated extraction (two WebFetch passes returned only the two
   "Public Extended Access" Claude 3.5 Sonnet rows). Re-check with
   `scripts/bedrock-price-list-claude.sh`, which prints `CHANGED` if AWS ever publishes them.
2. **It is not the whole bank.** 1,336 of 1,356 cells recorded usage; the other 20 are `api_error`
   rows that recorded none. So this is the cost of the completed part of the leg. A full 1,356 would
   be roughly $2.91 if the missing cells resemble the rest — which is an inference, not a
   measurement.
3. **It is not a checked rate against the actual bill.** Nothing here was reconciled against what
   AWS charged the account. Cost Explorer would settle it; that was not done.

**Bedrock is exempt from the thinking-token correction above, and that is measured.** Anthropic bills
reasoning *inside* `outputTokens` rather than alongside it, so there is no Gemini-shaped field to
miss. Converse's `usage` carries exactly five fields — inputTokens, outputTokens, totalTokens,
cacheReadInputTokens, cacheWriteInputTokens — `inputTokens + outputTokens == totalTokens` holds on
**all 1,336** rows, and both cache counts are 0 on every one. A live probe of
`us.anthropic.claude-sonnet-4-6` at temperature 0.0 / maxTokens 12288 returned a single `text` block
with no `reasoningContent`: extended thinking on Bedrock is opt-in through
`additionalModelRequestFields`, which this transport does not send. `direct.unaccounted_tokens` now
reconciles `bedrock_usage` the same way it reconciles `gemini_usage`, with a positive-control test.
**The unproven part:** the two cache fields bill at their own rates and are not part of
`inputTokens`; whether a nonzero cache count would land inside `totalTokens` and so be caught is
untested, because testing it means paying to enable caching.

**Where leg 8 lands now that it has a price.** $2.871 for 88.8% is dominated on both axes by
`gemini-3.5-flash-lite` — **4.3× the cost for 8.5 points less accuracy**. It is cheaper all-in than
Mistral OCR 4-0 ($3.25 with its extractor) and less accurate than it too. Pricing this leg did not
change any ranking; it removed the last reason to describe the ranking as incomplete on cost.

---

## What the cost column compares — and the transcript it never charges for (2026-08-12)

Two billing bases sit in one column, and the difference is not a rate difference but a difference in
what the money buys.

| basis | legs | rate |
|---|---|---|
| **per page**, flat, regardless of the question | docstrange | $0.0100/page |
| | mistral-ocr-4-0 / 4-1 | $0.0040/page |
| | mistral-ocr-4-0 + annotation | $0.0050/page (unverified) |
| **per token** | all Gemini, both Claude, Doc QnA | in + out |

### A vlm-chat leg's bill is the page image, not the answer

Gemini reports input tokens by modality, so this is measured per row, not inferred.
`gemini-3.5-flash-lite`, over its 1,356 rows:

| component | tokens | cost | share |
|---|---|---|---|
| **IMAGE input** | 1,484,575 | $0.4454 | **66.8%** |
| TEXT input | 213,087 | $0.0639 | 9.6% |
| output | 62,790 | $0.1570 | 23.6% |

So the cheap leg is cheap because of a **$0.30/Mtok input rate on a ~1,095-token page image**, not
because its structured answer is small. Contrast `gemini-3.6-flash`, where thinking inverts the
shape: **68.4% output**. Both Claude legs are input-dominated too (14.4% / 13.7% output).

### The bank overcharges the vlm-chat legs by re-sending each image ~2.3×

581 pages carry 1,356 questions — mean **2.33 per page**, median 2, max 8, and 157 pages carry only
one. Each vlm-chat question is its own call with the image re-attached; a transcriber OCRs the page
once and every question reads the same transcript.

Summing each page's image tokens **once** instead of per question: 1,484,575 → 635,984 image tokens,
and flash-lite's $0.6663 → **$0.4117 (−38%)**, or $0.00030/question. The per-page legs cannot benefit
— they are already one call per page. **This is arithmetic on measured tokens, not a leg that ran:**
batching 8 questions into one call could move accuracy in either direction and that is untested.

### The transcriber shape cannot win on this task at any question density

The second stage is itself token-billed. The measured extractor cost is **~$0.00068/question** —
**39% more than an entire flash-lite call** ($0.00049) and 2.3× the batched figure. So even
amortising a page rate to zero over unlimited questions per page, a two-call pipeline still costs
more per answer than a one-call one. On this task the page rate is pure additional cost.

### …but that is because it never writes the transcript, and writing one is the expensive part

**ESTIMATE, not a measurement. No Gemini transcriber leg has ever been run** — `gemini-3.5-flash@google`
(`registry.yaml:88`, `shape: transcriber`, `upstream_parser: gemini_3_5_flash`) is registered and
priced but unrun, and there is no flash-**lite** transcriber entry at all.

Sizing one from this run's own data. Per page, flash-lite would send the same ~1,095 image tokens
plus a transcription prompt (~$0.216 input for all 581 pages) and emit a whole transcript instead of
46 tokens of JSON:

| output sizing | chars/token | output tokens | output cost | **stage-1 total** | **per page** |
|---|---|---|---|---|---|
| high-token floor | **2.29** (measured on flash-lite's own JSON output) | 1,231,669 | $3.079 | **$3.295** | $0.0057 |
| typical prose | 4.00 (assumed) | 705,130 | $1.763 | **$1.979** | $0.0034 |

against **mistral-ocr-4-0 at $0.0040/page** and **docstrange at $0.0100/page**.

**Mistral's page rate sits inside the range a token-billed model would charge to do the same work.**
It is not a markup. DocStrange's is 2–3× above it. And that is transcription alone — a flash-lite
transcriber leg would then pay an extractor call per question on top, landing at or above
mistral-ocr-4-0's $3.25 all-in.

**So `gemini-3.5-flash-lite`'s ~5× advantage over the OCR legs comes entirely from skipping
transcription, not from being cheaper at it.** A transcript is 26–46× more output than an answer, and
output bills at 8.3× the input rate ($2.50 vs $0.30).

Sources of error in the estimate, in the order they matter: the transcript length is DocStrange's
(2,820,521 chars over 581 pages), and flash-lite's could differ materially; the 2.29 chars/token
figure is real but measured on JSON, so it over-counts tokens for prose and is used deliberately as
the expensive bound; and no transcription prompt has been written, so its input contribution is
approximate. **A 20-page smoke would replace all of this with a measurement for roughly $0.07–0.11.**

### What this means for the ranking

Nothing in the table moves — every figure prices the same job, *answer this question about this
page*, and the ranking for that job stands. What changes is what may be concluded from it: **these
numbers do not price obtaining a reusable transcript**, and no leg here scores transcript quality as
a deliverable (only whether the checkbox answer derived from it was right). A pipeline that needs the
transcript for other reasons should not attribute the whole page rate to the checkbox question.

---

## Caveats that travel with this table

1. **Three sampling regimes in one table.** Gemini rows are at vendor default (temp 1.0/top_p 0.95);
   Haiku, the qwen family and ollama are at `temperature: 0.0`; leg 13 omits sampling params
   entirely; leg 14 is temp 0.0. A cross-provider delta is not a pure capability comparison.
2. **Contamination.** `gemini-3.6-flash`, `gemini-3.5-flash-lite` (2026-07-21) and all three Mistral
   OCR 4 legs (2026-06-23) post-date `CONTAMINATION_CUTOFF` (2026-05-24) — **ceiling anchors, not
   clean baselines**. `gemini-3.1-pro-preview` (2026-02-19) and `mistral-small-4` (2026-03) are
   pre-cutoff.
3. **Shapes are not interchangeable.** `vlm-chat` answers in one call; `transcriber` is OCR + a
   separate extractor; leg 7 is three stages; Doc QnA is `vlm-chat` with provider-side OCR. Ranking
   them in one table is convenient, not rigorous.
4. **pdf-direct free ride.** Doc QnA and the OCR legs upload the PDF, so a born-digital file may be
   read from its embedded text layer rather than from pixels — materially easier than OCR-ing a scan.
5. **`gemini-3.1-pro-preview` is a preview endpoint** and can be withdrawn without notice, so its
   row may not be reproducible.
6. **STALE-RENDER.** Four legs (both ollama rows, qwen3-vl-8b@openrouter, qwen3.5-9b) carry rows
   whose stored `image_sha` no longer matches a fresh re-render; `report.md` renders them
   `⚠STALE-RENDER` and requires `--allow-stale-render` to build.
7. **DocStrange is the reference transcript, not ground truth,** wherever a Mistral-vs-DocStrange
   glyph ratio is quoted. A page where DocStrange over-emits registers as Mistral "losing" glyphs.
8. **Two billing bases in one cost column, buying different things.** The per-page legs deliver a
   full transcript; the vlm-chat legs deliver only the answer and never transcribe. The cost column
   prices *answering a question about a page*, never *obtaining a reusable transcript* — and
   flash-lite's advantage is a consequence of that, not of a better transcription rate. See
   §What the cost column compares.

---

## Session spend

| Item | Cost |
|---|---|
| Mistral OCR, both plain pins | $4.648 |
| Mistral OCR + annotation (leg 7) | $2.905 |
| gemini-3.1-pro-preview | $9.509 |
| gemini-3.6-flash | $8.060 |
| gemini-3.5-flash-lite | $0.666 |
| Doc QnA × 2 conditions | $0.618 |
| Extractor (measured, 2,712 cells) | ~$1.850 |
| Probes (models endpoint, blocks, annotation) | ~$0.04 |
| **Total measured, 2026-08-10 session** | **≈$28.3** |

Plus two legs of unmetered prior spend (§Cost gaps).

**2026-08-11 session, separately authorised:**

| Item | Cost |
|---|---|
| claude-sonnet-4.6 full bank (leg 15) | $8.6917 |
| ↳ of which the 20-cell smoke that authorised it | $0.1181 |
| Bedrock invokability probes (4 models, ~15 tokens each) | <$0.001 |
| **Total** | **≈$8.69** |

Leg 8's $2.8706 is **not** new spend — it was paid on 2026-08-04 and only priced on 2026-08-11, so
it belongs in the all-sessions total and in neither table above.

**All sessions:** $28.3 (through 2026-08-10) + $8.69 (2026-08-11) + $2.87 (leg 8, paid 2026-08-04)
= **≈$39.9 measured**, plus the two unmetered qwen transcriber legs at roughly $0.70 each.
