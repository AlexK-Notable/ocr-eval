# Checkbox state — final results, 14 legs (2026-08-10)

Supersedes `results-checkbox-accuracy-cost-2026-08-06.md` as the current numbers table. That file
is retained: its "Correction: why Mistral OCR 4 loses checkboxes" section is the diagnostic
write-up and is **not** duplicated here.

Every leg: **1,356 cells**, pinned dataset revision `906170ab201d7b8238a32a9115fc66b4b72e0710`.
**Checkbox denominator: 258 boolean fields — 165 `true`, 93 `false`. Majority-class baseline
64.0%** (always answer `true`). Any leg below that is worse than a constant.

`cb-acc` is accuracy-over-**all** (an errored cell counts wrong) — this repo's ranking key.
`strict` is the share of cells where *every* field was right, over all 1,356 cells, so it moves
independently. All figures below were recomputed from the cache rows on 2026-08-10, not carried
forward from an earlier report.

---

## The table

| # | Leg | Shape | cb-acc | ✓chk | ✓unchk | Strict | Err | Total cost | Per call | Cost basis |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | gemini-3.6-flash | vlm-chat | **97.7%** | 100.0% | 93.5% | 92.0% | 0.0% | $8.060 | $0.00594 | token rates, incl. thinking |
| 2 | gemini-3.5-flash-lite | vlm-chat | 97.3% | 98.8% | 94.6% | 86.2% | 0.0% | $0.666 | $0.00049 | token rates |
| 3 | docstrange@nanonets | transcriber | **93.8%** | 95.8% | 90.3% | 84.4% | 0.0% | $5.810 | $0.00428 | per-page, exact |
| 4 | **mistral-ocr-4-0** | transcriber | **90.7%** | 91.5% | 89.2% | 82.2% | 0.0% | $2.324 | $0.00171 | per-page, exact |
| 5 | qwen3-vl-8b (OpenRouter) | vlm-chat | 90.7% | 95.8% | 81.7% | 75.1% | 0.0% | $0.390 | $0.00029 | provider cost field |
| 6 | qwen3-vl-32b | transcriber | 90.3% | 93.3% | 84.9% | 79.4% | 0.0% | unknown | unknown | §Cost gaps |
| 7 | **mistral-ocr-4-0 + annotation** | transcriber³ | **89.9%** | 87.9% | **93.5%** | 81.7% | 0.0% | $2.905 | $0.00214 | per-page, rate unverified |
| 8 | claude-haiku-4.5 (Bedrock) | vlm-chat | 88.8% | 96.4% | 75.3% | 71.7% | 0.0% | unknown | unknown | §Cost gaps |
| 9 | gemini-3.1-pro-preview | vlm-chat | 88.8% | 89.7% | 87.1% | 89.0% | **10.1%** | $9.509 | $0.00701 | token rates, incl. thinking |
| 10 | qwen3-vl-8b-instruct (ollama) | vlm-chat | 87.2% | 93.3% | 76.3% | 70.3% | 0.0% | $0 (local) | $0 | no API billing |
| 11 | **mistral-ocr-4-1** | transcriber | 85.7% | 86.1% | 84.9% | 80.7% | 0.0% | $2.324 | $0.00171 | per-page, exact |
| 12 | qwen3-vl-8b | transcriber | 81.8% | 77.6% | 89.2% | 72.7% | 0.0% | unknown | unknown | §Cost gaps |
| 13 | mistral-small-4 Doc QnA *(vendor default)* | vlm-chat | 80.6% | 88.5% | 66.7% | 68.7% | 0.0% | $0.309 | $0.00023 | token rates |
| 14 | mistral-small-4 Doc QnA *(temp 0.0)* | vlm-chat | 79.5% | 87.3% | 65.6% | 67.3% | 0.0% | $0.309 | $0.00023 | token rates |

³ Leg 7 is **three stages** (Mistral OCR → Mistral vision LLM → gemini extractor), not two. See
§Leg 7 below — it is not a like-for-like against leg 4.

Rows 13 and 14 are the **same model under two sampling conditions**, kept as separate rows under
distinct condition hashes (`859a42584aea` vendor-default, `a37c368fa220` temperature 0.0).

**Transcriber legs understate cost.** Add ~**$0.00068/cell** (~$0.92/leg) for the per-cell
`gemini-3.6-flash` extractor call, which is not recorded per row and so cannot be attributed
exactly.

---

## Cost gaps — read before quoting any per-call figure

Three legs report `unknown`, **not `$0`** — a printed `$0.0000` would read as "free".

* **qwen3-vl-8b / qwen3-vl-32b transcribers** — registry carries per-token rates, but the parse
  records persisted `cost_usd: 0.0` (upstream writes `result.cost_estimate_usd or 0.0`, converting
  unknown into a literal zero) and token counts were never stored. Real spend ≈$0.70/leg.
  **Not recoverable from disk.**
* **claude-haiku-4.5** — unpriced by design: Bedrock Converse returns no cost field and
  `pricing:GetProducts` is IAM-denied here. Token counts **are** recorded
  (**2,252,308 in / 71,464 out**), so it can be priced later from published rates.
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
| **Total measured** | **≈$28.3** |

Plus three legs of unmetered prior spend (§Cost gaps).
