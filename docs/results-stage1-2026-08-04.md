# Stage 1 results and findings — 2026-08-04

First session in which Section B (transcribe-then-extract) has real rows. Three transcribers
parsed and scored over the full 1,356-question bank against all 581 corpus documents.

Everything below is measured, not estimated, unless a line says otherwise. Where a claim is an
assumption or remains unverified, it says so.

---

## 1. Section B — the three scored transcriber rows

| row | checkbox acc-over-all | blank/null | hallucination | general/field | strict/question | transcript-recall | median latency |
|---|---|---|---|---|---|---|---|
| `docstrange@nanonets` | **93.0%** [89.1, 96.2] | 93.6% | 6.4% | 89.6% | **82.1%** | **97.6%** | 30.3 s |
| `qwen3-vl-32b@openrouter-transcriber` | 91.5% [87.2, 95.1] | 94.7% | 5.3% | 87.4% | 76.8% | 92.7% | 14.8 s |
| `qwen3-vl-8b@openrouter-transcriber` | 80.6% [74.0, 86.3] | **95.2%** | **4.8%** | 82.3% | 70.1% | 94.3% | 8.0 s |

All three: extractor `gemini-3.1-flash-lite`, input `raster-png`, `beats majority: yes`,
**0 errors** across 4,068 scored cells (1,356 × 3).

### The headline is a tie, not a ranking

Paired-bootstrap separability on checkbox acc-over-all:

| comparison | verdict |
|---|---|
| docstrange vs qwen3-vl-32b | **not separable** — Δ CI [−3.0pp, +6.1pp] spans 0 |
| docstrange vs qwen3-vl-8b | separable, docstrange ahead, Δ [6.1pp, 19.1pp] |
| qwen3-vl-32b vs qwen3-vl-8b | separable, 32b ahead, Δ [4.9pp, 17.1pp] |

**"DocStrange is the best checkbox transcriber" is not supported by this data.** It leads
numerically and has the best transcript-recall, but an Apache-2.0 open-weights 32B ties it inside
the confidence interval — at roughly **one eighth of the transcription cost** ($5.81 vs ~$0.70
by token count) and **half the latency**. Any write-up should state the tie, not the ordering.

### The checkbox thesis, validated

Nanonets was selected because its OCR model prompt is the only surveyed one that explicitly asks
for checkbox glyphs ("Prefer using ☐ and ☑ for check boxes", from the `docstrange` SDK's
`pipeline/nanonets_processor.py`). That evidence came from the SDK's **local** pipeline, and the
hosted endpoint's checkpoint is undisclosed, so it was a prior rather than a fact.

Confirmed: **97.6% transcript-recall** — the fraction of checkbox-bucket documents whose
transcript contains both a checkbox glyph and the gold field key — converting to **93.0%**
checkbox accuracy. The hosted service does emit checkbox state.

### transcript-recall earned its keep

`qwen3-vl-8b` has **higher** transcript-recall (94.3%) than `qwen3-vl-32b` (92.7%) yet is
separably **worse** on checkbox accuracy (80.6% vs 91.5%). It emits checkbox glyphs reliably and
reads their *state* wrong.

That is exactly the distinction the diagnostic exists to draw — "never emitted it" versus "read it
wrong" — and it means the 8B's deficit is not a formatting problem that prompt-tuning would fix.

---

## 2. DocStrange adapter (`docstrange_sync`)

Full corpus: **581/581 ok, 0 failures, 59.1 min, $5.81** — the pre-run estimate was exact to the
cent. Transcript lengths min 344 / median 4,206 / max 32,622 chars, no empty or truncated output.

Two live deviations from the published OpenAPI schema, both handled in the adapter. **Do not
"correct" them back toward the schema without re-checking against the live API:**

1. `result.markdown` is an **object** `{"content": ..., "metadata": ...}`, not the bare string the
   schema documents. Coding to the spec fails on the first real call.
2. `pages_processed` returns **null** from `/extract/sync`, though the same record fetched from
   `GET /api/v1/extract/results/{record_id}` carries `1`. Billing falls back to one page per
   request and records `billed_pages_source` so an audit knows which number it read.

Concurrency: **`--workers 8` is the throughput ceiling.** At 8, median per-call latency (54.9 s) is
indistinguishable from a lone call (52.9 s) — nothing queues. At 16 it rises to 68 s with a 141 s
tail while throughput does *not* improve (6.8 vs 7.1 pages/min). Server-side limit, enforced by
queuing rather than rejection: no 429s at either level.

---

## 3. Extractor: divergence D11

`score.py`'s `DEFAULT_MODEL` moved `gemini-3-flash-preview` → **`gemini-3.1-flash-lite`**,
overriding a spec-ratified decision ("exactly as upstream — required for published-number
comparability"). Grounds:

**Durability.** A *preview* endpoint can be withdrawn. `gemini-2.0-flash-lite` returned
HTTP 404 "no longer available" the same day this was decided. An extractor that vanishes
mid-project destroys reproducibility far more thoroughly than a judge swap.

**Measured equivalence.** Paired A/B, 300 real bank items on real transcripts, identical items per
model, McNemar on discordant pairs:

| extractor | match | vs incumbent | cost/row |
|---|---|---|---|
| `gemini-3.1-flash-lite` | 244/300 (81.3%) | +14/−11, **p=0.690** | **$0.91** |
| `gemini-3-flash-preview` | 241/300 (80.3%) | — (incumbent) | $1.83 |
| `gemini-3.5-flash-lite` | 240/300 (80.0%) | +13/−14, p=1.000 | $1.12 |
| `gemini-2.5-flash` | 235/300 (78.3%) | +12/−18, p=0.362 | $1.12 |

Nothing separates them, so the choice fell to cost, speed and GA status. The full-corpus row later
scored **82.0%**, closely matching the n=300 estimate.

**Cost of the change:** DoD #2 compares our absolute numbers to upstream's published Table 3,
produced with the old extractor, so a reproduction gap now carries one extra uncontrolled
variable. Re-pin `DEFAULT_MODEL` to reproduce upstream exactly.

> ⚠️ **`selftest --extractor` cannot adjudicate this.** All four candidates score 5/5 on its
> fixtures. It is a floor, not a discriminator — passing it is not evidence of equivalence.

Extractor economics, measured: **2,507 input / 31 output tokens per call** — an input-dominated
read, not a generation. Over 1,356 calls that is 3.40M input tokens, $1.83 at
`gemini-3-flash-preview`'s verified Standard rate ($0.50/M in, $3.00/M out). A Flash-Lite swap
saves only $0.70–$0.92/row; the real lever if cost ever bites is the **Batch tier at exactly half
Standard**, with no model change.

---

## 4. Completion budget: 1024/4096 → 12288

Set from measurement, after an intermediate 32k baseline used purely to observe demand.

| model | peak completion tokens (4 densest pages) | seconds/page |
|---|---|---|
| `qwen3-vl-8b` | 3,599 | 8–27 |
| `qwen3-vl-32b` | 3,117 | 9–33 |

Longest transcript produced corpus-wide: 32,622 chars ≈ 8k tokens — the practical ceiling on
legitimate single-page output. **12288 clears that with headroom.**

**Bigger is not free.** A non-terminating model bills whatever ceiling it is given: at 32k,
`qwen3.5-9b` consumed the *entire* budget on 2 of 4 dense pages and still returned a 9-byte
transcript, at 351 s and 1,024 s. Raising the cap converted a cheap failure into an expensive one.

Local vLLM must be served at `--max-model-len 16384` to fit 12288 completion + ~2.5k prompt/image.
The old 8192 window no longer works.

---

## 5. `qwen3.5-9b` is not currently viable as a transcriber

It is the only Qwen entry advertising `reasoning`/`include_reasoning` (verified via the OpenRouter
models API; the two `-instruct` VL models expose neither).

- At **every** budget tried — 4k, 16k, 32k — it consumed the ceiling and returned a 9-byte
  transcript on roughly half the dense pages. This is non-termination, not truncation.
- `reasoning: {max_tokens: 4096}` **is honored** (3,701 reasoning tokens, clean `finish=stop`,
  145 s) on pages where it works.
- `reasoning: {enabled: false}` is **worse**: runaway generation, 32,971 chars, `finish=length`,
  466 s. Disabling thinking is not the fix.
- Its vlm-chat row shows the same disease: ~15% `error_class: "empty"` in its first 153 cells,
  against 3 parse_errors across `qwen3-vl-8b`'s full 1,356.

**Provider caveat:** DeepInfra returned `429 engine_overloaded`
(`limit_source: upstream_provider_shared_pool`) during this investigation. Some of the 335 s and
1,024 s page times are queuing, not model speed — treat those latencies as confounded. Routing
around it is not free: this entry is pinned to DeepInfra with `allow_fallbacks: false` **on ToS
grounds** (SiliconFlow bars commercial use, Together bars sensitive-data transmission, Venice and
Parasail were never assessed), so a provider switch needs a ToS review first.

---

## 6. Two harness defects found and fixed

**Truncated response bodies were classified permanent.** A body cut off mid-stream arrives as a
bare `json.JSONDecodeError` — the openai SDK wraps transport failures *during* the request into
`APIConnectionError`, but a body that arrives and then stops is parsed outside that wrapping.
It matched neither branch of `_is_retryable`. Live cost: a page died after 335 s and was never
retried, while the identical request succeeded in 145 s by hand. Now retried; the check is
deliberately narrow (`json.JSONDecodeError`, not its `ValueError` parent).

**The transcription leg reported `$0.0000` for rows that cost real money.** Dynamically-registered
parser names can never appear in the static `catalog.yaml`, so `parse_cost`/`agent_cost` both miss,
cost returns `None`, and upstream `_parse_one` writes `cost_estimate_usd or 0.0` — turning
"unknown" into "free" on disk. `report_md`'s M1 guard is designed to catch this but only fires when
the entry carries no `pricing`; these entries **do** carry per-token rates, so it concluded the
zero was real. The leg now prices itself from those rates, returning `None` (never `0.0`) when
rates are absent or unusable so the M1 guard still works.

> ⚠️ **Forward-looking only.** The existing Qwen sidecars already have `0.0` written and do not
> store token counts, so no backfill is possible. Those two rows will keep showing `$0.0000` until
> re-parsed (~$1.40 to re-bill). Real cost by token count is ~$0.70 each.

---

## 7. Renting GPU vs hosted per-token APIs

Full survey with sourced prices: `~/notes/claude-things/gpu-pricing-2026-08-04.md` (13 providers,
every figure carrying a URL and check date; unverifiable ones marked `UNVERIFIED`).

Break-even against $0.60/pass hosted, where `h = 581 × s/page ÷ 3600`:

| s/page | break-even $/hr |
|---|---|
| 2 | $1.86 |
| 5 | $0.74 |

**Anything above ~$1.86/hr never beats the hosted API for the 8B, at any volume.** Renting wins on
cheap consumer silicon (RunPod Community RTX 4090 24GB at $0.34/hr → $0.165/pass, ahead from the
first pass). The HF RTX PRO 6000 96GB ($2.75/hr) and H200 ($5.00/hr) are both verified-real prices
but sit permanently above break-even for the 8B — they are 32B cards. For the 32B, Nebius sells
the identical RTX PRO 6000 at $1.80 on-demand / $0.95 preemptible.

**The strongest argument for renting is not price.** `GLM-OCR` and `dots.ocr` are not served
per-token by OpenRouter, so for those entries renting is not the cheaper option — it is the only
one. Self-hosting also removes the ToS analysis entirely and lets `precision` be asserted rather
than recorded as `provider-default` (Section B currently prints "precision unasserted across all
rows").

**Load-bearing unknown: throughput.** Every figure above swings on 2 vs 5 s/page, which is
unmeasured. One hour on a $0.34/hr card settles it for about 34 cents.

---

## 8. Open items

- `qwen3.5-9b` transcriber and vlm-chat rows: blocked on non-termination, plus a provider rate
  limit, plus no ToS-cleared alternative provider.
- Direct leg (Section A) needs a full re-run under the new 12288 condition — the earlier run was
  stopped and its condition hash is superseded (~$1.04).
- The two Qwen transcriber rows show `$0.0000` until re-parsed (see §6).
- Measured GPU throughput (§7) — the one experiment that would resolve the build-vs-buy question.
- `gemini-3-flash-preview` reproduction comparison, if DoD #2 is to be evaluated against upstream's
  exact construction.
