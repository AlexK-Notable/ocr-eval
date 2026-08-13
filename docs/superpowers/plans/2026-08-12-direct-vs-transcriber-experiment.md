# Direct vs transcriber: experiment design for a definitive answer (2026-08-12)

**Status: design only. Nothing here has been run.** This is the detailed plan the Stage 2–3 roadmap
deferred — Stage 3 item 3 ("direct-vs-two-stage formalized") merged with item 4 ("extractor
sensitivity"), written now because Stage 1's results are in, which was the roadmap's stated
precondition.

**Standing instruction for this experiment: spend is not a constraint. Rigor is.** Every design
choice below should be read as "what would make this answer hold up", not "what is affordable".

---

## 1. The question, and the estimand

Informally: *asking a VLM to transcribe a whole page is a different task from asking it one targeted
question about that page, and the difference should show up in accuracy, not just cost.*

Formally, the quantity to estimate is a **within-model paired shape delta**:

> Δ_m = (field accuracy of model *m* answering directly from the page image)
>     − (field accuracy of model *m* transcribing the page, with a fixed extractor answering from
>        that transcript)

and then the **population-level shape penalty** across models, `Δ = E[Δ_m]`, with an interval that
accounts for between-model heterogeneity. Two distinct claims, and they need distinct evidence:

* **Claim A (per model):** "this model loses X points in transcriber shape." Paired within model.
* **Claim B (the shape):** "the transcriber shape costs X points on average." Requires several models
  and a random-effects estimate over per-model deltas — one model cannot support it.

The existing `qwen3-vl-8b` pair supports A for one model. Nothing yet supports B.

---

## 2. What is already measured

`qwen3-vl-8b`, identical weights, both shapes, the same 258 boolean checkbox fields (this is the
repo's DoD #3 calibration pair — already run and scored):

| | cb-acc | ✓checked | ✓unchecked |
|---|---|---|---|
| direct (image → answer) | **90.7%** (234/258) | 95.8% | 81.7% |
| transcriber (image → markdown → gemini-3.6-flash) | **81.8%** (211/258) | 77.6% | 89.2% |

Naive McNemar: discordant **43 / 20, p=0.0052**, agreement 75.6%. **−8.9 points.**

Two things make this more than a single number:

1. **The polarity inverts** — −18.2 pt on checked boxes, **+7.5 pt on unchecked**. That is the
   glyph-omission signature: the transcript drops the mark, the extractor reads absence as "unchecked",
   so the leg gains on empty boxes while collapsing on filled ones. Same mechanism the Mistral
   investigation isolated (`results-checkbox-accuracy-cost-2026-08-06.md` §Correction).
2. **The transcription prompt is not a strawman.** `cloud_vlm.py:_MARKDOWN_PROMPT` explicitly says
   *"Render checkboxes inline as ☒ (checked) or ☐ (unchecked), paired with their adjacent label on the
   same line."* The shape loses anyway. A null result on a *weak* prompt would prove nothing; this is
   not that.

---

## 3. Threats to a definitive answer

This section is the point of the document. Each threat has a required countermeasure.

### 3.1 The 258 booleans are clustered, and every published p-value is anti-conservative

258 boolean fields sit in **123 documents** (mean 2.10 fields/doc, max 5). Fields on one page share a
render, a transcript, and a glyph-rendering decision, so they are not independent draws. McNemar
assumes they are.

**Measured ICC of per-field correctness, clustered by document** (one-way ANOVA on binary
correctness, computed 2026-08-12 from the committed rows):

| leg | ICC | design effect | effective n (of 258) |
|---|---|---|---|
| qwen3-vl-8b **direct** | 0.214 | 1.23 | 209 |
| qwen3-vl-8b **transcriber** | **0.420** | **1.46** | **177** |
| gemini-3.5-flash-lite (direct) | 0.106 | 1.12 | 231 |
| claude-sonnet-4.6 (direct) | 0.302 | 1.33 | 194 |
| docstrange (transcriber) | 0.187 | 1.20 | 214 |
| mistral-ocr-4-0 (transcriber) | 0.328 | 1.36 | 190 |

**Two consequences, and the second is a finding in its own right.**

* Effective n is **177–231, never 258**. Standard errors are understated by √(design effect) — up to
  **21%** on the z-scale. Every McNemar p-value quoted in `results-checkbox-final-2026-08-10.md` is
  optimistic by roughly that factor. None of the strong results (p=8e-7, p=0.0052) flips, but the
  marginal ones (Mistral 4-0 vs 4-1 at p=0.060, leg 7 at p=0.82) must not be re-quoted without this.
* **Transcriber legs cluster roughly 2× harder than direct legs** (ICC 0.42/0.33 vs 0.11/0.21). That
  is mechanistically expected — one transcript governs every field on its page, so a transcription
  failure takes out a whole document's fields together, while a direct leg's errors are drawn
  per-question. It also means the shape comparison has **shape-dependent clustering**, so a
  cluster-robust method is not optional and the transcriber arm needs it more.

**Countermeasure — and most of it already exists.** `ocr_eval_ext/stats.py` already implements
`cluster_bootstrap_ci` and `paired_delta_ci`, both resampling **documents**, and `report.md`'s
Section A already uses them (that is what its `Δ CI … spans 0` lines are). What fell behind is the
*prose*: every McNemar figure quoted in the results docs, and `scripts/compare-legs-checkbox.py`
written on 2026-08-11, use the naive field-level test only. Fix the script, quote both, label the
naive one naive. A document-level paired **permutation** test is the one genuinely new piece worth
adding, as an exact complement to the bootstrap.

**Run cluster-robust, the anchor holds — but barely, and that shapes the whole design**
(10,000 document resamples, computed 2026-08-12):

| pair | Δ CI (cluster-robust) | verdict |
|---|---|---|
| **qwen3-vl-8b direct − its own transcriber** | **[+1.5pp, +16.4pp]** | **separable** |
| sonnet-4.6 − gemini-3.6-flash | [−1.6pp, +2.5pp] | spans 0 |
| sonnet-4.6 − gemini-3.5-flash-lite | [−1.5pp, +3.2pp] | spans 0 |
| sonnet-4.6 − haiku-4.5 | [+4.9pp, +14.7pp] | separable |
| mistral-ocr-4-0 − 4-1 | [+0.0pp, +10.3pp] | spans 0 |

Two readings, both load-bearing:

* **The shape penalty is real** — it survives clustering, unlike the marginal Mistral 4-0/4-1 result,
  which does not.
* **It is badly estimated.** A point estimate of 8.9 pt with a CI of **[1.5, 16.4]** is an order-of-
  magnitude uncertainty: this data can say "the shape costs something" and cannot say whether that
  something is 2 points or 16. **Getting from "real" to "how big" is the actual job of this
  experiment**, and it is why §3.2 (more fields) and §3.3 (repeats) are not optional extras.

### 3.2 The checkbox subset is too small and the good models are at a ceiling

`gemini-3.5-flash-lite` scores 251/258 direct. **Only 7 fields are available to win back**, so a
paired test against its transcriber arm is nearly one-directional: it measures loss well and gain
essentially not at all. Minimum detectable effect at n=258, α=0.05, naive:

| transcriber wins back | must lose | net detectable |
|---|---|---|
| 0 fields | 6 | 2.3 pt |
| 2 fields | 10 | 3.1 pt |
| 5 fields | 15 | 3.9 pt |

Inflate by the design effect and the real floor is **≈3–5 points**. A 2-point shape penalty is
invisible to this instrument, and "not detected" must never be written as "no penalty".

**Countermeasure — use the whole bank, not the checkbox slice.** The bank carries **3,742 gold
fields** over 1,356 cells, **14.5× the boolean subset**. The shape question applies to every field
type, not only booleans.

* **Primary endpoint:** per-field accuracy over **all 3,742 fields**, cluster-robust by document.
  Highest power available without new data.
* **Co-primary (interpretability):** the **258 booleans** — clean boolean gold, exact matching, and
  the metric every prior table reports. Direction must agree with the primary; disagreement is
  itself a result and must be reported, not resolved by picking one.
* **Secondary:** `strict` whole-cell accuracy (n=1,356 cells) and the **188 null-gold** fields, where
  the transcriber shape has a specific predicted failure (a dropped field reads as absent).
* Pre-specify the polarity split (checked vs unchecked) as a **mechanism** measure, not an endpoint.

Caveat to record with the primary: the non-boolean fields are scored by upstream's string/number
matching, which is noisier than boolean gold. More n, dirtier scorer. That is exactly why the
boolean endpoint stays co-primary rather than being demoted.

### 3.3 Every number in this repo is a single draw, and the Gemini legs are sampled at temperature 1.0

This is the largest unquantified risk to *any* "definitive" claim here, and it is not specific to the
shape question.

Every cell has been run **once**. The Gemini legs run at **vendor sampling — temperature 1.0,
top_p 0.95** (deliberately: Google warns that sub-1.0 temperature causes looping on 3.x). So
`flash-lite`'s 251/258, `3.6-flash`'s 252/258, and the three-way tie at the top of the results table
are **single draws from a stochastic sampler whose run-to-run variance has never been measured.**
Claude and qwen legs are at temperature 0.0, which is more stable but not guaranteed deterministic
server-side. Mistral OCR *is* known deterministic (byte-identical glyph counts on re-probe).

**The code says so itself** — `direct.py`'s `GEMINI_SAMPLING` comment lists it as an accepted cost:
*"Reruns are no longer deterministic. The same cell can answer differently on a second pass, so a
cache hit is not reproducible-on-demand the way a greedy row is. `sample_index` stays 0: this is one
sample per cell, not a K-sample design."* That was the right call for Stage 1. It is the binding
limitation on any definitive claim now.

Without a noise floor, an 8.9-point shape delta cannot be distinguished from a large-variance model,
and a 2-point delta certainly cannot. Note the interaction with §3.1: the anchor's CI is already
[1.5, 16.4] from clustering alone, and run-to-run variance is **additional** to that.

**Countermeasure — K-sample repeats, which is Stage 2 item 4 brought forward.** `sample_index: 0..K-1`
already exists as a condition axis, so this needs no schema change: each repeat is a new condition
value and therefore new cache cells. **K=5 per arm.** Deliverables:

* A **within-arm variance estimate** — the noise floor. Report it before any delta.
* **Error bars on the existing three-way tie**, which currently has none. This may be the single most
  valuable output of the whole exercise regardless of what the shape question returns.
* A shape delta expressed as **effect size relative to run-to-run SD**, not just percentage points.

Sampling regime is itself a confound to keep straight: comparing a temp-1.0 direct arm to a temp-1.0
transcriber arm is internally consistent, and that is what matters for the paired delta. Do **not**
mix regimes within a pair.

### 3.4 The transcriber arm's loss is confounded: transcription vs extraction

`Δ_m` as defined lumps two mechanisms — information destroyed during transcription, and the extractor
failing to use information that is present. A definitive answer must separate them.

**Countermeasure 1 — mechanism decomposition on the transcript text (free, no new spend).** For every
field where direct is right and transcriber is wrong, classify against the derived mark alphabet
`☐☑☒□▢■✓✔●○√✕×▪`:

| observation | attribution |
|---|---|
| no glyph anywhere in the transcript | **transcription loss** — destroyed before extraction |
| glyph present, correct polarity, answer wrong | **extraction loss** |
| glyph present, wrong polarity | **transcription error**, distinct from omission |
| glyph present but not adjacent to its label | **serialization loss** — the flattening hypothesis, now testable |

Run this on the existing qwen pair **first**. If the losses are overwhelmingly glyph-absent, the
mechanism is settled before a dollar is spent, and the paid arms become confirmation rather than
discovery. The mark alphabet was derived empirically once (an earlier assumed alphabet missed
`●○√✕×▪` and mis-scored a document) and **was never saved as a script** — that gap must be closed
first, with the wrong-alphabet canary retained as a test.

**Countermeasure 2 — two extractors over identical transcripts.** Run every transcript through
`gemini-3.6-flash` (the pinned extractor) **and** a second, different extractor. If both land in the
same place, the loss is in the transcript and the finding is extractor-independent; if they diverge,
the "shape penalty" is partly an extractor property and must be reported that way. This folds in
Stage 3 item 4 and needs new CLI surface (see §6).

**Never self-grade.** A `gemini-3.6-flash` transcriber arm scored by the `gemini-3.6-flash` extractor
grades its own output; the registry already flags this. Any arm that transcribes with a model must be
extracted by a different one.

### 3.5 Contamination biases the delta upward

`gemini-3.5-flash-lite` and `gemini-3.6-flash` are post-cutoff (`CONTAMINATION_CUTOFF` 2026-05-24).
A memorized answer helps the **direct** arm, which is asked the question; it helps the transcriber arm
much less, because the extractor only ever sees the transcript. **So a contaminated model's shape
penalty is overstated by an unknown amount.**

**Countermeasure.** The clean pre-cutoff pairs carry the claim about the *shape*
(`qwen3-vl-8b` 2025-10-11, `qwen3-vl-32b` 2025-10-19, `claude-sonnet-4.6` 2025-11-14,
`gemini-3.1-pro-preview` 2026-02-19); the contaminated pairs carry the claim about *the specific
model you would deploy*. Report both, always labelled, and never let a contaminated pair alone
support Claim B.

### 3.6 Multiplicity and post-hoc slicing

With spend unconstrained, the real threat shifts from cost to analytic freedom: many arms × many
endpoints × many subgroups will produce a significant result by chance, and this repo has already
paid for that once — the annotation probe scored **6/6** on pages *selected because plain OCR failed
there*, was mistaken for an effect size, and cost **$3.83** to disprove at full scale.

**Countermeasures.**

* **Pre-register** the arms, the primary/co-primary endpoints, the test, and the subgroup list in
  this file, with a commit hash, **before the first paid call**. Anything decided afterwards is
  labelled exploratory in the write-up, permanently.
* **One primary comparison per model pair.** Holm across the per-model deltas for the family; the
  meta-estimate is its own single test.
* **No outcome-based selection, ever.** Subgroups must be defined by document properties knowable in
  advance (glyph density, page ink coverage, domain, field type), never by which arm got it wrong.
* Report **all** pre-registered endpoints including nulls.

---

## 4. Design

### Arms

`D` = direct (vlm-chat, one call, image → answer). `T` = transcriber (image → markdown → extractor).

| model | D | T | contamination | note |
|---|---|---|---|---|
| qwen3-vl-8b | **have** | **have** | clean | the existing anchor; re-run under K-sampling |
| qwen3-vl-32b | registry entry, unrun | **have** | clean | cheapest new pair |
| gemini-3.5-flash-lite | **have** | needs parser | post-cutoff | **the decision-relevant pair** — the current recommendation |
| gemini-3.1-pro-preview | **have** | needs parser | clean | clean pair on a frontier model |
| claude-sonnet-4.6 | **have** | needs transcriber transport | clean | best direct arm; T needs real code (§6) |
| gemini-3.6-flash | **have** | needs parser + non-self extractor | post-cutoff | only with a different extractor |

Minimum for Claim B: **three pairs, at least two of them clean.** More is better; spend is not the
constraint.

### Analysis plan

1. **Noise floor first.** From K=5 repeats per arm, report within-arm SD per endpoint. Publish this
   before any delta, and put the resulting error bars on the existing three-way tie.
2. **Per-model paired delta** (Claim A): document-level paired permutation, cluster-bootstrap CI,
   naive McNemar reported alongside and labelled naive.
3. **Meta-estimate** (Claim B): random-effects over per-model deltas; report between-model
   heterogeneity explicitly. If heterogeneity dominates, the honest answer is "the penalty is
   model-specific", and that is a real finding, not a failure.
4. **Mechanism split** — transcription vs extraction vs serialization loss, from §3.4, per pair.
5. **Extractor sensitivity** — the same transcripts under two extractors.
6. **Pre-specified subgroups only.**

### Stopping rule

Fix K and the arm list up front. **No adding arms after seeing results**, and no stopping early
because a comparison reached p<0.05 — with K-sampling there is a real optional-stopping hazard here.

---

## 5. Sequence

| # | step | spend | why here |
|---|---|---|---|
| 0 | glyph-decomposition script + wrong-alphabet canary; run on the existing qwen pair | **$0** | may settle the mechanism before any spend |
| 1 | pre-register §4 in this file; commit | $0 | the countermeasure for §3.6 |
| 2 | K=5 repeats of the **existing** direct arms | moderate | the noise floor; independently valuable — it error-bars the published tie |
| 3 | `qwen3-vl-32b` direct (completes the cheapest clean pair) | small | second pair for Claim B |
| 4 | `gemini-3.5-flash-lite` transcriber, K=5 | larger | the decision-relevant pair; also converts the $2.0–3.3 transcription **estimate** into a measurement |
| 5 | second extractor over all transcripts | moderate | extractor independence |
| 6 | `gemini-3.1-pro-preview` transcriber; optionally sonnet-4.6 | larger | clean frontier pairs |

Step 2 is worth doing even if the shape question is abandoned: the top of the results table is
currently three single draws from a temperature-1.0 sampler with no error bars.

---

## 6. Build list

| item | effort | note |
|---|---|---|
| glyph-decomposition script | small | new; reusable; the Mistral analysis was ad hoc and lost |
| **wire `compare-legs-checkbox.py` to `stats.paired_delta_ci`** | **trivial** | the cluster bootstrap **already exists and the report already uses it** — the script and the results prose use naive McNemar. This is a reporting bug, not missing machinery |
| document-level paired permutation test | small | the one genuinely new statistic; exact complement to the existing bootstrap; goes in `ocr_eval_ext/stats.py` |
| all-fields + null-gold endpoints in the comparison script | small | `scripts/compare-legs-checkbox.py` currently does the 258 booleans only; the bank has 3,742 fields |
| `gemini_3_5_flash_lite` parser | ~5 lines | subclass `GeminiVisionParserBase`, set `model`/`pricing_key`/`prompt`; catalog rate `{input 0.0003, output 0.0025}` per 1k; registry entry `shape: transcriber` |
| `gemini_3_1_pro_preview` parser | ~5 lines | same shape |
| K-sample plumbing | **moderate — not merely unwired, deliberately pinned** | `sample_index` is a condition-dict key but is hardcoded to `0` (`direct.py:59`) with a comment stating "this is one sample per cell, not a K-sample design". It needs a CLI flag, a loop, and confirmation that each K lands in its own cache cell (it will — `sample_index` is in the condition hash). Do **not** add a new key to `STAGE1_CONDITION`; the value already exists, only the driver is missing |
| `--extractor` override / `rescore` | **moderate — does not exist** | spec'd as Stage 3 item 4; extractor is pinned by design (D11 rev 3) |
| bedrock-converse transcriber path | **moderate–large** | only if a sonnet-4.6 T arm is wanted; no transcriber support on that transport today |

---

## 7. What would leave the answer non-definitive

State these in the write-up regardless of outcome:

* **A ≤2-point penalty is below this instrument's floor** at n=258; the all-fields endpoint lowers
  that but does not remove it.
* **One instrument.** Everything is RealDocBench. Replication on a second instrument is Stage 3
  item 1 (**CheckboxQA**, CC BY-NC, loader + ANLS\* unbuilt). Any claim about "the shape" rather than
  "the shape on this bank" needs it.
* **The transcription prompt is one prompt.** It is a fair one (it names the glyphs), but per-model
  prompt optimization is explicitly out of scope, so the T arm is a lower bound on what the shape
  could achieve with tuning.
* **Between-model heterogeneity may exceed the effect.** Then the answer is "model-specific", and
  three pairs will not fix it.
* **Contamination inflates the delta** on post-cutoff pairs (§3.5), by an unquantified amount.

---

## 8. State at handoff (2026-08-12)

**Nothing in §4–§6 has been built or run.** What exists is this design, the free analyses quoted
above, and the 2026-08-11/12 work below.

### Uncommitted at handoff

13 modified files, 3 new scripts, and **1,356 `claude-sonnet-4.6@bedrock` cache rows that exist only
on disk** — no cache rows for that leg are tracked by git. Committing was not requested; the working
tree is the only copy of the Sonnet leg. Also still unstaged from an earlier session: 394 deletions of
abandoned temp-0 Gemini cache rows, deliberately left alone.

Modified: `configs/registry-bedrock.yaml`, `ocr_eval_ext/{bedrock,direct,report_md}.py`,
`tests_ext/test_{bedrock,report_md}.py`, `docs/{api,cost-benefit-2026-08-10,
results-checkbox-accuracy-cost-2026-08-06,results-checkbox-final-2026-08-10}.md`,
`runs/stage1/{report.md,eval/results.json}`.
New: `scripts/{bedrock-price-list-claude.sh,compare-legs-checkbox.py}`, this plan.

**518 tests pass; scoped lint at the 14 `B008` baseline.**

### Landed 2026-08-11/12, and worth not re-deriving

* **Leg 15 — `claude-sonnet-4.6@bedrock`, full bank, $8.6917.** 98.1% (253/258), the nominal best leg,
  **not separable from either Gemini flash** (p=1.000 / p=0.727; cluster-robust CIs span 0). Beats
  Haiku by 9.3 pt (p=8.0e-7). **Pre-cutoff**, so a clean-baseline leg now matches the contaminated
  ceiling — the best evidence yet that ~97–98% is not memorisation.
* **`us.anthropic.claude-sonnet-5` is AccessDenied to this role.** 4-6 and 4-5 are invokable; 4-5 and
  4-6 price identically.
* **Leg 8 (Haiku) priced at $2.8706** — single-source rate ($1.10/$5.50 per Mtok, Bedrock pricing
  page). AWS's *public* bulk price list carries **no Claude 4.x row at all**;
  `scripts/bedrock-price-list-claude.sh` re-checks and prints `CHANGED` if that changes.
* **Bedrock has no hidden thinking tokens** — Anthropic bills reasoning inside `outputTokens`;
  `in + out == total` on all 2,672 rows across both legs; `unaccounted_tokens` now covers
  `bedrock_usage` with a positive control. Untested: whether nonzero cache tokens would be caught.
* **Bedrock's 5 MB image cap is enforced on the base64 payload** — real PNG ceiling 3,932,160 bytes.
  Costs both Bedrock legs the **same 20 of 1,356 cells** across 10 documents, deterministically, and
  **zero** checkbox fields. Not throttling; `--workers` is irrelevant. This was already in
  `docs/api.md` from 2026-08-04 and was rediscovered from error rows — which is why it is now also in
  `bedrock.py`'s docstring.
* **Two qwen transcriber legs were rendering `$0.0000`** in `report.md` on legs that really cost
  ~$0.70 each; the M1 guard required the entry to be *unpriced* and these are priced. Now
  `n/a (unrecorded)`.
* **Cost-column framing corrected** (both results docs): per-page legs deliver a transcript, vlm-chat
  legs never transcribe. flash-lite's bill is **66.8% page image, 23.6% output**; the bank overcharges
  vlm-chat legs ~2.3× by re-sending each image per question (batching would cut 38%, untested);
  and **flash-lite transcribing would cost $0.0034–0.0057/page against Mistral's $0.0040** — so its
  ~5× advantage comes entirely from skipping transcription, not from being cheaper at it. That
  estimate is unmeasured and step 4 of §5 replaces it with a measurement.

### Traps this experiment will walk into

* `.venv/bin/python` and `.venv/bin/ocr-eval` — bare `python3` fails collection, and
  `python -m ocr_eval_ext.cli` **silently produces no output** (no `__main__` guard); it looks like a
  successful no-op run.
* `report` needs every registry passed (`--registry` ×5) plus `--allow-stale-render`, or five legs
  render `(unregistered)`.
* Check liveness with `[ -d /proc/$PID ]`, never `grep` on `ps` — a non-matching pattern reads as
  "finished".
* Exit **2** means "completed with errored cells" (fail-visible), **1** is a crash. Do not `set -e`
  over `direct`/`parse`/`score`.
* A cached error is terminal unless its message contains `"no answer"`. Delete the affected rows to
  re-attempt; `--force` re-bills everything.
* `direct` does **not** refresh `results.json` — call `aggregate_results` afterwards, then `report`.
