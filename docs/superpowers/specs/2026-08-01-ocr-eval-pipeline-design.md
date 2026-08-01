# OCR Extraction Eval Pipeline — Design (rev 2)

**Date:** 2026-08-01 (rev 2, same day)
**Status:** Draft for review
**Provenance:** Direct successor to the small-VLM OCR survey (`~/notes/claude-things/gate*.md`, znote project `ocr-vlm-survey`). Rev 1 was reviewed by two independent agents (`reviews/2026-08-01-review-{fable,opus}.md`); the Opus review verified claims against the live upstream artifacts and found material errors that this revision corrects. Where rev 2 states upstream facts (bucket composition, scorer behavior, DPI), they are the Opus review's measurements against dataset revision `906170ab` and harness commit `fb26a687` — the implementer re-verifies them at fork time as a precondition, not as optional diligence.

**User decisions ratified 2026-08-01:** fork upstream rather than rebuild · gemini-3-flash-preview as primary extractor with a local sensitivity check · run the full 1,356-question bank.

---

## Purpose

Produce comparable, licence-annotated, uncertainty-quantified numbers for document/form extraction — checkbox **state** first — across any model reachable by API or runnable on the local GPU (RTX 4070 Ti SUPER, 16 GB, CUDA). One harness, three build stages.

| Stage | Question answered |
|---|---|
| 1 | On real scanned forms, which reachable model reads checkbox state best (boolean-restricted metric, n=258), and who hallucinates into blank fields (bank-wide nulls, n=188)? |
| 2 | Does the degradation inversion survive deskew? Does a lenient output contract change rankings vs the typed contract? Does sample-agreement predict correctness? |
| 3 | Does the ranking replicate on CheckboxQA? At what human-review rate does deployment become viable? |

## Non-goals

No web UI, no database server, no leaderboard publishing, no fine-tuning support, no per-model prompt optimization (explicitly out of scope; may become a condition axis later). Own-forms loader deferred (would slot into Stage 3).

---

## Build strategy: fork upstream

`ocr-eval` becomes a fork of `extend-hq/realdoc-bench` (Apache-2.0), pinned at harness commit `fb26a687` with dataset revision `906170ab201d7b8238a32a9115fc66b4b72e0710` (HF `Extend-AI/RealDoc-Bench`, CC-BY-4.0). These are **two independently versioned pins** (`harness_commit`, `dataset_revision`); both are recorded in run metadata and asserted at load.

Upstream already ships: registry, cache, pricing meter, runner, HTML/leaderboard reporting, page-fanout parser base, and working adapters for `mistral_ocr`, `azure_di`, `aws_textract`, `reducto`, `llamaparse`, and a generic `cloud_vlm` VLM-as-parser. **Every divergence from upstream code is a comparability break with the only published numbers on this instrument** — so we extend, and diverge only deliberately with the divergence documented.

Our additions (the genuinely new work):
1. An OpenAI-compatible **`vlm-chat`** provider (direct QA: image + question → answer) — a new measurement shape upstream does not have.
2. A **local-vLLM transcriber** adapter (OCR specialists served on the local GPU).
3. The **condition dict** and its axes (Stage 2).
4. **Boolean-restricted and null-restricted metrics**, trivial baselines, document-clustered bootstrap CIs, report segregation by shape.
5. Fail-closed **cardinality preconditions** and the reproduction gate.

Fork hygiene: upstream's `.env` secret loading is disabled at fork time (environment-only keys is a hard constraint); the current paper Table 3 reproduction targets are snapshotted into the repo (the README leaderboard has already drifted from the paper — reproduction targets come from the paper only).

---

## Measurement definitions

This section exists because rev 1 got them wrong.

### Primary metric — checkbox-state accuracy
Per-field accuracy restricted to **boolean-typed gold fields within checkbox-tagged questions**: n = 258 fields (165 True / 93 False) over 263 docs. Typed via `isinstance(v, bool)` on `gold_dict`. Reported with a full 2×2 polarity split (gold-checked vs gold-unchecked accuracy) — polarity inversion is the founding thesis and an aggregate hides it.

### Secondary metrics
- **Blank-field hallucination rate:** over **all 188 null-gold fields bank-wide** (the `blank_field` bucket alone contains only 34 — 82% of nulls live outside it). Bucket-level n=34 figures, if shown, carry their ±12-point interval inline.
- **General extraction:** per-field and strict per-question accuracy over the full 1,356-question bank — the leaderboard-comparable numbers, clearly labelled as general extraction, never as checkbox performance.
- Bucket overlap is stated in the report (40 questions are in both the checkbox and blank-field buckets).

### Denominators
Two accuracies per cell, always with n printed: **accuracy-over-all** (error rows count as incorrect — this is the ranking key) and accuracy-over-answered (diagnostic). Ruling on nulls: a null-gold field is scored correct only on an explicit null/blank answer per the typed contract; a refusal is `error_class=refusal` and counts incorrect in accuracy-over-all.

### Trivial baselines — in every report
Synthetic rows at zero cost: **always-true**, **always-false**, **majority-class** (= 64.0% on the checkbox metric — printed with the class balance beside the table), plus a **no-image control** (question text only) for at least one model to expose language-prior guessing. A model is not ranked above "competent" unless it beats the majority-class row by more than its CI.

### Uncertainty
**Document-clustered bootstrap 95% CIs** on every accuracy (questions cluster on 263 docs; naive binomial is wrong), **paired bootstrap** for model-vs-model deltas, and the report prints "not separable" instead of an ordering where intervals overlap. The hard-case tags (filled_bubble 16, circled_choice 7, ambiguous_mark 6, crossed_out 1) are reported qualitatively only, sample sizes printed beside every mention.

---

## Core architecture

### Two measurement shapes

- **`vlm-chat`** — model sees page image(s) + question, answers directly under the typed contract. One model measured per row. OpenAI-compatible chat completions; hosted and local (vLLM/SGLang/Ollama/LM Studio) are the same code path with different `base_url`.
- **`transcriber`** — system produces a page transcript; the pinned extractor answers questions from the transcript alone (upstream's construction — all published numbers are this shape). A row measures (transcriber ∘ extractor). `transcriber` is a shape, not a transport: a Python protocol `(images, condition) → TranscriptResult` with per-provider adapters (upstream's existing parsers; our local-vLLM adapter; classical engines in Stage 2). **Raster-only enforcement:** every transcriber receives rendered PNGs, never a PDF path — 2 of 16 sampled docs carry a text layer that PDF-accepting tools would free-ride on.

**Report segregation (hard rule):** `vlm-chat` and `transcriber` rows appear in separate report sections. Transcriber rows are labelled `(<transcriber> + extractor <id>)`. Cross-shape deltas are pipelines-vs-pipelines, and the report auto-prints that warning. One VLM is registered under **both shapes** in Stage 1 as the direct-vs-two-stage calibration cell.

**Transcription contract:** upstream's `_MARKDOWN_PROMPT` verbatim (it mandates `☒`/`☐` inline with labels and `**<label>:** <blank>` for empty fields), hash recorded per row. Endpoints that accept no prompt are flagged `contract: not-promptable` — an unavoidable asymmetry that is labelled, not hidden. A **transcript-recall diagnostic** runs on every transcriber row: does the transcript contain a checkbox glyph and the target field label at all — separating "read it wrong" from "never emitted it".

### Extractor policy
Primary: **`gemini-3-flash-preview`**, exactly as upstream — required for published-number comparability. Recorded per row. Same-family pairings (Gemini transcriber judged by Gemini extractor) are auto-flagged. **Extractor validation fixture:** before any run set, the extractor must score near-ceiling on a small hand-written set of transcripts that verifiably contain the answers; failure blocks the run. **Local sensitivity check:** a pinned local extractor re-scores a sample of transcripts; disagreement rate is reported. Extractor changes bump `extractor_generation` (invalidates extractions, never transcripts — see cache).

### Condition dict — in every row and cache key
```yaml
condition:
  preprocess: raw            # stage 2: deskew
  output_contract: schema    # Stage 1 BASELINE = the bank's typed response_format template.
                             # stage 2 adds: lenient (novel condition), split as
                             # schema_native / schema_prompted (distinct values — a model
                             # gaining native support must not collide with prompted rows)
  render: {engine: pymupdf, dpi: 150}   # matches upstream exactly; pixel dims + bytes recorded per row
  sampling: {temperature: 0.0, top_p: 1.0, max_tokens: 1024, seed: null}  # IN the dict, IN the key; max_tokens fixed for all Stage 1 cells
  sample_index: 0            # stage 2: 0..K-1 at one COMMON temperature across all registries
```
**Per-axis applicability table (maintained in the spec dir):** which pipeline step each axis binds to per shape — e.g. `output_contract` binds to the extractor for transcriber rows and to the model for vlm-chat rows; K-sampling binds to the answering step and is **skipped for deterministic engines** (a Tesseract row at K=5 is five identical samples and would poison the agreement table). The runner refuses to expand inapplicable cells.

### Registry
One entry per (model, serving) pair; same weights served two ways = two rows, never merged. Fields: `id`, `shape`, `base_url`, `model`, `precision` (exact format: `bf16` / `fp8-vllm` / `q8-gguf` — FP8 and Q8 are different schemes and are never pooled under one label), `weights_licence`, `provider_tos_commercial: ok|blocked|conditional` + `tos_note` (the survey's finding: provider ToS binds harder than weights licence — Novita/Z.ai/Fal/Datalab all carry blockers; the report stamps commercially-blocked serving paths **in code, not from memory**), `provenance`, `release_date` (vs dataset publication 2026-06-03 — post-dataset models get a contamination flag), `pricing_source`.

**Hosted serving identity:** OpenRouter routes per-request across providers — so OpenRouter entries pin the provider (`provider.order`, `allow_fallbacks: false`) and every row records the **provider actually returned in the response body** plus `retrieved_at`. The report warns when one registry's rows span >7 days. "Old cache entries stay valid" holds only per resolved serving identity, which is part of the cache key.

### Precision policy (v2 — protects the headline comparison, not just same-weights pairs)
Any model in the headline table appears at **one precision throughout**. ≤4B specialists: BF16 local. 8–9B candidates: hosted at provider-pinned full precision by default; a local FP8 run is a separate labelled row, never pooled. If a candidate can only be reached at a precision unlike its comparison group, the report labels the group mixed-precision and prints the caveat — or a same-model both-precision delta run is added. vLLM version pinned and recorded (kernel changes alter outputs).

### Scoring — upstream verbatim, fail-closed
- Upstream's actual scoring is used as-is at the pinned commit: `score.py` + `jsonify.py` template generation + the 26-line extractor `SYSTEM_PROMPT` + `deep_equal` with `fuzzy_equal` (rapidfuzz ratio ≥ 92, both sides ≥ 5 words) on plain string fields. Rev 1's "normalize-then-exact" description was wrong; nothing about scoring is reimplemented.
- **Fail-closed cardinality preconditions before any spend** (`capability_buckets.yaml` silently skips unknown tags — the fail-open shape of standing lesson lrn-ea833a5b): checkbox bucket == 429 q / 263 docs / 258 boolean fields; blank_field == 122 q / 34 nulls; bank == 1,356 items / 3,742 fields / 188 nulls. Any mismatch aborts.
- **Bucket policy:** upstream buckets verbatim for comparability, **plus** an extended checkbox tag set (checkbox_grid_alignment 8, four_way_checkbox_grid 5, n_a_vs_n_o 5, singleton checkbox_* tags, hard-case tags) reported as a separate clearly-labelled slice. Both, never a silent merge.
- **Scorer self-test at startup** (golden fixtures, hand-written — never drawn from the bank): known-correct, known-wrong including a polarity inversion, known-missing, and the null-boundary cases (explicit null, "field is blank" phrasing, empty response, refusal). Any fixture failure aborts before API spend.
- **Non-blank render check** (ink-coverage floor) per rendered page.

### Records, cache, resumability
- One JSONL row per (question × registry × condition × sample): ids, bucket tags, condition dict, prompt hashes, **rendered-image content hash**, resolved serving identity, raw response, extractor output (transcriber rows), parsed/scored fields, `error_class ∈ {none, parse_error, refusal, api_error, empty}`, latency, tokens, realized cost **from provider `usage` fields** (no hand-maintained price tables; upstream's meter used where it exists).
- **Two-level cache:** transcripts keyed on (image content hash, transcriber identity, preprocess, render); extractions keyed on (transcript hash, question id, extractor generation, output contract). vlm-chat responses keyed on (image hash, registry + resolved serving identity, question id, condition). Same page is never re-transcribed per question; extractor swaps never re-pay transcription.
- **`ocr-eval rescore`:** scores are derived data — a command recomputes parse/score from stored raw responses so scoring fixes cost zero API calls (upstream already re-scores on cache hit; we keep that behavior).
- `--dry-run` prices **both legs** (transcription + extraction) per cell; `--max-spend` aborts a live run that overshoots.

### Keys and secrets
Environment only (`bws run … --`), never config files or JSONL. Upstream `.env` loading disabled. **Prerequisites before first hosted run:** rotate the OpenRouter key and the `gemini-paid` key (both exposed in a prior session transcript; the extractor makes Google a required dependency).

---

## Stage 1 — comparative baseline (build now)

**Data:** full RealDocBench bank — 1,356 questions, 581 PDFs, at the pinned dataset revision, checksummed and mirrored on first fetch. Loader asserts `page_count == 1` per doc (all sampled docs are single-page and the bank has no page index — a multi-page doc appearing means the assumption broke; fail loudly, no silent page selection). Rendering: pymupdf @ 150 DPI (upstream-identical); pixel dimensions and encoded bytes recorded per row; provider-side downscaling noted as an uncontrolled factor with per-provider caps documented in the report.

**Data caveats auto-printed on every report:** `origin=None` for all items (real vs synthetic filled documents cannot be separated) **and** the ~18% born-digital composition estimate (gate2's 40-doc stratified sample: 74.4% scanned / 17.9% born-digital), with the note that per-class filtering is not possible from the released bank. CC-BY-4.0 attribution line generated in code.

**Contract:** the bank's typed `response_format` template — schema is the Stage 1 baseline (it types the booleans and signals nullability; "lenient" is a Stage 2 novel condition).

**Backends:** `vlm-chat` hosted (provider-pinned) + local; `transcriber` via upstream's existing hosted adapters + our local-vLLM adapter. Classical engines are Stage 2.

**Definition of done:**
1. Cardinality preconditions, scorer self-test, and extractor validation fixture all green (fail-closed).
2. **Reproduction gate:** one published open-weight row from the paper's Table 3 reproduced within its CI — candidates: dots.ocr 70.6±3.6 / 61.4±3.5, olmOCR-2 79.5±2.6 / 67.9±3.0, PaddleOCR-VL 59.6±4.0 / 48.5±3.6 — or the discrepancy explained in the report. This is the pipeline-level positive control; the scorer self-test alone cannot catch a wrong render, prompt, or template.
3. Report generated for ≥3 hosted VLMs + ≥1 local specialist (BF16) + ≥1 hosted OCR endpoint + **≥1 frontier ceiling anchor** (Gemini 3.5 Flash — also upstream's `cloud_vlm`, so it doubles as a secondary reproduction point), + the direct-vs-two-stage calibration cell (one VLM, both shapes).
4. Trivial-baseline rows and CIs present in the report; shape-segregated sections; polarity split shown.
5. A second invocation of the same run completes with 100% cache hits and zero API calls.

## Stage 2 — conditions and classical engines

- Axes gain values: `deskew` (OpenCV, correction magnitude recorded per page), `lenient` output contract (the novel condition — its synonym/parsing table versioned as data with golden fixtures per mapping, in particular false-vs-null), `schema_native`/`schema_prompted` as distinct values, K-sample consistency at **one common temperature across all registries** (per-registry deviations are separate labelled cells).
- Classical transcribers: Tesseract and the **PP-OCR classical pipeline** (named so — distinct from the PaddleOCR-VL model already in Stage 1). Deterministic engines are excluded from K-sampling cells by the applicability table.
- Reports gain per-condition delta columns (paired bootstrap) and the agreement-vs-accuracy table.

## Stage 3 — second instrument and decision layer

- **CheckboxQA:** loader (annotations from repo; documents mirrored from DocumentCloud on first fetch with checksums — link rot is a live risk), ANLS* scorer. Reports containing CheckboxQA numbers are stamped `CC BY-NC — internal model selection only` in code.
- Hard-case slice as qualitative appendix; extended-tag checkbox slice compared against the primary metric.
- Extractor sensitivity formalized: full local-extractor re-score of one complete run; disagreement analysis published.
- HITL routing analysis from Stage 2 rows (agreement threshold → review rate vs residual error), zero new API cost.

---

## Error handling

Per-cell isolation with `api_error` rows; bounded retries with backoff honoring `Retry-After`; run exits nonzero if any cell errored; **the report asserts per-row cell completeness and marks incomplete rows** — a partial matrix must be visibly partial, never silently rendered as complete. All subprocess exit statuses checked unpiped (PIPESTATUS discipline).

## Testing

Scorer golden fixtures (hand-written; provenance documented; never from the bank) at startup + CI. Loader tests against a committed `qa_bank.json` schema snapshot + the cardinality assertions. Backend contract tests against a mock OpenAI server. Extractor validation fixture. End-to-end smoke (3 questions, stub model → JSONL → report). The reproduction gate (DoD #2) is the systematic-error control.

## Repo layout

Fork of `extend-hq/realdoc-bench` with our additions in-tree:
```
ocr-eval/                        (fork; upstream history preserved; upstream remote kept)
  realdoc_bench/ ...             (upstream, minimally touched; .env loading disabled)
  ocr_eval_ext/                  (our code: vlm_chat provider, local_vllm transcriber,
                                  conditions/, metrics/ (boolean-restricted, CIs, baselines),
                                  preconditions.py, rescore.py, report_ext.py)
  configs/registry.yaml          (extended registry schema)
  docs/local-serving.md          (per-model vLLM launch lines, versions pinned)
  docs/superpowers/specs/        (this doc + reviews/ + snapshot of paper Table 3 targets)
  docs/applicability-table.md    (axis × shape binding + skipped cells)
```

## Risks

| Risk | Handling |
|---|---|
| Upstream data/harness drift (README leaderboard already drifted from paper) | Two pins asserted at load; Table 3 targets snapshotted in-repo |
| Bucket filtering fail-open on tag rename | Cardinality preconditions abort the run |
| Systematic harness error (render/prompt/template) | Reproduction gate in DoD |
| Contamination (dataset public 2026-06-03; candidates post-date it) | `release_date` per registry entry; flag printed per row |
| Hosted serving mutates under a stable id | Provider pinning, resolved-identity in cache key, `retrieved_at`, staleness warning |
| Extractor (Gemini) silently updated by Google | Validation fixture before each run set; local sensitivity check; generation bump on change |
| Provider-side image downscaling | Pixel dims + bytes recorded; per-provider caps documented; noted as uncontrolled |
| Local OOM (vLLM `gpu_memory_utilization` on 16 GB with 6.8 MP pages) | Documented per-model launch lines with tested settings; OOM is `api_error`, isolated |
| Temp-0 non-determinism on batched serving | Recorded as known limit; agreement analysis interprets accordingly |
| Runaway spend | `--dry-run` both legs + `--max-spend` hard cap |
| Renderer/vLLM version drift | pymupdf + vLLM versions in run metadata |
| Mixed-provenance + born-digital share in data | Auto-printed caveats (origin=None; ~18% born-digital) |
| DocumentCloud link rot (Stage 3) | Mirror-on-first-fetch with checksums |

## Effort estimate

Rev 1's 600–800 lines was roughly 2× optimistic even before forking. Fork strategy shifts effort from plumbing to integration: expect 3–5 focused sessions for Stage 1 (fork bring-up + vlm-chat provider + local vLLM serving + metrics/CIs/baselines + preconditions + reproduction gate), of which local vLLM bring-up for the first specialist is a session on its own. Hosted spend for the full Stage 1 matrix: low tens of dollars; local passes free after setup.
