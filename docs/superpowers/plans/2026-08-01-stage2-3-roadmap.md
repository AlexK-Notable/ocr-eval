# Stage 2–3 Roadmap (direction-level — not an implementation plan)

Stage 1's plan is `2026-08-01-stage1-eval-pipeline.md`. This file records the committed direction
for Stages 2–3 so Stage 1 implementers know which seams are load-bearing. Each stage gets its own
detailed TDD plan **after** the previous stage's results are in — several Stage 2 choices
(temperature for K-sampling, deskew library settings) should be informed by Stage 1's error patterns,
so planning them now would be planning blind.

## Stage 2 — conditions and classical engines

**Question answered:** does the degradation inversion survive deskew; does the lenient contract change
rankings vs typed; does sample-agreement predict correctness?

Work items, in dependency order:
1. **Deskew axis.** `preprocess: deskew` — OpenCV (`minAreaRect` on binarized content, rotate; record
   correction angle per page in the row). New condition value → new cache cells; zero schema change.
   Applies to both shapes (see `docs/applicability-table.md`).
2. **Lenient contract axis.** `output_contract: lenient` for vlm-chat and the extractor: free-text
   answer, then a versioned synonym/parsing table (data file + golden fixtures per mapping, esp.
   false-vs-null). The Stage 1 typed contract stays the baseline column.
3. **`schema_native` axis value.** Provider-native structured outputs where supported
   (`response_format` json_schema on OpenAI-compat; vLLM guided JSON). Per-registry capability flag;
   unsupported cells are skipped, not downgraded (applicability table).
4. **K-sample consistency.** `sample_index: 0..K-1` (default K=5) at ONE common temperature across
   all registries (chosen after Stage 1; recorded in the spec when chosen). Agreement rate per
   question; agreement-vs-accuracy table; binds to the transcriber for transcriber rows, extractor
   pinned at temp 0 (applicability table). Deterministic engines skipped.
5. **Classical engines.** Tesseract + PP-OCR classical pipeline as `transcriber` entries
   (local subprocess adapters implementing the ParseProvider contract; raster-only input enforced).
   Named `tesseract_local`, `pp-ocr-classical_local` — never confusable with PaddleOCR-VL.
6. **Report additions.** Per-condition delta columns (paired bootstrap per doc), agreement-vs-accuracy
   table, deskew-angle-vs-accuracy scatter data.

Cost note: K=5 × 2 contracts × 2 preprocess on the full bank is ~20× Stage 1 volume per model —
this is where local serving pays for itself; hosted cells should be budgeted per-run with
`--max-spend` and possibly restricted to the checkbox+blank union (~511 questions) rather than the
full bank. That scope decision is Stage 2 planning input, not made here.

## Stage 3 — second instrument and decision layer

**Question answered:** does the ranking replicate on CheckboxQA; at what review rate is deployment viable?

Work items:
1. **CheckboxQA loader + ANLS\*.** Annotations from the Snowflake repo; documents mirrored from
   DocumentCloud on first fetch (checksums; link rot is live). ANLS* scorer with its own golden
   fixtures. Every report containing these numbers auto-stamps
   `CC BY-NC — internal model selection only; not usable to validate a commercial product`.
2. **Hard-case + extended-tag slices.** filled_bubble 16 / circled_choice 7 / ambiguous_mark 6 /
   crossed_out 1 as a qualitative appendix; extended checkbox tag set (checkbox_grid_alignment 8,
   four_way_checkbox_grid 5, n_a_vs_n_o 5, singletons) reported beside — never merged into — the
   upstream-bucket primary metric.
3. **Direct-vs-two-stage formalized.** Stage 1 ships one calibration pair; Stage 3 runs every
   dual-capable model both ways and reports the systematic two-stage delta.
4. **Extractor sensitivity.** Full local-extractor re-score of one complete run
   (`ocr-eval rescore --extractor <local-id>`); disagreement analysis published.
5. **HITL routing.** From Stage 2 K-sample rows: agreement-threshold sweep → human-review rate vs
   residual field-error curve, per model. Zero new API cost.
6. **(Deferred, optional) own-forms loader** — same dataset interface if revisited.

## Standing constraints that carry through both stages

- No cache-key schema changes — new axes add values, never rename fields.
- Fail-closed preconditions extend to every new dataset (CheckboxQA gets its own cardinality
  assertions measured at first fetch and pinned).
- Every new metric ships with golden fixtures before its first real run.
- Keys via environment only; licence/ToS/contamination stamps generated in code.
