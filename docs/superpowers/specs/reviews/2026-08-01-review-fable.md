# Independent design review — Fable 5 agent

**Date:** 2026-08-01 · **Reviewer:** Claude Fable 5 subagent (design review, no upstream verification pass)
**Artifact:** `docs/superpowers/specs/2026-08-01-ocr-eval-pipeline-design.md`

---

## Findings, most severe first

**1. The Stage 1 headline table ranks two incommensurable measurement procedures against each other.**
Stage 1's single report ranks `vlm-chat` rows (model sees image + question) alongside `transcriber` rows (question-blind transcription, then pinned extractor answers from text). The spec correctly notes the extractor "cancels out of transcriber-vs-transcriber comparisons" — but it does *not* cancel out of the cross-shape comparison the Stage 1 table performs. Transcriber rows carry two extra error sources vlm-chat rows don't: extractor QA error, and the transcriber's checkbox-rendering policy. The second is fatal for the headline capability: most OCR specialists' markdown does not reliably render checkbox *state* at all (the survey itself found Nanonets is the only model advertising ☐/☑/☒ output). A transcriber's checkbox score is therefore substantially a measure of its transcript format, and its blank-field score a measure of whether the extractor guesses when the transcript is silent. Meanwhile the direct-vs-two-stage control that would quantify this is deferred to Stage 3 — Stage 1 runs the comparison uncontrolled and reports it as a model ranking.
**Fix:** split the Stage 1 report into two sections by backend shape (or add a hard visual separator plus an auto-printed warning that cross-shape deltas measure pipelines, not models). Pull one cell of the Stage 3 direct-vs-two-stage control into Stage 1 (one VLM registered both ways) as a cheap calibration of how much the two-stage penalty is worth.

**2. "Error classes never folded into accuracy" creates a denominator-gaming metric.**
If accuracy is computed only over rows with `error_class = none`, a model with 40% parse failures is scored on its parseable 60% — likely the easier questions — and outranks a model that answers everything. The rule as written rewards failing unparseably on hard questions. This is the exact fail-shape of a headline column that reports success while the thing measured is broken.
**Fix:** report *two* accuracies: strict-over-all-questions (errors count as incorrect — this is the primary ranking column) and accuracy-over-valid-responses (diagnostic). Keep the error-class columns as is. The current text will lead an implementer to build the misleading version.

**3. Sampling parameters are not in the condition dict or cache key.**
Temperature appears only as "per-registry" (Stage 2) and is entirely unspecified for Stage 1 candidates (hosted defaults are often 1.0 and differ per provider — Stage 1 rankings would be irreproducible noise). Worse, because temperature lives on the registry entry, not the condition dict, changing a registry's temperature does *not* change the cache key: old samples at the old temperature are silently reused alongside new ones — a cache-poisoning bug baked into the design. And per-registry temperature makes the Stage 2 agreement-vs-accuracy analysis incomparable across models (agreement at T=0.3 vs T=1.0 are different quantities).
**Fix:** put sampling params (temperature, top_p, max_tokens, seed if supported) in the condition dict / cache key; fix Stage 1 to temperature 0 explicitly; fix one temperature for the K-sample axis across all registries, with per-registry deviation recorded and flagged.

**4. No uncertainty quantification anywhere.**
RealDocBench itself publishes 95% document-clustered bootstrap CIs, and gate2 records that questions cluster on 263 documents (correlated errors — naive binomial CIs would be too tight). The spec vendors the normalization rules but not the CI machinery, and the report columns are point estimates. With 429 questions on 263 docs, few-point model differences are plausibly noise; the report will be read as a ranking regardless.
**Fix:** implement document-clustered bootstrap CIs from Stage 1 and print them beside every accuracy column. This is ~50 lines and it is the difference between "a number" and "a defensible number."

**5. Prompt policy is unspecified — the single largest known variance source in the survey's own findings.**
The spec records a prompt hash per row but never says whose prompt: one fixed prompt for all vlm-chat models, or per-model tailored prompts? These produce different rankings, and the survey's own diagnosis of GLM-OCR's ParseBench 29.6 vs OmniDocBench 95.22 was precisely that scores can measure format conformance rather than reading ability. Same gap for transcribers: vLLM-served OCR specialists take a task prompt ("convert to markdown" vs "transcribe including checkbox states"), and that choice will single-handedly determine their checkbox scores (see finding 1). Two reasonable implementers diverge here, incompatibly.
**Fix:** state the policy: one fixed prompt per backend shape per stage (RealDocBench's own "performance of a reasonable default" framing, carried as a report caveat); per-specialist transcription prompts taken from each model's own documentation, recorded verbatim in the registry entry; per-model prompt optimization explicitly out of scope or a future condition axis.

**6. Model input policy — page selection and image encoding — is unspecified.**
"Receives page image(s)" leaves open: whole document vs gold page only; and PNG vs JPEG, resize policy, OpenAI `detail` parameter. Whole-doc changes the task (adds page-finding) and is infeasible for 16 GB local models — if hosted models get all pages and local ones get one, that's a silent confound. Encoding matters on scanned forms (JPEG artifacts). Cost estimates also depend on it.
**Fix:** specify: gold page(s) only (the eval measures reading, not retrieval — state this in the report), fixed encoding (PNG from `pdftoppm`, no recompression), `detail` fixed and recorded, identical across hosted and local.

**7. No end-to-end calibration against published numbers.**
The scorer self-test is excellent, but it validates only the scorer. Nothing validates the whole pipeline — rendering, encoding, prompting, parsing — against external truth. RealDocBench Table 3 contains published rows for models this harness can run (dots.ocr 70.6/61.4, olmOCR-2 79.5/67.9, PaddleOCR-VL 59.6/48.5). Reproducing one of them within CI tolerance is the pipeline-level positive control the design is missing; without it, a systematic harness error (wrong DPI, lossy encoding, prompt mismatch) produces confidently wrong novel numbers with a green self-test.
**Fix:** add to Stage 1 definition of done: one published Table 3 row reproduced within its published confidence interval (or the discrepancy explained in the report).

**8. The cache granularity is wrong for transcribers.**
The cache is keyed at question level, but the expensive artifact for transcriber rows is the per-page transcript, shared across every question on that document (263 docs, 429 questions in the checkbox bucket alone, plus heavy page overlap with the blank-field bucket). Implemented as written, each question re-transcribes its pages — roughly doubling-plus transcription cost — and the `extractor_generation` bump compounds it: changing extractors should invalidate extractions, not transcripts, but a single-level cache re-pays both.
**Fix:** specify a two-level cache: transcripts keyed on (doc/page hash, registry id, preprocess, dpi); extractions keyed on (transcript key, question id, extractor generation, output contract). This also makes the dry-run cost model honest.

**9. Condition axes have backend-shape-dependent semantics the spec claims are uniform.**
"Later stages add values, never schema changes" — but at least two axes crack: (a) *schema* output contract: for transcriber rows, does it constrain the transcript (mostly meaningless for Tesseract) or the extractor's answer? Presumably the latter, but unstated, so Stage 2 cell semantics are ambiguous. (b) *sample_index*: does K-sampling resample the transcriber or the extractor? For deterministic classical engines (Tesseract), K samples are identical, yielding degenerate 100%-agreement rows that poison the agreement-vs-accuracy table. The runner as specced would happily run these meaningless cells.
**Fix:** add a per-axis applicability table: which pipeline step each axis binds to per backend shape, and which (shape × axis-value) cells are skipped or collapsed. This is exactly the kind of thing that must be settled before the abstraction is "fixed from first commit."

**10. Null-equivalence for the blank-field metric is undefined.**
"Correctly returned nothing" is one of the two headline metrics, and the spec never defines "nothing." Empty string, `null`, "N/A", "not filled", "the field is blank", a refusal? The boundary between correct-null, `error_class=empty`, and `refusal` decides the hallucination-rate column, and two implementers will draw it differently.
**Fix:** enumerate the null-equivalence set in the normalization spec, and make the golden fixtures include each boundary case (a correct "field is blank" phrasing, a refusal, an empty response).

**11. No checked/unchecked polarity breakdown — ironic, given inversion is the founding thesis.**
Gate3's own build-your-own criteria warned: without both polarities, "a model that answers 'checked' to everything scores well." The class balance of RealDocBench's 258 boolean golds is unknown; if it skews checked, aggregate accuracy hides exactly the bias this project exists to expose. The scorer self-test has a polarity-inversion fixture, but the *report* has no polarity-conditioned column.
**Fix:** add per-class accuracy (gold-checked vs gold-unchecked) or a 2×2 confusion split for the checkbox bucket, from Stage 1. Near-zero cost, directly on-thesis.

**12. Hosted-model drift and OpenRouter routing are under-handled.**
Two gaps beyond the risk table's "registry snapshot per run": (a) OpenRouter routes *per request* to different underlying providers with different precisions — `precision: bf16 (provider default)` is not a fact you can assert for an OpenRouter entry, and variance is within-run, not between-run. (b) "Old results stay valid forever" interacts badly with mutable hosted endpoints: expanding a matrix months later mixes cached rows from the old served version with fresh rows from the updated one, in the same report, invisibly.
**Fix:** pin the provider on OpenRouter requests (provider preference/allow-list) and record the served provider per row (OpenRouter returns it); add `retrieved_at` per row; have the report warn when rows for the same hosted registry span more than N days.

**13. The extractor's ceiling is never measured, and its serving stability is unaddressed.**
The risk table's mitigation ("pinned + recorded") doesn't mitigate a shared downward bias: a weak extractor caps every transcriber's score, and drags transcriber rows relative to vlm-chat rows. Also, if the extractor is a hosted model, the provider can silently update it — invalidating all transcriber comparability with no detectable signal, since `extractor_generation` is bumped manually.
**Fix:** add an extractor validation fixture — run the extractor over a small set of transcripts known to contain the answer (hand-verified or synthetic) and require near-ceiling accuracy before any run set; prefer serving the extractor locally (it's small by design) so its weights are actually pinned.

**14. Stage 1 report has no ceiling-baseline requirement.**
The research plan's locked assumption 1 states closed frontier models are included as clearly-labelled ceiling baselines; gate3 identified Gemini as the calibration point. The Stage 1 definition of done (≥3 hosted VLMs + ≥1 local specialist + ≥1 hosted OCR endpoint) omits any baseline row. Without one, a 70% top score is uninterpretable — model weakness or benchmark difficulty?
**Fix:** add "≥1 frontier ceiling baseline, flagged non-candidate in the licence column" to the Stage 1 definition of done.

**15. Missing risks (risk-table omissions), bundled:**
- **Benchmark contamination:** RealDocBench is public on HF since ~June 2026; any model trained after that may have seen it. Record model release/training-cutoff dates per registry entry and flag post-benchmark models.
- **DocumentCloud link rot** (Stage 3): CheckboxQA documents are fetched from a third-party service the project doesn't control; mirror-on-first-fetch with checksums, same as RealDocBench.
- **Renderer version:** `pdftoppm` output differs across poppler versions; the image content-hash cache hides this until a reinstall silently changes every page. Pin/record poppler version in run metadata.
- **Provider-side image transforms:** some hosted providers downscale/recompress images above pixel limits — effective DPI silently varies per provider despite the fixed-DPI condition. At minimum, record and note as an uncontrolled factor.
- **"FP8/Q8" conflation:** FP8 (vLLM) and Q8 (GGUF/Ollama) are different quantization schemes with different quality, and FP8 on an Ada consumer card typically runs via marlin-style W8A16 fallback, not native. The precision policy should name the exact allowed formats per server, or the "precision" column will compare unlike things under one label.
- **No runtime budget cap:** `--dry-run` estimates, but nothing aborts a live run that overshoots the estimate. A `--max-spend` guard is cheap insurance.

**16. Mild overbuild (Stage 1 YAGNI) — small list, mostly defensible:**
Per-row cost USD requires maintaining provider pricing tables; at "single-digit dollars per model per pass," token counts alone would do for Stage 1, with pricing joined at report time (and OpenRouter returns cost directly — prefer provider-reported cost over a maintained table). The per-registry circuit breaker is arguably Stage 2 polish for a matrix this small. Neither is expensive; flag, don't fight.

**17. The auto-printed data caveat understates the composition issue.**
It covers `origin=None` (real vs synthetic) but not the measured ~18% born-digital share (gate2's 40-doc stratified sample: 74.4% scanned, 17.9% born-digital) — and born-digital documents are explicitly out of the project's scope. Cheap fix: fold the composition estimate into the same auto-caveat; note that per-class filtering is not possible from the released bank.

## What is sound

The two-backend-shape discipline with same-weights-two-servings as distinct registry entries, the fail-closed scorer self-test with a polarity-inversion fixture, error classes as first-class row data, the precision policy's refusal to let Q4-local masquerade as a model comparison, the never-changing cache-key semantics, the code-generated CC BY-NC stamp for CheckboxQA reports, the honest qualitative-only treatment of the ~30 hard-case questions, PIPESTATUS discipline, and the phasing itself (conditions specced day one as single-value axes) are all right, and several of them fix real failures the survey documented. The findings above are almost entirely about making the *numbers* mean what the table headers will claim they mean — the harness skeleton is good.
