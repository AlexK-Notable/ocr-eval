# Scoring rubric

Two layers: upstream's scorer, used verbatim, and this fork's metrics layer on top of it. This doc
covers what each layer measures, the error taxonomy, baselines, uncertainty quantification, and the
reproduction gate that catches a systematically-wrong harness. **Who should read this:** anyone
interpreting a `report.md` number, or changing `metrics.py`/`stats.py`. For the CLI commands that
produce these numbers, see [`cli.md`](cli.md); for the full divergence text, see the
[design spec](superpowers/specs/2026-08-01-ocr-eval-pipeline-design.md) and
[implementation plan](superpowers/plans/2026-08-01-stage1-eval-pipeline.md).

## Layer 1 — upstream scorer (used verbatim)

Every bank item's `response_format` string is turned into a typed JSON template by
`build_template()` (`realdoc_bench/evaluate/score.py`), which recognizes three dialects
(`k=<tok>` pairs, `||`-separated pairs, or list-style) and falls back to inferring shape from the
gold value. `string_keys(template)` extracts which fields are typed plain `<string>` — these are
the *only* fields eligible for fuzzy comparison.

`score_typed(answer, gold_dict, str_keys)` scores each field:
- **`deep_equal`** (exact, type-tolerant) for everything except plain-string fields — booleans
  compare `is`, numbers compare as floats, dates/enums compare as normalized strings. It also
  tolerates parser-style single-element list wrapping, smart-quote/dash variants, markdown styling,
  and OCR character-spacing.
- **`fuzzy_equal`** (rapidfuzz `ratio >= 92`) applies *only* to plain `<string>` fields, and only
  when **both** sides have at least 5 words (`FUZZ_THRESHOLD = 92`, `FUZZ_MIN_WORDS = 5` in
  `score.py`) — short strings (IDs, amounts) never get fuzzy tolerance, since a 90+ ratio would
  forgive a single wrong digit.

For transcriber rows, the extractor is **`gemini-3.5-flash-lite`** (`score.py`'s `DEFAULT_MODEL`)
— one call per (question, parser), reading the transcript markdown and the typed template, never
outside knowledge. The construction is upstream's; nothing about it is reimplemented.

Every scored transcriber row records the extractor that produced it in an **`"extractor"`** field.
Absence of that key means the row predates the stamp (added 2026-08-05), never that it was scored by
whatever is currently pinned — the stamp is deliberately not backfilled on the cache-hit rescore
path, since labelling an old row with today's pin would assert exactly the provenance the field
exists to establish. Before it existed, the grader's identity lived only in `run_meta.json`, and a
disagreement about which extractor had scored 4,068 rows could not be settled from the data.

**Divergence D11 (2026-08-04, user-decided) — the extractor model itself.** Upstream and this
project's own spec both pinned `gemini-3-flash-preview`, ratified "exactly as upstream — required
for published-number comparability"
(`specs/2026-08-01-ocr-eval-pipeline-design.md:79`). Overridden on two grounds:

- **Durability.** The instrument has to outlive the project. `gemini-3-flash-preview` is a
  *preview* endpoint, and preview/older models get retired — `gemini-2.0-flash-lite` returned
  HTTP 404 "no longer available" the same day this was decided. An extractor that vanishes
  mid-project destroys reproducibility more completely than a judge swap does.
- **Measured equivalence.** Paired A/B over 300 real bank items on real DocStrange transcripts,
  identical items per model, McNemar on discordant pairs: `gemini-3.1-flash-lite` 81.3%
  (+14/−11 vs incumbent, p=0.690), `gemini-3-flash-preview` 80.3%, `gemini-3.5-flash-lite` 80.0%
  (p=1.000), `gemini-2.5-flash` 78.3% (p=0.362). Nothing separates them *at n=300*. Cost per
  full-corpus transcriber row falls $1.83 → $0.91.

**D11 rev 2 (2026-08-05, user-decided).** The pin moved again, to `gemini-3.5-flash-lite` — newest
GA flash-lite generation, fixed id rather than the `gemini-flash-lite-latest` alias (a floating
alias silently changes the instrument between runs while stamping the same name). Re-scoring all
4,068 transcriber rows then **contradicted the equivalence finding above**: pooled McNemar over
4,068 cells gives b=108 / c=144, **p=0.027 in favour of the OLDER 3.1 extractor** — 36 fewer
fully-correct cells (~0.9pp), with 93.8% verdict agreement. No individual leg reaches significance;
only the pooled test does.

The effect is real but small, and it reorders nothing — every Section B ranking and separability
verdict survives. It is also not uniform across metrics: DocStrange's *checkbox* accuracy rose
93.0% → 95.0% while its overall fully-correct count fell, so the losses sit on non-checkbox fields.
Full table in [`results-stage1-2026-08-04.md`](results-stage1-2026-08-04.md#d11-rev-2-2026-08-05-repin-to-gemini-35-flash-lite-and-what-re-scoring-revealed).

**The lesson worth carrying:** the n=300 A/B was underpowered to detect a ~1pp effect, and p=1.000
was misread as evidence of interchangeability. A non-significant result at small n rejects a *large*
difference; it does not establish equivalence. Quantify power before treating a null as a green
light.

**What it costs us:** DoD #2 compares our absolute numbers against upstream's published Table 3,
which upstream produced with `gemini-3-flash-preview` — so a reproduction gap now carries one
extra uncontrolled variable. Re-pin `DEFAULT_MODEL` to reproduce upstream exactly.

**Do not treat `selftest --extractor` as evidence here.** All four candidates score 5/5 on its
fixtures; the gate is a floor, not a discriminator. The n=300 paired run is the evidence.

## Layer 2 — this fork's metrics (`metrics.py`)

`field_outcomes(records, fields)` builds one `FieldOutcome(qid, key, doc, gold, status)` per
requested (question, field) pair, with `status ∈ {correct, incorrect, error}`:

- **`error`** — no scorable answer at all: the qid is absent, the row has an `"error"` key (even if
  an `answer` is *also* present — error wins), the row has no `"answer"` key, or the answer isn't a
  dict.
- **Null-gold fields** — correct *only* on a key-present, explicit `None` answer. This is stricter
  than upstream: upstream's own `deep_equal(None, None)` scores a **missing** key the same as an
  **explicit** null, which rewards extractor collapse — an extractor that stops emitting a field
  entirely would otherwise score as if it had correctly recognized the field was blank. This
  fork's rule requires the key to be present.
- Otherwise, correct iff the row's own `field_matches[key]` (upstream's `score_typed` output) says
  so.

**Ranking key vs diagnostic:** every `MetricBlock` carries both `acc_over_all` (errors count as
incorrect — the denominator is every requested field; this is the ranking key used everywhere in
`report.md`) and `acc_over_answered` (errors excluded — a diagnostic only, never used to rank).

**Checkbox metric:** `checkbox_metrics(outcomes)` — n=258 boolean-typed checkbox-bucket fields
(165 checked / 93 unchecked), always reported polarity-split (`checked`/`unchecked` sub-blocks) plus
a full confusion matrix (`tt`/`tf`/`ft`/`ff`/`err`). It raises if it's ever handed a non-boolean
gold — defence-in-depth against a future caller mis-bucketing.

**Blank-field headline:** `null_metrics(outcomes)`'s `overall.acc_over_all` over all 188 null-gold
fields bank-wide (not just the 34 in the `blank_field`-tagged bucket) is the headline number — it
is fail-safe by construction: a collapsed extractor (answer always null-key-absent) scores `error`
under the null-gold rule above, never `correct`, and an inventive extractor scores `incorrect`.
Either failure mode drives it down. `hallucination_rate` (`incorrect / n_answered`) is a narrower
diagnostic — the propensity to invent a value *given that the extractor answered at all* — and is
**never rendered alone**: `report_md.py`'s `_fmt_hallucination` always prints it alongside
`n_answered` and `error_rate`, because a fully-collapsed extractor and a well-behaved one can both
report `hallucination_rate == 0.0`.

## Error-class taxonomy

Set on every `vlm-chat` row by `direct.py`'s `_one`/`do()`:

| `error_class` | Produced when | Counted as |
|---|---|---|
| `none` | A well-formed JSON answer was parsed | scorable — status per-field via `score_typed` |
| `parse_error` | Response text wasn't valid/extractable JSON, and wasn't a refusal | `error` in `field_outcomes` |
| `refusal` | Unparseable response matching a refusal marker (`"i cannot"`, `"i can't"`, ...) | `error` |
| `empty` | `message.content` was empty after stripping (commonly: thinking model exhausted `max_tokens`) | `error` |
| `api_error` | Retries exhausted or a non-retryable HTTP status | `error` |
| `render_error` | The document itself failed: `_render_page`/`ink_coverage` raised, or the render was blank | `error` |
| `harness_error` | Anything else in the cell (prompt/template construction, `_one`'s own plumbing raising) | `error` |

`render_error` vs `harness_error` is a deliberate split (D10 in the divergence ledger): only a
genuine document-render failure gets `render_error`; everything else that goes wrong in the cell is
a harness bug, not a bad scan, and must not look like one.

## Baselines

`baseline_rows(fields)` computes, from the checkbox-bucket class balance (165 True / 93 False, n=258):
**always-true** (63.9%... reported as `165/258`), **always-false**, and **majority-class**
(`max(always_true, always_false)` — the printed number is exactly the checked-fraction, ~64.0%).
Every report also carries a **no-image control** (question text only, image omitted via
`--no-image`) to expose language-prior guessing. **Beats-majority rule**
(`report_md.py`'s `_beats_majority`): a model is never labelled above baseline unless a paired
bootstrap delta against the majority row is separable from zero (`ci[0] > 0`); below the cluster
floor it reports `"insufficient clusters (n_docs=N)"` rather than guessing.

## Uncertainty

Questions cluster on documents (429 checkbox questions / 263 docs) — a naive binomial interval
would be too tight. `stats.py` resamples **documents with replacement** (not individual questions):
`cluster_bootstrap_ci` for single-row CIs, `paired_delta_ci` for model-vs-model deltas (the
resampled document index is shared across both arms, preserving the pairing). `separable(ci)` is
strict: `lo > 0 or hi < 0` — an interval that spans zero is never called a win. Both bootstrap
functions are gated by `MIN_CLUSTER_DOCS = 20` in `report_md.py`: below that document count,
`separable()`/paired deltas are never computed at all, and every consumer (`report.md`'s
"beats majority" column, the separability appendix) prints `n_docs` alongside every claim.

## The reproduction gate

Two different constructions exist side by side in `report.md`, and they diverge on purpose:

- **Ranking key** (`general/field`/`strict/question` columns, Section B leaderboard table) — this
  fork's own D7 null-gold rule (key-present-explicit-null only).
- **Reproduction gate** (`_upstream_construction_metrics`, the separate "Reproduction gate (upstream
  construction)" block) — recomputed straight from each row's stored `field_matches`/`match`
  (upstream's own `score_typed`/`deep_equal` semantics, including its more lenient null handling),
  restricted to `ok` rows only, with **no** D7 re-scoring applied.

Both exist because comparing the wrong one against a paper- or README-sourced number produces a
phantom gap: on the full RealDoc-Bench bank, 188/3742 = **5.02%** of all fields are null-gold, and
upstream's `deep_equal(None, None)` scores a missing-key answer correct where this fork's ranking
key scores it incorrect. That is wider than the ±2.5pp tolerance the runbook applies to the Gemini
frontier-anchor reproduction check. **The reproduction gate (DoD #2) must always compare against
the "Reproduction gate (upstream construction)" block — never the ranking-key columns** — see
[`table3-snapshot.md`](superpowers/specs/table3-snapshot.md) for the pinned target numbers and the
full D7 divergence note.

## Divergence ledger digest

Full text and rationale live in the [design spec's rev 2.1 appendix](superpowers/specs/2026-08-01-ocr-eval-pipeline-design.md#rev-21-amendments-2026-08-01-from-plan-review)
and the [plan's Global Constraints](superpowers/plans/2026-08-01-stage1-eval-pipeline.md). One line
each:

- **D1** Per-cell JSON cache records, not JSONL (matches upstream's own cache shape).
- **D2** `max_tokens: 1024` scopes to vlm-chat cells only; transcription uses upstream's 12,000 default (a page of markdown doesn't fit in 1024).
- **D3** Rendered-image hash lives in the row (`image_sha`), not the cache key; STALE-RENDER is a report-time gate.
- **D4** Resolved serving identity is not in the cache key; the report hard-fails on a parser key spanning >1 provider.
- **D5** Raster-only for parsers this fork implements; upstream's PDF-uploading adapters are permitted and labelled `input: pdf-direct`.
- **D6** `--dry-run` prices the direct leg only; the transcriber scoring leg has no automated cost preview (budgeted manually in the runbook).
- **D7** Null-gold scoring is stricter than upstream (key-present-explicit-null only) — see Layer 2 above.
- **D8** The local-extractor sensitivity check is deferred to Stage 3; Stage 1's mitigation is the blocking extractor-validation fixture.
- **D9** The blank-field headline is `acc_over_all` over null-gold fields (fail-safe), not `hallucination_rate` alone.
- **D10** `error_class` splits `render_error` (document failure) from `harness_error` (everything else) — see the taxonomy above.

## Known metric caveats

- **CI-below-floor rendering:** a cluster-bootstrap CI renders on any row with `n_docs >= 1` — the
  `MIN_CLUSTER_DOCS = 20` floor only gates *paired* comparisons (beats-majority, the separability
  appendix), not single-row CI display. A CI shown beside a small `n_docs` is not equally
  trustworthy just because it renders (`report_md.py`'s `METRIC_DEFINITIONS`).
- **Transcript-recall underscore semantics:** the transcript-recall diagnostic
  (`_transcript_recall`/`_field_tokens`) splits snake_case field keys into tokens, but the
  word-boundary regex treats `_` as a word character, not a boundary — a transcript that emits a
  compound identifier like `signature_present` verbatim is recalled via *neither* half. This is a
  conservative undercount by design, never an overcount.
- **Contamination borderline margins:** `CONTAMINATION_CUTOFF` (`config.py`, `"2026-05-24"`) is the
  dataset's HF `createdAt` — corrected once already from the spec's initial citation of `lastModified
  2026-06-03`. `release_date` in the registry is a coarse `YYYY-MM-DD` string comparison, and for
  closed models it's sourced from approximate public-announcement dates (no HF card to check) — a
  registry entry dated close to the cutoff deserves a manual date check before trusting the
  contamination flag either way.
