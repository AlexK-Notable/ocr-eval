# Stage 1 OCR Evaluation Report

Generated: 2026-08-05T15:57:52.291428+00:00
Dataset revision: 906170ab201d7b8238a32a9115fc66b4b72e0710 (revision_source: hf_metadata)
Harness commit: fb26a6876481de76dc293f722ab4efa71279904d
Extractor: gemini-3.1-flash-lite
Renders verified: True
pymupdf version: 1.28.0
Bank items in this report: 1356

**Dataset licence: CC-BY-4.0** — Extend-AI/RealDoc-Bench. Figures reproduced/derived from this dataset in this report are attributed to Extend-AI/RealDoc-Bench under CC-BY-4.0.

## Caveats

- `origin=None` for every bank item: real vs synthetic (filled-in) documents cannot be distinguished from the released bank — provenance is mixed and unrecoverable per-item.
- Composition is an estimated ~18% born-digital (gate2's 40-doc stratified sample: 74.4% scanned / 17.9% born-digital) — per-class filtering is not possible from the released bank alone.
- Provider-side image downscaling is UNCONTROLLED: hosted providers may resize or cap the rendered page before inference; per-provider caps are not independently verified per request.

## Section A — Direct QA (vlm-chat)

precision unasserted across all rows in this section

| model | checkbox acc-over-all | polarity checked/unchecked | blank/null acc-over-all | hallucination_rate | general/field | strict/question | beats majority |
|---|---|---|---|---|---|---|---|
| qwen3-vl-8b@openrouter | 90.7% [86.3%, 94.6%] (n_docs=123) | checked 95.8% / unchecked 81.7% | 77.1% [68.5%, 84.9%] (n_docs=94) | 22.9% (n_answered=188, error_rate=0.0%) | 86.4% | 75.1% | yes |
| **INCOMPLETE (369/1356)** qwen3.5-9b@openrouter | 36.0% [28.0%, 45.4%] (n_docs=123) | checked 45.5% / unchecked 19.4% | 0.5% [0.0%, 1.8%] (n_docs=94) | 66.7% (n_answered=3, error_rate=98.4%) | 13.1% | 19.1% | no |
| **INCOMPLETE (1035/1356)** vlm__qwen3-vl-8b-instruct@ollama-validation-ctx8k__4caf39e9ac1f (unregistered) | 87.2% [81.8%, 92.1%] (n_docs=123) | checked 94.5% / unchecked 74.2% | 23.9% [15.4%, 32.9%] (n_docs=94) | 19.6% (n_answered=56, error_rate=70.2%) | 54.4% | 57.9% | yes |
| vlm__qwen3-vl-8b-instruct@ollama-validation__4caf39e9ac1f (unregistered) | 87.2% [81.9%, 91.8%] (n_docs=123) | checked 93.3% / unchecked 76.3% | 62.2% [51.9%, 72.0%] (n_docs=94) | 20.4% (n_answered=147, error_rate=21.8%) | 78.0% | 70.3% | yes |
| **INCOMPLETE (60/1356)** vlm__qwen3-vl-8b@ollama-validation-ctx16k__8b1b841fc49b (unregistered) | 14.3% [7.6%, 21.7%] (n_docs=123) | checked 17.0% / unchecked 9.7% | 0.0% [0.0%, 0.0%] (n_docs=94) | 0.0% (n_answered=0, error_rate=100.0%) | 2.0% | 3.3% | no |
| **INCOMPLETE (60/1356)** vlm__qwen3-vl-8b@ollama-validation__42581bd83e8a (unregistered) | 10.5% [5.3%, 16.1%] (n_docs=123) | checked 10.9% / unchecked 9.7% | 0.0% [0.0%, 0.0%] (n_docs=94) | 0.0% (n_answered=0, error_rate=100.0%) | 1.6% | 2.9% | no |
| **INCOMPLETE (60/1356)** vlm__qwen3-vl-8b@ollama-validation__4caf39e9ac1f (unregistered) | 8.9% [4.5%, 14.2%] (n_docs=123) | checked 10.3% / unchecked 6.5% | 0.0% [0.0%, 0.0%] (n_docs=94) | 0.0% (n_answered=0, error_rate=100.0%) | 1.4% | 2.6% | no |

- **qwen3-vl-8b@openrouter** — precision: unknown (not asserted) · licence: apache-2.0 · ToS: ok · contaminated: no · contract: promptable · providers seen: Alibaba · median latency: 1.49s · realized cost: $0.3897 · n: 1356/1356 · error classes: none=1353, parse_error=3
- **qwen3.5-9b@openrouter** — precision: unknown (not asserted) · licence: apache-2.0 · ToS: ok — pinned DeepInfra: cleanest commercial ToS in survey; Silicon · contaminated: no · contract: promptable · providers seen: DeepInfra · median latency: 11.10s · realized cost: $0.0657 · n: 369/1356 · error classes: api_error=6, empty=51, missing=987, none=311, parse_error=1
- **vlm__qwen3-vl-8b-instruct@ollama-validation-ctx8k__4caf39e9ac1f (unregistered)** — precision: unknown (unregistered) · licence: unknown (unregistered) · ToS: unknown (unregistered) · contaminated: unknown (unregistered) · contract: unknown (unregistered) · providers seen: (none recorded) · median latency: 2.31s · realized cost: n/a · n: 1035/1356 · error classes: missing=321, none=1035
- **vlm__qwen3-vl-8b-instruct@ollama-validation__4caf39e9ac1f (unregistered)** — precision: unknown (unregistered) · licence: unknown (unregistered) · ToS: unknown (unregistered) · contaminated: unknown (unregistered) · contract: unknown (unregistered) · providers seen: (none recorded) · median latency: 2.27s · realized cost: n/a · n: 1356/1356 · error classes: api_error=5, none=1307, parse_error=44
- **vlm__qwen3-vl-8b@ollama-validation-ctx16k__8b1b841fc49b (unregistered)** — precision: unknown (unregistered) · licence: unknown (unregistered) · ToS: unknown (unregistered) · contaminated: unknown (unregistered) · contract: unknown (unregistered) · providers seen: (none recorded) · median latency: 4.73s · realized cost: n/a · n: 60/1356 · error classes: empty=5, missing=1296, none=55
- **vlm__qwen3-vl-8b@ollama-validation__42581bd83e8a (unregistered)** — precision: unknown (unregistered) · licence: unknown (unregistered) · ToS: unknown (unregistered) · contaminated: unknown (unregistered) · contract: unknown (unregistered) · providers seen: (none recorded) · median latency: 4.29s · realized cost: n/a · n: 60/1356 · error classes: empty=13, missing=1296, none=47
- **vlm__qwen3-vl-8b@ollama-validation__4caf39e9ac1f (unregistered)** — precision: unknown (unregistered) · licence: unknown (unregistered) · ToS: unknown (unregistered) · contaminated: unknown (unregistered) · contract: unknown (unregistered) · providers seen: (none recorded) · median latency: 4.26s · realized cost: n/a · n: 60/1356 · error classes: empty=19, missing=1296, none=41

### Section A (vlm-chat) — per-domain (general per-field)

_Cells are acc-over-all with a document-clustered 95% CI; `n` = gold fields, `d` = documents. Read `d` before trusting an interval — 20 documents is the floor below which paired comparisons are refused elsewhere in this report._

> ⚠️ **These are per-row intervals, NOT a paired test.** Non-overlapping CIs do imply a real difference, but *overlapping* CIs do not imply the opposite — a paired bootstrap on the same resampled documents is strictly more sensitive. Read an overlap as "not established here", never as "no difference". The separability appendix is the paired test, and it covers checkbox accuracy only.

> ⚠️ **A partial run renders as a low score, not as missing data.** Every gold field in the bucket stays in the denominator, and a question with no cache row scores as an error, i.e. wrong. A row that was interrupted, capped by `--max-spend`, or never run to completion will therefore show a near-zero cell that is indistinguishable here from a model that answered and got it wrong. Check the row's `n: X/Y` and error classes in the section detail above before reading any low cell as a capability finding.

| row | mortgage | finance | supply_chain | medical_healthcare |
|---|---|---|---|---|
| vlm__qwen3-vl-8b-instruct@ollama-validation-ctx8k__4caf39e9ac1f (unregistered) | 76.6% [71.6%, 81.1%] (n=1502, d=222) | 79.7% [75.2%, 84.0%] (n=703, d=172) | 8.2% [5.3%, 11.8%] (n=1215, d=113) | 69.6% [58.4%, 79.9%] (n=322, d=74) |
| vlm__qwen3-vl-8b-instruct@ollama-validation__4caf39e9ac1f (unregistered) | 72.6% [66.8%, 78.4%] (n=1502, d=222) | 80.4% [75.8%, 84.5%] (n=703, d=172) | 84.1% [79.9%, 87.7%] (n=1215, d=113) | 74.5% [66.8%, 82.1%] (n=322, d=74) |
| qwen3-vl-8b@openrouter | 91.1% [88.9%, 93.0%] (n=1502, d=222) | 81.5% [77.3%, 85.5%] (n=703, d=172) | 86.4% [82.4%, 90.1%] (n=1215, d=113) | 75.2% [67.6%, 82.5%] (n=322, d=74) |
| qwen3.5-9b@openrouter | 0.0% [0.0%, 0.0%] (n=1502, d=222) | 66.1% [60.7%, 71.7%] (n=703, d=172) | 0.0% [0.0%, 0.0%] (n=1215, d=113) | 7.8% [3.0%, 13.8%] (n=322, d=74) |
| vlm__qwen3-vl-8b@ollama-validation-ctx16k__8b1b841fc49b (unregistered) | 0.0% [0.0%, 0.0%] (n=1502, d=222) | 10.8% [6.8%, 15.4%] (n=703, d=172) | 0.0% [0.0%, 0.0%] (n=1215, d=113) | 0.0% [0.0%, 0.0%] (n=322, d=74) |
| vlm__qwen3-vl-8b@ollama-validation__42581bd83e8a (unregistered) | 0.0% [0.0%, 0.0%] (n=1502, d=222) | 8.4% [5.1%, 12.1%] (n=703, d=172) | 0.0% [0.0%, 0.0%] (n=1215, d=113) | 0.0% [0.0%, 0.0%] (n=322, d=74) |
| vlm__qwen3-vl-8b@ollama-validation__4caf39e9ac1f (unregistered) | 0.0% [0.0%, 0.0%] (n=1502, d=222) | 7.4% [4.4%, 10.7%] (n=703, d=172) | 0.0% [0.0%, 0.0%] (n=1215, d=113) | 0.0% [0.0%, 0.0%] (n=322, d=74) |

### Section A (vlm-chat) — per-capability (general per-field)

_Top 12 capability tags by item count. The bank tags far more than this; the rest are omitted for readability, not because they were scored differently — every question was scored._

_Cells are acc-over-all with a document-clustered 95% CI; `n` = gold fields, `d` = documents. Read `d` before trusting an interval — 20 documents is the floor below which paired comparisons are refused elsewhere in this report._

> ⚠️ **These are per-row intervals, NOT a paired test.** Non-overlapping CIs do imply a real difference, but *overlapping* CIs do not imply the opposite — a paired bootstrap on the same resampled documents is strictly more sensitive. Read an overlap as "not established here", never as "no difference". The separability appendix is the paired test, and it covers checkbox accuracy only.

> ⚠️ **A partial run renders as a low score, not as missing data.** Every gold field in the bucket stays in the denominator, and a question with no cache row scores as an error, i.e. wrong. A row that was interrupted, capped by `--max-spend`, or never run to completion will therefore show a near-zero cell that is indistinguishable here from a model that answered and got it wrong. Check the row's `n: X/Y` and error classes in the section detail above before reading any low cell as a capability finding.

| row | field_value_pairing | checkbox_state | column_alignment | row_binding | table_structure | form_region | parallel_columns | line_binding | scanned_form | multi_column_grid | handdrawn_check | blank_field |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vlm__qwen3-vl-8b-instruct@ollama-validation-ctx8k__4caf39e9ac1f (unregistered) | 56.3% [49.8%, 63.8%] (n=1239, d=335) | 61.3% [54.0%, 69.0%] (n=1049, d=252) | 63.0% [56.2%, 69.9%] (n=768, d=250) | 62.7% [55.5%, 69.9%] (n=790, d=226) | 45.0% [36.0%, 54.9%] (n=993, d=190) | 64.1% [56.1%, 72.0%] (n=515, d=180) | 78.0% [70.4%, 85.1%] (n=451, d=157) | 73.3% [64.7%, 81.0%] (n=397, d=147) | 64.5% [54.3%, 75.2%] (n=408, d=121) | 59.4% [49.6%, 68.8%] (n=404, d=119) | 85.7% [79.5%, 91.2%] (n=238, d=83) | 68.5% [57.6%, 79.0%] (n=270, d=97) |
| vlm__qwen3-vl-8b-instruct@ollama-validation__4caf39e9ac1f (unregistered) | 85.5% [82.7%, 88.0%] (n=1239, d=335) | 76.0% [70.2%, 81.2%] (n=1049, d=252) | 80.5% [75.5%, 85.2%] (n=768, d=250) | 81.0% [75.9%, 85.9%] (n=790, d=226) | 75.8% [68.3%, 82.3%] (n=993, d=190) | 85.2% [81.4%, 88.9%] (n=515, d=180) | 83.8% [77.1%, 89.6%] (n=451, d=157) | 84.9% [80.2%, 89.0%] (n=397, d=147) | 82.6% [77.5%, 87.4%] (n=408, d=121) | 74.0% [66.1%, 81.3%] (n=404, d=119) | 85.3% [79.2%, 90.6%] (n=238, d=83) | 83.3% [78.6%, 87.7%] (n=270, d=97) |
| qwen3-vl-8b@openrouter | 87.6% [85.2%, 89.8%] (n=1239, d=335) | 84.4% [80.2%, 87.9%] (n=1049, d=252) | 83.3% [79.0%, 87.3%] (n=768, d=250) | 85.7% [81.3%, 89.6%] (n=790, d=226) | 87.2% [83.2%, 90.7%] (n=993, d=190) | 83.5% [78.8%, 87.8%] (n=515, d=180) | 90.2% [86.4%, 93.8%] (n=451, d=157) | 86.1% [81.6%, 90.2%] (n=397, d=147) | 81.9% [75.4%, 87.6%] (n=408, d=121) | 81.9% [76.1%, 87.4%] (n=404, d=119) | 89.1% [82.9%, 94.2%] (n=238, d=83) | 82.6% [77.2%, 87.1%] (n=270, d=97) |
| qwen3.5-9b@openrouter | 15.1% [11.6%, 18.8%] (n=1239, d=335) | 13.2% [9.5%, 17.1%] (n=1049, d=252) | 15.0% [10.8%, 19.3%] (n=768, d=250) | 17.7% [13.0%, 22.9%] (n=790, d=226) | 8.0% [5.1%, 11.5%] (n=993, d=190) | 16.3% [11.1%, 22.0%] (n=515, d=180) | 15.3% [10.1%, 21.1%] (n=451, d=157) | 24.2% [17.6%, 31.4%] (n=397, d=147) | 16.2% [10.6%, 23.1%] (n=408, d=121) | 13.6% [8.1%, 19.6%] (n=404, d=119) | 38.7% [28.0%, 49.8%] (n=238, d=83) | 18.1% [11.1%, 26.4%] (n=270, d=97) |
| vlm__qwen3-vl-8b@ollama-validation-ctx16k__8b1b841fc49b (unregistered) | 1.7% [0.7%, 2.9%] (n=1239, d=335) | 5.1% [2.7%, 7.6%] (n=1049, d=252) | 2.0% [0.7%, 3.4%] (n=768, d=250) | 3.9% [1.7%, 6.5%] (n=790, d=226) | 2.8% [1.2%, 4.9%] (n=993, d=190) | 1.2% [0.2%, 2.7%] (n=515, d=180) | 2.0% [0.5%, 4.1%] (n=451, d=157) | 0.0% [0.0%, 0.0%] (n=397, d=147) | 0.7% [0.0%, 2.4%] (n=408, d=121) | 6.2% [2.5%, 10.5%] (n=404, d=119) | 17.6% [9.9%, 26.8%] (n=238, d=83) | 1.5% [0.0%, 3.7%] (n=270, d=97) |
| vlm__qwen3-vl-8b@ollama-validation__42581bd83e8a (unregistered) | 1.6% [0.6%, 2.8%] (n=1239, d=335) | 3.7% [2.1%, 5.6%] (n=1049, d=252) | 1.3% [0.4%, 2.4%] (n=768, d=250) | 2.7% [0.9%, 4.8%] (n=790, d=226) | 2.2% [0.8%, 4.2%] (n=993, d=190) | 0.6% [0.0%, 1.3%] (n=515, d=180) | 1.3% [0.2%, 2.8%] (n=451, d=157) | 0.0% [0.0%, 0.0%] (n=397, d=147) | 0.7% [0.0%, 2.4%] (n=408, d=121) | 3.5% [0.9%, 6.9%] (n=404, d=119) | 13.0% [7.2%, 19.8%] (n=238, d=83) | 1.5% [0.0%, 3.7%] (n=270, d=97) |
| vlm__qwen3-vl-8b@ollama-validation__4caf39e9ac1f (unregistered) | 1.6% [0.6%, 2.8%] (n=1239, d=335) | 3.2% [1.7%, 5.0%] (n=1049, d=252) | 1.3% [0.4%, 2.4%] (n=768, d=250) | 2.2% [0.7%, 4.0%] (n=790, d=226) | 1.8% [0.6%, 3.4%] (n=993, d=190) | 0.4% [0.0%, 1.0%] (n=515, d=180) | 1.3% [0.2%, 2.8%] (n=451, d=157) | 0.0% [0.0%, 0.0%] (n=397, d=147) | 0.7% [0.0%, 2.4%] (n=408, d=121) | 3.5% [0.9%, 6.9%] (n=404, d=119) | 10.5% [5.5%, 16.4%] (n=238, d=83) | 1.5% [0.0%, 3.7%] (n=270, d=97) |

## Baseline rows

- always-true: 64.0%
- always-false: 36.0%
- majority-class: 64.0% (class balance: true=165, false=93, n=258)

## Cross-shape comparison note

**Cross-shape comparability warning:** Section A (`vlm-chat`) rows answer the question directly from the rendered page image in one model call. Section B (transcribe-then-extract) rows first transcribe the page to markdown, then a SEPARATE `gemini-3.1-flash-lite` extractor call answers the question from that markdown. A Section B score reflects BOTH stages at once — a low score can come from a bad transcription OR a bad extraction — so a Section A and a Section B number for the "same" underlying model are not directly comparable rankings. Use the direct-vs-two-stage calibration pair (below, when both rows exist) to gauge how much of a two-stage gap is pipeline overhead versus genuine transcription quality.

40 questions appear in both the checkbox and blank-field buckets (of 429 checkbox-bucket and 122 blank-field-bucket questions) — a field counted in both buckets contributes to both metrics independently.

**Metric definitions:**

- **checkbox acc-over-all**: correct / all boolean-typed checkbox-bucket fields (errors count as wrong, never excluded from the denominator).
- **blank/null acc-over-all**: correct-null / all null-gold fields bank-wide — the headline blank-field number (D9 ruling); a collapsed extractor (answer: null) and an inventive one both drive this DOWN, never up.
- **hallucination_rate**: incorrect / n_answered on null-gold fields ONLY — the propensity to invent a value given that the extractor answered at all. Meaningless read alone; always shown with n_answered and error_rate.
- **general per-field**: acc-over-all across every gold field of every bank item.
- **strict per-question**: fraction of ALL bank items fully correct (row `match is True`).
- **beats majority**: paired-bootstrap delta vs the majority-class predictor on checkbox acc-over-all — "not separable" (or "insufficient clusters" below n_docs=20) never counts as "yes".
- **CI-below-floor caveat**: a cluster-bootstrap CI is rendered whenever n_docs≥1, but MIN_CLUSTER_DOCS=20 only gates PAIRED comparisons (beats-majority, the separability appendix) — a single-row CI shown beside a small n_docs is not equally trustworthy just because it renders; read n_docs before trusting the interval.
- **Section B staleness is not assessed**: the D3 STALE-RENDER check only applies to vlm__ rows (the only rows carrying `image_sha`) — a swapped/stale PDF behind a transcriber or upstream-parser row is NOT detected by this report.
- the blank/null column's n is BANK-WIDE (every null-gold field in the whole corpus, not just the blank-field-tagged bucket) — e.g. n=188 on the full RealDoc-Bench corpus.
- **transcript-recall**: fraction of checkbox-bucket docs whose transcript contains both a checkbox glyph and a word-boundary match of a gold field key token. Snake_case field keys (e.g. `signature_present`) are split into separate tokens, but the boundary matcher treats underscore as a word character, not a boundary — a transcript that emits the compound identifier verbatim will NOT be recalled via either half. This is a conservative undercount, never an overcount.

**Direct-vs-two-stage calibration pair detected:** `qwen3-vl-8b@openrouter` (Section A, direct) vs `qwen3-vl-8b@openrouter-transcriber` (Section B, transcribe-then-extract) — same underlying model (`qwen/qwen3-vl-8b-instruct`); see the separability appendix for the paired delta between their rows.

## Section B — Transcribe-then-extract

_Cost/latency below describe the TRANSCRIPTION LEG only — the extractor (`gemini-3.1-flash-lite`) call is a separate, shared cost not attributed per-row._

precision unasserted across all rows in this section

| row | checkbox acc-over-all | blank/null acc-over-all | hallucination_rate | general/field | strict/question | transcript-recall | input | beats majority |
|---|---|---|---|---|---|---|---|---|
| docstrange@nanonets + extractor gemini-3.1-flash-lite | 93.0% [89.1%, 96.2%] (n_docs=123) | 93.6% [87.8%, 97.8%] (n_docs=94) | 6.4% (n_answered=188, error_rate=0.0%) | 89.6% | 82.1% | 97.6% | raster-png | yes |
| qwen3-vl-32b@openrouter-transcriber + extractor gemini-3.1-flash-lite | 91.5% [87.2%, 95.1%] (n_docs=123) | 94.7% [90.9%, 97.6%] (n_docs=94) | 5.3% (n_answered=188, error_rate=0.0%) | 87.4% | 76.8% | 92.7% | raster-png | yes |
| qwen3-vl-8b@openrouter-transcriber + extractor gemini-3.1-flash-lite | 80.6% [74.0%, 86.3%] (n_docs=123) | 95.2% [90.8%, 98.5%] (n_docs=94) | 4.8% (n_answered=188, error_rate=0.0%) | 82.3% | 70.1% | 94.3% | raster-png | yes |

- **docstrange@nanonets** · precision: unknown (not asserted) · licence: closed · ToS: ok — NOT zero-retention: nanonets.com/terms 6.6 reserves the righ · contaminated: no · contract: not-promptable · provider: n/a (not persisted) · median latency: 30.31s · realized cost: $5.8100 · n: 1356/1356 · error classes: none=1356
- **qwen3-vl-32b@openrouter-transcriber** · precision: unknown (not asserted) · licence: apache-2.0 · ToS: ok · contaminated: no · contract: promptable · provider: n/a (not persisted) · median latency: 14.84s · realized cost: $0.0000 · n: 1356/1356 · error classes: none=1356
- **qwen3-vl-8b@openrouter-transcriber** · precision: unknown (not asserted) · licence: apache-2.0 · ToS: ok · contaminated: no · contract: promptable · provider: n/a (not persisted) · median latency: 7.95s · realized cost: $0.0000 · n: 1356/1356 · error classes: none=1356

### Section B (transcribe-then-extract) — per-domain (general per-field)

_Cells are acc-over-all with a document-clustered 95% CI; `n` = gold fields, `d` = documents. Read `d` before trusting an interval — 20 documents is the floor below which paired comparisons are refused elsewhere in this report._

> ⚠️ **These are per-row intervals, NOT a paired test.** Non-overlapping CIs do imply a real difference, but *overlapping* CIs do not imply the opposite — a paired bootstrap on the same resampled documents is strictly more sensitive. Read an overlap as "not established here", never as "no difference". The separability appendix is the paired test, and it covers checkbox accuracy only.

> ⚠️ **A partial run renders as a low score, not as missing data.** Every gold field in the bucket stays in the denominator, and a question with no cache row scores as an error, i.e. wrong. A row that was interrupted, capped by `--max-spend`, or never run to completion will therefore show a near-zero cell that is indistinguishable here from a model that answered and got it wrong. Check the row's `n: X/Y` and error classes in the section detail above before reading any low cell as a capability finding.

| row | mortgage | finance | supply_chain | medical_healthcare |
|---|---|---|---|---|
| docstrange@nanonets | 96.2% [94.8%, 97.4%] (n=1502, d=222) | 84.5% [80.5%, 88.2%] (n=703, d=172) | 87.2% [82.6%, 91.4%] (n=1215, d=113) | 78.9% [70.4%, 86.6%] (n=322, d=74) |
| qwen3-vl-32b@openrouter-transcriber | 91.8% [89.7%, 93.8%] (n=1502, d=222) | 79.4% [75.0%, 83.6%] (n=703, d=172) | 88.9% [85.0%, 92.2%] (n=1215, d=113) | 78.9% [71.7%, 85.8%] (n=322, d=74) |
| qwen3-vl-8b@openrouter-transcriber | 88.5% [85.4%, 91.3%] (n=1502, d=222) | 72.5% [68.3%, 76.8%] (n=703, d=172) | 83.0% [79.1%, 87.0%] (n=1215, d=113) | 71.4% [62.5%, 80.0%] (n=322, d=74) |

### Section B (transcribe-then-extract) — per-capability (general per-field)

_Top 12 capability tags by item count. The bank tags far more than this; the rest are omitted for readability, not because they were scored differently — every question was scored._

_Cells are acc-over-all with a document-clustered 95% CI; `n` = gold fields, `d` = documents. Read `d` before trusting an interval — 20 documents is the floor below which paired comparisons are refused elsewhere in this report._

> ⚠️ **These are per-row intervals, NOT a paired test.** Non-overlapping CIs do imply a real difference, but *overlapping* CIs do not imply the opposite — a paired bootstrap on the same resampled documents is strictly more sensitive. Read an overlap as "not established here", never as "no difference". The separability appendix is the paired test, and it covers checkbox accuracy only.

> ⚠️ **A partial run renders as a low score, not as missing data.** Every gold field in the bucket stays in the denominator, and a question with no cache row scores as an error, i.e. wrong. A row that was interrupted, capped by `--max-spend`, or never run to completion will therefore show a near-zero cell that is indistinguishable here from a model that answered and got it wrong. Check the row's `n: X/Y` and error classes in the section detail above before reading any low cell as a capability finding.

| row | field_value_pairing | checkbox_state | column_alignment | row_binding | table_structure | form_region | parallel_columns | line_binding | scanned_form | multi_column_grid | handdrawn_check | blank_field |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| docstrange@nanonets | 91.0% [88.5%, 93.2%] (n=1239, d=335) | 88.1% [83.7%, 91.9%] (n=1049, d=252) | 88.4% [84.1%, 92.3%] (n=768, d=250) | 89.6% [86.6%, 92.5%] (n=790, d=226) | 89.0% [84.9%, 92.7%] (n=993, d=190) | 86.4% [82.4%, 90.1%] (n=515, d=180) | 92.2% [88.9%, 95.4%] (n=451, d=157) | 87.2% [82.2%, 91.5%] (n=397, d=147) | 85.3% [80.1%, 90.0%] (n=408, d=121) | 91.1% [86.5%, 95.1%] (n=404, d=119) | 92.4% [88.3%, 96.0%] (n=238, d=83) | 87.0% [82.6%, 91.4%] (n=270, d=97) |
| qwen3-vl-32b@openrouter-transcriber | 86.7% [83.6%, 89.6%] (n=1239, d=335) | 88.9% [85.6%, 91.8%] (n=1049, d=252) | 86.8% [83.0%, 90.4%] (n=768, d=250) | 85.2% [80.5%, 89.2%] (n=790, d=226) | 88.6% [84.8%, 92.0%] (n=993, d=190) | 86.0% [82.1%, 89.5%] (n=515, d=180) | 87.6% [83.3%, 91.3%] (n=451, d=157) | 84.9% [79.3%, 89.7%] (n=397, d=147) | 79.7% [72.9%, 85.7%] (n=408, d=121) | 90.6% [86.6%, 93.9%] (n=404, d=119) | 89.1% [82.1%, 94.9%] (n=238, d=83) | 78.1% [71.0%, 85.0%] (n=270, d=97) |
| qwen3-vl-8b@openrouter-transcriber | 82.1% [78.7%, 85.3%] (n=1239, d=335) | 81.8% [77.5%, 85.6%] (n=1049, d=252) | 80.9% [76.0%, 85.4%] (n=768, d=250) | 84.3% [79.4%, 88.4%] (n=790, d=226) | 82.7% [77.8%, 87.2%] (n=993, d=190) | 80.8% [76.3%, 84.9%] (n=515, d=180) | 89.1% [85.3%, 92.7%] (n=451, d=157) | 80.1% [74.2%, 85.8%] (n=397, d=147) | 77.2% [70.7%, 83.7%] (n=408, d=121) | 82.2% [76.0%, 87.6%] (n=404, d=119) | 79.8% [72.9%, 86.3%] (n=238, d=83) | 78.5% [72.7%, 83.8%] (n=270, d=97) |

## Reproduction gate (upstream construction)

_upstream construction — for the reproduction gate only; not the ranking key._ Recomputed straight from each row's stored `field_matches`/`match` (upstream's own `score_typed`/`deep_equal` semantics — a null-gold key-absent answer counts as upstream scored it), over ok rows only, with NO D7 re-scoring applied. Point the paper/README reproduction-target comparison at THIS table, never at the `general/field`/`strict/question` columns above.

| row | field% (upstream) | question% (upstream) | n (ok rows) |
|---|---|---|---|
| docstrange@nanonets | 89.6% | 82.1% | 1356 |
| qwen3-vl-32b@openrouter-transcriber | 87.4% | 76.8% | 1356 |
| qwen3-vl-8b@openrouter-transcriber | 82.3% | 70.1% | 1356 |

## Separability appendix

Pairwise paired-bootstrap deltas on checkbox acc-over-all, WITHIN each section only (Section A vs Section B pipelines are not directly comparable — see the cross-shape note above). Sign convention: Δ = first label minus second label (a positive Δ favors the FIRST name listed in each comparison).

### Section A
- vlm__qwen3-vl-8b-instruct@ollama-validation-ctx8k__4caf39e9ac1f (unregistered) [INCOMPLETE 1035/1356] vs vlm__qwen3-vl-8b-instruct@ollama-validation__4caf39e9ac1f (unregistered): not separable — Δ CI [-2.4pp, 2.3pp] spans 0
- vlm__qwen3-vl-8b-instruct@ollama-validation-ctx8k__4caf39e9ac1f (unregistered) [INCOMPLETE 1035/1356] vs qwen3-vl-8b@openrouter: separable — qwen3-vl-8b@openrouter ahead, Δ=[-6.8pp, -0.4pp]
- vlm__qwen3-vl-8b-instruct@ollama-validation-ctx8k__4caf39e9ac1f (unregistered) [INCOMPLETE 1035/1356] vs qwen3.5-9b@openrouter [INCOMPLETE 369/1356]: separable — vlm__qwen3-vl-8b-instruct@ollama-validation-ctx8k__4caf39e9ac1f (unregistered) [INCOMPLETE 1035/1356] ahead, Δ=[41.4pp, 60.1pp]
- vlm__qwen3-vl-8b-instruct@ollama-validation-ctx8k__4caf39e9ac1f (unregistered) [INCOMPLETE 1035/1356] vs vlm__qwen3-vl-8b@ollama-validation-ctx16k__8b1b841fc49b (unregistered) [INCOMPLETE 60/1356]: separable — vlm__qwen3-vl-8b-instruct@ollama-validation-ctx8k__4caf39e9ac1f (unregistered) [INCOMPLETE 1035/1356] ahead, Δ=[64.7pp, 80.8pp]
- vlm__qwen3-vl-8b-instruct@ollama-validation-ctx8k__4caf39e9ac1f (unregistered) [INCOMPLETE 1035/1356] vs vlm__qwen3-vl-8b@ollama-validation__42581bd83e8a (unregistered) [INCOMPLETE 60/1356]: separable — vlm__qwen3-vl-8b-instruct@ollama-validation-ctx8k__4caf39e9ac1f (unregistered) [INCOMPLETE 1035/1356] ahead, Δ=[69.6pp, 83.6pp]
- vlm__qwen3-vl-8b-instruct@ollama-validation-ctx8k__4caf39e9ac1f (unregistered) [INCOMPLETE 1035/1356] vs vlm__qwen3-vl-8b@ollama-validation__4caf39e9ac1f (unregistered) [INCOMPLETE 60/1356]: separable — vlm__qwen3-vl-8b-instruct@ollama-validation-ctx8k__4caf39e9ac1f (unregistered) [INCOMPLETE 1035/1356] ahead, Δ=[71.7pp, 84.8pp]
- vlm__qwen3-vl-8b-instruct@ollama-validation__4caf39e9ac1f (unregistered) vs qwen3-vl-8b@openrouter: separable — qwen3-vl-8b@openrouter ahead, Δ=[-6.6pp, -0.8pp]
- vlm__qwen3-vl-8b-instruct@ollama-validation__4caf39e9ac1f (unregistered) vs qwen3.5-9b@openrouter [INCOMPLETE 369/1356]: separable — vlm__qwen3-vl-8b-instruct@ollama-validation__4caf39e9ac1f (unregistered) ahead, Δ=[41.1pp, 60.5pp]
- vlm__qwen3-vl-8b-instruct@ollama-validation__4caf39e9ac1f (unregistered) vs vlm__qwen3-vl-8b@ollama-validation-ctx16k__8b1b841fc49b (unregistered) [INCOMPLETE 60/1356]: separable — vlm__qwen3-vl-8b-instruct@ollama-validation__4caf39e9ac1f (unregistered) ahead, Δ=[64.2pp, 81.1pp]
- vlm__qwen3-vl-8b-instruct@ollama-validation__4caf39e9ac1f (unregistered) vs vlm__qwen3-vl-8b@ollama-validation__42581bd83e8a (unregistered) [INCOMPLETE 60/1356]: separable — vlm__qwen3-vl-8b-instruct@ollama-validation__4caf39e9ac1f (unregistered) ahead, Δ=[69.5pp, 83.5pp]
- vlm__qwen3-vl-8b-instruct@ollama-validation__4caf39e9ac1f (unregistered) vs vlm__qwen3-vl-8b@ollama-validation__4caf39e9ac1f (unregistered) [INCOMPLETE 60/1356]: separable — vlm__qwen3-vl-8b-instruct@ollama-validation__4caf39e9ac1f (unregistered) ahead, Δ=[71.5pp, 84.8pp]
- qwen3-vl-8b@openrouter vs qwen3.5-9b@openrouter [INCOMPLETE 369/1356]: separable — qwen3-vl-8b@openrouter ahead, Δ=[44.8pp, 63.5pp]
- qwen3-vl-8b@openrouter vs vlm__qwen3-vl-8b@ollama-validation-ctx16k__8b1b841fc49b (unregistered) [INCOMPLETE 60/1356]: separable — qwen3-vl-8b@openrouter ahead, Δ=[68.2pp, 84.0pp]
- qwen3-vl-8b@openrouter vs vlm__qwen3-vl-8b@ollama-validation__42581bd83e8a (unregistered) [INCOMPLETE 60/1356]: separable — qwen3-vl-8b@openrouter ahead, Δ=[73.7pp, 86.5pp]
- qwen3-vl-8b@openrouter vs vlm__qwen3-vl-8b@ollama-validation__4caf39e9ac1f (unregistered) [INCOMPLETE 60/1356]: separable — qwen3-vl-8b@openrouter ahead, Δ=[75.6pp, 87.8pp]
- qwen3.5-9b@openrouter [INCOMPLETE 369/1356] vs vlm__qwen3-vl-8b@ollama-validation-ctx16k__8b1b841fc49b (unregistered) [INCOMPLETE 60/1356]: separable — qwen3.5-9b@openrouter [INCOMPLETE 369/1356] ahead, Δ=[14.6pp, 29.8pp]
- qwen3.5-9b@openrouter [INCOMPLETE 369/1356] vs vlm__qwen3-vl-8b@ollama-validation__42581bd83e8a (unregistered) [INCOMPLETE 60/1356]: separable — qwen3.5-9b@openrouter [INCOMPLETE 369/1356] ahead, Δ=[18.6pp, 33.5pp]
- qwen3.5-9b@openrouter [INCOMPLETE 369/1356] vs vlm__qwen3-vl-8b@ollama-validation__4caf39e9ac1f (unregistered) [INCOMPLETE 60/1356]: separable — qwen3.5-9b@openrouter [INCOMPLETE 369/1356] ahead, Δ=[19.9pp, 35.5pp]
- vlm__qwen3-vl-8b@ollama-validation-ctx16k__8b1b841fc49b (unregistered) [INCOMPLETE 60/1356] vs vlm__qwen3-vl-8b@ollama-validation__42581bd83e8a (unregistered) [INCOMPLETE 60/1356]: separable — vlm__qwen3-vl-8b@ollama-validation-ctx16k__8b1b841fc49b (unregistered) [INCOMPLETE 60/1356] ahead, Δ=[1.1pp, 7.1pp]
- vlm__qwen3-vl-8b@ollama-validation-ctx16k__8b1b841fc49b (unregistered) [INCOMPLETE 60/1356] vs vlm__qwen3-vl-8b@ollama-validation__4caf39e9ac1f (unregistered) [INCOMPLETE 60/1356]: separable — vlm__qwen3-vl-8b@ollama-validation-ctx16k__8b1b841fc49b (unregistered) [INCOMPLETE 60/1356] ahead, Δ=[1.9pp, 9.5pp]
- vlm__qwen3-vl-8b@ollama-validation__42581bd83e8a (unregistered) [INCOMPLETE 60/1356] vs vlm__qwen3-vl-8b@ollama-validation__4caf39e9ac1f (unregistered) [INCOMPLETE 60/1356]: not separable — Δ CI [0.0pp, 4.5pp] spans 0

### Section B
- docstrange@nanonets vs qwen3-vl-32b@openrouter-transcriber: not separable — Δ CI [-3.0pp, 6.1pp] spans 0
- docstrange@nanonets vs qwen3-vl-8b@openrouter-transcriber: separable — docstrange@nanonets ahead, Δ=[6.1pp, 19.1pp]
- qwen3-vl-32b@openrouter-transcriber vs qwen3-vl-8b@openrouter-transcriber: separable — qwen3-vl-32b@openrouter-transcriber ahead, Δ=[4.9pp, 17.1pp]

