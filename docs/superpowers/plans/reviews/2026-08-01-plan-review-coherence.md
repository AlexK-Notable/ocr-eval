# Plan review — coherence & spec coverage (Opus agent)

**Date:** 2026-08-01 · **Scope:** Stage 1 plan + Stage 2/3 roadmap + applicability table vs spec rev 2.
**Disposition:** all findings triaged and applied to the plan (see git history for the fix commit); items
1–5 were blocking; the divergence ledger (plan Global Constraints D1–D8 + spec rev 2.1 appendix) resolves
the documentation-debt findings. Kept verbatim for the audit trail.

---

## 1. The blank-field hallucination metric (headline #2, n=188) scores extractor failure as success [verified upstream]

`score_typed` does `av = answer.get(k) if isinstance(answer, dict) else None`, and `deep_equal(None, None) → True`. Upstream's `_worker` writes `{"answer": None, ...}` whenever `gemini_extract` returns None. The plan's `field_outcomes` only checked `"answer" not in rec` — so `answer: null` and key-omitted answers both scored **correct on every null-gold field**. A transcriber whose extractor collapses gets a 100% "does not hallucinate" score and ranks top. The spec's ruling is explicit and stricter. **Fix (applied):** answer-None → error; null-gold correct only on key-present explicit None; two new metrics tests; documented as divergence D7 (stricter than upstream).

## 2. Every fail-closed gate was skippable on the paths that actually spend money

`direct` ran only the offline selftest (no `check_bank`); `parse`/`score` ran no gate at all (score = the paid Gemini leg); the extractor-validation fixture was wired to a manual runbook step; `verify` warned (not failed) on a harness-pin mismatch and never checked `dataset_revision`. This is the lrn-ea833a5b shape the spec was written to prevent. **Fix (applied):** `_preflight()` (cardinalities + pins + scorer selftest) at the top of `direct`/`parse`/`score`/`report`/`rescore`; blocking extractor gate in `score` cached per (extractor, fixture-hash); pins hard-fail; `run_meta.json` records and re-checks `dataset_revision`.

## 3. `--max-spend` could not stop a run, and there were no retries

`usage.cost` absent on vLLM (and the OpenRouter enablement comment was stale); `pool.map` submits everything up front; `_one` and `_call_page` had zero retries; one transient 429 froze a cell permanently behind `--force`. **Fix (applied):** cost fallback = token counts × registry `pricing` rates; bounded retries with Retry-After honoring; overshoot documented (≤ workers, measured); `--retry-errors` noted for follow-up. (Plan-review execution found OpenRouter now always includes usage cost — `usage: {include}` is deprecated/no-op.)

## 4. Transcriber rows carried no condition dict and no condition in any cache key [verified upstream]

Upstream transcript key = `(parser, stem)`, extraction key = `(qid, parser)` — no condition, no extractor generation. Stage 2 deskew would overwrite raw transcripts in place; `lenient` extractions would collide with `schema_prompted`; `extractor_generation` was implemented nowhere. **Fix (applied):** registered parser names carry `__<condition-hash>`; `parses/<parser>/condition.json` sidecar; extractor id recorded in `run_meta.json` with `--new-extractor-generation` archival; upstream-parser rows (published-comparability) keep upstream names and never vary conditions.

## 5. Stage 1 DoD #3 was not achievable with the artifacts the plan built

Only two hosted `vlm-chat` entries; no transcriber-shaped VLM for the calibration pair (and `register_openai_parsers` skips non-transcribers); the frontier anchor existed only in Section B. **Fix (applied):** added `qwen3-vl-32b@openrouter`, `qwen3-vl-8b@openrouter-transcriber`, `gemini-3.5-flash@google-vlmchat` (Google's OpenAI-compat endpoint); registry test asserts DoD-category satisfiability.

## 6. Two runbook commands wrong; one stated upstream fact false [verified upstream]

`mistral_ocr_4` not `mistral_ocr`; `--dataset` not `--repo-id`; `evaluate score` does NOT validate parser names (only `parse`/`run` do). **Fix (applied):** all corrected; rescore rationale restated; registry step verifies `upstream_parser` names against `evaluate list`.

## 7. Upstream's dashboard.html renders a single ranked table mixing both shapes [verified upstream]

`report.py` globs all cache records with no shape filter; `evaluate run` regenerates it. **Fix (applied):** runbook uses `ocr-eval parse`/`score` (never `evaluate run` after direct rows exist); `ocr-eval report` renames dashboard.html to `dashboard-upstream-UNSEGREGATED.html`.

## 8. Resolved serving identity not in the cache key — undocumented divergence

**Fix (applied):** documented as D4 with the hard-fail-on-multi-provider report guard (chosen over post-hoc key rewriting).

## 9. Task 3 → Task 4 interface mismatch: `null_fields()` 3-tuples vs 4-tuple unpack

**Fix (applied):** 4-tuple `(qid, key, None, source_file)`; tests exercise the real seam.

## 10. The image-hash divergence was acceptable only with a drift check no task built

**Fix (applied):** report task now includes the STALE-RENDER gate with `--allow-stale-render` escape (D3).

## 11. Spec-required row and registry fields absent

prompt hashes per row · pixel dims + encoded bytes · `contract: not-promptable` flag · extractor id from run_meta (not hardcoded) · same-family pairing flag · `pricing_source`/pricing · `refusal` error class (was dropped; `render_error` added) · local sensitivity check deferral unstated. **Fix (applied):** all added to Tasks 2/6/9; sensitivity check recorded as deferral D8; `refusal` classified via marker heuristics.

## 12. Neither spend-preview leg was implemented

**Fix (applied):** direct-leg labelled estimate in `--dry-run`; transcriber leg documented as D6 with runbook budgeting.

## 13. Raster-only, single-page, and ink floor covered only the direct path [verified upstream]

mistral_ocr_4 uploads the PDF; `assert_single_page`/`ink_coverage` ran lazily on direct cells only. **Fix (applied):** `verify` sweeps every PDF (single-page + non-blank render, warming the PNG cache); pdf-direct labelled as D5.

## 14. Report gaps vs measurement rules

beats-majority comparison rule · bucket-overlap line · transcript-recall needs glyph AND label · precision policy unenforced, `provider-default` undeclared. **Fix (applied):** all added to Task 9; `provider-default` renders "unknown (not asserted)".

## 15. Stage 2 forward-compat bugs cheap to prevent now

PNG cache path ignored `preprocess` · write race (also found by the other reviewer, confirmed) · `no_image` added as a new key (violating "values, never keys") · `seed` missing. **Fix (applied):** all in STAGE1_CONDITION/_render_page.

## 16. Undocumented naming/parameter divergences

`schema` → `schema_prompted` · `max_tokens` scope · JSONL → per-cell JSON · Section B cost/latency columns can only describe the transcription leg. **Fix (applied):** divergence ledger D1/D2 + spec rev 2.1 appendix; Section B column scoping in Task 9.

## 17. Cardinality constants: guard against the "fix the constant" failure mode

**Fix (applied):** stop-and-investigate guard comment with derivation-command requirement.

## Coverage verdict (verbatim)

"Implementable after fixes — not structurally deficient. […] What blocks a start is a coherent cluster, not scattered nits: the fail-closed gates are not wired into the spending paths (2), both spend controls are inert (3), the null metric scores extractor failure as success (1), and the transcriber half of the matrix carries no condition in any key (4)."
