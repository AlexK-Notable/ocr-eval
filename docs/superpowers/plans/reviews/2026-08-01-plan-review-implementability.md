# Plan review — implementability vs upstream code (Opus agent, executed verification)

**Date:** 2026-08-01 · **Method:** the reviewer `uv sync`'d a clone of upstream at the pinned commit
`fb26a687` and **ran the plan's code and tests against it** — findings marked CONFIRMED were reproduced,
not inferred. **Disposition:** all findings applied to the plan (see git history). Kept for the audit trail.

---

## Findings (all applied)

1. **CONFIRMED — `null_fields()` 3-tuple vs `field_outcomes()` 4-tuple unpack.** Reproduced `ValueError: not enough values to unpack`. The plan's own test hid it by hand-building a 4-tuple. Blocks the blank-field headline number. *Fixed: 4-tuple shape + test exercises the real seam.*
2. **CONFIRMED — PNG render-cache race aborts the run.** `write_bytes` truncates first; concurrent reader sees 0 bytes → `PIL.UnidentifiedImageError` → unhandled through `pool.map`. Stress-reproduced: 5 exceptions / 30 trials, byte-lengths `[0, 9792]`. Near-certain in production (adjacent bank items share `source_file`). *Fixed: temp-file + `os.replace`, empty-read retry, per-cell try/except so a bad doc costs one cell.*
3. **CONFIRMED — the claim that `evaluate score` rejects our parser names is false; `--force` destroys direct rows.** `evaluate_score` never calls `_validate_plugin_names` (only `parse`/`run` do). Benign corollary: upstream rescoring works on `vlm__*` records and **preserves every extra key**. Destructive corollary (observed): `evaluate score --force -p vlm__…` overwrites the row with `error: markdown missing` — answer, raw response, usage gone. *Fixed: rationale corrected; runbook prohibition; ocr-eval rescore iterates cache files directly.*
4. **`ocr-eval rescore` declared but never implemented in any step.** *Fixed: full implementation added to the CLI task.*
5. **CONFIRMED — upstream dashboard.html cross-ranks both shapes.** `vlm__…` rows appear ranked in the same summary table as parsers. Runbook's `evaluate run` regenerates it. *Fixed: ocr-eval wrappers everywhere; report renames the artifact.*
6. **Runbook flag `--repo-id` does not exist** — upstream declares `--dataset`. *Fixed.*
7. **CONFIRMED — Task 3 test regex `"bank items"` never matches** (message says `bank_items`). *Fixed: match on "cardinality mismatch".*
8. **CONFIRMED — Task 4 float-equality test failure** (`1 - 2/3 != 1/3`). *Fixed: count-based `always_false` + pytest.approx.*
9. **Task 9 mapping drops upstream-parser entries** — `safe_name("gemini-3.5-flash@google")` ≠ upstream's `gemini_3_5_flash`; anchor and Mistral rows would render `(unregistered)`, losing licence/ToS stamps. *Fixed: map by `e.upstream_parser` exactly.*
10. **`--dry-run` priced nothing** despite the test name and spec requirement. *Fixed: labelled estimate from registry rates; ledger D6.*
11. **CONFIRMED — `parsers_openai.py` NameError:** `parser_registry`, `Path`, `RegistryEntry` unimported. Adding the imports makes all Task 8 tests pass. *Fixed.*
12. **`verify` checked neither page_count nor dataset revision; nothing stamped pins into run metadata.** *Fixed: full-corpus sweep in verify + run_meta.json.*
13. **`verify` was cwd-locked** (relative `configs/pins.yaml`, ambient-cwd git). *Fixed: package-relative paths + `cwd=REPO_ROOT`.*
14. **Cost structurally unavailable for local transcriber rows** — `parse_cost("")→None` coerced to `$0.0000`, a plausible-looking fake. *Fixed: `n/a` rendering for `local: true`; registry pricing fallback.*
15. **`--max-spend` brakes but overshoots ≤ workers in-flight calls** (measured 12/50 cells at workers=8 — the generator-cleanup cancel works); the OpenRouter `usage: {include}` comment was stale (cost now always included; SDK preserves the key: `model_dump()["usage"]["cost"]`). *Fixed: documented overshoot; comment corrected; single-threaded `track` needs no lock (verified).*
16. **PLAUSIBLE — extractor fixture false-fail:** gold `"Rivera"` vs verbatim-span extraction `"Maria Rivera"`; fuzzy fallback blocked by FUZZ_MIN_WORDS=5 → a correct extraction fails a fail-closed gate. *Fixed: gold = full verbatim span.*
17. **Task 1 ordering breaks `uv sync`** (hatchling errors on declared-but-missing `ocr_eval_ext/`); `import os` is NOT present in upstream cli.py. *Fixed: explicit skeleton step before pyproject; import noted.*
18. **Rich console wrapping makes CLI output assertions luck-dependent.** *Fixed: `Console(width=200)`.*

## What checked out (do not re-verify)

- **The plan's "Verified upstream interfaces" block is accurate symbol-for-symbol at `fb26a687`** — every RunLayout member, all score.py functions, all nine ParseResult fields, the parser registry surface, `DEFAULT_DPI=150`, `_render_pdf_pages`, `_MARKDOWN_PROMPT`, `download_dataset`, `cli._env`.
- **All 7 scorer selftest fixtures produce the expected verdicts against real upstream code** (traced AND executed): `<text|null>` → `"string | null"`; `string_keys` strips `| null` so nullable strings are fuzzy-eligible; textual `"blank"` scores wrong; booleans compare by identity so polarity inversion and None-vs-True both score False. The flagged "semantic risk" abort will not fire.
- **Cache-record compatibility holds:** extras survive upstream rescoring; `aggregate_results` picks up `vlm__*` files; upstream report doesn't crash on them (it just mixes shapes — finding 5).
- **Fork mechanics clean:** zero path overlap; `git merge --allow-unrelated-histories` conflict-free; upstream baseline = **85 passed in ~1s, no keys, no network**.
- **Tasks 2, 6 and 8 run green as written** (Task 8 after the import fix). MockOpenAI works with openai-python's lenient response construction; `extra_body` provider pin reaches the wire verbatim; pymupdf fixture calls all valid.
- Task 5's bootstrap tests pass, including clustered-wider-than-independent.

## Verdict (verbatim)

"Implementable after fixes — the architecture is sound, the upstream reading is unusually accurate, and the defects are concentrated and cheap to repair. […] I would fix #1–#5 before Task 1 starts, since #1 and #9 change interfaces that Tasks 3–4 freeze and #2 changes code Task 6 writes."
