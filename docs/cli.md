# CLI reference

Every `ocr-eval` command: purpose, key flags, the gates it runs before doing anything, and exit
codes. Plus which upstream `realdoc-bench` commands are still used directly, which are
deliberately avoided, and a worked keyless end-to-end example. **Who should read this:** anyone
running the pipeline. For *why* the gates exist, see [`architecture.md`](architecture.md); for the
full numbered run procedure and DoD checklist, see
[`runbook-stage1.md`](runbook-stage1.md) — this doc does not duplicate it.

All commands below are verified against `uv run ocr-eval <command> --help` and `ocr_eval_ext/cli.py`
at commit `b3f4703`.

## Exit code convention

- **0** — clean pass, nothing to investigate.
- **1** — refused before (or without) spend: a precondition, pin, gate, or configuration problem.
- **2** — spend happened but the result is fail-visible: some cells errored, or a prior run left
  unresolved error cells behind. The run's data is still written; the nonzero code means "look at
  it before treating this as done."

## Commands

| Command | Purpose | Key flags | Gates run first | Exit codes |
|---|---|---|---|---|
| `verify --run-dir DIR [--skip-renders]` | Full fail-closed sweep: pins, bank cardinality, every PDF single-page + non-blank render. Warms `docs_png/`. Stamps `run_meta.json`. | `--skip-renders` (pins+cardinality only, fast path — the full sweep is still required at least once) | `_preflight` (self-test, pins, cardinality, corpus completeness) | 0 pass; 1 on any red (multi-page/blank docs, precondition failure) |
| `selftest [--extractor]` | Offline scorer fixtures; with `--extractor`, also 5 Gemini calls against known-answer fixtures. | `--extractor` | none (this *is* the gate) | 0 pass; 1 on any fixture failure |
| `direct --run-dir DIR -m ID [-m ID ...]` | Run the vlm-chat shape: image + question → typed JSON answer, cached under `vlm__<id>__<cond>`. | `--registry PATH`, `--dry-run`, `--max-spend USD`, `--limit N`, `--no-image`, `--workers N` (default 8), `--force`, `--allow-partial-corpus` | `_preflight`; rejects any requested entry that isn't `shape=vlm-chat, transport=openai-compat` | 0 clean; 2 if `cached_error>0` or any live `error` occurred; `--dry-run` always returns 0 (no spend) |
| `rescore --run-dir DIR` | Recompute `field_matches`/`match` from stored answers with the *current* scoring code — both shapes, zero API calls. | none | `_preflight(..., skip_corpus_check=True)` (never reads `docs/`) | 0 pass (prints examined/skipped/changed counts); 1 on a preflight failure (self-test/pins/cardinality — rescore still runs the full gate minus the corpus check) |
| `preflight ENTRY_ID [--registry PATH]` | `GET {base_url}/models`, confirm the registry's `model` is actually being served. Run before any local vLLM spend. | `--registry PATH` | none (openai-compat entries only; errors on upstream-parser entries) | 0 serving confirmed; 1 mismatch or unreachable |
| `parse --run-dir DIR -p PARSER [-p PARSER ...]` | Register this run's dynamically-built openai-compat transcriber parsers, then delegate to upstream `run_parse`. Writes `condition.json` sidecars. | `--registry PATH`, `--workers N` (auto: 1 if any requested parser is `local: true`, else 8), `--force`, `--limit N`, `--allow-partial-corpus` | `_preflight`; parser-name registration/collision check | 0 clean; 1 unknown parser name or registration collision; 2 any parse failed, or a transcript is too short to score (content floor, `md_length <= 16*page_count`) |
| `score --run-dir DIR [-p PARSER ...]` | Register transcriber parsers, then delegate to upstream `run_score` + `aggregate_results`. Requires `GEMINI_API_KEY`/`GOOGLE_API_KEY`. | `--registry PATH`, `--force`, `--workers N` (default 16), `--limit N`, `--new-extractor-generation`, `--allow-partial-corpus` | `_preflight`; `require_api_key()`; **blocking** `require_extractor_gate` (5 Gemini calls, cached per run dir/model/fixture-hash) | 0 all cells answered; 1 unknown parser name, or extractor validation failed; 2 some cells errored |
| `report --run-dir DIR [--registry PATH] [--allow-stale-render] [--iters N]` | Build the shape-segregated `report.md`. Renames any pre-existing `dashboard.html` to `dashboard-upstream-UNSEGREGATED.html`. | `--allow-stale-render`, `--iters` (bootstrap resamples, default 2000) | `_preflight`; D3 STALE-RENDER (fails unless `--allow-stale-render`); D4 serving-identity (unconditional, no escape hatch) | 0 written; 1 `ReportError` (stale render without the flag, or a serving-identity violation) |

**On `_preflight` failures:** every row above whose "Gates run first" column includes
`_preflight` can also exit **1** purely from that shared gate (scorer self-test failure, harness-pin
ancestry mismatch, dataset-revision mismatch, bank cardinality mismatch, or missing corpus without
`--allow-partial-corpus`) before the command's own logic ever runs — the per-command reasons listed
above are in addition to that, not instead of it. `verify` *is* this gate plus the render sweep, so
its own row already covers the full set.

## Upstream `realdoc-bench` commands

**Still used directly:**
- `realdoc-bench evaluate download --run-dir DIR --dataset REPO --revision REV [--limit N]` — pulls
  `qa_bank.json` + `docs/` from the pinned HF dataset revision. No `ocr-eval` wrapper exists or is
  needed for this step. Note the flag name: `--dataset`, mapped internally onto
  `download_dataset(repo_id=dataset, ...)` — the function's own kwarg name is not the flag name.
- `realdoc-bench evaluate list` — lists parse providers registered in the running process. This
  fork's openai-compat transcriber parsers are registered dynamically, on demand, only inside
  `ocr-eval parse`/`ocr-eval score`'s own process (`ocr_eval_ext.parsers_openai.register_openai_parsers`)
  — the separate `realdoc-bench` binary never imports `ocr_eval_ext` at all, so `realdoc-bench
  evaluate list` can never show them, in any invocation. Read the registered names off `ocr-eval
  parse`'s own console output instead (it prints them, and an `unknown parser name(s)` error prints
  them too).

**Deliberately not used:**
- **`realdoc-bench evaluate score --force` against `vlm__*` keys.** `_worker` (`score.py`) looks for
  a transcript at `parses/<parser>/<stem>.md`; a `vlm__` key has no such file (it's a direct-QA row,
  not a transcriber row), so `--force` overwrites the cached answer with `{"error": "markdown
  missing"}` — destroying paid-for direct-QA answers. Use `ocr-eval rescore` instead (zero API
  calls, safe on both shapes). See the runbook's Warnings section and `cli.py`'s `rescore`
  docstring.
- **`realdoc-bench evaluate run`** (the upstream parse→score→report one-shot). Its report phase
  writes `dashboard.html`, which globs *every* cache record and ranks both shapes — vlm-chat direct
  answers and transcribe-then-extract scores — in one table. That is exactly the comparison
  `report_md.py`'s module docstring rules out as not the same measurement. `ocr-eval report`
  renames any `dashboard.html` it finds to `dashboard-upstream-UNSEGREGATED.html` on sight, so it
  can never be mistaken for the authoritative output — `report.md` is the only authoritative
  artifact.

## Worked example: the keyless local flow

No API keys anywhere in the environment. Uses
[`configs/registry-local-validation.yaml`](../configs/registry-local-validation.yaml) against a
local Ollama endpoint. Full narrative (including the corpus prerequisite and a narrowed-run-dir
variant) is in the runbook's
[local-validation section](runbook-stage1.md#local-validation-path-no-keys-needed); this is the
condensed command sequence with what to expect at each step.

```bash
uv run ocr-eval selftest
# -> "offline scorer self-test: PASS"; exit 0

uv run ocr-eval preflight qwen3-vl-8b@ollama-validation \
  --registry configs/registry-local-validation.yaml
# -> "preflight PASS — http://localhost:11434/v1 serves qwen3-vl:8b"; exit 0

uv run ocr-eval direct --run-dir runs/stage1 -m qwen3-vl-8b@ollama-validation \
  --registry configs/registry-local-validation.yaml --dry-run
# -> {'cells': N, 'priced_cells': 0, 'unpriced_cells': N, 'estimated_usd': 0.0, ...}
#    (no registry pricing on a local entry — cells are counted, never silently priced at $0)

uv run ocr-eval direct --run-dir runs/stage1 -m qwen3-vl-8b@ollama-validation \
  --registry configs/registry-local-validation.yaml --limit 5
# -> {'ok': N, 'error': 0, 'cached': 0, 'cached_ok': 0, 'cached_error': 0}; exit 0
#    (or exit 2 with a nonzero 'error' — see the caveat below)

uv run ocr-eval report --run-dir runs/stage1 \
  --registry configs/registry-local-validation.yaml
# -> runs/stage1/report.md written
```

**Thinking-model empty-response caveat:** observed live against `qwen3-vl:8b` via Ollama —
`STAGE1_CONDITION`'s default `max_tokens: 1024` can be consumed entirely by the model's own
reasoning tokens, leaving an empty `message.content` and `finish_reason: "length"`. `direct.py`'s
`_one` handles this as designed: an `error_class: "empty"` row, never a crash. This is why the
`direct` summary above can show a nonzero `error` count on a thinking-enabled local model at the
default budget — it is not a harness bug.

**The amended-condition workaround:** to get non-empty answers, widen the completion budget. There
is **no CLI flag** for this today — `run_direct`'s `condition` parameter (which `STAGE1_CONDITION`
fills by default) is not exposed by `cli.py`'s `direct()` command. The current mechanism is a short
direct call into the library function:

```python
from ocr_eval_ext.config import get_entry, load_registry
from ocr_eval_ext.direct import STAGE1_CONDITION, run_direct
from realdoc_bench.evaluate.runs import RunLayout

layout = RunLayout.at("runs/stage1")
entries = load_registry("configs/registry-local-validation.yaml")
entry = get_entry(entries, "qwen3-vl-8b@ollama-validation")
cond = {**STAGE1_CONDITION, "sampling": {**STAGE1_CONDITION["sampling"], "max_tokens": 8192}}
run_direct(layout, [entry], condition=cond, limit=5)
```

This lands under a distinct cache key by design (`condition_hash` folds the whole condition dict),
so it never overwrites the default-budget attempt's error rows — see
[`architecture.md`](architecture.md#the-condition-dict) for the live-validated two-condition
example this produced. A CLI flag for this (e.g. `direct --max-tokens N`) is future work, not yet
built. One caveat discovered after this example was first written: against Ollama, raising
`max_tokens` alone is bounded by the serving side — the shim silently clamps generation to
`num_ctx − prompt` (default `num_ctx=4096`) regardless of the request. See
[`api.md`](api.md#known-provider-behavior-caveats-from-validation) for the verified diagnosis and
the derived-model (`PARAMETER num_ctx`) fix.
