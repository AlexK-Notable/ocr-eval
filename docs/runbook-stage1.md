# Stage 1 run procedure

The operational sequence for a full Stage 1 comparative-baseline run, written against the CLI as
actually built (`ocr_eval_ext/cli.py`; verified against `uv run ocr-eval --help` /
`uv run ocr-eval <command> --help` output at commit `39e3908`). Reproduction targets are pinned in
[`docs/superpowers/specs/table3-snapshot.md`](superpowers/specs/table3-snapshot.md) — read that
file's gate rule before step 4 or step 6 below. Registry entries referenced by id come from
[`configs/registry.yaml`](../configs/registry.yaml); pinned harness/dataset revisions come from
[`configs/pins.yaml`](../configs/pins.yaml).

## Prerequisites

- **Keys rotated before first hosted call (DoD #6).** `OPENROUTER_API_KEY` and `GEMINI_API_KEY`
  were both exposed in a prior session transcript — rotation is **mandatory** before step 4 below
  makes the first real hosted call. Do not proceed past step 3 on un-rotated keys.
- `MISTRAL_API_KEY` is also required (step 5, `mistral_ocr_4`) — no prior-exposure flag on this
  one, but it gets the same environment-only handling as the rotated keys, never a config file or
  `.env` (upstream `.env` loading is disabled — Task 1).
- **Keys are exported in the operator's shell profile.** Every `uv run ocr-eval ...` invocation
  below inherits them from the environment; no wrapper command is assumed. The Stage 1 variable
  list, and which step first needs each one:
  | Variable | First needed | Used by |
  |---|---|---|
  | `GEMINI_API_KEY` | step 1 | extractor gate, every transcriber `score`, and the Section A frontier anchor |
  | `OPENROUTER_API_KEY` | step 7 | the 3 hosted VLM rows + the transcriber calibration pair |
  | `MISTRAL_API_KEY` | step 5 | `mistral_ocr_4` hosted OCR endpoint |

  **Export `GEMINI_API_KEY` under that exact name**, not just `GOOGLE_API_KEY`: upstream's scorer
  accepts either alias, but `direct.py` resolves `api_key_env` by exact name with no fallback, so
  `gemini-3.5-flash@google-vlmchat` (step 7) fails on the Google alias alone while the step-4
  transcriber entry succeeds in the same shell. See [`api.md`](api.md#keys).

  **`.env` does not work for `ocr-eval`** — `RDB_ALLOW_DOTENV=1` gates only upstream
  `realdoc_bench/cli.py`'s `_env()`, which `ocr_eval_ext/` never calls. Keys must be in the real
  process environment.
- `uv sync` completed. Working tree at or descended from `configs/pins.yaml`'s `harness_commit`
  (`fb26a6876481de76dc293f722ab4efa71279904d`) — `_preflight`'s `git merge-base` ancestry check
  (every run-dir-scoped command below: `verify`/`direct`/`rescore`/`parse`/`score`/`report` — NOT
  `selftest` or `preflight`, neither of which touches a run dir or calls `_preflight` at all)
  fails closed on this automatically; no manual check needed beyond making sure you haven't
  checked out something unrelated.
- **Harness-pin ruling (L-f, carried from Task 7):** a deliberate re-pin of `harness_commit` in
  `configs/pins.yaml` bricks every already-stamped run dir BY DESIGN (`_preflight`'s
  `harness_commit` cross-check fails closed the instant the pin changes) — this is not a bug to
  route around; migrate by starting a new run dir at the new pin, or hand-edit that run dir's
  `run_meta.json` `harness_commit` field if you specifically intend to carry it forward.

## The numbered sequence

**1. Self-test + extractor validation gate** (fail-closed; 5 Gemini calls — needs `GEMINI_API_KEY`)
```
uv run ocr-eval selftest --extractor
```
Expect `offline scorer self-test: PASS` and `extractor validation: PASS (5/5)`. This is DoD #1's
positive-pass check; the checklist at the end of this file also asks you to observe the
*fail-closed* direction at least once.

**2. Download the pinned dataset revision** (no key needed — the dataset is public)
```
uv run realdoc-bench evaluate download --run-dir runs/stage1 \
  --dataset Extend-AI/RealDoc-Bench --revision 906170ab201d7b8238a32a9115fc66b4b72e0710
```
NB: the CLI flag is `--dataset` (`realdoc_bench/cli.py`'s `evaluate_download` maps it onto
`download_dataset(repo_id=dataset, ...)` — the function kwarg `repo_id` is not the flag name). The
revision above is `configs/pins.yaml`'s `dataset_revision` — `ocr-eval verify` cross-checks the
on-disk HF snapshot metadata against this pin on every later command and refuses to proceed on
drift, so an accidental `--revision` typo here surfaces immediately at step 3, not silently later.

**3. Full fail-closed sweep** (pins + cardinalities + every PDF single-page + non-blank render;
warms the PNG cache)
```
uv run ocr-eval verify --run-dir runs/stage1
```
Minutes, over the full 581-doc corpus. Abort and investigate on any red. This is the run dir's
first successful `_preflight` pass — it stamps `run_meta.json` (`dataset_revision`,
`harness_commit`, `renders_verified: true`) that every subsequent command cross-checks against the
pins before doing anything else.

**4. Gemini 3.5 Flash anchor** (environment-reproduction + frontier ceiling anchor; needs
`GEMINI_API_KEY`)
```
uv run ocr-eval parse --run-dir runs/stage1 -p gemini_3_5_flash
uv run ocr-eval score --run-dir runs/stage1 -p gemini_3_5_flash
```
Use these `ocr-eval` wrappers, **not** upstream `realdoc-bench evaluate run` — see Warnings below.
The `-p` argument above is the raw upstream parser name, but `report.md` labels the resulting
Section B row by the RESOLVED REGISTRY id instead (`report_md.py`'s `_row_label` renders
`entry.id`, never the parser key) — after step 9 builds `report.md`, look for the
**`gemini-3.5-flash@google`** row (`configs/registry.yaml`'s entry with
`upstream_parser: gemini_3_5_flash`), not a literal `gemini_3_5_flash` string, under the
**"Reproduction gate (upstream construction)"** block, and compare its `field% (upstream)` /
`question% (upstream)` columns against
[`table3-snapshot.md`](superpowers/specs/table3-snapshot.md)'s README row (89.3% / 82.2%) —
investigate before spending on any hosted candidate below if either is outside ±2.5pp. **Never**
compare the Section B leaderboard table's `general/field`/`strict/question` columns here instead
— those are this report's ranking key, scored under this harness's stricter D7 null-gold rule, and
diverge from upstream's own (README-published) numbers by up to ~5pp on null-gold fields alone
(see `table3-snapshot.md`'s D7 divergence note).

**5. Hosted OCR endpoint** (needs `MISTRAL_API_KEY`; upstream parser name is `mistral_ocr_4` — NOT
`mistral_ocr`)
```
uv run ocr-eval parse --run-dir runs/stage1 -p mistral_ocr_4
uv run ocr-eval score --run-dir runs/stage1 -p mistral_ocr_4
```

**6. Local specialists** (see [`docs/local-serving.md`](local-serving.md); one model resident at a
time — 16 GB VRAM). Start the server in its own terminal, preflight, then parse+score:

> **Requires a CUDA GPU — absent on the current host (probed 2026-08-03, see
> [`host-setup.md`](host-setup.md#vllm--cuda-gpu--the-local-bf16-specialists)).** This step and both
> of its DoD consequences — DoD #2's `dots.ocr` open-weight reproduction check and DoD #3's
> "≥1 local specialist (BF16)" row — are **blocked**, not merely unrun, until a GPU host is
> available or the legs are re-pinned/descoped by an explicit decision. Do not mark either DoD item
> green from a run made on this host.
```
vllm serve zai-org/GLM-OCR --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --max-model-len 8192
uv run ocr-eval preflight glm-ocr@local-vllm
uv run ocr-eval parse --run-dir runs/stage1 -p glm-ocr_local-vllm__<cond>
uv run ocr-eval score --run-dir runs/stage1 -p glm-ocr_local-vllm__<cond>
```
NB on `<cond>`: our registered transcriber parsers exist only inside the `ocr-eval` CLI process —
upstream `realdoc-bench` never imports `ocr_eval_ext`, so these names don't exist in that process
at all — and the registered name carries the transcriber-condition hash
(`safe_name(entry.id) + "__" + condition_hash(TRANSCRIBER_CONDITION)`, `ocr_eval_ext/
parsers_openai.py`). Don't hand-compute it: `ocr-eval parse`'s own console output prints the
registered names, and an `unknown parser name(s)` error (wrong guess) prints them too.

Stop the server, then repeat for dots.ocr:
```
vllm serve dots-studio/dots.ocr --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --max-model-len 8192 --trust-remote-code
uv run ocr-eval preflight dots-ocr@local-vllm
uv run ocr-eval parse --run-dir runs/stage1 -p dots-ocr_local-vllm__<cond>
uv run ocr-eval score --run-dir runs/stage1 -p dots-ocr_local-vllm__<cond> --limit 20   # smoke first
uv run ocr-eval score --run-dir runs/stage1 -p dots-ocr_local-vllm__<cond>              # full corpus
```
`dots-ocr@local-vllm`'s row is the **open-weight reproduction check** — after step 9 builds
`report.md`, compare its row under the **"Reproduction gate (upstream construction)"** block
(`field% (upstream)` / `question% (upstream)`), **not** the Section B leaderboard table's
`general/field`/`strict/question` columns (ranking key, D7-diverged — see
`table3-snapshot.md`'s D7 divergence note), against
[`table3-snapshot.md`](superpowers/specs/table3-snapshot.md) §1's `dots.ocr` entry
(70.6±3.6 / 61.4±3.5). Setup caveats apply: this is our serving stack (vLLM, this GPU, this
`--max-model-len`), not the paper's — document any residual gap in `report.md` rather than
treating a miss as an automatic fail (see that file's "CI overlap is the pass condition" note).

**7. Direct QA candidates + Section A ceiling anchor** (needs `OPENROUTER_API_KEY`,
`GEMINI_API_KEY`)
```
uv run ocr-eval direct --run-dir runs/stage1 -m qwen3-vl-8b@openrouter -m qwen3.5-9b@openrouter \
  -m qwen3-vl-32b@openrouter -m gemini-3.5-flash@google-vlmchat --dry-run
```
Inspect the printed per-entry cell counts and the labelled ±2x estimate, then re-run for real with
a spend cap:
```
uv run ocr-eval direct --run-dir runs/stage1 -m qwen3-vl-8b@openrouter -m qwen3.5-9b@openrouter \
  -m qwen3-vl-32b@openrouter -m gemini-3.5-flash@google-vlmchat --max-spend 40
```
Calibration pair (same weights, transcriber shape — registry id
`qwen3-vl-8b@openrouter-transcriber`):
```
uv run ocr-eval parse --run-dir runs/stage1 -p qwen3-vl-8b_openrouter-transcriber__<cond>
uv run ocr-eval score --run-dir runs/stage1 -p qwen3-vl-8b_openrouter-transcriber__<cond>
```
Once both rows exist, `report.md`'s cross-shape comparison note auto-detects and calls out this
pair (`_build_cross_shape_note` — no manual step needed to enable it).

**8. No-image control** (language-prior baseline; one model is enough)
```
uv run ocr-eval direct --run-dir runs/stage1 -m qwen3-vl-8b@openrouter --no-image --max-spend 10
```

**9. Build the report**
```
uv run ocr-eval report --run-dir runs/stage1
```
→ `runs/stage1/report.md`. Renames any pre-existing `dashboard.html` to
`dashboard-upstream-UNSEGREGATED.html` (see Warnings). This is the point to actually do the
step-4 and step-6 snapshot comparisons called out above. R-c: the report header also prints
`pymupdf version: <...>` straight from `run_meta.json`'s `pymupdf_version` (stamped by `verify`'s
full sweep, step 3) — worth checking alongside the D3 STALE-RENDER section if a run's renders ever
need to be reproduced on a different machine.

**10. Cache-hit rerun (DoD #5)** — re-run step 7's exact `--max-spend` command verbatim:
```
uv run ocr-eval direct --run-dir runs/stage1 -m qwen3-vl-8b@openrouter -m qwen3.5-9b@openrouter \
  -m qwen3-vl-32b@openrouter -m gemini-3.5-flash@google-vlmchat --max-spend 40
```
Expect the printed summary's `cached` count to equal every `(model × bank item)` cell and `ok`/
`error` both `0` — e.g. `{'ok': 0, 'error': 0, 'cached': 5424, 'cached_ok': 5424, 'cached_error': 0}` for 4 models × 1,356 items. Zero
HTTP calls made: `run_direct`'s cell-building loop (`if force or not cpath.exists():
cells.append(...)`) excludes a cell from the dispatch list entirely whenever its cache file
already exists — `--force` overrides that and includes it unconditionally regardless of cache
state, so a plain rerun with no `--force` never reaches `do()`/the HTTP client for a cached cell
at all.

## Bedrock path (no API keys — AWS credential chain)

An alternative source of `vlm-chat` candidate rows when `OPENROUTER_API_KEY` is unavailable.
Semantics, constraints and caveats: [`api.md`](api.md#aws-bedrock-transport-bedrock-converse);
access probing: [`host-setup.md`](host-setup.md#aws-bedrock).

Substitutes for **step 7's direct-QA candidates only**. It does *not* replace step 4/5 (both are
transcriber rows) and does *not* remove the `GEMINI_API_KEY` requirement — the extractor is
Gemini-pinned, so any transcriber scoring still needs it.

```bash
# 1. Confirm the role can actually invoke each entry (one tiny real call each; listing is NOT
#    availability — 119 models listed vs 7 invokable on the probed account).
for m in nova-lite@bedrock nova-pro@bedrock claude-haiku-4.5@bedrock gemma-3-27b@bedrock; do
  uv run ocr-eval preflight "$m" --registry configs/registry-bedrock.yaml
done

# 2. Cell count, no spend.
uv run ocr-eval direct --run-dir runs/stage1 --registry configs/registry-bedrock.yaml \
  -m nova-lite@bedrock -m claude-haiku-4.5@bedrock --dry-run

# 3. Real run. NB --max-spend FAILS CLOSED on these entries (unpriced by design — Converse returns
#    no cost field and pricing:GetProducts is denied to this role). Either add verified pricing to
#    the registry entry, or run uncapped and rely on --limit for a bounded smoke.
uv run ocr-eval direct --run-dir runs/stage1 --registry configs/registry-bedrock.yaml \
  -m nova-lite@bedrock -m claude-haiku-4.5@bedrock --limit 20

# 4. Report (registry flag so report_md.py can resolve the entries' stamp columns).
uv run ocr-eval report --run-dir runs/stage1 --registry configs/registry-bedrock.yaml
```

Three things to expect in the output rather than treat as faults:

- **~20 `api_error` rows from oversize page images.** Bedrock enforces its 5 MB per-image limit
  against the **base64-encoded** payload, so the real ceiling is ~3.75 MiB of raw PNG. At
  `render.dpi: 150` this hits 20 of 1,356 cells across 10 dense documents (measured 2026-08-04 —
  see [`api.md`](api.md#the-5-mb-image-cap-counts-base64-not-raw-bytes-measured-2026-08-04)). They
  are honest permanent failures (one attempt each, not four), and `direct` exits 2 because of them,
  which is the fail-visible contract working, not a broken run. Lowering dpi to fit would create a
  different `condition_hash` — i.e. a separate row, not a repair of this one.
- **`usage.sampling_note` on Anthropic rows.** Anthropic-on-Bedrock rejects `temperature` and
  `top_p` in one request; `top_p=1.0` is the identity value so it is omitted, and the row says so.
  A non-identity `top_p` raises instead of being dropped.
- **`precision: provider-default`** renders as "unknown (not asserted)" — Bedrock documents no
  serving precision. A Bedrock row is a serving-stack measurement, not a clean weights comparison
  against a locally-served BF16 row.

SSO sessions expire; a stale one fails fast as `ExpiredTokenException` (classified permanent, so it
does not burn 4 retries per cell). Refresh with `aws sso login`.

## Local-validation path (no keys needed)

Before spending on any hosted API, the whole `run_direct` → `build_markdown_report` seam — the
same seam `tests_ext/test_e2e_smoke.py` exercises at the unit level — can be smoke-tested
end-to-end against a fully local OpenAI-compatible endpoint (Ollama or vLLM serving any small
vision model), with **zero** API keys anywhere in the environment. Registry entries for this live
in **[`configs/registry-local-validation.yaml`](../configs/registry-local-validation.yaml)**
(committed, orchestrator-authored) rather than hand-written per run:

```yaml
- id: qwen3-vl-8b@ollama-validation             # shape: vlm-chat — direct leg
- id: qwen3-vl-8b@ollama-transcriber-validation # shape: transcriber — parse leg (score needs GEMINI_API_KEY)
```

Both point at `http://localhost:11434/v1` (Ollama) with `api_key_env: null`. **Read the file's own
header comment before using it**: `qwen3-vl:8b` served via Ollama is Q4_K_M-quantized (verified
`ollama show`), below the spec's precision-policy floor for a headline comparison ("if only Q4
fits, run hosted instead") — `precision: provider-default` here is a deliberate declaration of
"not asserted" (renders as such in `report.md`), and the `tos_note` flags this registry as
**harness-validation only, never a comparative-table candidate**. Never point Stage 1's real
`configs/registry.yaml` `--registry` invocations at this file, and never merge these two entries
into it.

### Corpus prerequisite

Every command below still goes through `_preflight`, which enforces `docs/` ⊇ every bank
`source_file` (`_check_corpus_completeness`) unless told otherwise. **The path below assumes the
same fully-downloaded, fully-verified run dir from steps 2–3** (`runs/stage1`) — `--limit` on
`direct` only bounds how many *new cells* get spent, it does not shrink the underlying corpus, so
the completeness check passes trivially and no extra flag is needed. This also means it's safe to
run against `runs/stage1` directly (validation-entry cache rows land under their own
`vlm__qwen3-vl-8b@ollama-validation__<cond>` key, never colliding with any real candidate's rows).

If you instead want a *narrower* smoke dir to avoid the full 581-doc download (e.g.
`realdoc-bench evaluate download --run-dir runs/stage1-smoke ... --limit 5`), `direct`/`parse`/
`score` all expose `--allow-partial-corpus` to skip the completeness check (with a warning, and a
sticky `partial_corpus: true` stamp in `run_meta.json`) — but note `verify` never exposes that
flag at all (by design — see `_preflight`'s docstring), so skip straight to `direct
--allow-partial-corpus` on a narrowed dir rather than trying `verify` first. **`report` has no
`--allow-partial-corpus` flag either** (`cli.py`'s `report()` calls `_preflight(layout)` with no
override, so it re-runs the full completeness check unconditionally, `partial_corpus: true` in
`run_meta.json` notwithstanding) — this is a known, previously-flagged gap (progress ledger, Task
9 minor: "report CLI --allow-partial-corpus symmetry gap"), not a documentation oversight. A
narrowed run dir can validate `direct`/`parse`/`score` end-to-end but **cannot** currently produce
a `report.md` — use the full-corpus path above if you need to see an actual rendered report.

### Steps (against `runs/stage1`, full corpus)

1. Start Ollama with the model pulled (`ollama serve`; see
   [`docs/local-serving.md`](local-serving.md) for the vLLM equivalent).
2. `uv run ocr-eval preflight qwen3-vl-8b@ollama-validation --registry configs/registry-local-validation.yaml`
3. `uv run ocr-eval direct --run-dir runs/stage1 -m qwen3-vl-8b@ollama-validation --registry configs/registry-local-validation.yaml --limit 5`
4. `uv run ocr-eval report --run-dir runs/stage1 --registry configs/registry-local-validation.yaml`

Neither `direct` (vlm-chat direct-answer — no separate extractor call) nor `report`
(`build_markdown_report` is a pure disk-reading function; `_preflight` itself needs no API key —
only `st.run_offline()`, the pins/ancestry check, and corpus completeness) touches
`GEMINI_API_KEY` or `OPENROUTER_API_KEY`. This validates render/cache/report plumbing before any
hosted spend and is a good first check after any harness code change. (Step 4's `--registry`
points at the local-validation file so `report_md.py` can resolve the entry and render its stamp
columns; a plain `ocr-eval report --run-dir runs/stage1` against the default registry would still
succeed and show the row as `(unregistered)` instead.)

### Real-world note: thinking models exhaust `max_tokens` and return empty content

**First check the variant.** Ollama's default `qwen3-vl:8b` tag resolves to the *Thinking* build
(`RENDERER`/`PARSER qwen3-vl-thinking` in its Modelfile) — not the Instruct build the hosted
comparison entries pin. For validation runs, use `qwen3-vl:8b-instruct` (the
`qwen3-vl-8b-instruct@ollama-validation` registry entry): verified 2026-08-02, it answers 60/60
at stock `STAGE1_CONDITION` with zero error rows and a median of 22 completion tokens — no
amended condition, no `num_ctx` surgery. Everything below is retained for anyone who must run a
thinking-enabled model through this path.

Observed live against `qwen3-vl:8b` via Ollama: `STAGE1_CONDITION`'s `sampling.max_tokens: 1024`
is consumed entirely by the model's own reasoning/thinking tokens, leaving nothing for the actual
answer — the response comes back with empty `message.content` and `finish_reason: "length"`. The
harness handles this exactly as designed: `direct.py`'s `_one` sees an empty `text` and writes an
`error_class: "empty"` row (never a crash, never a silently-wrong answer) — this is a normal,
expected outcome for a thinking-enabled local model at Stage 1's default budget, not a bug to
chase.

To get real (non-empty) answers for a full validation pass, widen the completion budget. The
`ocr-eval direct` CLI has **no flag for this** — `run_direct`'s `condition` parameter (which
`STAGE1_CONDITION` fills by default) isn't exposed by `cli.py`'s `direct()` command at all — so
this needs a short direct call into `ocr_eval_ext.direct.run_direct`, e.g.:
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
This lands under a **distinct cache key by design** (`condition_hash` folds the whole condition
dict, so a different `max_tokens` is a different, trackable `vlm__...__<cond>` key, never a silent
overwrite of the `error_class: "empty"` rows from the default-budget attempt). Also note two
Ollama shim limits, both verified live:

1. It ignores a `think: false` request field — there is no way to suppress thinking through
   that route.
2. The request-level `max_tokens` is **silently clamped to `num_ctx − prompt_tokens`** — the
   server's context window wins, and Ollama's default `num_ctx` is 4096. Asking for 8192
   completion tokens behind a 4096 window is a no-op that surfaces as empty-content rows whose
   `usage` sums to exactly 4096 (this was live-diagnosed 2026-08-02; it had earlier been misread
   as answers stranded in `message.reasoning` — see `docs/api.md`'s caveats section for the
   corrected diagnosis). The OpenAI-compat endpoint cannot set `num_ctx` per-request; bake it
   into a derived model instead:
   ```bash
   printf 'FROM qwen3-vl:8b\nPARAMETER num_ctx 16384\n' > /tmp/Modelfile.ctx16k
   ollama create qwen3-vl:8b-ctx16k -f /tmp/Modelfile.ctx16k
   ```
   then point a registry entry's `model:` at the derived name (see
   `qwen3-vl-8b@ollama-validation-ctx16k` in `configs/registry-local-validation.yaml`) and widen
   `max_tokens` in the amended condition. The clamp also bites the *Instruct* build at full-bank
   scale — 5 pages exceed 4096 prompt tokens outright (`api_error`) and 44 dense-page cells
   truncate mid-JSON at the wall (`parse_error`) — so full-bank local validation uses the
   `qwen3-vl-8b-instruct@ollama-validation-ctx8k` entry (num_ctx=8192; worst observed prompt
   ~4.2k tokens at 150 dpi). Verified result: 55/60 answered vs 47/60 at the 4096
   window, recovering 11 of 13 clamped cells; the residual 5 were runaway reasoning loops that
   consumed the full 16k window (~14k completion tokens, ~3–4 min per cell) — no finite window
   guarantees a thinking model converges, and which cells go runaway is not stable across
   serving configs even at `temperature 0.0`.

## Warnings

- **Never run upstream `evaluate score --force` against `vlm__*` keys directly** (bypassing the
  `ocr-eval` wrappers) — it overwrites the cached direct-QA answers with `'markdown missing'` and
  destroys paid-for rows (verified during plan review; see `cli.py`'s `rescore` docstring). Use
  `uv run ocr-eval rescore --run-dir runs/stage1` instead — it recomputes `field_matches`/`match`
  from stored answers with **zero** API calls and is safe against both shapes.
- **Never run upstream `realdoc-bench evaluate run` on a run dir after any direct rows exist.** Its
  report phase writes `dashboard.html`, which globs *every* cache record and cross-ranks both
  shapes (vlm-chat direct answers vs. transcribe-then-extract two-stage scores) into one table —
  exactly the comparison `report_md.py`'s module docstring rules out as not-the-same-measurement.
  `ocr-eval report` renames any `dashboard.html` it finds to
  `dashboard-upstream-UNSEGREGATED.html` precisely so it can never be mistaken for the
  authoritative output; **`report.md` is the only authoritative artifact.**
- **Scoring-leg cost has no automated preview (D6).** Each scored question costs one
  `gemini-3-flash-preview` extractor call per transcriber (~1,356 calls per full-corpus score run
  — low-single-digit dollars per transcriber at current Gemini pricing). Unlike the direct leg,
  `--dry-run` has no cost estimate for this leg at all (upstream exposes no hook for it — a
  documented divergence, see `docs/superpowers/plans/2026-08-01-stage1-eval-pipeline.md`). Budget
  it manually before scoring a new transcriber, and smoke with `--limit 20` first (as in step 6).

## Definition-of-done checklist

Every DoD item from the plan's Task 11, with its verifying command. Run these against
`runs/stage1` after completing the sequence above.

**1. Preconditions, scorer self-test, and extractor validation all green — fail-closed observed at
least once (deliberately corrupt a fixture, then restore).**
```
uv run ocr-eval verify --run-dir runs/stage1 --skip-renders   # cardinality + fail-closed
                                                                # pins/preconditions; renders were
                                                                # already verified in step 3, no
                                                                # need to re-sweep 581 PDFs here
uv run ocr-eval selftest --extractor              # green pass
grep '"renders_verified": true' runs/stage1/run_meta.json      # R-a: UNPIPED — read the tool's
                                                                # own pass/fail line directly, per
                                                                # lrn-ea833a5b (never trust an exit
                                                                # code read downstream of a pipe)
```
Fail-closed demonstration: temporarily flip one `expect_match` (or `gold`) value in
`ocr_eval_ext/selftest.py`'s `FIXTURES`/`EXTRACTOR_FIXTURES`, re-run `uv run ocr-eval selftest
--extractor` and confirm a red `FAIL` + non-zero exit, then `git checkout --
ocr_eval_ext/selftest.py` and re-run to confirm green again.

**2. Reproduction: `gemini-3.5-flash@google` within tolerance of snapshot; `dots.ocr` local vs.
paper CI with caveats documented.**
```
uv run ocr-eval report --run-dir runs/stage1
grep -A2 "gemini-3.5-flash@google" runs/stage1/report.md
grep -A2 "dots-ocr@local-vllm" runs/stage1/report.md
```
NB: `report.md` labels rows by registry id, not by the `-p`/upstream parser name used to score them
— `gemini_3_5_flash` (the CLI arg from step 4) never appears verbatim in the report; look for
`gemini-3.5-flash@google` instead (see step 4's note). This also matches the frontier-anchor
entry `gemini-3.5-flash@google-vlmchat` (Section A) since it shares that prefix — harmless, both
are relevant Gemini reproduction points; the Section B row is the one to compare against §2 below.

Compare against
[`docs/superpowers/specs/table3-snapshot.md`](superpowers/specs/table3-snapshot.md) (§2 for
Gemini, ±2.5pp; §1 for dots.ocr, CI-overlap-with-documented-caveats — see that file's gate rule).

**3. ≥3 hosted VLM rows, ≥1 local specialist (BF16), ≥1 hosted OCR endpoint, frontier anchor,
calibration pair.**
```
grep -E "^- \*\*qwen3-vl-8b@openrouter( \[cond [0-9a-f]{12}\])?\*\* —|^- \*\*qwen3\.5-9b@openrouter( \[cond [0-9a-f]{12}\])?\*\* —|^- \*\*qwen3-vl-32b@openrouter( \[cond [0-9a-f]{12}\])?\*\* —" \
  runs/stage1/report.md   # 3 hosted VLM (Section A, direct-QA detail bullets: `report_md.py`'s
                           # `_build_section_a` renders `- **<id>** — <stamp>...`)
grep -E "^- \*\*gemini-3\.5-flash@google-vlmchat( \[cond [0-9a-f]{12}\])?\*\* —" runs/stage1/report.md   # frontier ceiling anchor (also Section A)
grep -E "^- \*\*(glm-ocr@local-vllm|dots-ocr@local-vllm)\*\* ·" runs/stage1/report.md   # local
                           # specialist (BF16) — Section B (transcribe-then-extract) detail
                           # bullets use a DIFFERENT separator than Section A's (`_build_section_b`
                           # joins its bullet fields with " · ", not " — ") — `- **<id>** · <stamp>...`
grep -E "^- \*\*mistral-ocr@mistral\*\* ·" runs/stage1/report.md   # hosted OCR endpoint (also Section B)
grep "calibration pair detected" runs/stage1/report.md            # calibration pair
```
R-b: every grep above is anchored on the STABLE detail-bullet line (`- **<id>** ` + section-specific
separator), never a table cell. A table cell's leading text can carry an `[INCOMPLETE k/N]` marker
or (F4) a `[cond <hash>]` disambiguation suffix before its closing `" |"`, either of which breaks a
naive `"<id> |"`-anchored grep; `[INCOMPLETE k/N]` decorates only the table cell, but `[cond <hash>]` is baked INSIDE the `**...**` label — patterns below anchor on the opening `- **<id>` only, which survives both and still
excludes e.g. the `qwen3-vl-8b@openrouter-transcriber` calibration row and the "calibration pair
detected" prose line, both of which contain `qwen3-vl-8b@openrouter` as a substring but are never
followed directly by the closing `**`.

**4. Baselines + CIs + polarity split + shape segregation present in `report.md`.**
```
grep -E "^## Baseline rows|^## Section A|^## Section B" runs/stage1/report.md
grep "polarity checked" runs/stage1/report.md
grep -E "\[[0-9]+\.[0-9]%, [0-9]+\.[0-9]%\]" runs/stage1/report.md | head -3   # bracketed CIs present
```

**5. Cache-hit rerun: 100% cached, zero API calls.**
```
uv run ocr-eval direct --run-dir runs/stage1 -m qwen3-vl-8b@openrouter -m qwen3.5-9b@openrouter \
  -m qwen3-vl-32b@openrouter -m gemini-3.5-flash@google-vlmchat --max-spend 40
```
(step 10 above — confirm the printed `cached` count equals total cells, `ok`/`error` both `0`, AND
`cached_error` is `0` — F3: a non-zero `cached_error` means a previous run left error cells behind
that this plain rerun never re-attempted; `direct` now exits 2 in that case with a message to
rerun with `--force` to re-attempt them).

**6. Keys were rotated before first hosted call.**

Manual sign-off, not a code-checkable gate — and not checkable from this repo at all, since keys
live only in the environment. Confirm at the **issuing provider's own console** that the
`OPENROUTER_API_KEY` and `GEMINI_API_KEY` currently exported were created *after* the
prior-exposure incident and *before* step 4's first hosted call:

- OpenRouter → Keys page (each key shows its creation date)
- Google AI Studio → API keys (creation date per key)

Record the confirmation (who, when, and each key's creation date) alongside the run's `report.md`.
Rotating means **creating a new key and revoking the old one** — not just re-exporting the same
value into a new shell.
