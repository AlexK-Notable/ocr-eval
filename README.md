# ocr-eval

Stage 1 of a benchmark/eval pipeline for document/form extraction — **checkbox state first** —
measuring VLMs (direct QA) and OCR systems (transcribe-then-extract) on RealDocBench. This is a
fork of [`extend-hq/realdoc-bench`](https://github.com/extend-hq/realdoc-bench) (Apache-2.0),
pinned at harness commit `fb26a687` with dataset revision `906170ab` (HF
`Extend-AI/RealDoc-Bench`, CC-BY-4.0). Upstream's own README is preserved verbatim at
[`docs/upstream-README.md`](docs/upstream-README.md).

## Why this exists

A prior model-selection survey (the design spec's cited provenance —
[`docs/superpowers/specs/2026-08-01-ocr-eval-pipeline-design.md`](docs/superpowers/specs/2026-08-01-ocr-eval-pipeline-design.md))
audited the document-AI benchmark landscape and found that no published checkbox-*state* accuracy
number exists anywhere: across sixteen benchmarks checked by name, not one scores checkbox or
selection-mark state as its own metric. RealDocBench is the exception that makes this project
possible — a public, CC-BY-4.0, real-scanned-forms bank that tags checkbox questions at all.

But the upstream "checkbox" bucket doesn't measure what it's named for. This repo's own review of
the bucket ([`docs/superpowers/specs/reviews/2026-08-01-review-opus.md`](docs/superpowers/specs/reviews/2026-08-01-review-opus.md))
found it holds 1,117 gold fields, of which only 258 (23.1%) are boolean — the rest are strings,
numbers, nulls, and lists. A "checkbox accuracy" number computed the naive way (all fields in the
bucket) is ~77% determined by string/number extraction, not checkbox reading. This pipeline's
primary metric is instead **per-field accuracy restricted to the 258 boolean golds** (165 checked /
93 unchecked), always reported **polarity-split** — a page-level or bucket-level metric cannot see
checkbox *inversion* (answering the opposite of what's checked), which is the failure mode this
project exists to expose.

## Two measurement shapes — never ranked in one table

- **`vlm-chat`** — a model sees the page image + question and answers directly, one call.
- **`transcriber`** — a system transcribes the page to markdown; a separate pinned Gemini extractor
  answers the question from the transcript alone (upstream's own construction). A transcriber row
  measures *(transcriber ∘ extractor)*, not the transcriber alone.

A transcriber's checkbox score is dominated by whether its markdown renders checkbox glyphs at
all — that's a different measurement than a VLM reading the page directly, so the two shapes never
share a leaderboard table. See [`docs/architecture.md`](docs/architecture.md) and
[`docs/scoring.md`](docs/scoring.md) for the full rationale.

## Host prerequisites

Verified on the current host 2026-08-03 — what runs as-is, and what needs installing first. See
[`docs/host-setup.md`](docs/host-setup.md) for the probe commands behind each row.

| Requirement | Needed for | Status on this host |
|---|---|---|
| `uv sync --extra dev`, `pytest`, `ocr-eval selftest` | everything offline | **Working** (282 tests pass; offline scorer gate green) |
| Pinned dataset revision reachable (581 docs, 538 MB) | steps 2–3 | **Working** (public; sha matches `pins.yaml`; `check_bank` passes on the real bank) |
| `GEMINI_API_KEY` exported | extractor gate, all `score`, Section A anchor | **Working** — `selftest --extractor` passes `5/5`. NB export the exact name, not only `GOOGLE_API_KEY` |
| `OPENROUTER_API_KEY` / `MISTRAL_API_KEY` exported | hosted rows (steps 5, 7) | **Not set** — hosted legs blocked until exported |
| AWS credentials + Bedrock model access | `vlm-chat` rows with no API key ([`registry-bedrock.yaml`](configs/registry-bedrock.yaml)) | **Working** — 7 models invokable, live-validated end-to-end |
| Ollama on `localhost:11434` | the keyless quickstart below | **Not installed** — quickstart cannot run as written |
| CUDA GPU + `vllm` (16 GB) | local BF16 specialists (step 6) | **Absent** — no GPU on this host; see below |

**No GPU means two DoD items are currently unsatisfiable.** `glm-ocr@local-vllm` and
`dots-ocr@local-vllm` cannot be served here, which blocks DoD #2's open-weight reproduction check
(`dots.ocr` vs the paper's 70.6±3.6 / 61.4±3.5) and DoD #3's "≥1 local specialist (BF16)" row. This
is recorded rather than routed around, per the repo's fail-closed convention — nothing unrunnable
should read as a pass. Resolution (rent a GPU, re-pin to hosted serving with an honest precision
stamp, or formally descope both items) is an open decision, not yet made.

## Quickstart — keyless local validation

**Requires a local Ollama install** (not present on this host — see Host prerequisites above). No
API keys required. Exercises render → cache → report end-to-end against a local
OpenAI-compatible endpoint. Full details, caveats, and the thinking-model empty-response
gotcha are in [`docs/runbook-stage1.md`](docs/runbook-stage1.md#local-validation-path-no-keys-needed).

```bash
uv run ocr-eval selftest                                                    # offline scorer gate
uv run ocr-eval preflight qwen3-vl-8b@ollama-validation \
  --registry configs/registry-local-validation.yaml                        # confirm Ollama serves it
uv run ocr-eval direct --run-dir runs/stage1 -m qwen3-vl-8b@ollama-validation \
  --registry configs/registry-local-validation.yaml --dry-run              # cell count, no spend
uv run ocr-eval direct --run-dir runs/stage1 -m qwen3-vl-8b@ollama-validation \
  --registry configs/registry-local-validation.yaml --limit 5              # real (free) local calls
uv run ocr-eval report --run-dir runs/stage1 \
  --registry configs/registry-local-validation.yaml                        # -> runs/stage1/report.md
```

## Status

| Area | State |
|---|---|
| Full-corpus download + `verify` (581 docs, cardinality preconditions) | Implemented, live-validated |
| Keyless local direct-QA smoke (Ollama vlm-chat → cache → `report.md`) | Implemented, live-validated |
| Condition-hash disambiguation (same registry id, two condition hashes) | Implemented, live-validated |
| Hosted `vlm-chat` candidates (OpenRouter Qwen3-VL/Qwen3.5) | Implemented; needs `OPENROUTER_API_KEY`, unrun |
| Bedrock `vlm-chat` candidates (`transport: bedrock-converse`) | Implemented, live-validated (no API key — SigV4); transcriber leg not wired |
| Gemini extractor validation gate + scoring leg | **Run** — 4,068 cells scored, 0 errors; extractor `gemini-3.6-flash` (divergence D11 rev 3 — best of three graders measured head-to-head, p<1e-8), stamped per row |
| Hosted transcriber (Mistral OCR 4) | Implemented; needs `MISTRAL_API_KEY`, unrun |
| Hosted transcriber (Nanonets DocStrange) | **Run + scored** — 581/581 parsed ($5.81), 93.0% checkbox, 97.6% transcript-recall, 82.1% strict ([results](docs/results-stage1-2026-08-04.md)) |
| Hosted transcribers (Qwen3-VL 8B / 32B) | **Run + scored** — 1,162/1,162 parsed, 0 fail; 32B ties DocStrange on checkboxes at ~1/8 the cost |
| Transcriber `qwen3.5-9b` | Registered but **not viable** — non-terminating on ~half of dense pages, any budget; see results doc §5 |
| Local vLLM specialists (GLM-OCR, dots.ocr) | Implemented ([local-serving.md](docs/local-serving.md)); **blocked on this host — no GPU** |
| Reproduction gate (DoD #2) | Has its first scored transcriber rows, but the comparison against upstream Table 3 is not yet done (D11 changed the extractor — see results doc §3); `dots.ocr` leg blocked (no GPU) |
| Stage 2 (conditions, classical engines) / Stage 3 (CheckboxQA, HITL) | Not started — see [roadmap](docs/superpowers/plans/2026-08-01-stage2-3-roadmap.md) |

## Doc map

| Doc | Covers |
|---|---|
| [`docs/host-setup.md`](docs/host-setup.md) | Host prerequisites with the probe command behind each claim — toolchain, dataset, keys, local serving |
| [`docs/architecture.md`](docs/architecture.md) | Module map, run-dir anatomy, condition dict, cache semantics, fork boundaries, fail-closed gate inventory |
| [`docs/cli.md`](docs/cli.md) | Every `ocr-eval` command, upstream commands used/avoided, a worked keyless example |
| [`docs/scoring.md`](docs/scoring.md) | The scoring rubric — upstream scorer, our metrics layer, baselines, uncertainty, reproduction gate |
| [`docs/api.md`](docs/api.md) | Provider contract, retry policy, cost control, per-provider serving notes and caveats |
| [`docs/runbook-stage1.md`](docs/runbook-stage1.md) | The operational, numbered Stage 1 run procedure and DoD checklist |
| [`docs/local-serving.md`](docs/local-serving.md) | vLLM launch lines and context-budget arithmetic for local specialists |
| [`docs/results-stage1-2026-08-04.md`](docs/results-stage1-2026-08-04.md) | Measured Stage 1 results: the three scored transcriber rows, extractor A/B, budget sizing, harness defects found |
| [`docs/applicability-table.md`](docs/applicability-table.md) | Which condition axis binds to which pipeline step, per shape |
| [`docs/superpowers/specs/2026-08-01-ocr-eval-pipeline-design.md`](docs/superpowers/specs/2026-08-01-ocr-eval-pipeline-design.md) | The design spec (rev 2 + rev 2.1 divergence appendix) |
| [`docs/superpowers/plans/2026-08-01-stage1-eval-pipeline.md`](docs/superpowers/plans/2026-08-01-stage1-eval-pipeline.md) | The implementation plan and full divergence ledger (D1–D10) |
| [`docs/superpowers/plans/2026-08-01-stage2-3-roadmap.md`](docs/superpowers/plans/2026-08-01-stage2-3-roadmap.md) | Stage 2/3 direction |
| [`docs/superpowers/specs/table3-snapshot.md`](docs/superpowers/specs/table3-snapshot.md) | Pinned reproduction targets (paper Table 3 + README leaderboard snapshot) |
| [`docs/upstream-README.md`](docs/upstream-README.md) | Upstream's own README, preserved verbatim |

## Repo layout

```
ocr-eval/
  realdoc_bench/         upstream (minimally touched — see docs/architecture.md's fork boundaries)
  ocr_eval_ext/          this fork's additions:
    config.py              registry schema + loader (RegistryEntry, load_registry, get_entry)
    preconditions.py       fail-closed cardinality/render gates (check_bank, assert_single_page)
    metrics.py             boolean/null-restricted per-field outcomes, baselines
    stats.py               document-clustered bootstrap CIs, paired deltas, separability
    direct.py              vlm-chat runner (run_direct) — image + question -> typed JSON answer
    parsers_openai.py      openai-compat transcriber adapter, registered into upstream's parser registry
    bedrock.py             AWS Bedrock vlm-chat transport (SigV4 Converse; no OpenAI-compat endpoint exists)
    selftest.py            scorer + extractor fail-closed self-tests
    report_md.py           shape-segregated markdown report builder
    cli.py                 the `ocr-eval` CLI (verify/selftest/direct/parse/score/rescore/report)
  configs/                pins.yaml, registry.yaml, registry-local-validation.yaml, registry-bedrock.yaml
  docs/                   this documentation suite + runbook + superpowers/ provenance chain
  tests_ext/, tests/      this fork's tests + upstream's own test suite
```

## Licences

- **Code:** Apache-2.0 (upstream `realdoc_bench/`, and this fork's additions in `ocr_eval_ext/`
  under the same terms — see [`LICENSE`](LICENSE)).
- **Dataset:** RealDocBench is CC-BY-4.0 — any figures reproduced or derived from it must be
  attributed to `Extend-AI/RealDoc-Bench` (done automatically in every generated `report.md`).
- **CheckboxQA (Stage 3, not yet built):** CC BY-NC. Any report containing CheckboxQA numbers will
  be stamped `CC BY-NC — internal model selection only` in code, never merged into a commercial
  comparison.

## Keys policy

Environment-only — never a config file, never committed. Every registry entry naming a key
requirement declares the variable via `api_key_env`, and
the variable must be present in the real process environment before the command runs. The full
Stage 1 variable list is in [`docs/runbook-stage1.md`](docs/runbook-stage1.md)'s prerequisites.

Injection is **on demand, not a profile export**: `gemkey` loads a `0600`
`~/.config/ocr-eval/secrets.env` into the current shell only, so a fresh terminal starts clean and
no secret sits in a dotfile that gets backed up or synced. The harness reads `os.environ.get()`, so
any mechanism that populates the environment works identically — see
[`docs/api.md`](docs/api.md#on-demand-injection-not-a-profile-export-host-convention-adopted-2026-08-05).

Two things that bite in practice, both documented in [`docs/api.md`](docs/api.md#keys):

- **`.env` does not reach the `ocr-eval` CLI.** Upstream's `.env`/`.env.local` loading is disabled
  by default and re-enabled only by `RDB_ALLOW_DOTENV=1` — but that flag gates `_env()` in
  `realdoc_bench/cli.py` (the one upstream file this fork modifies), and `ocr_eval_ext/` never
  calls it, not even where `ocr-eval score` imports upstream's scorer in-process. `.env` therefore
  only ever reaches `realdoc-bench` commands, none of which need a key.
- **Export `GEMINI_API_KEY` under that exact name.** Upstream's scorer accepts
  `GEMINI_API_KEY` *or* `GOOGLE_API_KEY`, but the `vlm-chat` runner reads `api_key_env` by exact
  name with no fallback — so the Section A frontier anchor fails on the Google alias alone.
