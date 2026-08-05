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

## Quickstart — keyless local validation

No API keys required. Exercises render → cache → report end-to-end against a local
OpenAI-compatible endpoint (Ollama). Full details, caveats, and the thinking-model empty-response
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
| Gemini extractor validation gate + scoring leg | Implemented; needs `GEMINI_API_KEY`, unrun |
| Hosted transcriber (Mistral OCR 4) | Implemented; needs `MISTRAL_API_KEY`, unrun |
| Hosted transcriber (Nanonets DocStrange) | Implemented, single-page live smoke passed; full corpus unrun — needs `DOCSTRANGE_API_KEY`, $0.01/page (581 pages ≈ $5.81), ~55 s/page |
| Local vLLM specialists (GLM-OCR, dots.ocr) | Implemented ([local-serving.md](docs/local-serving.md)); unrun |
| Reproduction gate (DoD #2) | Implemented; needs a scored transcriber row to evaluate |
| Stage 2 (conditions, classical engines) / Stage 3 (CheckboxQA, HITL) | Not started — see [roadmap](docs/superpowers/plans/2026-08-01-stage2-3-roadmap.md) |

## Doc map

| Doc | Covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Module map, run-dir anatomy, condition dict, cache semantics, fork boundaries, fail-closed gate inventory |
| [`docs/cli.md`](docs/cli.md) | Every `ocr-eval` command, upstream commands used/avoided, a worked keyless example |
| [`docs/scoring.md`](docs/scoring.md) | The scoring rubric — upstream scorer, our metrics layer, baselines, uncertainty, reproduction gate |
| [`docs/api.md`](docs/api.md) | Provider contract, retry policy, cost control, per-provider serving notes and caveats |
| [`docs/runbook-stage1.md`](docs/runbook-stage1.md) | The operational, numbered Stage 1 run procedure and DoD checklist |
| [`docs/local-serving.md`](docs/local-serving.md) | vLLM launch lines and context-budget arithmetic for local specialists |
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
    selftest.py            scorer + extractor fail-closed self-tests
    report_md.py           shape-segregated markdown report builder
    cli.py                 the `ocr-eval` CLI (verify/selftest/direct/parse/score/rescore/report)
  configs/                pins.yaml, registry.yaml, registry-local-validation.yaml
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

Environment-only. Keys are injected via `bws run --project-id <id> -- <cmd>`
([`docs/runbook-stage1.md`](docs/runbook-stage1.md) prerequisites) — never a config file, never
committed. Upstream's `.env`/`.env.local` loading is disabled by default; the one upstream file
this fork modifies (`realdoc_bench/cli.py`'s `_env()`) makes it opt-in only via
`RDB_ALLOW_DOTENV=1`.
