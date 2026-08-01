# OCR Extraction Eval Pipeline — Design

**Date:** 2026-08-01
**Status:** Draft for review
**Provenance:** Direct successor to the small-VLM OCR survey (`~/notes/claude-things/gate*.md`, znote project `ocr-vlm-survey`). The survey established that no published checkbox-accuracy number exists for any model (16 benchmarks verified negative) and specced — but did not run — the eval that would produce one (`gate3-eval-spec.md`). A direction review then upgraded that flat eval into a factorial pipeline experiment. This project builds the harness for both.

---

## Purpose

Produce comparable, licence-annotated numbers for document/form extraction — checkbox state first — across any model reachable by API or runnable on the local GPU. One harness, three build stages.

**The question each stage answers:**

| Stage | Question answered |
|---|---|
| 1 | On real scanned forms, which reachable model reads checkboxes best, and who hallucinates into blank fields? |
| 2 | Does the degradation inversion survive deskew? Was cross-benchmark variance format conformance? Does sample-agreement predict correctness? |
| 3 | Does the ranking replicate on an independent instrument (CheckboxQA)? At what human-review rate does deployment become viable? |

## Non-goals

No web UI, no database server, no eval framework dependency (inspect-ai etc.), no leaderboard publishing, no fine-tuning support. Own-forms dataset support was considered and deferred (would slot into Stage 3 as a fourth loader if revisited).

---

## Core architecture (fixed from first commit)

### Two backend shapes — never a third

- **`vlm-chat`** — receives page image(s) + question, returns an answer. Implementation: OpenAI-compatible chat completions with image content parts. Covers hosted providers (OpenRouter, DeepInfra, Baseten, Fireworks, …) and local servers (vLLM, SGLang, Ollama, LM Studio) identically — a local model is a registry entry whose `base_url` points at localhost.
- **`transcriber`** — receives page image(s), returns markdown/text. Covers dedicated OCR endpoints (Mistral OCR, Marker/Surya on Replicate), OCR specialists served locally via vLLM/SGLang (OpenAI-compatible transport, but the output is a transcript, so the shape is transcriber), local classical engines (Tesseract, PaddleOCR pipeline — Stage 2), and pre-computed transcript files. A single **pinned extractor** model then answers each question from the transcript.

**Extractor pinning rule:** one extractor for all transcribers within a run set — exact model id, temperature 0, prompt template hash recorded on every row. Because it is constant, it cancels out of transcriber-vs-transcriber comparisons. The extractor model is chosen at implementation time (small, cheap, strong instruction-following); changing it invalidates cross-run transcriber comparisons and therefore bumps a `extractor_generation` tag in the cache key.

### Condition dict — present from day one

Every row and every cache key carries a condition dict. Stage 1 has a single value per axis; later stages add values, never schema changes:

```yaml
condition:
  preprocess: raw          # stage 2 adds: deskew
  output_contract: lenient # stage 2 adds: schema (native structured output or prompt-enforced; mechanism actually used is recorded per row)
  dpi: 200                 # fixed; recorded because DPI is a hidden condition
  sample_index: 0          # stage 2: 0..K-1 at temperature > 0
```

### Model registry

`configs/models.yaml` — one entry per (model, serving) pair:

```yaml
- id: qwen3-vl-8b@openrouter
  backend: vlm-chat
  base_url: https://openrouter.ai/api/v1
  model: qwen/qwen3-vl-8b-instruct
  precision: bf16 (provider default)
  licence: apache-2.0
  provenance: Alibaba
- id: glm-ocr@local-vllm
  backend: transcriber
  base_url: http://localhost:8000/v1
  model: glm-ocr
  precision: bf16
  serving_note: vLLM, dedicated architecture; launch per docs/local-serving.md
  licence: MIT
```

The same weights served two ways are two registry entries; reports never merge them silently.

### Local inference (Stage 1 capability)

Hardware: RTX 4070 Ti SUPER, 16 GB, CUDA.

- **Why it matters beyond cost:** several OCR specialists are hosted nowhere or ToS-blocked for commercial use (Novita bars commercial endeavors; Z.ai bars document-market verticals; dots.ocr unhosted). Local vLLM/SGLang — which ship dedicated architectures for GLM-OCR, PaddleOCR-VL, DeepSeek-OCR(-2), dots.ocr — is the only compliant access path to much of Tier B.
- **Precision policy (comparative-validity rule):**
  - ≤4B models (all the OCR specialists: 0.9–3B) run **BF16** locally — no quantization confound.
  - 8–9B models fit at **FP8/Q8**; allowed, but `precision` is a mandatory report column and any local-vs-hosted comparison of the same weights at different precision is flagged in the report.
  - If only Q4 fits, run it hosted instead — a Q4-local vs BF16-hosted table misattributes quantization loss to the model.
- **Scheduling:** one local model resident at a time. The runner groups cells by `base_url`; local groups run sequentially, hosted groups run concurrently. Stage 1: server launch is manual, documented per-model in `docs/local-serving.md`, with a preflight ping (model id check against `/v1/models`) before any cells fire. Automated model swapping is Stage 2 if manual proves annoying.
- vLLM is the primary local server (specialist architectures, full precision, matches production serving). Ollama/LM Studio are acceptable for generalists already in their libraries — same registry mechanism, precision recorded.

### Scoring — with positive controls

- Vendored from RealDocBench's Apache-2.0 harness **as data**: normalization rules and `capability_buckets.yaml` bucket definitions, pinned to a recorded upstream commit hash.
- Metric: normalize-then-exact; strict per-question accuracy and per-field accuracy reported separately. Blank-field bucket scores "correctly returned nothing" (null gold).
- **Startup self-test (mandatory, fail-closed):** before any run, the scorer executes golden fixtures — known-correct, known-wrong (including a checkbox polarity inversion), known-missing, and null-gold cases. If any fixture scores wrong, the run aborts before API spend. A scorer that cannot flag a known-wrong answer is worse than no scorer.
- Parse failures are not wrong answers: every row carries `error_class ∈ {none, parse_error, refusal, api_error, empty}` and these are reported as separate columns, never folded into accuracy.

### Records, cache, and resumability

- One JSONL row per (question × model × condition × sample): run id, dataset, question id, doc/page ids, bucket tags (hard-case tags preserved from Stage 1 even though unreported until Stage 3), registry id, condition dict, prompt hash, raw response, parsed answer, normalized answer, gold, score, error class, latency ms, tokens in/out, cost USD, extractor id (transcriber rows).
- **Content-addressed response cache** keyed on (dataset, question id, registry id, condition dict — which includes sample index, prompt hash, extractor generation). Crashes resume free; expanding a matrix pays only for new cells; old results stay valid forever because keys never change meaning.
- `--dry-run` prices the full cell matrix (cells × estimated tokens × registry pricing) before anything fires.

### Keys and secrets

API keys reach the harness via environment variables only (`bws run --project-id … -- ocr-eval run …`); no keys in config files, no keys in JSONL rows. **Prerequisite before first hosted run: rotate the OpenRouter key** (exposed in a prior session transcript).

---

## Stage 1 — comparative baseline (build now)

**Data:** RealDocBench only. Checkbox bucket (429 questions, 263 docs) + blank-field bucket (122 questions, 188 null golds). Loader fetches PDFs + `qa_bank.json` over plain HTTPS (no auth), verifies checksums, pins the upstream commit. Pages render via `pdftoppm` at the fixed DPI, cached to disk by content hash.

**Known data caveat carried into every report:** `origin` is `None` for all 1,356 items — real filled documents cannot be separated from synthetic-persona renders. Scores are over a mixed-provenance set and reports say so.

**Backends:** `vlm-chat` (hosted + local) and `transcriber` for hosted OCR endpoints + local vLLM-served specialists. Classical engines are Stage 2.

**Conditions:** single cell — `raw` / `lenient` / fixed DPI / one sample. Lenient parsing: request JSON in the prompt, parse tolerantly (fenced JSON, bare value, yes-no synonyms), normalize, then exact-match.

**Output:** one markdown report — rows are registry entries; columns: checkbox accuracy (strict per-question), per-field accuracy, blank-field hallucination rate, error-class rates, $/1k questions, median latency, precision, licence, provenance. CheckboxQA columns don't exist yet; licence column mechanism does.

**Definition of done:** report generated for ≥3 hosted VLMs + ≥1 local specialist + ≥1 hosted OCR endpoint; scorer self-test green; a second invocation of the same run completes with 100% cache hits and zero API calls.

## Stage 2 — conditions and local engines

- Condition axes gain values: `deskew` (OpenCV, correction recorded per page), `schema` output contract (native structured outputs where supported, prompt-enforced elsewhere, mechanism recorded), K-sample consistency (K configurable, default 5, temperature per-registry).
- Local classical transcribers: Tesseract, PaddleOCR pipeline — implement the existing `transcriber` interface.
- Run configs become cell matrices `{registry ids} × {conditions} × {buckets}`; cache dedups overlap with Stage 1.
- Report adds per-condition delta columns and the agreement-vs-accuracy table (per-question agreement rate binned against correctness).

## Stage 3 — second instrument and decision layer

- **CheckboxQA:** loader (annotations from repo, documents from DocumentCloud), ANLS* scorer. **Licence gate in code:** every report containing CheckboxQA-derived numbers is stamped `CC BY-NC — internal model selection only; not usable to validate a commercial product` — the flag is generated, not remembered.
- Hard-case slice report (~30 questions: filled_bubble 16, circled_choice 7, ambiguous_mark 6, crossed_out 1) as a qualitative appendix with sample sizes printed next to every number.
- Direct-vs-two-stage: same VLM registered under both backend shapes; tests whether transcribe-then-extract beats direct QA.
- HITL routing analysis: agreement-threshold sweep → human-review rate vs residual-error curve, from Stage 2 rows at zero new API cost.

---

## Error handling

- Per-cell isolation: an exception in one cell records an `api_error` row and continues; a run summarizes failures at the end and exits nonzero if any cell errored (fail-visible, not fail-open).
- HTTP: bounded retries with exponential backoff on 429/5xx honoring `Retry-After`; a registry-level circuit breaker (N consecutive failures pauses that registry's cells, run continues for others).
- All shell/subprocess steps check exit status directly, never downstream of a pipe (`PIPESTATUS` discipline).

## Testing

- Scorer golden tests = the positive-control fixtures (run in CI and at every run start).
- Loader tests against a committed snapshot of `qa_bank.json` structure (schema drift upstream fails loudly).
- Backend contract tests against a local mock OpenAI server (correct request shape, image encoding, error mapping).
- One end-to-end smoke test: 3 questions through a stub model → JSONL → report, asserting on the rendered table.

## Repo layout

```
ocr-eval/                      (~/repos/ocr-eval, uv-managed, Python 3.12)
  pyproject.toml
  configs/models.yaml          # registry
  configs/runs/*.yaml          # named cell matrices
  docs/local-serving.md        # per-model vLLM/Ollama launch lines
  docs/superpowers/specs/      # this document
  src/ocr_eval/
    datasets/   backends/   scoring/   conditions/
    runner.py   cache.py    records.py report.py   cli.py
  tests/
  vendored/realdocbench/       # normalization rules + bucket defs + upstream commit hash
```

## Risks

| Risk | Handling |
|---|---|
| RealDocBench repo changes/moves | Pin commit hash + checksums at first fetch; loader fails loudly on mismatch |
| Extractor choice biases transcriber scores | Pinned + recorded; sensitivity check (second extractor on a sample) is a Stage 3 option |
| Provider silently changes served precision/version | Registry snapshot recorded per run; hosted rows carry provider + date |
| 16 GB insufficient for a wanted local model | Precision policy escalates to hosted; never silent Q4 |
| Mixed-provenance data (origin=None) | Caveat auto-printed on every report |

## Effort estimate

Stage 1 ≈ 600–800 lines + tests; Stage 2 and Stage 3 each a comparable increment. Stage 1 hosted spend: single-digit dollars per model per full pass; local passes free after setup.
