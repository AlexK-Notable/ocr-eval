# Local serving (RTX 4070 Ti SUPER, 16 GB, CUDA)

> **Host requirement:** everything in this file assumes a CUDA GPU with ~16 GB VRAM and a `vllm`
> install in its own serving environment. Neither is present on the host probed 2026-08-03 — see
> [`host-setup.md`](host-setup.md#vllm--cuda-gpu--the-local-bf16-specialists) for what that blocks
> (runbook step 6, DoD #2's `dots.ocr` reproduction check, DoD #3's local-specialist row).

Pin and record: `uv run python -c "import vllm; print(vllm.__version__)"` per run — vLLM runs in
its own serving environment, not this repo's `uv` env, so its version is never captured by
`run_meta.json`; this file is the operative record for local rows.

One model resident at a time (16 GB VRAM). Start, preflight, run, stop:

1. `vllm serve ...` (below, per model) — leave running in its own terminal/session.
2. `uv run ocr-eval preflight <entry-id>` — confirms the server is actually serving the model id
   the registry expects before a single page is transcribed against it. Never skip this: a wrong
   checkpoint resident, or a server that wasn't restarted after a config change, produces
   transcripts that look plausible but are silently off-model.
3. `uv run ocr-eval parse --run-dir <run-dir> -p <parser-name>` (and `ocr-eval score` after) —
   `configs/registry.yaml` has THREE dynamically-registered openai-compat transcribers in total:
   two run locally on this box (`glm-ocr@local-vllm`, `dots-ocr@local-vllm` — this runbook covers
   these two) and one is hosted (`qwen3-vl-8b@openrouter-transcriber`, served over OpenRouter —
   no local `vllm serve` needed for that one). The registered parser name is
   `safe_name(entry.id) + "__" + condition_hash(TRANSCRIBER_CONDITION)` (see
   `ocr_eval_ext/parsers_openai.py`) — read it off the wrapper's own output rather than guessing.
4. Stop the server (`Ctrl-C` / kill the `vllm serve` process) before switching models — the next
   model's launch line assumes 16 GB is fully free.

### Concurrency knobs — two separate ones, don't confuse them

- **`ocr-eval parse --workers N`** controls how many *documents* are parsed concurrently
  (upstream `run_parse`'s own `ThreadPoolExecutor`). Against a `local: true` entry this defaults
  to **1** automatically (no flag needed) — the local vLLM server is single-resident, and
  `page_concurrency=1` on `OpenAICompatVisionParser` only serializes requests *within* one
  document's `parse()` call, not *across* documents, so `--workers` must also be 1 to actually
  keep only one request in flight against the server at a time. Pass `--workers` explicitly to
  override (e.g. running two independently-served local models on separate GPUs); the CLI prints
  which default it picked and why.
- **`RDB_PAGE_CONCURRENCY`** (env var, read by upstream `VisionParserBase.parse`) controls how
  many *pages of the same document* are requested concurrently — irrelevant for Stage 1's
  single-page bank, relevant once Stage 2 processes multi-page documents. Leave unset (defaults
  to `page_concurrency=1` on this parser) for local serving regardless of `--workers`.

## GLM-OCR (0.9B, BF16)

Registry entry: `glm-ocr@local-vllm` (`configs/registry.yaml`), model id `zai-org/GLM-OCR`.

```
vllm serve zai-org/GLM-OCR --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --max-model-len 8192
```

**Context budget:** `--max-model-len 8192` must cover prompt + image + completion tokens
combined. A rendered page at 150dpi costs roughly 1.3-1.9k image tokens; the markdown-extraction
prompt text is a few hundred more. `OpenAICompatVisionParser.max_tokens` is pinned to **4096**
(via `TRANSCRIBER_CONDITION["sampling"]["max_tokens"]`, not upstream's inherited 12000-token
default) specifically so `~2k` prompt/image tokens `+ 4096` completion tokens fits inside 8192
with headroom — the original 12000-token default blew straight through the window, and the
server's resulting 400 was (correctly) treated as a *permanent* failure by `_is_retryable`, so
every local page failed on its first attempt with no retry. Raise `--max-model-len` (and
`max_tokens` in `TRANSCRIBER_CONDITION`, which changes the registered parser's condition hash —
a deliberate, tracked change, not a silent one) if a model routinely needs longer completions.

Verify the exact HF id + any `--trust-remote-code` / processor flags against the model card at
first use, and update this block with whatever flags actually worked.

## dots.ocr (3B, BF16)

Registry entry: `dots-ocr@local-vllm` (`configs/registry.yaml`), model id
`dots-studio/dots.ocr` (registry.yaml notes: the org that originally published this model as
`rednote-hilab/dots.ocr` renamed/moved on HF — that repo now 307-redirects to `dots-studio/dots.ocr`;
confirm at bring-up whether upstream branding should read `dots-studio` instead of RedNote).

```
vllm serve dots-studio/dots.ocr --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --max-model-len 8192 --trust-remote-code
```

## Preflight (always, before any cells)

```
uv run ocr-eval preflight glm-ocr@local-vllm
uv run ocr-eval preflight dots-ocr@local-vllm
```

Each call does a plain `GET {base_url}/models` and checks the registry's `model` id is in the
served list — fails loudly (non-zero exit, red `preflight FAILED`) on any mismatch instead of
silently transcribing against whatever happens to be resident.

---

The exact flags per model are verified against each model card / vLLM docs at first bring-up, and
the working command line is committed back into this file — this file is the record of what
actually ran, not a plan of what should run.
