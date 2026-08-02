# Local serving (RTX 4070 Ti SUPER, 16 GB, CUDA)

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
   these are the two dynamically-registered openai-compat transcribers in `configs/registry.yaml`
   (`glm-ocr@local-vllm`, `dots-ocr@local-vllm`); the registered parser name is
   `safe_name(entry.id) + "__" + condition_hash(TRANSCRIBER_CONDITION)` (see
   `ocr_eval_ext/parsers_openai.py`) — read it off the wrapper's own output rather than guessing.
4. Stop the server (`Ctrl-C` / kill the `vllm serve` process) before switching models — the next
   model's launch line assumes 16 GB is fully free.

## GLM-OCR (0.9B, BF16)

Registry entry: `glm-ocr@local-vllm` (`configs/registry.yaml`), model id `zai-org/GLM-OCR`.

```
vllm serve zai-org/GLM-OCR --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --max-model-len 8192
```

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
