# Host setup and environment probes

What this repository needs from the machine it runs on, each claim paired with the command that
verifies it. **Who should read this:** anyone bringing the pipeline up on a new host, or diagnosing
a "works on the other box" failure. For the operational run sequence see
[`runbook-stage1.md`](runbook-stage1.md); for key semantics see [`api.md`](api.md#keys).

Results below were captured on the current host **2026-08-03** (Linux 6.8, Python 3.13.11,
uv 0.9.23). Re-run the probes rather than trusting the recorded values — that is the point of
writing them as commands.

## Toolchain

```bash
uv --version                      # 0.9.23
uv sync --extra dev               # installs runtime + pytest/ruff/basedpyright
uv run pytest -q                  # expect: 282 passed, 1 skipped
uv run ocr-eval selftest          # expect: offline scorer self-test: PASS
```

The one skipped test (`tests_ext/test_cli.py:70`) skips on `real 581-PDF corpus not present in this
checkout` — expected until step 2 of the runbook has downloaded the corpus.

`requires-python = ">=3.11"`; 3.13 is fine. Resolved versions worth recording for a run:
pymupdf 1.28.0 (MuPDF 1.29.0), Pillow 12.3.0, openai 2.52.0, huggingface_hub 1.26.0, numpy 2.5.1,
scipy 1.18.0. `verify` stamps `pymupdf_version` into `run_meta.json` and `report.md` prints it —
renders are only bit-reproducible against the same pymupdf build.

### Lint expectations

```bash
uv run ruff check .
```

**Non-zero exit is expected and is not an environment fault.** `CONTRIBUTING.md` records that the
Typer `B008` warnings are intended: every `typer.Option(...)` in a signature default trips it. On
this host: 147 findings, 133 of them in upstream `realdoc_bench/`/`tests/`, and all 14 in
`ocr_eval_ext/` are `B008`. The count is also ruff-version-dependent (117 under ruff 0.5.7, the
`pyproject.toml` floor, vs 147 under 0.16.1) — treat a *delta against your own baseline* as signal,
never the absolute number.

## Git pin ancestry

Every run-dir-scoped command (`verify`/`direct`/`rescore`/`parse`/`score`/`report`) fails closed
unless `configs/pins.yaml`'s `harness_commit` is an ancestor of `HEAD`:

```bash
git merge-base --is-ancestor \
  $(python3 -c "import yaml;print(yaml.safe_load(open('configs/pins.yaml'))['harness_commit'])") \
  HEAD && echo OK
```

A shallow clone can fail this by lacking the pinned commit — verify the object exists
(`git cat-file -t <sha>`) before suspecting the check itself.

## Dataset

Public, ungated, CC-BY-4.0 — no `HF_TOKEN` required, though setting one raises rate limits and the
hub will warn without it.

```bash
curl -s https://huggingface.co/api/datasets/Extend-AI/RealDoc-Bench | head -c 200
```

Confirm the returned `sha` equals `pins.yaml`'s `dataset_revision`
(`906170ab201d7b8238a32a9115fc66b4b72e0710`). **Size: 581 PDFs, 538 MB** under `docs/`, plus
`qa_bank.json` — budget disk for that per run dir, since `download` materializes real files into the
run dir (not symlinks into the shared HF cache).

The bank's cardinality contract (`ocr_eval_ext/preconditions.py`'s `EXPECTED`) was verified against
the real bank at this revision on this host — all 12 counts match (1,356 items · 3,742 fields ·
258 checkbox booleans · 165 checked / 93 unchecked). `verify` re-checks this on every run.

### huggingface_hub deprecation (harmless, worth knowing)

`realdoc_bench/evaluate/download.py` passes `local_dir_use_symlinks=False` to both
`hf_hub_download` and `snapshot_download`. That parameter was **removed** from both signatures in
huggingface_hub 1.x; it now raises a `UserWarning` and is ignored. Downloads still work, and — the
part that actually matters — the `.cache/huggingface/download/qa_bank.json.metadata` sidecar that
`cli.py`'s `_observed_dataset_revision` reads is still written, with the resolved commit hash on
its first line (verified live). The `trees/*.json` fallback path is *not* produced by this
huggingface_hub version, so the sidecar is the only working source; treat its absence as "unknown
revision", never as "matches the pin".

## Keys

See [`api.md`](api.md#keys) for the full semantics. **This host loads keys on demand** — run
`gemkey` first (`~/.zshrc` function → `~/bin/ocr-eval-keys.sh` → `0600`
`~/.config/ocr-eval/secrets.env`), which populates the current shell only. A reference copy of the
loader lives at [`scripts/ocr-eval-keys.sh`](../scripts/ocr-eval-keys.sh) so it can be reinstalled
if `~/bin` is lost; the secrets file itself is never in the repo. Rationale for on-demand over a
profile export is in
[`api.md`](api.md#on-demand-injection-not-a-profile-export-host-convention-adopted-2026-08-05).

Probe what the environment actually has:

```bash
for v in GEMINI_API_KEY GOOGLE_API_KEY OPENROUTER_API_KEY MISTRAL_API_KEY; do
  [ -n "${!v:-}" ] && echo "$v set" || echo "$v MISSING"
done                                            # run under bash, not zsh (${!v} indirection)
```

Confirm the Google key reaches Gemini and that the pinned model ids exist — the extractor pins
`realdoc_bench/evaluate/score.py`'s `DEFAULT_MODEL` and the anchors use `gemini-3.5-flash`.
**Send the key as a header, never a URL query parameter** — a URL reaches shell history, proxy logs,
and any error message that echoes the request URL, which is precisely the leak removed from
`score.py`:

```bash
curl -s -H "x-goog-api-key: $GEMINI_API_KEY" \
  "https://generativelanguage.googleapis.com/v1beta/models" \
  | grep -o 'gemini-3[^"]*' | sort -u
```

Then run the real gate (5 Gemini calls, cents):

```bash
uv run ocr-eval selftest --extractor            # expect: extractor validation: PASS (5/5)
```

**Reminder:** export `GEMINI_API_KEY` under that exact name. `GOOGLE_API_KEY` alone satisfies the
extractor gate and the `score` leg but **not** `gemini-3.5-flash@google-vlmchat`, because
`direct.py` resolves `api_key_env` exactly with no alias fallback.

OpenRouter model availability and pinned providers can be checked without a key:

```bash
curl -s https://openrouter.ai/api/v1/models/qwen/qwen3-vl-8b-instruct/endpoints \
  | python3 -c "import json,sys;print(sorted({e['provider_name'] for e in json.load(sys.stdin)['data']['endpoints']}))"
```

Verified 2026-08-03 — every `configs/registry.yaml` provider pin is still served:
`qwen3-vl-8b-instruct` → Alibaba (also Parasail), `qwen3-vl-32b-instruct` → Alibaba (sole),
`qwen3.5-9b` → DeepInfra (also Parasail, SiliconFlow, Together, Venice). The registry's ToS-driven
DeepInfra pin for `qwen3.5-9b` remains valid.

## AWS Bedrock

An alternative to the hosted-key path: authenticates via the AWS credential chain, so no
`OPENROUTER_API_KEY`/`MISTRAL_API_KEY` is needed for `vlm-chat` candidate rows. Full semantics in
[`api.md`](api.md#aws-bedrock-transport-bedrock-converse).

```bash
aws sts get-caller-identity                     # confirm credentials resolve
aws bedrock list-foundation-models --region us-east-1 --query 'length(modelSummaries)'
```

**Listing is not availability** — access is per-role. On this host (2026-08-04, account
381492199098, `AWSReservedSSO_Developer`, us-east-1) 119 vision-capable models were listed and 7
were invokable. Probe what the calling role can actually invoke:

```bash
uv run python -c "
from ocr_eval_ext.bedrock import probe_invokable
for m, s in probe_invokable('us-east-1', [
    'amazon.nova-lite-v1:0', 'amazon.nova-pro-v1:0', 'google.gemma-3-27b-it',
    'us.anthropic.claude-haiku-4-5-20251001-v1:0']).items():
    print(f'{m:48} {s}')"
```

Then preflight each registry entry before spending a bank against it:

```bash
uv run ocr-eval preflight nova-lite@bedrock --registry configs/registry-bedrock.yaml
```

Verified invokable on this host, each also confirmed to read checkbox polarity correctly on a
synthetic form (checked → true, empty → false): `amazon.nova-lite-v1:0`, `amazon.nova-pro-v1:0`,
`google.gemma-3-12b-it`, `google.gemma-3-27b-it`, `mistral.mistral-large-3-675b-instruct`,
`moonshotai.kimi-k2.5`, `us.anthropic.claude-haiku-4-5-20251001-v1:0`. Denied to this role at that
time — re-probe before adding: all bare `anthropic.*` ids, `us.amazon.nova-2-lite-v1:0`,
`us.amazon.nova-premier-v1:0`, `us.meta.llama4-*`, `us.mistral.pixtral-large-*`.

SSO credentials expire; a stale session surfaces as `ExpiredTokenException`, classified permanent
(so it fails fast rather than retrying 4× per cell). Refresh with `aws sso login`.

Two things Bedrock does **not** solve: the scoring leg still needs `GEMINI_API_KEY` (the extractor
is Gemini-pinned), and there is no Bedrock transcriber yet, so it covers `vlm-chat` rows only.

## Local serving

Two separate local dependencies, neither bundled, neither present on the current host.

### Ollama — the keyless validation path

Required by the README quickstart and all five entries in
[`configs/registry-local-validation.yaml`](../configs/registry-local-validation.yaml), which point
at `http://localhost:11434/v1`.

```bash
curl -s http://localhost:11434/api/tags >/dev/null && echo up || echo "not reachable"
```

Bring-up needs three things, not just the install: `ollama serve` running, `qwen3-vl:8b-instruct`
pulled, **and** the derived `qwen3-vl:8b-instruct-ctx8k` model created locally. That derived model
is a host-local artifact that does not travel with this repo, and it is the entry the docs
recommend for full-bank validation (Ollama's default `num_ctx` of 4096 otherwise produces 5
`api_error` and 44 `parse_error` cells — see [`api.md`](api.md#known-provider-behavior-caveats-from-validation)):

```bash
printf 'FROM qwen3-vl:8b-instruct\nPARAMETER num_ctx 8192\n' > Modelfile.ctx8k
ollama create qwen3-vl:8b-instruct-ctx8k -f Modelfile.ctx8k
```

CPU-only inference works (this host: 8 cores, 30 GB RAM) but is slow — the path is for plumbing
validation, and these entries are barred from any comparative table by policy regardless.

### vLLM + CUDA GPU — the local BF16 specialists

[`local-serving.md`](local-serving.md) is written against a 16 GB RTX 4070 Ti SUPER.

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv    # absent on this host
python -c "import vllm; print(vllm.__version__)"         # in the SERVING env, not this repo's
```

**Absent on the current host**, which blocks runbook step 6 entirely and with it two
definition-of-done items:

- **DoD #2** — `dots-ocr@local-vllm` is the open-weight reproduction check (paper target
  70.6±3.6 / 61.4±3.5).
- **DoD #3** — requires "≥1 local specialist (BF16)"; both candidates are local-only entries.

Recorded here rather than worked around, per the fail-closed convention: an unrunnable leg must not
read as a pass. The resolution is an open decision — rent a GPU, add hosted entries for these
weights with precision and serving identity stamped honestly (they would no longer be the paper's
serving stack, and the report must carry that caveat), or formally descope both DoD items.

Docker is installed on this host but the invoking user cannot reach `/var/run/docker.sock`
(`permission denied`) — moot for this leg without a GPU, but worth knowing before reaching for a
containerized vLLM.
