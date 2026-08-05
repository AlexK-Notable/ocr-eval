# API and provider considerations

The wire contract this fork speaks to model providers, the retry policy, cost controls, and
per-provider serving notes and caveats gathered during validation. **Who should read this:** anyone
adding a registry entry, debugging a provider-side error, or reasoning about spend. For the CLI
flags that trigger these calls, see [`cli.md`](cli.md); for local vLLM launch commands, see
[`local-serving.md`](local-serving.md).

## Provider contract

**OpenAI-compatible chat completions is the primary transport** — both the
vlm-chat runner (`direct.py`) and the transcriber adapter (`parsers_openai.py`) go through the
`openai` Python client's `chat.completions.create`, so any provider (OpenRouter, a local vLLM
server, Ollama, Google's OpenAI-compat endpoint) is the same code path with a different `base_url`.
Upstream's own non-OpenAI-compat adapters (Gemini via its native API, Mistral's PDF-upload
endpoint) are used as-is, unmodified, via `transport: upstream-parser` registry entries.
**AWS Bedrock is a third transport** (`transport: bedrock-converse`, `ocr_eval_ext/bedrock.py`) —
see [its own section](#aws-bedrock-transport-bedrock-converse) for why it cannot be an
openai-compat entry.

**What we send:**
- **Messages:** a `system` message (vlm-chat only — the transcriber sends a single `user` message)
  plus a `user` message whose `content` is a list: a text part, then (unless `no_image`) an
  `image_url` part with the page PNG as a base64 `data:image/png;base64,...` URL.
- **Sampling:** vlm-chat sends `temperature`, `top_p`, `max_tokens`, and `seed` *only when set*
  (`STAGE1_CONDITION`'s default `seed: None` stays unsent on the wire — no `seed` key at all rather
  than a literal `null`). The transcriber adapter sends only `temperature` and `max_tokens` — no
  `top_p`, no `seed` — a real asymmetry between the two shapes, not an oversight to reconcile.
- **`extra_body`:** when a registry entry sets `provider_pin`, it's forwarded verbatim as
  `extra_body["provider"]` — this is how OpenRouter provider pinning
  (`{order: [...], allow_fallbacks: false}`) reaches the wire.

**What we read:** `resp.choices[0].message.content`; `resp.model_dump()["usage"]` (OpenRouter's
response always includes a `cost` field inside `usage` — read opportunistically, never assumed
present for other providers); and `raw.get("provider")`, the resolved serving identity OpenRouter
attaches to the response body, recorded per row as `resolved_provider`.

## Retry policy

Three legs, three implementations — the candidate legs (`direct.py`, shared by
`parsers_openai.py`), the Bedrock leg (`bedrock.py`, delegated to from `direct.py`), and the
**scoring leg** (`score.py`'s `_post_with_retry`, added 2026-08-04 — upstream had no retry at all,
so one transient 503 aborted the whole extractor gate). All three use 4 attempts, honour
`Retry-After`, clamp the wait to `[0, 60]`s, and treat 408/429/5xx plus connection-level failures as
retryable and everything else as permanent. They are deliberately separate implementations rather
than one shared helper: upstream modules must not import the fork's package (see
[`architecture.md`](architecture.md#fork-boundaries)), and httpx, `openai` and botocore surface
status codes and headers in three incompatible ways.

The candidate-leg policy, in detail — owned entirely by this module, not the SDK: every
`OpenAI(...)` client is constructed with
**`max_retries=0`** (both `direct.py` and `parsers_openai.py`) — leaving the SDK's own default
retry-on-408/429/5xx enabled would nest under this module's retries, observed as up to
`MAX_RETRIES` outer attempts × the SDK's own retries each (up to 12 real HTTP attempts for what
should be at most 4).

- **`MAX_RETRIES = 4`** total attempts per cell.
- **Retryable:** `408`, `429`, any `5xx`, and connection-level failures (DNS, refused connection,
  timeout — `APIConnectionError` and its `APITimeoutError` subclass). Everything else (400, 401,
  403, 404, 409, 422, ...) fails permanently after one attempt — `_is_retryable` in `direct.py`,
  reused by `parsers_openai.py`.
- **Backoff:** `Retry-After` is honored when present — parsed as either a numeric seconds value or
  an HTTP-date (RFC 7231) — and falls back to exponential backoff (`2.0 * 2**attempt`) on any parse
  failure. Either way the wait is **clamped to `[0, 60]` seconds** (`MAX_RETRY_WAIT_SEC`), so a
  misbehaving or hostile header can never cost more than a minute of real wall-clock time per
  attempt (`_retry_wait`).

## Cost control

- **`--dry-run`** (vlm-chat only — D6): estimates using a fixed per-cell token guess (~2,000 image
  tokens/page, ~400 prompt, ~120 completion) against registry `pricing` rates, explicitly labelled
  "±2x — token counts are guesses." Cells whose entry carries no `pricing` are counted as
  `unpriced_cells` and **excluded** from `estimated_usd` — never silently folded in as $0.
- **`--max-spend` fail-closed rule:** if a cell's realized cost can't be established (no
  `usage.cost` from the provider, and the registry entry has no `pricing` to fall back on),
  `track()` in `direct.py` **raises** rather than treating the cell as free. This only fires when a
  spend cap is actually active — an uncapped run with unpriced cells simply doesn't track spend for
  them.
- **Overshoot bound:** bounded by `workers` in-flight calls, not by the whole cell queue.
  `_dispatch_bounded` keeps at most `workers` cells in flight at a time; once `track()` raises
  (cap exceeded), no further cells are submitted, but the ≤`workers` already in flight are allowed
  to finish (their rows are still written) before the error propagates.
- **Registry pricing fields:** `pricing: {input_per_mtok, output_per_mtok}` — USD per million
  tokens, used both for `--dry-run` estimates and as a fallback when a provider's `usage` lacks a
  `cost` field.

## Serving notes

- **OpenRouter:** pin the provider via `provider_pin: {order: [...], allow_fallbacks: false}` in
  the registry entry. Every answered row records the actually-resolved provider
  (`resolved_provider`); `report_md.py`'s D4 gate hard-fails if one parser key's rows resolve to
  more than one distinct provider (no escape hatch — a cache key that doesn't pin serving identity
  must not silently compare across serving stacks).
- **Ollama:** validation-only, never a comparison candidate.
  [`configs/registry-local-validation.yaml`](../configs/registry-local-validation.yaml)'s own header
  documents why: `qwen3-vl:8b` served via Ollama is Q4_K_M-quantized, below the project's precision
  policy floor for a headline row ("if only Q4 fits, run hosted instead"). The registry declares
  `precision: provider-default`, which renders as "unknown (not asserted)" in `report.md` rather
  than a false claim of a specific quant level. **Check the variant before anything else:**
  Ollama's default `qwen3-vl:8b` tag resolves to the *Thinking* build (its Modelfile carries
  `RENDERER`/`PARSER qwen3-vl-thinking` and the Thinking edition's sampling defaults), while the
  hosted comparison entries pin `qwen/qwen3-vl-8b-instruct` — use `qwen3-vl:8b-instruct` for
  validation (verified 2026-08-02: 60/60 answered at stock `STAGE1_CONDITION`, median 22
  completion tokens, vs 41/60 for the thinking build at the same condition). Vendor guidance
  (both models' HF cards) also discourages greedy decoding outright — Instruct VL tasks:
  temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, max output 16,384; Thinking VL
  tasks: temperature 1.0, top_p 0.95, top_k 20, max output 40,960. `STAGE1_CONDITION` pins
  greedy (temperature 0.0, top_p 1.0) deliberately for reproducibility — a documented divergence
  from vendor guidance that is empirically benign for the Instruct build at Stage-1 budgets but
  pathological for the Thinking build (greedy is the documented trigger for endless-repetition
  loops in Qwen thinking models, matching the runaway signature observed live).
  See [`cli.md`](cli.md)'s worked example for the
  thinking-model `max_tokens` exhaustion caveat this serving path surfaces live; Ollama's
  OpenAI-compat shim has no code-level way to suppress thinking in this codebase — there is no
  `think` field anywhere in `STAGE1_CONDITION`'s sampling dict. A second shim limit, verified
  live 2026-08-02: the request-level `max_tokens` is **silently clamped to `num_ctx −
  prompt_tokens`** — the server's context window wins, and Ollama's default `num_ctx` is 4096.
  The endpoint cannot set `num_ctx` per-request; the working fix is a derived model with
  `PARAMETER num_ctx <N>` baked in via a Modelfile (see the caveats section below and the
  runbook).
- **vLLM:** see [`local-serving.md`](local-serving.md) for exact launch lines and versions pinned.
  Context-budget arithmetic matters: `OpenAICompatVisionParser.max_tokens` is pinned to **4096** via
  `TRANSCRIBER_CONDITION` (not upstream's inherited 12,000-token default) specifically so
  ~2k prompt/image tokens + 4096 completion tokens fits inside a `--max-model-len 8192` server with
  headroom — the 12,000-token default previously blew through the window, and the server's
  resulting 400 was (correctly) treated as a *permanent* failure by `_is_retryable`, so every local
  page failed on its first attempt with no retry.
- **Google's OpenAI-compat endpoint** (`https://generativelanguage.googleapis.com/v1beta/openai`) is
  how the Gemini frontier anchor (`gemini-3.5-flash@google-vlmchat`) is reached as a `vlm-chat`
  entry — the same `openai`-client code path as everything else, alongside the separate
  `transport: upstream-parser` Gemini transcriber entry (`gemini-3.5-flash@google`) that goes
  through upstream's native Gemini adapter instead.

## AWS Bedrock transport (`bedrock-converse`)

`ocr_eval_ext/bedrock.py`. A `vlm-chat` transport alongside `openai-compat`; the transcriber
(`parse`/`score`) leg is **not** wired to Bedrock — see the limitations at the end of this section.

### Why it isn't just another `base_url`

Bedrock exposes no OpenAI-compatible endpoint reachable from the probed account (2026-08-04):
`/openai/v1/models` returns `404 UnknownOperationException`; `bedrock-runtime` advertises only
`Converse`/`ConverseStream`/`InvokeModel`/… with no chat-completions operation;
`bedrock-runtime get-api-key` is not permitted for the calling role; and an unauthenticated POST
returns `403` — every request must be SigV4-signed. The `openai` client therefore cannot reach
Bedrock at all.

`BedrockConverseClient` is deliberately **not** an `openai.OpenAI` look-alike. Faking
`chat.completions.create` and a `model_dump()` shape would make two genuinely different wire
protocols look interchangeable at the call site; `_one` branches on `entry.transport` explicitly
instead. Everything *after* the call is shared, so a Bedrock cache row has the same shape as an
openai-compat row — `condition_hash`, `parser_key`, `score_typed`, `track()` and every
`report_md.py` gate are untouched.

### Registry entry

```yaml
- id: nova-lite@bedrock
  shape: vlm-chat
  transport: bedrock-converse
  model: amazon.nova-lite-v1:0     # the Bedrock modelId (or inference-profile id)
  region: us-east-1                # REQUIRED — never inherited from AWS_REGION
  precision: provider-default      # Bedrock does not document serving precision
```

`RegistryEntry`'s validator **rejects** `api_key_env`, `base_url`, and `provider_pin` on a
bedrock entry (auth is SigV4, the endpoint is derived from region, and `provider_pin` is an
OpenRouter routing concept), and requires `region` explicitly rather than defaulting it: region is
part of the serving identity recorded on every row (`resolved_provider: "bedrock:<region>"`), and
an ambient env var would let one registry id silently produce rows from two serving stacks.
`report_md.py`'s D4 gate hard-fails a parser key whose rows span more than one provider, so a run
accidentally split across regions is caught rather than averaged.

Committed entries live in [`configs/registry-bedrock.yaml`](../configs/registry-bedrock.yaml).

### Model access is per-role, and listing is not availability

On the probed account, **119 vision-capable models are listed and 7 are invokable** by the calling
role. `AccessDeniedException` on invoke is the normal state for a listed-but-not-enabled model, so
a control-plane `ListFoundationModels` check answers the wrong question. Two identity traps:

- Models with `inferenceTypesSupported: ON_DEMAND` take the bare id (`amazon.nova-lite-v1:0`).
- Models marked `INFERENCE_PROFILE` are **not** invokable by their bare id and need the geo-prefixed
  cross-region profile id (`us.anthropic.claude-haiku-4-5-20251001-v1:0`). The bare Anthropic id
  fails with `ValidationException`.

`ocr-eval preflight <id>` sends one real minimal Converse call (a few tokens) and reports the
concrete error code — run it before spending a bank against any Bedrock entry.

### Anthropic-on-Bedrock rejects `temperature` + `top_p` together

Verified live across all six invokable models: **only** the Anthropic one raises
`ValidationException: `temperature` and `top_p` cannot both be specified for this model`. Nova
Lite/Pro, Gemma 3 27B, Mistral Large 3 and Kimi K2.5 all accept both.

`STAGE1_CONDITION` pins `top_p: 1.0`, which is the **identity** value (consider the full
distribution), so for that value omitting it is provably a no-op.
`narrow_sampling_for_model` drops `top_p` only when it is exactly `1.0` and **raises** for any
other value rather than silently altering what the condition claims to pin. When the narrowing
happens the row records `usage.sampling_note`, so a row stays honest about what actually went on
the wire instead of implying the condition was honored verbatim.

### No `seed`

Converse's `inferenceConfig` exposes `temperature`/`topP`/`maxTokens`/`stopSequences` only.
`run_direct` **refuses** a Bedrock entry under a condition whose `sampling.seed` is set, rather than
dropping it — a silently-dropped seed would let the condition hash assert a reproducibility
guarantee the wire never provided. Stage 1's default (`seed: None`) is unaffected.

### Retries and cost

botocore raises `ClientError` with a string error code and no `.status_code`, so `_is_retryable`
delegates botocore exceptions to `is_retryable_bedrock`. `ThrottlingException`,
`ServiceUnavailableException`, `ModelTimeoutException`, `ModelNotReadyException` and friends are
retryable; `AccessDeniedException`, `ValidationException` and friends are permanent (one attempt).
**An unrecognised code is treated as permanent** — the deliberate direction, since an unenumerated
transient code costs one honest error row, whereas an unknown permanent code treated as retryable
would quadruple a misconfigured full-bank run's wall-clock and spend. `retry_after_bedrock` reads
`ResponseMetadata.HTTPHeaders` (botocore's `.response` is a dict, so `_retry_wait`'s `.headers`
probe finds nothing), still clamped to `[0, 60]`. The botocore client is built with
`retries={"max_attempts": 1}` — the same no-nested-retries discipline as `OpenAI(max_retries=0)`.

**Bedrock returns token counts but no cost field**, and `pricing:GetProducts` is denied to this
role, so per-token rates cannot be verified from inside this environment. The committed entries are
therefore **unpriced**, with two honest consequences: `--dry-run` counts their cells as
`unpriced_cells` and excludes them from `estimated_usd`, and `--max-spend` **fails closed** on them
(`track()` raises rather than treating an unknowable cost as free). Add a verified
`pricing: {input_per_mtok, output_per_mtok}` with its retrieval date to use a spend cap.

### The 5 MB image cap counts BASE64, not raw bytes (measured 2026-08-04)

Converse takes the PNG as **raw bytes** (`{"image": {"source": {"bytes": png}}}` — botocore does the
base64 encoding), but the service's documented 5 MB per-image limit is enforced against the
**encoded** payload. So the effective ceiling on what this harness may hand it is
`5 MiB × 3/4 ≈ 3.75 MiB` of raw PNG, not 5 MB. Exceeding it is a `ValidationException` — classified
permanent, so it costs one attempt per cell, not four.

Measured on the full 1,356-cell bank at `render.dpi: 150`:

| | raw bytes | implied base64 |
|---|---|---|
| largest **succeeding** image | 3,776,968 | 5,035,957 |
| smallest **failing** image | 3,933,782 | 5,245,042 |
| the limit | — | 5,242,880 (5 MiB) |

Rows above `5 MiB × 3/4` numbered exactly 20 — precisely the run's 20 `api_error` rows, across 10
dense documents. The error text quotes the *encoded* size ("6164064 bytes > 5242880 bytes") against a
raw image of 4.6 MB, which is what makes this confusing to diagnose from one row.

Consequences to plan around, none of them currently automated:

- There is **no preflight size check** — an oversize cell fails at the wire, honestly (an
  `api_error` row with the service's message), but only after it is dispatched.
- **Downscaling is not a free fix.** Lowering `render.dpi`, or capping bytes, is a change to the
  condition dict, so it produces a different `condition_hash` and therefore a *different row* — by
  design (see [`architecture.md`](architecture.md#the-condition-dict)). It cannot be applied to
  patch up part of an existing run's matrix without splitting that matrix across two conditions.
- The affected documents are correlated with page density, so the loss is **not** random with
  respect to difficulty — treat a run with oversize failures as missing its hardest pages, and read
  the report's `error classes:` tally rather than assuming the denominator is uniform.

### Current limitations

- **vlm-chat only.** `parsers_openai.py`'s transcriber adapter is built on the `openai` client, so
  Bedrock cannot yet serve a `transcriber`-shape row. A Bedrock transcriber would need a
  `VisionParserBase` subclass calling Converse per page.
- **Comparability.** `precision: provider-default` renders as "unknown (not asserted)" —
  Bedrock does not document these checkpoints' serving precision. Treat a Bedrock row as a
  serving-stack measurement, not a clean weights comparison against a locally-served BF16 row.
- **The scoring leg still needs Gemini.** Bedrock covers the *candidate* side only; every
  transcriber row's extraction still goes through `GEMINI_API_KEY` (next section).

## The Gemini extractor dependency

`gemini-3-flash-preview` (`realdoc_bench/evaluate/score.py`'s `DEFAULT_MODEL`) is required in two
places: the `score` leg (every transcriber row's extraction call) and `ocr-eval selftest
--extractor` (5 calls against `EXTRACTOR_FIXTURES`). `score` runs the **blocking**
`require_extractor_gate` (`cli.py`) before any real extraction spend — it re-runs those same 5
fixtures unless a stamp file (`.extractor_ok_<model>_<fixture-hash>` in the run dir) already
confirms a pass for this exact (run dir, model, fixture-set) combination; a failure raises
`PreconditionError` and blocks the run entirely, with no way to skip it. Switching Gemini
extractor generations requires `--new-extractor-generation`, which archives extractor-dependent
cache rows to `eval/cache@<old-id>/` first (see [`architecture.md`](architecture.md#cache-semantics)).
**Key rotation prerequisite:** the runbook's prerequisites section requires `GEMINI_API_KEY` (and
`OPENROUTER_API_KEY`) to be rotated before the first hosted call in any session, since both were
exposed in a prior session transcript.

## Keys

Environment-only. Every registry entry with a key requirement names the variable via
`api_key_env` (e.g. `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`); the value itself
never appears in any config file. Keys are exported in the operator's shell profile — the variable
must be present in the real process environment before the command runs. No secrets-manager
wrapper is assumed or required: `bws`/`bitwarden` appear in **zero lines of code** in this
repository (verified by grep over `ocr_eval_ext/`, `realdoc_bench/`, `configs/`, `tests/`,
`tests_ext/`); every key is read through a plain `os.environ.get()`, so any injection mechanism
that populates the environment works identically.

### `.env` does not reach the `ocr-eval` CLI

Upstream's `.env`/`.env.local` loading is disabled by default and only re-enabled by setting
`RDB_ALLOW_DOTENV=1` (the one upstream modification — see
[`architecture.md`](architecture.md#fork-boundaries)). **That flag only gates `_env()` in
`realdoc_bench/cli.py`, which `ocr_eval_ext/` never calls** — there is no `load_dotenv` reference
anywhere in this fork's own modules, including the code path where `ocr-eval score` imports
upstream's scorer in-process. So `RDB_ALLOW_DOTENV=1` + a `.env` file reaches `realdoc-bench`
commands only, and the sole `realdoc-bench` command Stage 1 actually uses
(`evaluate download`) needs no key at all. Do not rely on `.env` for any `ocr-eval` invocation.

### Credential hygiene in the scoring leg (fixed 2026-08-04)

`gemini_extract` sends the key as an **`x-goog-api-key` header**. Upstream sent it as a URL query
parameter (`params={"key": ...}`), which leaked a live key into a session transcript when a
transient 503 raised an `httpx.HTTPStatusError` — httpx embeds the full request URL in the exception
message, and URLs additionally reach proxy and server access logs. Two consequences worth knowing:

- **Do not reintroduce a URL-param credential** anywhere in this codebase.
  `tests_ext/test_score_credentials.py` asserts the key is absent from both the URL and the
  exception message, using a canary value — that test failing means a secret is leaking again.
- **Beware shell idioms that print values.** `${VAR:+SET}` expands to the *value*, not the literal
  `SET`. To check presence without printing, use
  `[ -n "$VAR" ] && echo set || echo unset`, or compare a hash.

Any key that has appeared in a transcript must be **revoked at the issuing provider**, not merely
unset — unsetting removes local access, it does not invalidate the credential.

### `GEMINI_API_KEY` vs `GOOGLE_API_KEY` — not interchangeable everywhere

Upstream's `require_api_key()` (`realdoc_bench/evaluate/score.py:203`) and its Gemini vision
adapter both accept `GEMINI_API_KEY` **or** `GOOGLE_API_KEY`, so the extractor gate
(`ocr-eval selftest --extractor`) and the whole transcriber `score` leg work off either name.

But `direct.py`'s `run_direct` resolves `os.environ.get(e.api_key_env)` **by exact name with no
fallback** and raises `env var GEMINI_API_KEY not set` otherwise. The affected entry is
`gemini-3.5-flash@google-vlmchat` — Section A's frontier ceiling anchor — which therefore fails on
a `GOOGLE_API_KEY`-only environment even though the transcriber entry `gemini-3.5-flash@google`
succeeds in the same shell. Export `GEMINI_API_KEY` under that exact name. The asymmetry is
deliberate at the registry layer (an entry declares the one variable it wants, so a shared alias
can never silently reroute a row to different credentials) — it is a configuration requirement,
not a bug to route around.

## Known provider-behavior caveats from validation

- **Thinking exhaustion:** observed live against `qwen3-vl:8b` via Ollama — a thinking-enabled
  model can consume the entire `max_tokens` budget on its own reasoning, returning
  `finish_reason: "length"` and empty `message.content`. Handled as designed: `direct.py`'s `_one`
  writes an `error_class: "empty"` row, never a crash and never a silently-wrong answer (see
  [`cli.md`](cli.md)'s worked example).
- **Request `max_tokens` silently clamped by Ollama's `num_ctx` (corrected diagnosis):** in the
  same live validation (60 cells, `qwen3-vl:8b` via Ollama, `max_tokens` raised to 8192), 13
  cells still returned empty `message.content` at ~1,865 completion tokens. This was initially
  misread as the model voluntarily stopping with its answer stranded in the non-standard
  `message.reasoning` field — but every one of those 13 rows sums to **exactly**
  `prompt_tokens + completion_tokens = 4096`: Ollama's default `num_ctx`, which caps generation
  at `num_ctx − prompt` regardless of the requested `max_tokens`. The model was truncated
  mid-reasoning; the cut-off chain (sometimes containing answer text) sits in
  `message.reasoning`, which is what made it look like a channel-routing failure. Verified
  2026-08-02 by re-running the matrix against a derived model with `PARAMETER num_ctx 16384`
  baked in (`qwen3-vl:8b-ctx16k` — the OpenAI-compat endpoint has no per-request way to set
  `num_ctx`): 55/60 answered vs 47/60, recovering 11 of the 13 clamped cells. The 5 residual
  empties each consumed ~14,100 completion tokens and hit exactly `total_tokens = 16384` —
  runaway reasoning loops that no finite window is guaranteed to satisfy. Notably, 3 of those 5
  had answered fine under the 4096 window: at `temperature 0.0`, whether a given cell goes
  runaway is not stable across serving configs. Harness policy is unchanged: these are honest
  `error_class: "empty"` rows with the token counts as evidence. The same 4096 window also bites
  the *Instruct* build at full-bank scale, in two further costumes (quantified on the complete
  1,356-cell run, 2026-08-02): 5 mortgage pages exceed 4096 prompt tokens outright (HTTP 400 →
  `api_error`), and 44 dense-page multi-field cells get their JSON truncated at exactly
  `total_tokens = 4096` (→ `parse_error`, none of them at the 1024 completion cap). Full-bank
  local validation therefore needs `num_ctx=8192` (the `...-ctx8k` registry entry); observed
  prompt maximum is ~4.2k tokens at 150 dpi, and observed clean completions max out at 338
  tokens (p99 = 126), so `STAGE1_CONDITION`'s 1024 completion cap has ~3× headroom and needs no
  change for hosted providers, which serve the model's full context. Parsing an answer out of the
  reasoning channel would be a semantic change (scoring thinking output) and is deliberately not
  done in Stage 1; if Stage 2 adds a reasoning-fallback, it must be a distinct `output_contract`
  condition value, never a silent widening.
- **Empty transcripts from thinking transcribers:** the same failure mode on the transcriber side
  (a transcript reduced to near-nothing by a thinking model's budget exhaustion) would otherwise
  pass `parse`'s ok/fail split with a false-green result and then spend one Gemini extractor call
  per bank item against no content. `cli.py`'s `parse()` content floor (`md_length <= 16 *
  page_count`) catches this and fails the parse step (exit 2) before any scoring spend — see the
  fail-closed inventory in [`architecture.md`](architecture.md#fail-closed-inventory).

No other provider-behavior quirks are currently documented in this repository beyond what's cited
above — anything else encountered during a hosted run should be recorded here (or in the runbook)
with the same "what was observed, where it's handled in code" discipline, not asserted from memory.
