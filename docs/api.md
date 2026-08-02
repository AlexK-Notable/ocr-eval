# API and provider considerations

The wire contract this fork speaks to model providers, the retry policy, cost controls, and
per-provider serving notes and caveats gathered during validation. **Who should read this:** anyone
adding a registry entry, debugging a provider-side error, or reasoning about spend. For the CLI
flags that trigger these calls, see [`cli.md`](cli.md); for local vLLM launch commands, see
[`local-serving.md`](local-serving.md).

## Provider contract

**OpenAI-compatible chat completions is the only transport this fork implements** — both the
vlm-chat runner (`direct.py`) and the transcriber adapter (`parsers_openai.py`) go through the
`openai` Python client's `chat.completions.create`, so any provider (OpenRouter, a local vLLM
server, Ollama, Google's OpenAI-compat endpoint) is the same code path with a different `base_url`.
Upstream's own non-OpenAI-compat adapters (Gemini via its native API, Mistral's PDF-upload
endpoint) are used as-is, unmodified, via `transport: upstream-parser` registry entries.

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

Owned entirely by this module, not the SDK: every `OpenAI(...)` client is constructed with
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
  than a false claim of a specific quant level. See [`cli.md`](cli.md)'s worked example for the
  thinking-model `max_tokens` exhaustion caveat this serving path surfaces live; Ollama's
  OpenAI-compat shim has no code-level way to suppress thinking in this codebase — there is no
  `think` field anywhere in `STAGE1_CONDITION`'s sampling dict, and the runbook records that raising
  `max_tokens` is the only available workaround.
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
never appears in any config file. The standard invocation pattern is `bws run --project-id <id> --
<cmd>` (Bitwarden Secrets Manager injection — see the runbook's prerequisites). Upstream's
`.env`/`.env.local` loading is disabled by default and only re-enabled by setting
`RDB_ALLOW_DOTENV=1` (the one upstream modification — see
[`architecture.md`](architecture.md#fork-boundaries)).

## Known provider-behavior caveats from validation

- **Thinking exhaustion:** observed live against `qwen3-vl:8b` via Ollama — a thinking-enabled
  model can consume the entire `max_tokens` budget on its own reasoning, returning
  `finish_reason: "length"` and empty `message.content`. Handled as designed: `direct.py`'s `_one`
  writes an `error_class: "empty"` row, never a crash and never a silently-wrong answer (see
  [`cli.md`](cli.md)'s worked example).
- **Empty transcripts from thinking transcribers:** the same failure mode on the transcriber side
  (a transcript reduced to near-nothing by a thinking model's budget exhaustion) would otherwise
  pass `parse`'s ok/fail split with a false-green result and then spend one Gemini extractor call
  per bank item against no content. `cli.py`'s `parse()` content floor (`md_length <= 16 *
  page_count`) catches this and fails the parse step (exit 2) before any scoring spend — see the
  fail-closed inventory in [`architecture.md`](architecture.md#fail-closed-inventory).

No other provider-behavior quirks are currently documented in this repository beyond what's cited
above — anything else encountered during a hosted run should be recorded here (or in the runbook)
with the same "what was observed, where it's handled in code" discipline, not asserted from memory.
