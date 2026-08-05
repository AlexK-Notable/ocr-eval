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

- **Nanonets DocStrange** (`docstrange@nanonets`, parser `docstrange_sync`) posts one rendered
  150-dpi PNG per page to `POST /api/v1/extract/sync` on `https://extraction-api.nanonets.com`
  with `output_format=markdown`, Bearer-authed from `DOCSTRANGE_API_KEY`. Notes:
  - **Raster-only despite accepting PDFs.** The endpoint would take a PDF upload; the adapter
    never sends one (D5). This is why the entry declares `input_mode: raster-png` — without it
    the report would infer `pdf-direct` from `transport: upstream-parser` and hang the
    embedded-text-layer caveat on a row that only ever sees a PNG.
  - **Page-billed.** `$1 = 100 credits`, `1 credit = 1 page` → `$0.010/page` in `catalog.yaml`;
    the 581-page corpus is $5.81 at list. Cost is metered from pages *billed*, not pages
    rendered, so a billing surprise lands in the artifact instead of being papered over — but
    see the next bullet for how reliable that number is. Set `DOCSTRANGE_MAX_PAGES=<n>` to
    refuse further billable calls once `n` pages have been billed in the process (unset = no
    cap); already-written transcripts are cached and are never re-billed on resume. The cap is
    a guardrail, not an exact quota: it is checked before each call, so up to `--workers`
    threads can pass the same pre-cap count, overshooting by at most `workers` pages ($0.08 at
    the default 8). Same semantics as `--max-spend`'s documented overshoot bound above.
  - **Two live deviations from the published OpenAPI schema** (verified 2026-08-04, both handled
    by the adapter — do not "fix" them back toward the schema without re-checking):
    1. `result.markdown` is an **object** `{"content": "<markdown>", "metadata": {...}}`, not the
       bare string the schema documents. Both shapes are accepted; a 200 carrying neither raises
       rather than writing an empty transcript.
    2. `pages_processed` comes back **null on the sync response**, even though the same record
       fetched from `GET /api/v1/extract/results/{record_id}` carries `1`. When the API declines
       to say, the adapter bills one page per request and records
       `billed_pages_source: "assumed-1-per-request"` in the parse metadata, so a spend audit
       never has to guess which number it is reading.
  - **~55 s per page**, and **`--workers 8` is already the throughput ceiling** — measured
    2026-08-04 against distinct corpus docs (repeating one page would risk server-side caching
    flattering the numbers):

    | workers | wall | median per-call | throughput | 581 pages |
    |---|---|---|---|---|
    | 1 | 52.9 s | 52.9 s | 1.1 pages/min | ~9 h |
    | 8 | 67.8 s / 8 pages | 54.9 s | **7.1 pages/min** | **~82 min** |
    | 16 | 141.5 s / 16 pages | 68.2 s (max 141.5 s) | 6.8 pages/min | ~86 min |

    At 8, median per-call latency is indistinguishable from a lone call — nothing is queuing.
    At 16 it climbs to 68 s with a 141 s tail while throughput does *not* improve, which is the
    signature of a server-side concurrency ceiling around 8. Raising `--workers` past the
    default buys nothing and lengthens the tail. No 429s and no empty transcripts at either
    level, so the ceiling is enforced by queuing rather than by rejection.
  - **A 429 whose body reads as exhausted credits fails immediately** rather than being retried
    as rate limiting — otherwise the run burns four attempts per page and buries the real reason
    it stopped. Transient 408/429/5xx and connection errors retry with backoff (`Retry-After`
    honoured, clamped to 60 s); 4xx other than 408/429 fail the document at once.
  - **Stock prompt, by choice.** The endpoint accepts `custom_instructions` (+ `prompt_mode`),
    but the Stage 1 parser sends none, measuring the service as a deployer would get it. That
    means the row does **not** carry the pinned `_MARKDOWN_PROMPT` contract, which is what
    `promptable: false` communicates in this row's report stamp. A contract-prompted variant is
    a subclass setting `custom_instructions` (folded into `config_hash()`, so the two never share
    a cache identity).
  - **Why this provider is interesting:** Nanonets' OCR model prompt is the only surveyed one
    that explicitly asks for checkbox glyphs ("Prefer using ☐ and ☑ for check boxes", from the
    `docstrange` SDK's `pipeline/nanonets_processor.py`), and Stage 1's bank is checkbox-heavy.
    That evidence is from the SDK's **local** pipeline and the hosted endpoint's checkpoint is
    undisclosed — but a live smoke transcript of `finance_1` did emit `☑`, so the behaviour
    carries over to the hosted service on at least one real corpus page. Same transcript:
    tables rendered as embedded **HTML** (`<table>`, including `rowspan`), blank form fields as
    runs of underscores, and a `<header>` pseudo-tag. The underscore convention is worth
    watching — the pinned contract asks for `**<label>:** <blank>` on empty fields, and a
    stock-prompt row does not follow it.
  - The pip `docstrange` SDK is **not** used and is not a guide to this endpoint: it still calls
    a legacy `/extract` route with `output_type` rather than v1's `output_format`.

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
`api_key_env` (e.g. `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`,
`DOCSTRANGE_API_KEY`); the value itself
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
