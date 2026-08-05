"""OpenAI-compatible transcriber parser: bridges `ocr_eval_ext`'s registry model into upstream's
`ParseProvider`/`VisionParserBase` machinery so `evaluate parse`/`evaluate score` work unchanged
on transcriber rows served behind any OpenAI-compatible /chat/completions endpoint — local vLLM
servers foremost. Upstream's own `realdoc-bench` CLI never imports `ocr_eval_ext`, so parsers
built here only become reachable in-process via `ocr_eval_ext.cli`'s `parse`/`score` wrapper
commands (they call `register_openai_parsers` before delegating to upstream `run_parse`/
`run_score`)."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from ocr_eval_ext.config import RegistryEntry
from realdoc_bench.evaluate.parsers._vision_base import VisionParserBase
from realdoc_bench.evaluate.parsers.base import ParseResult, register_parser
from realdoc_bench.evaluate.parsers.base import registry as parser_registry
from realdoc_bench.evaluate.parsers.cloud_vlm import _MARKDOWN_PROMPT

# The condition every openai-compat transcriber is scored under in Stage 1 — raw preprocessing,
# pymupdf@150dpi renders (the same render leg as direct.py's STAGE1_CONDITION), temperature 0,
# 12288-token completion budget (C1 — see OpenAICompatVisionParser.max_tokens below), single
# sample. Folded into the registered parser NAME itself (see `register_openai_parsers` below),
# not just recorded as metadata: Stage 2's deskew preprocessing will register a NEW parser name
# (different condition -> different hash) instead of silently overwriting these raw transcripts
# under the same `parses/<name>/` dir and cache keys. Decided now because it is unwindable later
# without discarding cache.
TRANSCRIBER_CONDITION = {
    "preprocess": "raw",
    "render": {"engine": "pymupdf", "dpi": 150},
    "sampling": {"temperature": 0.0, "max_tokens": 12288},
    "sample_index": 0,
}
# max_tokens 4096 -> 12288 (2026-08-04, user-decided, set FROM measurement — see direct.py's
# STAGE1_CONDITION for the demand figures). A thinking transcriber spends the budget
# on reasoning and emits nothing: qwen3.5-9b produced a 9-byte transcript ("## Page 1" and no
# content) on the denser of two smoke pages while transcribing the other correctly at 3.9kB —
# page-dependent, so it would have salted a full run with silent holes rather than failing
# outright. `parse` does fail closed on it (the md_length <= 16 chars/page gate) but only after
# the pages are billed. Entries whose provider supports it also cap thinking via the registry's
# `reasoning` field, so it cannot consume the whole allowance. Raising the cap costs nothing for
# non-thinking transcribers, which stop when the page is done.

MAX_RETRIES = 4


class OpenAICompatVisionParser(VisionParserBase):
    """Per-page markdown transcription against any OpenAI-compatible /chat/completions
    endpoint. Subclasses are built per registry entry by `register_openai_parsers` below with a
    zero-arg `__init__` (the contract `ParseProvider.build`/`registry.get(name)()` requires) —
    this base class is never registered directly.

    I7: `dpi` and `max_tokens` are pinned FROM `TRANSCRIBER_CONDITION` itself (not left to
    silently inherit `VisionParserBase`'s own `DEFAULT_DPI`/12000-token defaults) so the
    condition dict actually drives what gets rendered/requested rather than being decorative
    metadata that can drift out of sync with it."""

    prompt = _MARKDOWN_PROMPT
    page_concurrency = 1        # local GPU: one in-flight request; hosted entries may override
    dpi = TRANSCRIBER_CONDITION["render"]["dpi"]
    max_tokens = TRANSCRIBER_CONDITION["sampling"]["max_tokens"]
    # C1 (SUPERSEDED 2026-08-04 — read this before serving a local transcriber). The budget was
    # originally 4096 to fit local-serving.md's `--max-model-len 8192`: a full-page prompt
    # (markdown-extraction instructions + one rendered page image, ~1.3-1.9k image tokens at
    # 150dpi) plus upstream's inherited 12000-token completion budget blows through that window,
    # the server 400s, `_is_retryable` correctly classifies a 400 as permanent (not a transient
    # 408/429/5xx), and EVERY local page fails on its first attempt, never retried.
    # The budget is now 12288 (see TRANSCRIBER_CONDITION), which does NOT fit inside 8192.
    # Local vLLM entries (glm-ocr, dots-ocr) must therefore be served with a larger
    # `--max-model-len` — docs/local-serving.md now specifies 16384. Serving them at 8192 against
    # this condition reproduces exactly the failure described above, on every page.

    def __init__(self, *, base_url: str, model: str, api_key_env: str | None = None,
                 provider_pin: dict | None = None, reasoning: dict | None = None):
        import os

        from openai import OpenAI

        key = os.environ.get(api_key_env) if api_key_env else "none"
        if api_key_env and not key:
            raise RuntimeError(f"env var {api_key_env} not set")
        # max_retries=0: this module owns retries end-to-end (`_call_page` below), reusing
        # direct.py's `_is_retryable`/`_retry_wait`. Leaving the SDK's own default retry-on-
        # 408/429/5xx enabled would nest under ours — the same double-retry hazard `run_direct`
        # documents (up to MAX_RETRIES outer attempts x the SDK's own retries each).
        self._client = OpenAI(base_url=base_url, api_key=key or "none", max_retries=0)
        self._model = model
        self._reasoning = reasoning            # caps thinking so it cannot eat the whole
                                                 # completion budget (see TRANSCRIBER_CONDITION)
        self._provider_pin = provider_pin      # I2: hosted transcribers (e.g. OpenRouter) honor
                                                 # this exactly like direct.py's `_one` does
        # N1: upstream `run_parse` builds ONE `ParseProvider` instance per parser name and shares
        # it across every document — `_parse_one` calls `pinst.parse(pdf)` directly inside a
        # `ThreadPoolExecutor(max_workers=workers)` worker thread (confirmed against
        # `realdoc_bench/evaluate/parse.py`), and hosted (non-local) transcribers are exactly the
        # ones with no `_default_parse_workers` override, so they run with the real default of 8
        # concurrent document-threads on this SAME instance. `page_concurrency=1` only serializes
        # requests WITHIN one document's `parse()` call — it says nothing about two DIFFERENT
        # documents' `parse()` calls running concurrently on separate threads. Plain instance
        # state here (a bare `set()`) would let two concurrently-parsing documents stomp on each
        # other's provider set — `threading.local()` gives every thread its own storage, and
        # since upstream never calls `parse()` more than once at a time on a given thread, a
        # fresh set assigned at the top of `parse()` and read back at the bottom is fully
        # isolated from whatever a DIFFERENT thread's concurrent `parse()` call is doing.
        self._tl = threading.local()
        # NEW-1 (round 3): the `threading.local()` isolation above assumes `_call_page` only ever
        # runs on the SAME thread that is currently inside `parse()` for that document — true at
        # `page_concurrency=1` (Stage 1's enforced reality: single-page corpus), but NOT true the
        # moment `page_concurrency > 1` or `RDB_PAGE_CONCURRENCY` fans a document's pages out
        # across upstream `VisionParserBase.parse`'s own NESTED `ThreadPoolExecutor` — a pattern
        # this project's own docs/comments anticipate even though Stage 1 never exercises it.
        # Those page-worker threads never ran `self._tl.providers = set()` themselves (only
        # `parse()` does that, on the DOCUMENT thread), so `self._tl.providers` is simply absent
        # on them — `_call_page` would raise `AttributeError`, which propagates up through
        # upstream's per-page future and fails the whole document AFTER the API call was already
        # paid for. Proportionate fix for a single-page contract: never crash. `_call_page` falls
        # back to this lock-protected, per-instance set whenever thread-local storage isn't
        # primed; `parse()` drains it after the fact and flags the result as attribution-degraded
        # rather than pretending per-document precision was preserved. Exact per-page-thread
        # attribution would need a context-propagating rewrite of the capture mechanism itself —
        # deferred to Stage 2, which already owns multi-page support on its roadmap.
        self._fanout_providers: set[str] = set()
        self._fanout_lock = threading.Lock()

    def _call_page(self, png_bytes: bytes) -> tuple[str, int, int]:
        import base64

        from ocr_eval_ext.direct import _is_retryable, _retry_wait

        b64 = base64.b64encode(png_bytes).decode()
        extra_body = {}
        if self._provider_pin:
            extra_body["provider"] = self._provider_pin
        if self._reasoning:
            extra_body["reasoning"] = self._reasoning
        kwargs = {
            "model": self._model, "temperature": 0.0, "max_tokens": self.max_tokens,
            "extra_body": extra_body or None,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": self.prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
        }
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except Exception as e:
                # Retry only 408/429/5xx and connection-level failures (per `_is_retryable`) —
                # everything else (400/401/403/404/...) is permanent and must fail this page
                # immediately rather than burn the retry budget. `VisionParserBase.parse` has no
                # per-page try/except of its own, so an exhausted/unretryable error here
                # correctly propagates and fails the whole document, rather than silently
                # producing an empty page section.
                if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(_retry_wait(e, attempt))
                continue
            raw = resp.model_dump()
            usage = raw.get("usage") or {}
            provider = raw.get("provider")
            if provider:
                # NEW-1: `self._tl.providers` only exists on the DOCUMENT thread that called
                # `parse()` (which primes it). At page_concurrency=1 that's always the thread
                # `_call_page` is running on. Under page fan-out (page_concurrency>1 or
                # RDB_PAGE_CONCURRENCY), this runs on a page-worker thread that never primed its
                # own thread-local storage — `getattr` with a default detects that case instead of
                # raising, and the provider is recorded into the lock-protected fallback set
                # instead of being lost or crashing the document after its API call was paid for.
                bucket = getattr(self._tl, "providers", None)
                if bucket is not None:
                    bucket.add(provider)
                else:
                    with self._fanout_lock:
                        self._fanout_providers.add(provider)
            return ((resp.choices[0].message.content or ""),
                    usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        raise RuntimeError("unreachable")  # loop above always returns or raises

    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        """I2/N1: wraps `VisionParserBase.parse` purely to surface the resolved provider(s)
        collected across pages by `_call_page` — that base method's per-page loop only threads
        `(text, in_tok, out_tok)` through, with no room for a 4th field, and it's upstream code
        this project doesn't own, so provider capture happens as a thread-local side channel
        (see `__init__`'s N1 comment) instead of changing that tuple shape.

        NEW-1 (round 3): after the thread-local read-back, also drains `_fanout_providers` — the
        lock-protected fallback `_call_page` uses when it runs on a page-worker thread that never
        primed `self._tl.providers` (page_concurrency>1 / RDB_PAGE_CONCURRENCY fan-out). Anything
        captured there is merged into `resolved_providers`, but the result is explicitly flagged
        `provider_attribution: "degraded"`.

        DISCLOSURE (corrects an understatement in the NEW-1 description above): contamination of
        `_fanout_providers` is NOT limited to documents that are concurrently fanning out pages
        "at that moment" — it also required both docs to fan out AND overlap in time, but the
        drain below used to run only on the SUCCESS path (no `try`/`finally`). A document whose
        `super().parse()` raised partway through fan-out left its already-captured providers
        sitting in the shared set undrained; the NEXT document to drain it — even a later,
        non-concurrent, non-fanning one — would silently inherit them. L2 fix: the drain now runs
        in a `finally` block so a failed document's providers are cleared unconditionally and can
        never leak into a later, unrelated document's result."""
        self._tl.providers = set()
        try:
            result = super().parse(pdf_path, cache_dir=cache_dir)
        finally:
            # L2: drain _fanout_providers unconditionally — including when parse() raised — so a
            # document whose fan-out failed never leaves residual providers for a LATER
            # document's drain to inherit (see the DISCLOSURE above).
            with self._fanout_lock:
                fanout = set(self._fanout_providers)
                self._fanout_providers.clear()
        providers = set(self._tl.providers) | fanout
        degraded = bool(fanout)
        if providers:
            raw = dict(result.raw or {})
            raw["resolved_providers"] = sorted(providers)
            if degraded:
                raw["provider_attribution"] = ("degraded (page fan-out; cross-document "
                                                "attribution not guaranteed)")
            result.raw = raw
        return result


def safe_name(entry_id: str) -> str:
    """Registry id -> filesystem/registry-safe parser-name component. Parser names become
    directory names under `parses/` and cache-key suffixes, so `@` -> `_`, `.` -> `-`, and
    `/` -> `_`."""
    return entry_id.replace("@", "_").replace(".", "-").replace("/", "_")


def register_openai_parsers(registry_path: Path) -> list[str]:
    """Build and register one `OpenAICompatVisionParser` subclass per transcriber+openai-compat
    registry entry, named `safe_name(entry.id) + "__" + condition_hash(TRANSCRIBER_CONDITION)`
    (see `TRANSCRIBER_CONDITION` above for why the hash is folded into the name). Idempotent: a
    name already present in the shared `parser_registry` with a MATCHING binding — e.g. a second
    call within the same process against the same `registry.yaml` — is left alone and just
    appended to the returned list, never re-registered (which upstream's `Registry.register`
    would raise `ValueError` on regardless).

    I3: idempotency is NOT unconditional. Two different `registry.yaml` files can produce the
    same registered name (same entry id + same TRANSCRIBER_CONDITION hash) while pointing at
    different endpoints — e.g. a stale/fake entry registered earlier in the same process, then a
    real one under the same id later. Silently keeping the first-registered (now stale) class
    bound to the wrong `(base_url, model, api_key_env, provider_pin)` would route all subsequent
    calls to the wrong server (or the wrong upstream provider) without any error. Every class
    built here carries its full binding as `_registry_binding`; a name collision with a DIFFERENT
    binding raises `ValueError` naming both endpoints instead of reusing the stale one."""
    from ocr_eval_ext.config import load_registry
    from ocr_eval_ext.direct import condition_hash

    suffix = condition_hash(TRANSCRIBER_CONDITION)
    names = []
    for e in load_registry(registry_path):
        if e.shape != "transcriber" or e.transport != "openai-compat":
            continue
        name = f"{safe_name(e.id)}__{suffix}"
        binding = (e.base_url, e.model, e.api_key_env, e.provider_pin, e.reasoning)
        if name in parser_registry:
            existing_binding = getattr(parser_registry.get(name), "_registry_binding", None)
            if existing_binding is not None and existing_binding != binding:
                raise ValueError(
                    f"parser {name!r} is already registered for a different endpoint — "
                    f"existing binding (base_url, model, api_key_env, provider_pin, reasoning)="
                    f"{existing_binding!r}, new binding from registry entry {e.id!r}="
                    f"{binding!r}. Refusing to silently reuse the stale registration.")
            names.append(name)          # idempotent re-import: same binding, safe to reuse
            continue

        def _init(self, *, _e=e):
            OpenAICompatVisionParser.__init__(self, base_url=_e.base_url, model=_e.model,
                                              api_key_env=_e.api_key_env,
                                              provider_pin=_e.provider_pin,
                                              reasoning=_e.reasoning)

        cls = type(name, (OpenAICompatVisionParser,), {
            "__init__": _init,
            "_registry_binding": binding,
        })
        register_parser(name, version=e.model)(cls)
        names.append(name)
    return names


def preflight(entry: RegistryEntry) -> str:
    """GET {base_url}/models and confirm the entry's `model` is actually being served. Run before
    any local vLLM spend (`ocr-eval preflight <entry-id>` wraps this) — a served-model mismatch
    (wrong checkpoint resident, server not restarted after a config change) must be caught before
    a single page is transcribed against the wrong weights."""
    import httpx

    r = httpx.get(f"{entry.base_url.rstrip('/')}/models", timeout=10)   # M9: avoid a doubled
                                                                          # slash when base_url
                                                                          # already ends in "/"
    r.raise_for_status()
    served = [m.get("id", "") for m in r.json().get("data", [])]
    if entry.model not in served:
        raise RuntimeError(f"{entry.id}: served models {served} do not include {entry.model}")
    return entry.model
