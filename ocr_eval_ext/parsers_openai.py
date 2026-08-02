"""OpenAI-compatible transcriber parser: bridges `ocr_eval_ext`'s registry model into upstream's
`ParseProvider`/`VisionParserBase` machinery so `evaluate parse`/`evaluate score` work unchanged
on transcriber rows served behind any OpenAI-compatible /chat/completions endpoint — local vLLM
servers foremost. Upstream's own `realdoc-bench` CLI never imports `ocr_eval_ext`, so parsers
built here only become reachable in-process via `ocr_eval_ext.cli`'s `parse`/`score` wrapper
commands (they call `register_openai_parsers` before delegating to upstream `run_parse`/
`run_score`)."""
from __future__ import annotations

import time
from pathlib import Path

from ocr_eval_ext.config import RegistryEntry
from realdoc_bench.evaluate.parsers._vision_base import VisionParserBase
from realdoc_bench.evaluate.parsers.base import register_parser
from realdoc_bench.evaluate.parsers.base import registry as parser_registry
from realdoc_bench.evaluate.parsers.cloud_vlm import _MARKDOWN_PROMPT

# The condition every openai-compat transcriber is scored under in Stage 1 — raw preprocessing,
# pymupdf@150dpi renders (the same render leg as direct.py's STAGE1_CONDITION), temperature 0,
# single sample. Folded into the registered parser NAME itself (see `register_openai_parsers`
# below), not just recorded as metadata: Stage 2's deskew preprocessing will register a NEW
# parser name (different condition -> different hash) instead of silently overwriting these raw
# transcripts under the same `parses/<name>/` dir and cache keys. Decided now because it is
# unwindable later without discarding cache.
TRANSCRIBER_CONDITION = {
    "preprocess": "raw",
    "render": {"engine": "pymupdf", "dpi": 150},
    "sampling": {"temperature": 0.0},
    "sample_index": 0,
}

MAX_RETRIES = 4


class OpenAICompatVisionParser(VisionParserBase):
    """Per-page markdown transcription against any OpenAI-compatible /chat/completions
    endpoint. Subclasses are built per registry entry by `register_openai_parsers` below with a
    zero-arg `__init__` (the contract `ParseProvider.build`/`registry.get(name)()` requires) —
    this base class is never registered directly."""

    prompt = _MARKDOWN_PROMPT
    page_concurrency = 1        # local GPU: one in-flight request; hosted entries may override

    def __init__(self, *, base_url: str, model: str, api_key_env: str | None = None):
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

    def _call_page(self, png_bytes: bytes) -> tuple[str, int, int]:
        import base64

        from ocr_eval_ext.direct import _is_retryable, _retry_wait

        b64 = base64.b64encode(png_bytes).decode()
        kwargs = {
            "model": self._model, "temperature": 0.0, "max_tokens": self.max_tokens,
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
            usage = resp.model_dump().get("usage") or {}
            return ((resp.choices[0].message.content or ""),
                    usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        raise RuntimeError("unreachable")  # loop above always returns or raises


def safe_name(entry_id: str) -> str:
    """Registry id -> filesystem/registry-safe parser-name component. Parser names become
    directory names under `parses/` and cache-key suffixes, so `@` -> `_`, `.` -> `-`, and
    `/` -> `_`."""
    return entry_id.replace("@", "_").replace(".", "-").replace("/", "_")


def register_openai_parsers(registry_path: Path) -> list[str]:
    """Build and register one `OpenAICompatVisionParser` subclass per transcriber+openai-compat
    registry entry, named `safe_name(entry.id) + "__" + condition_hash(TRANSCRIBER_CONDITION)`
    (see `TRANSCRIBER_CONDITION` above for why the hash is folded into the name). Idempotent: a
    name already present in the shared `parser_registry` — e.g. a second call within the same
    process — is left alone and just appended to the returned list, never re-registered (which
    upstream's `Registry.register` would raise `ValueError` on)."""
    from ocr_eval_ext.config import load_registry
    from ocr_eval_ext.direct import condition_hash

    suffix = condition_hash(TRANSCRIBER_CONDITION)
    names = []
    for e in load_registry(registry_path):
        if e.shape != "transcriber" or e.transport != "openai-compat":
            continue
        name = f"{safe_name(e.id)}__{suffix}"
        if name in parser_registry:      # idempotent re-import
            names.append(name)
            continue

        def _init(self, *, _e=e):
            OpenAICompatVisionParser.__init__(self, base_url=_e.base_url, model=_e.model,
                                              api_key_env=_e.api_key_env)

        cls = type(name, (OpenAICompatVisionParser,), {"__init__": _init})
        register_parser(name, version=e.model)(cls)
        names.append(name)
    return names


def preflight(entry: RegistryEntry) -> str:
    """GET {base_url}/models and confirm the entry's `model` is actually being served. Run before
    any local vLLM spend (`ocr-eval preflight <entry-id>` wraps this) — a served-model mismatch
    (wrong checkpoint resident, server not restarted after a config change) must be caught before
    a single page is transcribed against the wrong weights."""
    import httpx

    r = httpx.get(f"{entry.base_url}/models", timeout=10)
    r.raise_for_status()
    served = [m.get("id", "") for m in r.json().get("data", [])]
    if entry.model not in served:
        raise RuntimeError(f"{entry.id}: served models {served} do not include {entry.model}")
    return entry.model
