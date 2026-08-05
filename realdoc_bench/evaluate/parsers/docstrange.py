"""Nanonets DocStrange cloud transcriber — rendered PNG in, markdown out.

Talks to the documented v1 sync endpoint (``POST /api/v1/extract/sync`` on
``https://extraction-api.nanonets.com``) directly over ``httpx``, matching the repo's other
lightweight HTTP parse adapters rather than pulling in the ``docstrange`` pip SDK. That is
deliberate: the SDK still calls a **legacy** ``/extract`` endpoint with a different parameter
name (``output_type`` rather than v1's ``output_format``), so SDK examples are not a valid
guide to this API surface.

**Raster-only (divergence D5).** Every page is sent as a 150-dpi PNG rendered by
``VisionParserBase``/pymupdf — never the source PDF — even though the endpoint would happily
accept a PDF upload. 2 of 16 sampled corpus docs carry an embedded text layer that a
PDF-accepting parser would free-ride on, so a PDF upload here would not be measuring OCR.
D5 makes raster-only binding for every parser this fork implements.

**Why this provider is interesting.** Nanonets' OCR model prompt is the only surveyed one that
explicitly instructs checkbox-state glyphs ("Prefer using ☐ and ☑ for check boxes", from the
SDK's ``pipeline/nanonets_processor.py``) — and Stage 1's bank is checkbox-heavy. That evidence
is from the *local* pipeline's prompt and the hosted checkpoint is undisclosed, but a live smoke
transcript of ``finance_1`` did emit ``☑``, so the behaviour carries over to the hosted service
on at least one real corpus page. Same transcript: tables as embedded HTML (``<table>``, with
``rowspan``), blank form fields as runs of underscores, a ``<header>`` pseudo-tag. Budget ~55 s
per page.

**Prompt contract.** The endpoint accepts ``custom_instructions`` (+ ``prompt_mode``
append/replace), but this parser sends none by default: it measures the service's stock
behaviour, which is what a deployer would actually get. Set ``custom_instructions`` on a
subclass to register a contract-prompted variant — it is folded into ``config_hash()``, so the
two never share a cache identity.

Auth: ``DOCSTRANGE_API_KEY`` env var (checked at construction, so a missing key fails the whole
run before the first document rather than once per document). Pricing: ``docstrange_sync`` in
catalog.yaml — $1 = 100 credits and 1 credit = 1 page, so $0.01/page.

Spend guard: set ``DOCSTRANGE_MAX_PAGES`` to refuse further billable calls once that many pages
have been billed in this process. Unset means no cap.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from realdoc_bench.evaluate.parsers._vision_base import VisionParserBase
from realdoc_bench.evaluate.parsers.base import ParseResult, register_parser
from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.pricing.meter import parse_cost

DEFAULT_BASE_URL = "https://extraction-api.nanonets.com"
SYNC_PATH = "/api/v1/extract/sync"
API_KEY_ENV = "DOCSTRANGE_API_KEY"
MAX_PAGES_ENV = "DOCSTRANGE_MAX_PAGES"

MAX_RETRIES = 4
BACKOFF_BASE_SEC = 2.0
MAX_RETRY_WAIT_SEC = 60.0
REQUEST_TIMEOUT_SEC = 300.0      # matches the SDK's own per-request timeout

# A 429 is normally "slow down" and worth retrying. A 429 that is really "you are out of
# credits" is not: retrying it four times per page across a 581-page corpus burns wall-clock
# to no purpose and hides the real reason the run stopped. Body text decides which one it is.
_EXHAUSTED_RE = re.compile(r"credit|quota|insufficient|balance|exceed|limit reached", re.IGNORECASE)

_spend_lock = threading.Lock()
_billed_pages_total = 0


class DocStrangeSpendCapExceeded(RuntimeError):
    """Raised instead of issuing a billable call once DOCSTRANGE_MAX_PAGES is reached."""


def billed_pages_total() -> int:
    """Pages this process has been billed for, as reported by the API itself."""
    with _spend_lock:
        return _billed_pages_total


def reset_billed_pages() -> None:
    """Test hook — the counter is module-global and would otherwise leak across tests."""
    global _billed_pages_total
    with _spend_lock:
        _billed_pages_total = 0


def _reserve_page_budget() -> None:
    """Refuse a call that would push billed pages past DOCSTRANGE_MAX_PAGES.

    Checked *before* the request so the cap prevents spend rather than reporting it after the
    fact. Each sync call bills at least one page, so "already at the cap" is enough to refuse.

    **Overshoot bound:** the cap is approximate to within `--workers` pages. `run_parse` fans
    documents across a thread pool, so up to `workers` threads can pass this check on the same
    pre-cap count before any of them records a page. Same semantics as `--max-spend`'s
    documented overshoot bound in direct.py — at $0.01/page, a default 8-worker run can exceed
    the cap by at most $0.07. Set the cap as a budget guardrail, not as an exact quota.
    """
    raw = os.environ.get(MAX_PAGES_ENV)
    if not raw:
        return
    try:
        cap = int(raw)
    except ValueError as e:
        raise RuntimeError(f"{MAX_PAGES_ENV}={raw!r} is not an integer") from e
    with _spend_lock:
        spent = _billed_pages_total
    if spent >= cap:
        raise DocStrangeSpendCapExceeded(
            f"{MAX_PAGES_ENV}={cap} reached ({spent} pages billed this process) — refusing "
            f"further billable DocStrange calls. Raise or unset {MAX_PAGES_ENV} to continue; "
            f"transcripts already written are cached and will not be re-billed.")


def _record_billed(pages: int) -> None:
    global _billed_pages_total
    with _spend_lock:
        _billed_pages_total += pages


def _retry_wait(resp: httpx.Response | None, attempt: int) -> float:
    """Prefer the server's Retry-After (numeric seconds), else exponential backoff. Always
    clamped to [0, MAX_RETRY_WAIT_SEC] so a hostile header cannot stall the run."""
    fallback = BACKOFF_BASE_SEC * (2 ** attempt)
    wait = fallback
    if resp is not None:
        try:
            wait = float(resp.headers.get("retry-after", fallback))
        except (TypeError, ValueError):
            wait = fallback
    return max(0.0, min(MAX_RETRY_WAIT_SEC, wait))


def _markdown_from(body: dict[str, Any]) -> str:
    """Pull the transcript out of a 200 body.

    Observed live (2026-08-04), NOT what the published OpenAPI schema says: ``result.markdown``
    is an object ``{"content": "<markdown>", "metadata": {...}}``, not a string. The string form
    is still accepted in case the documented shape is what some deployments return. A body with
    neither is an error, not an empty transcript — silently writing "" would sail past the
    parse-stage length gate's siblings and burn one extractor call per bank item at scoring time
    against no content.
    """
    result = body.get("result")
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        md = result.get("markdown")
        if isinstance(md, str):
            return md
        if isinstance(md, dict) and isinstance(md.get("content"), str):
            return md["content"]
    raise RuntimeError(
        f"DocStrange 200 response carried no usable result.markdown "
        f"(keys={sorted(body)!r}, result_keys="
        f"{sorted(result) if isinstance(result, dict) else type(result).__name__})")


@register_parser("docstrange_sync", version="extract-v1-sync")
class DocStrangeSyncParser(VisionParserBase):
    """Nanonets DocStrange via ``POST /api/v1/extract/sync``, one rendered page per request.

    ``version`` names the API surface, not a model: the hosted endpoint does not disclose which
    checkpoint serves a request, so there is no model id to pin.
    """

    pricing_key = "docstrange_sync"
    page_concurrency = 1        # one page per request; cross-document concurrency is run_parse's
    dpi = 150                   # D5 render leg — same as every other Stage 1 transcriber
    output_format = "markdown"
    custom_instructions: str | None = None   # stock behaviour by default (see module docstring)
    prompt_mode = "append"                   # only sent alongside custom_instructions

    def __init__(self) -> None:
        self._api_key = os.environ.get(API_KEY_ENV)
        if not self._api_key:
            raise RuntimeError(f"{API_KEY_ENV} not set")
        self._base_url = os.environ.get("DOCSTRANGE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

    def config_hash(self) -> str:
        return sha256_text(
            f"{self.name}|{self.version}|format={self.output_format}|dpi={self.dpi}"
            f"|instructions={self.custom_instructions}|mode={self.prompt_mode}"
        )[:7]

    def _form_fields(self) -> dict[str, str]:
        data = {"output_format": self.output_format}
        if self.custom_instructions:
            data["custom_instructions"] = self.custom_instructions
            data["prompt_mode"] = self.prompt_mode
        return data

    def _call_page(self, png_bytes: bytes) -> tuple[str, int, int]:
        """One page → ``(markdown, pages_reported_by_api, billable_requests_issued)``.

        Those two counters ride the base class's ``input_tokens``/``output_tokens`` channels.
        DocStrange is page-billed and reports no tokens, and threading counts through the base
        class's own per-page aggregation is what makes billing attribution correct under any
        concurrency without a side channel. ``parse()`` below relabels both before they reach
        any artifact, so nothing downstream sees a page count wearing a token's name.

        Two counters rather than one because ``pages_processed`` is unreliable here: the live
        sync response returns it as **null** (verified 2026-08-04) even though the same record
        fetched back from ``GET /api/v1/extract/results/{record_id}`` carries ``1``. When the
        API declines to say, one billable request is one page and the row says so.
        """
        _reserve_page_budget()
        last_resp: httpx.Response | None = None
        for attempt in range(MAX_RETRIES):
            try:
                with httpx.Client(timeout=REQUEST_TIMEOUT_SEC) as client:
                    resp = client.post(
                        f"{self._base_url}{SYNC_PATH}",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        files={"file": ("page.png", png_bytes, "image/png")},
                        data=self._form_fields(),
                    )
            except httpx.TransportError:           # DNS / refused / timeout — no response at all
                if attempt == MAX_RETRIES - 1:
                    raise
                _sleep(_retry_wait(None, attempt))
                continue

            if resp.status_code < 400:
                body = resp.json()
                reported = int(body.get("pages_processed") or 0)
                _record_billed(reported or 1)
                return _markdown_from(body), reported, 1

            last_resp = resp
            detail = resp.text[:300]
            if resp.status_code == 429 and _EXHAUSTED_RE.search(detail):
                raise RuntimeError(
                    f"DocStrange 429 looks like exhausted credits, not rate limiting — "
                    f"stopping instead of retrying: {detail}")
            retryable = resp.status_code in (408, 429) or 500 <= resp.status_code < 600
            if not retryable or attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"DocStrange {resp.status_code}: {detail}")
            _sleep(_retry_wait(last_resp, attempt))
        raise RuntimeError("unreachable")   # loop always returns or raises

    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        """Relabel the two counters ``_call_page`` routed through the base class's token
        channels, and re-derive cost from pages billed rather than pages rendered. Those agree
        at one page per request, and a disagreement is exactly what is worth catching in a paid
        run. ``billed_pages_source`` records whether the number came from the API or from the
        one-request-is-one-page fallback, so a spend audit never has to guess which it read."""
        result = super().parse(pdf_path, cache_dir=cache_dir)
        raw = result.raw or {}
        reported = int(raw.get("input_tokens") or 0)
        requests = int(raw.get("output_tokens") or 0)
        billed = reported or requests
        if billed:
            result.pages_processed = billed
            result.cost_estimate_usd = parse_cost(self.pricing_key, pages=billed)
            result.raw = {
                "billed_pages": billed,
                "billed_pages_source": "api" if reported else "assumed-1-per-request",
                "pages_rendered": result.page_count,
                "requests": requests,
            }
        else:
            result.raw = None
        return result


def _sleep(seconds: float) -> None:
    """Indirection so tests can patch sleeping without patching the stdlib globally."""
    time.sleep(seconds)
