"""Tests for the Mistral OCR 4 adapter's fork changes (`realdoc_bench/evaluate/parsers/mistral_ocr.py`).

Two undeprecated pins — `mistral-ocr-4-0` and `mistral-ocr-4-1` — run as SEPARATE parser names so
both can be measured. Everything pinned here protects that split: a collision in either the config
hash or the pricing catalog would make one leg silently masquerade as the other, or report $0.

No network: `httpx.MockTransport` serves every response.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest

from realdoc_bench.evaluate.parsers import mistral_ocr as mo
from realdoc_bench.evaluate.parsers.base import build, registry
from realdoc_bench.shared.pricing.meter import parse_cost

PDF_BYTES = b"%PDF-1.4 fake"


def ok_body(md: str = "# page", pages: int = 1, model: str = "mistral-ocr-4-1") -> dict:
    return {"model": model,
            "pages": [{"index": i, "markdown": md, "blocks": []} for i in range(pages)],
            "usage_info": {"pages_processed": pages}}


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(PDF_BYTES)
    return p


def _patched(monkeypatch, handler) -> None:
    """Route the adapter's own `httpx.Client(...)` through a mock transport."""
    real = httpx.Client

    def fake(*a, **kw):
        kw.pop("timeout", None)
        return real(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(mo.httpx, "Client", fake)
    monkeypatch.setenv("MISTRAL_API_KEY", "TEST-KEY")


# ── the two pins must stay distinguishable ───────────────────────────────────────────────────

def test_both_undeprecated_pins_are_registered_under_distinct_names():
    """One name for two model versions would overwrite transcripts: upstream keys both
    `parses/<parser>/` and `eval/cache/<qid>__<parser>.json` by parser NAME."""
    assert "mistral_ocr_4" in registry
    assert "mistral_ocr_4_0" in registry
    assert build("mistral_ocr_4").model == "mistral-ocr-4-1"
    assert build("mistral_ocr_4_0").model == "mistral-ocr-4-0"


def test_no_pin_is_a_floating_alias():
    """`mistral-ocr-latest` resolved to 4-1 on 2026-08-06 and 4-0/4-1 return DIFFERENT markdown on
    the same page, so an alias can change what a measured number means without any code change.
    OCR 1 and OCR 2 are already retired — the aliases demonstrably move."""
    for name in ("mistral_ocr_4", "mistral_ocr_4_0"):
        p = build(name)
        assert "latest" not in p.model
        assert "latest" not in p.version


def test_model_reaches_the_config_hash():
    """Two configurations that hash alike share a cache key, so the second run would read the
    first's rows and report them as its own."""
    assert build("mistral_ocr_4").config_hash() != build("mistral_ocr_4_0").config_hash()


@pytest.mark.parametrize("field", ["table_format", "confidence_scores_granularity",
                                   "include_blocks", "extract_header", "extract_footer"])
def test_every_request_shaping_parameter_reaches_the_config_hash(field):
    base = build("mistral_ocr_4")
    changed = build("mistral_ocr_4", **{field: True if field.startswith(("include", "extract"))
                                        else "html"})
    assert base.config_hash() != changed.config_hash(), (
        f"{field} changes the request but not the hash — two conditions would collide")


def test_both_parser_names_are_priced():
    """REGRESSION: `mistral_ocr_4_0` inherits `name` from the registration decorator, and an
    unpriced name makes `parse_cost` return None rather than raise — a whole 581-page leg would
    have reported $0.00 spend. Same shape as the omitted-token-count crash: a silent None where a
    number belongs."""
    assert parse_cost("mistral_ocr_4", pages=581) == pytest.approx(2.324)
    assert parse_cost("mistral_ocr_4_0", pages=581) == pytest.approx(2.324)
    # the check can fail: an unknown name really does return None
    assert parse_cost("definitely_not_a_provider", pages=1) is None


# ── request shape ────────────────────────────────────────────────────────────────────────────

def test_unset_parameters_are_omitted_not_sent_as_null(monkeypatch, pdf):
    """An omitted key takes Mistral's server-side default; an explicit null is a different request
    and some values are rejected outright."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    _patched(monkeypatch, handler)
    build("mistral_ocr_4").parse(pdf)
    body = seen["body"]
    for k in ("table_format", "confidence_scores_granularity", "include_blocks",
              "extract_header", "extract_footer"):
        assert k not in body
    assert body["model"] == "mistral-ocr-4-1"


def test_configured_parameters_do_reach_the_wire(monkeypatch, pdf):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    _patched(monkeypatch, handler)
    build("mistral_ocr_4", include_blocks=True, extract_header=True, extract_footer=True,
          confidence_scores_granularity="word").parse(pdf)
    body = seen["body"]
    assert body["include_blocks"] is True
    assert body["extract_header"] is True and body["extract_footer"] is True
    assert body["confidence_scores_granularity"] == "word"


def test_key_travels_as_a_bearer_header_never_in_the_url(monkeypatch, pdf):
    """httpx embeds the full URL in HTTPStatusError, so a query-param key prints the live secret
    into any non-2xx traceback. This repo has leaked a key that way before."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=ok_body())

    _patched(monkeypatch, handler)
    build("mistral_ocr_4").parse(pdf)
    assert seen["auth"] == "Bearer TEST-KEY"
    assert "TEST-KEY" not in seen["url"]


def test_pdf_is_sent_as_a_base64_data_url(monkeypatch, pdf):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    _patched(monkeypatch, handler)
    build("mistral_ocr_4").parse(pdf)
    url = seen["body"]["document"]["document_url"]
    assert url.startswith("data:application/pdf;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == PDF_BYTES


# ── response mapping ─────────────────────────────────────────────────────────────────────────

def test_server_reported_model_is_recorded_not_the_requested_one(monkeypatch, pdf):
    """The only way to catch an alias resolving somewhere other than where we pinned it."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body(model="mistral-ocr-9-9"))

    _patched(monkeypatch, handler)
    res = build("mistral_ocr_4").parse(pdf)
    assert res.raw is not None and res.raw["model"] == "mistral-ocr-9-9"


def test_cost_is_metered_from_page_count(monkeypatch, pdf):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body(pages=3))

    _patched(monkeypatch, handler)
    res = build("mistral_ocr_4").parse(pdf)
    assert res.page_count == 3
    assert res.cost_estimate_usd == pytest.approx(0.012)


def test_unknown_parameter_is_rejected_not_silently_ignored():
    """A dropped `table_fmt=` typo would produce rows claiming a configuration they were never
    served at — and `config_hash` would agree with the false claim."""
    with pytest.raises(TypeError, match="unknown parameter"):
        build("mistral_ocr_4", table_fmt="html")


def test_missing_key_raises_before_any_request(monkeypatch, pdf):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=ok_body())

    _patched(monkeypatch, handler)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        build("mistral_ocr_4").parse(pdf)
    assert calls["n"] == 0
