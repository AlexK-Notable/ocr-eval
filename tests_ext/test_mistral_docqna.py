"""Tests for the `mistral-docqna` transport (`ocr_eval_ext/mistral_docqna.py`).

The transport exists to isolate ONE variable — whether Mistral's markdown flattening, not its OCR
engine, is what costs the transcriber legs their checkbox accuracy. So what is pinned here is that
the PDF reaches the provider intact, that the row's condition honestly records "no render happened",
and that this transport's rows can never be confused with a raster leg's.

No network: `httpx.MockTransport` serves every response.
"""
from __future__ import annotations

import base64
import json

import httpx
import pytest

from ocr_eval_ext import mistral_docqna as mdq
from ocr_eval_ext.config import RegistryEntry
from ocr_eval_ext.direct import (
    DOCQNA_INPUT,
    STAGE1_CONDITION,
    _condition_for,
    _is_retryable,
    condition_hash,
)

PDF = b"%PDF-1.4 fake-document-bytes"


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def ok_body(text: str = '{"a": true}', model: str = "mistral-small-2603") -> dict:
    return {"model": model,
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2778, "completion_tokens": 12, "total_tokens": 2790}}


def entry(**kw) -> RegistryEntry:
    base = dict(id="mistral-small-4@mistral-docqna", shape="vlm-chat", transport="mistral-docqna",
                model="mistral-small-2603", api_key_env="MISTRAL_API_KEY",
                input_mode="pdf-direct", precision="provider-default",
                weights_licence="apache-2.0", provider_tos_commercial="ok",
                provenance="Mistral", release_date="2026-03-01")
    base.update(kw)
    return RegistryEntry(**base)


# ── the PDF must arrive intact ────────────────────────────────────────────────────────────────

def test_pdf_is_sent_as_a_base64_data_url_and_round_trips():
    """Mistral's Document QnA doc says the URL must be "public and accessible by our API", which
    would rule out this corpus entirely. Probed live 2026-08-06: a base64 data URL IS accepted
    (HTTP 200, 2,778 prompt tokens on finance_4). Pinned so a docs-driven "fix" to public URLs
    cannot land silently and break every cell."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    c = mdq.MistralDocQnAClient("k", model="m", client=_client(handler))
    c.generate(system="s", prompt="p", pdf=PDF, temperature=0.0, top_p=1.0, max_tokens=64)
    parts = seen["body"]["messages"][1]["content"]
    doc = next(p for p in parts if p["type"] == "document_url")
    assert doc["document_url"].startswith("data:application/pdf;base64,")
    assert base64.b64decode(doc["document_url"].split(",", 1)[1]) == PDF


def test_the_question_and_the_document_are_both_present():
    """A request that lost the text part would score the model on "describe this document"."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    c = mdq.MistralDocQnAClient("k", model="m", client=_client(handler))
    c.generate(system="SYS", prompt="QUESTION-TEXT", pdf=PDF, temperature=0.0, top_p=1.0,
               max_tokens=64)
    body = seen["body"]
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    types = [p["type"] for p in body["messages"][1]["content"]]
    assert types == ["text", "document_url"]
    assert body["messages"][1]["content"][0]["text"] == "QUESTION-TEXT"
    assert body["response_format"] == {"type": "json_object"}


def test_key_travels_as_a_bearer_header_never_in_the_url():
    """httpx embeds the full URL in HTTPStatusError, so a query-param key prints the live secret
    into any non-2xx traceback. This repo has leaked a key that way before."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=ok_body())

    c = mdq.MistralDocQnAClient("SECRET-KEY", model="m", client=_client(handler))
    c.generate(system="s", prompt="p", pdf=PDF, temperature=0.0, top_p=1.0, max_tokens=8)
    assert seen["auth"] == "Bearer SECRET-KEY"
    assert "SECRET-KEY" not in seen["url"]


# ── condition honesty: no render happened ─────────────────────────────────────────────────────

def test_condition_drops_render_and_records_pdf_direct():
    """The provider rasterizes internally, so the run's pinned dpi describes nothing this row
    experienced. Keeping a `render` block would make the condition hash assert control over a
    resolution we never sent."""
    cond = _condition_for(entry(), STAGE1_CONDITION)
    assert "render" not in cond
    assert cond["input"] == DOCQNA_INPUT == "pdf-direct"
    assert "render" in STAGE1_CONDITION           # shared dict untouched


def test_docqna_condition_hash_is_distinct_from_every_other_transport():
    """Two transports sharing a hash would let `report` merge rows that saw different inputs."""
    dq = _condition_for(entry(), STAGE1_CONDITION)
    gem = _condition_for(
        RegistryEntry(id="g@google-native", shape="vlm-chat", transport="gemini-native",
                      model="gemini-3.6-flash", api_key_env="GEMINI_API_KEY",
                      precision="provider-default", weights_licence="closed",
                      provider_tos_commercial="ok", provenance="Google",
                      release_date="2026-07-21"), STAGE1_CONDITION)
    assert len({condition_hash(dq), condition_hash(gem),
                condition_hash(STAGE1_CONDITION)}) == 3


def test_other_transports_are_unaffected():
    bedrock = RegistryEntry(id="b@bedrock", shape="vlm-chat", transport="bedrock-converse",
                            model="m", region="us-east-1", precision="provider-default",
                            weights_licence="closed", provider_tos_commercial="ok",
                            provenance="X", release_date="2025-01-01")
    assert _condition_for(bedrock, STAGE1_CONDITION) == STAGE1_CONDITION


# ── response mapping ─────────────────────────────────────────────────────────────────────────

def test_usage_is_remapped_and_served_model_recorded():
    """`served_model` is the SERVER-reported id — the only way to catch an alias resolving somewhere
    other than where we pinned it. `mistral-small-latest`, `magistral-small-latest` and
    `mistral-vibe-cli-fast` all resolve to this same checkpoint today."""
    c = mdq.MistralDocQnAClient("k", model="mistral-small-2603",
                                client=_client(lambda r: httpx.Response(200, json=ok_body())))
    text, usage, provider = c.generate(system="s", prompt="p", pdf=PDF, temperature=0.0,
                                      top_p=1.0, max_tokens=64)
    assert text == '{"a": true}'
    assert usage["prompt_tokens"] == 2778 and usage["completion_tokens"] == 12
    assert usage["served_model"] == "mistral-small-2603"
    assert usage["finish_reason"] == "stop"
    assert provider == "mistral:api"


def test_omitted_token_fields_become_zero_not_none():
    """A provider may OMIT a token field rather than send 0. Gemini's omitted candidatesTokenCount
    killed a full-bank run ~300 paid cells in, because a dict default does not fire for a
    present-but-None key. Assume every new transport can do the same."""
    body = {"model": "m", "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100}}
    c = mdq.MistralDocQnAClient("k", model="m",
                                client=_client(lambda r: httpx.Response(200, json=body)))
    _t, usage, _p = c.generate(system="s", prompt="p", pdf=PDF, temperature=0.0, top_p=1.0,
                               max_tokens=8)
    assert usage["completion_tokens"] == 0 and usage["total_tokens"] == 0
    assert (usage["prompt_tokens"] / 1e6) + (usage["completion_tokens"] / 1e6) > 0   # used to raise


def test_empty_choices_yields_empty_text_not_a_crash():
    c = mdq.MistralDocQnAClient("k", model="m",
                                client=_client(lambda r: httpx.Response(200, json={"choices": []})))
    text, _u, _p = c.generate(system="s", prompt="p", pdf=PDF, temperature=0.0, top_p=1.0,
                              max_tokens=8)
    assert text == ""


# ── error taxonomy ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_raise_a_credential_error(status):
    c = mdq.MistralDocQnAClient("k", model="m", client=_client(
        lambda r: httpx.Response(status, json={"message": "Unauthorized"})))
    with pytest.raises(mdq.MistralCredentialError):
        c.generate(system="s", prompt="p", pdf=PDF, temperature=0.0, top_p=1.0, max_tokens=8)


def test_credential_error_is_never_retried():
    """A rejected key will be rejected on every attempt; at 1,356 cells that is slow waste."""
    assert _is_retryable(mdq.MistralCredentialError("nope")) is False


def test_server_error_stays_retryable():
    """Mistral's /v1/ocr returned two isolated 500s across 1,162 pages on 2026-08-06, and a re-run
    recovered both — so 5xx here is an observed transient, not a defensive guess."""
    assert mdq.is_retryable_status(500) is True
    assert mdq.is_retryable_status(429) is True
    assert mdq.is_retryable_status(400) is False


def test_missing_api_key_fails_before_any_request():
    with pytest.raises(mdq.MistralCredentialError):
        mdq.MistralDocQnAClient("", model="m")


# ── registry validation, both directions ─────────────────────────────────────────────────────

def test_registry_accepts_a_well_formed_entry():
    e = entry()
    assert e.transport == "mistral-docqna" and e.input_mode == "pdf-direct"


@pytest.mark.parametrize(("kw", "match"), [
    ({"model": None}, "requires model"),
    ({"api_key_env": None}, "requires api_key_env"),
    ({"region": "us-east-1"}, "takes no region"),
    ({"provider_pin": {"order": ["x"]}}, "provider_pin"),
    ({"input_mode": "raster-png"}, "must be 'pdf-direct'"),
])
def test_registry_rejects_malformed_entries(kw, match):
    """`input_mode: raster-png` is refused rather than corrected: it would attach the wrong caveat
    set in report.md and hide the embedded-text-layer free ride pdf-direct rows genuinely get."""
    with pytest.raises(ValueError, match=match):
        entry(**kw)
