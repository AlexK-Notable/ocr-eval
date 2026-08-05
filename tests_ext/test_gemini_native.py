"""Tests for the `gemini-native` transport (`ocr_eval_ext/gemini_native.py`).

The transport exists for ONE reason — to reach `mediaResolution`, which Google's OpenAI-compat shim
does not expose — so most of what is pinned here is about that parameter genuinely reaching the wire
and being recorded, plus the fail-closed directions around it. No network: `httpx.MockTransport`
serves every response.
"""
from __future__ import annotations

import base64
import json

import httpx
import pytest

from ocr_eval_ext import gemini_native as gn
from ocr_eval_ext.config import RegistryEntry
from ocr_eval_ext.direct import (
    GEMINI_MEDIA_RESOLUTION,
    STAGE1_CONDITION,
    _condition_for,
    _is_retryable,
    condition_hash,
    parser_key,
)

PNG = b"\x89PNG\r\n\x1a\nFAKE-IMAGE-BYTES"


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def ok_body(text: str = '{"a": true}', **usage) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1161, "candidatesTokenCount": 9,
                          "totalTokenCount": 1170, **usage},
    }


def entry(model: str = "gemini-2.5-flash", **kw) -> RegistryEntry:
    base = dict(id=f"{model}@google-native", shape="vlm-chat", transport="gemini-native",
                model=model, api_key_env="GEMINI_API_KEY", precision="provider-default",
                weights_licence="closed", provider_tos_commercial="ok",
                provenance="Google", release_date="2025-04-17")
    base.update(kw)
    return RegistryEntry(**base)


# ── the whole reason this transport exists: mediaResolution on the wire ───────────────────────

def test_media_resolution_is_sent_in_generation_config():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    c = gn.GeminiNativeClient("k", model="gemini-2.5-flash", client=_client(handler))
    c.generate(system="sys", prompt="p", png=PNG, temperature=0.0, top_p=1.0, max_tokens=64,
               media_resolution="MEDIA_RESOLUTION_HIGH")
    assert seen["body"]["generationConfig"]["mediaResolution"] == "MEDIA_RESOLUTION_HIGH"


def test_media_resolution_is_recorded_in_usage():
    """A row must say what it was actually served at — otherwise the comparison's central control is
    invisible in the data and only inferable from code at the time of the run."""
    c = gn.GeminiNativeClient("k", model="m",
                              client=_client(lambda r: httpx.Response(200, json=ok_body())))
    _text, usage, _prov = c.generate(system="s", prompt="p", png=PNG, temperature=0.0, top_p=1.0,
                                     max_tokens=64, media_resolution="MEDIA_RESOLUTION_MEDIUM")
    assert usage["media_resolution"] == "MEDIA_RESOLUTION_MEDIUM"


@pytest.mark.parametrize("bad", [None, "MEDIA_RESOLUTION_UNSPECIFIED", "HIGH", "high", "", 3])
def test_unspecified_or_bogus_media_resolution_is_refused(bad):
    """UNSPECIFIED and None are rejected as hard as a typo. Both mean "let the model choose", which
    silently restores the per-generation default gap (3.x HIGH ~1161 image tokens vs 2.5 MEDIUM
    ~317) — while the condition hash would still claim a pinned condition. That is the one failure
    mode this transport exists to prevent, so it must not be reachable by omission."""
    with pytest.raises(ValueError, match="media_resolution must be one of"):
        gn.validate_media_resolution(bad)


def test_client_refuses_a_call_with_an_invalid_media_resolution():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=ok_body())

    c = gn.GeminiNativeClient("k", model="m", client=_client(handler))
    with pytest.raises(ValueError):
        c.generate(system="s", prompt="p", png=PNG, temperature=0.0, top_p=1.0, max_tokens=64,
                   media_resolution="MEDIA_RESOLUTION_UNSPECIFIED")
    assert calls["n"] == 0          # never reached the wire


# ── request shape ────────────────────────────────────────────────────────────────────────────

def test_image_is_sent_as_base64_inline_data():
    """Native `inlineData.data` takes BASE64 — the opposite of Bedrock Converse, which takes raw
    bytes and silently degrades a base64 string handed to it as bytes. Pinned so the two transports
    can never be "fixed" into agreement by mistake."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    c = gn.GeminiNativeClient("k", model="m", client=_client(handler))
    c.generate(system="s", prompt="p", png=PNG, temperature=0.0, top_p=1.0, max_tokens=64,
               media_resolution=GEMINI_MEDIA_RESOLUTION)
    parts = seen["body"]["contents"][0]["parts"]
    inline = next(p["inlineData"] for p in parts if "inlineData" in p)
    assert inline["data"] == base64.b64encode(PNG).decode()
    assert base64.b64decode(inline["data"]) == PNG          # round-trips to the original bytes
    assert inline["mimeType"] == "image/png"


def test_no_image_sends_no_inline_data_part():
    """The `no_image` language-prior control must send NO image part at all, not an empty one."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    c = gn.GeminiNativeClient("k", model="m", client=_client(handler))
    c.generate(system="s", prompt="p", png=None, temperature=0.0, top_p=1.0, max_tokens=64,
               media_resolution=GEMINI_MEDIA_RESOLUTION)
    parts = seen["body"]["contents"][0]["parts"]
    assert not any("inlineData" in p for p in parts)


def test_sampling_and_system_reach_the_wire():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    c = gn.GeminiNativeClient("k", model="m", client=_client(handler))
    c.generate(system="SYSTEM-TEXT", prompt="p", png=PNG, temperature=0.25, top_p=0.9,
               max_tokens=777, media_resolution=GEMINI_MEDIA_RESOLUTION)
    gc = seen["body"]["generationConfig"]
    assert gc["temperature"] == 0.25 and gc["topP"] == 0.9 and gc["maxOutputTokens"] == 777
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "SYSTEM-TEXT"


def test_key_travels_as_header_never_in_url():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["hdr"] = request.headers.get("x-goog-api-key")
        return httpx.Response(200, json=ok_body())

    c = gn.GeminiNativeClient("SECRET-KEY", model="m", client=_client(handler))
    c.generate(system="s", prompt="p", png=None, temperature=0.0, top_p=1.0, max_tokens=8,
               media_resolution=GEMINI_MEDIA_RESOLUTION)
    assert seen["hdr"] == "SECRET-KEY"
    assert "SECRET-KEY" not in seen["url"]


# ── response mapping ─────────────────────────────────────────────────────────────────────────

def test_usage_is_remapped_to_openai_field_names_and_keeps_native_shape():
    """Downstream metrics/report code reads `prompt_tokens`/`completion_tokens` for every transport;
    the native shape is kept alongside rather than discarded."""
    c = gn.GeminiNativeClient("k", model="m",
                              client=_client(lambda r: httpx.Response(200, json=ok_body())))
    _t, usage, provider = c.generate(system="s", prompt="p", png=PNG, temperature=0.0, top_p=1.0,
                                     max_tokens=64, media_resolution=GEMINI_MEDIA_RESOLUTION)
    assert usage["prompt_tokens"] == 1161
    assert usage["completion_tokens"] == 9
    assert usage["gemini_usage"]["totalTokenCount"] == 1170
    assert usage["finish_reason"] == "STOP"
    assert provider == "google:generativelanguage"


def test_omitted_candidates_token_count_becomes_zero_not_none():
    """Google OMITS `candidatesTokenCount` entirely when the completion is empty — verified on a real
    cell, whose usageMetadata was `{promptTokenCount: 1928, totalTokenCount: 1928}` with no
    candidates field. Mapping that to None killed a full-bank run ~300 cells in, because
    `run_direct`'s cost tracker does `u.get("completion_tokens", 0) / 1e6` and a dict default does
    NOT fire for a present-but-None key: `TypeError: unsupported operand type(s) for /: 'NoneType'
    and 'float'`. Zero is also the honest value — no completion tokens were produced."""
    body = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"promptTokenCount": 1928, "totalTokenCount": 1928}}
    c = gn.GeminiNativeClient("k", model="m",
                              client=_client(lambda r: httpx.Response(200, json=body)))
    _t, usage, _p = c.generate(system="s", prompt="p", png=PNG, temperature=0.0, top_p=1.0,
                               max_tokens=64, media_resolution=GEMINI_MEDIA_RESOLUTION)
    assert usage["completion_tokens"] == 0
    assert usage["prompt_tokens"] == 1928
    # arithmetic that previously raised must now work
    assert (usage["prompt_tokens"] / 1e6) + (usage["completion_tokens"] / 1e6) > 0


def test_entirely_absent_usage_metadata_is_all_zeros():
    body = {"candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}]}
    c = gn.GeminiNativeClient("k", model="m",
                              client=_client(lambda r: httpx.Response(200, json=body)))
    _t, usage, _p = c.generate(system="s", prompt="p", png=None, temperature=0.0, top_p=1.0,
                               max_tokens=8, media_resolution=GEMINI_MEDIA_RESOLUTION)
    assert usage["prompt_tokens"] == 0 and usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0


def test_multi_part_text_is_joined():
    body = {"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]},
                            "finishReason": "STOP"}], "usageMetadata": {}}
    c = gn.GeminiNativeClient("k", model="m",
                              client=_client(lambda r: httpx.Response(200, json=body)))
    text, _u, _p = c.generate(system="s", prompt="p", png=None, temperature=0.0, top_p=1.0,
                              max_tokens=8, media_resolution=GEMINI_MEDIA_RESOLUTION)
    assert text == "a\nb"


def test_empty_candidates_yields_empty_text_not_a_crash():
    """A blocked/empty response is a row (`error_class: "empty"` upstream in `_one`), never an
    exception that kills the cell's neighbours."""
    c = gn.GeminiNativeClient("k", model="m",
                              client=_client(lambda r: httpx.Response(200, json={"candidates": []})))
    text, _u, _p = c.generate(system="s", prompt="p", png=None, temperature=0.0, top_p=1.0,
                              max_tokens=8, media_resolution=GEMINI_MEDIA_RESOLUTION)
    assert text == ""


# ── error taxonomy ───────────────────────────────────────────────────────────────────────────

def _err(status: int, reason: str | None = None, msg: str = "boom") -> httpx.Response:
    err: dict = {"code": status, "message": msg, "status": "INVALID_ARGUMENT"}
    if reason:
        err["details"] = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason}]
    return httpx.Response(status, json={"error": err})


def test_invalid_key_400_raises_credential_error_not_a_status_error():
    """Google returns an INVALID key as 400 (a MISSING key gives 403) — verified live 2026-08-05 by
    sending a UUID. Surfacing that as a bare 400 reads as "our request was malformed" and sends the
    reader to the wrong place; it already cost one debugging cycle elsewhere in this repo."""
    c = gn.GeminiNativeClient("k", model="m",
                              client=_client(lambda r: _err(400, "API_KEY_INVALID")))
    with pytest.raises(gn.GeminiCredentialError, match="API_KEY_INVALID"):
        c.generate(system="s", prompt="p", png=None, temperature=0.0, top_p=1.0, max_tokens=8,
                   media_resolution=GEMINI_MEDIA_RESOLUTION)


def test_credential_error_is_never_retried():
    """A rejected key will be rejected on all 4 attempts; at 1,356 cells that is slow waste."""
    assert _is_retryable(gn.GeminiCredentialError("nope")) is False


def test_non_credential_400_stays_a_status_error_with_the_body():
    c = gn.GeminiNativeClient("k", model="m",
                              client=_client(lambda r: _err(400, None, "Unknown name 'xyz'")))
    with pytest.raises(httpx.HTTPStatusError, match="Unknown name"):
        c.generate(system="s", prompt="p", png=None, temperature=0.0, top_p=1.0, max_tokens=8,
                   media_resolution=GEMINI_MEDIA_RESOLUTION)


@pytest.mark.parametrize(("status", "retryable"), [
    (408, True), (429, True), (500, True), (503, True), (529, True),
    (400, False), (403, False), (404, False), (422, False),
])
def test_status_retry_taxonomy(status, retryable):
    assert gn.is_retryable_status(status) is retryable


def test_direct_classifies_gemini_http_and_transport_errors():
    """`direct.py` owns the retry loop for every transport; httpx errors match none of the openai
    SDK branches, so they need their own classification or they would fall through to "permanent"."""
    req = httpx.Request("POST", "https://example.invalid")
    resp429 = httpx.Response(429, request=req)
    resp400 = httpx.Response(400, request=req)
    assert _is_retryable(httpx.HTTPStatusError("x", request=req, response=resp429)) is True
    assert _is_retryable(httpx.HTTPStatusError("x", request=req, response=resp400)) is False
    assert _is_retryable(httpx.ConnectError("refused", request=req)) is True
    assert _is_retryable(httpx.ReadTimeout("slow", request=req)) is True


def test_missing_api_key_is_a_credential_error_before_any_request():
    with pytest.raises(gn.GeminiCredentialError):
        gn.GeminiNativeClient("", model="m")


# ── condition-hash isolation: the cache-orphaning trap ───────────────────────────────────────

def test_gemini_condition_gets_its_own_hash_and_leaves_others_untouched():
    """`media_resolution` is folded in per-transport, NOT added to STAGE1_CONDITION. Adding a key to
    the shared dict would change `condition_hash` for every row on every transport at once and
    orphan the whole existing cache for a parameter three quarters of it cannot use."""
    gem = _condition_for(entry(), STAGE1_CONDITION)
    other = _condition_for(
        RegistryEntry(id="b@bedrock", shape="vlm-chat", transport="bedrock-converse",
                      model="m", region="us-east-1", precision="provider-default",
                      weights_licence="closed", provider_tos_commercial="ok",
                      provenance="X", release_date="2025-01-01"),
        STAGE1_CONDITION)
    assert gem["media_resolution"] == GEMINI_MEDIA_RESOLUTION
    assert "media_resolution" not in other              # untouched for every other transport
    assert other == STAGE1_CONDITION                     # ...byte-identical, not merely equivalent
    assert condition_hash(gem) != condition_hash(STAGE1_CONDITION)


def test_media_resolution_change_produces_a_distinct_parser_key():
    """A different image budget is a different experimental condition and must be a different,
    trackable row — never a silent overwrite of the previous one."""
    high = {**STAGE1_CONDITION, "media_resolution": "MEDIA_RESOLUTION_HIGH"}
    low = {**STAGE1_CONDITION, "media_resolution": "MEDIA_RESOLUTION_LOW"}
    assert parser_key("g@google-native", high) != parser_key("g@google-native", low)


# ── registry validation, both directions ─────────────────────────────────────────────────────

def test_registry_accepts_a_well_formed_gemini_native_entry():
    e = entry()
    assert e.transport == "gemini-native" and e.model == "gemini-2.5-flash"


@pytest.mark.parametrize(("kw", "match"), [
    ({"model": None}, "requires model"),
    ({"api_key_env": None}, "requires api_key_env"),
    ({"base_url": "https://example.invalid/v1"}, "takes no base_url"),
    ({"region": "us-east-1"}, "takes no region"),
    ({"provider_pin": {"order": ["x"]}}, "provider_pin"),
])
def test_registry_rejects_malformed_gemini_native_entries(kw, match):
    with pytest.raises(ValueError, match=match):
        entry(**kw)
