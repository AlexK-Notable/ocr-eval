"""Bedrock transport tests. No network, no AWS credentials — `BedrockConverseClient` is patched
out and botocore exceptions are constructed directly, so these run in CI and on a machine that has
never been near an AWS account.

What is deliberately pinned here:
  * the registry validator's fail-closed directions (a bedrock entry cannot smuggle in an
    api_key_env / base_url / provider_pin, and cannot omit region);
  * that a bedrock row lands in the cache with the SAME shape an openai-compat row does — the
    whole point of the transport split is that nothing downstream changes;
  * the retry taxonomy, including that an UNKNOWN error code is treated as permanent;
  * that a seeded condition is refused rather than silently sent without the seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import fitz  # pymupdf
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from ocr_eval_ext import bedrock as bd
from ocr_eval_ext.config import RegistryEntry
from ocr_eval_ext.direct import (
    STAGE1_CONDITION,
    _is_retryable,
    _retry_wait,
    parser_key,
    run_direct,
)
from realdoc_bench.evaluate.runs import RunLayout


def bedrock_entry(**over) -> RegistryEntry:
    kw: dict = {
        "id": "nova-lite@bedrock", "shape": "vlm-chat", "transport": "bedrock-converse",
        "model": "amazon.nova-lite-v1:0", "region": "us-east-1",
        "precision": "provider-default", "weights_licence": "closed",
        "provider_tos_commercial": "ok", "provenance": "Amazon", "release_date": "2024-12-03",
    }
    kw.update(over)
    return RegistryEntry(**kw)


def client_error(code: str, *, headers: dict | None = None) -> ClientError:
    resp: dict = {"Error": {"Code": code, "Message": f"mock {code}"}}
    if headers is not None:
        resp["ResponseMetadata"] = {"HTTPHeaders": headers}
    return ClientError(resp, "Converse")


def make_run_dir(tmp_path: Path) -> RunLayout:
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Is the box checked?  [X] Yes  [ ] No")
    page.draw_rect(fitz.Rect(72, 100, 220, 160), fill=(0, 0, 0))   # ink_coverage above the floor
    doc.save(layout.docs_dir / "doc_1.pdf")
    layout.bank_path.write_text(json.dumps({"items": [{
        "question_id": "q1", "source_file": "doc_1", "domain": "test",
        "question": "Is question 1 checkbox marked?", "capabilities": ["checkbox_state"],
        "gold_dict": {"a": True},
        "response_format": "Return exactly: a=<boolean>", "gold_answer": "a=true",
    }]}))
    return layout


class FakeBedrockClient:
    """Stands in for `BedrockConverseClient`. `script` is a list of either exceptions (raised) or
    reply strings (returned); the last entry repeats, matching MockOpenAI's convention."""

    def __init__(self, script, *, region="us-east-1"):
        self.script = script
        self.calls: list[dict] = []
        self._region = region

    @property
    def resolved_provider(self) -> str:
        return f"bedrock:{self._region}"

    def converse(self, *, system, prompt, png, temperature, top_p, max_tokens):
        i = min(len(self.calls), len(self.script) - 1)
        self.calls.append({"system": system, "prompt": prompt, "png": png,
                           "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens})
        item = self.script[i]
        if isinstance(item, BaseException):
            raise item
        return (item,
                {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                 "bedrock_usage": {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110},
                 "stop_reason": "end_turn"},
                self.resolved_provider)


@pytest.fixture
def patch_client(monkeypatch):
    """Returns a setter: `patch_client(script) -> FakeBedrockClient` already installed."""
    def _install(script):
        fake = FakeBedrockClient(script)
        monkeypatch.setattr(bd, "BedrockConverseClient", lambda **_kw: fake)
        return fake
    return _install


# ── registry validator: fail-closed directions ───────────────────────────────────────────────

def test_bedrock_entry_requires_region():
    with pytest.raises(ValueError, match="requires an explicit region"):
        bedrock_entry(region=None)


def test_bedrock_entry_requires_model():
    with pytest.raises(ValueError, match="requires model"):
        bedrock_entry(model=None)


def test_bedrock_entry_rejects_api_key_env():
    """Auth is SigV4 — an api_key_env would be silently ignored, implying a credential source that
    is never consulted."""
    with pytest.raises(ValueError, match="AWS credential chain"):
        bedrock_entry(api_key_env="AWS_SECRET_ACCESS_KEY")


def test_bedrock_entry_rejects_base_url():
    with pytest.raises(ValueError, match="takes no base_url"):
        bedrock_entry(base_url="https://bedrock-runtime.us-east-1.amazonaws.com")


def test_bedrock_entry_rejects_provider_pin():
    """provider_pin is OpenRouter routing; on Bedrock serving identity is (modelId, region)."""
    with pytest.raises(ValueError, match="no meaning"):
        bedrock_entry(provider_pin={"order": ["Amazon"]})


def test_committed_bedrock_registry_loads():
    from ocr_eval_ext.config import load_registry
    entries = load_registry(Path("configs/registry-bedrock.yaml"))
    assert entries, "registry-bedrock.yaml is empty"
    assert all(e.transport == "bedrock-converse" for e in entries)
    assert all(e.region and e.model for e in entries)
    # Unpriced by design — --max-spend must fail closed rather than assume $0 (see the file header).
    assert all(e.pricing is None for e in entries)


# ── retry taxonomy ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code", sorted(bd.RETRYABLE_ERROR_CODES))
def test_retryable_codes_are_retryable(code):
    assert _is_retryable(client_error(code)) is True


@pytest.mark.parametrize("code", sorted(bd.PERMANENT_ERROR_CODES))
def test_permanent_codes_are_not_retryable(code):
    assert _is_retryable(client_error(code)) is False


def test_unknown_error_code_is_permanent():
    """Deliberate direction: an unenumerated code costs one honest error row, whereas treating it
    as retryable would quadruple a misconfigured full-bank run's wall-clock and spend."""
    assert _is_retryable(client_error("SomeBrandNewException")) is False


def test_connection_level_botocore_error_is_retryable():
    """No HTTP response obtained — nothing established about permanence, same reasoning as
    APIConnectionError on the openai path."""
    exc = EndpointConnectionError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com")
    assert _is_retryable(exc) is True


def test_retry_wait_honors_bedrock_retry_after_header():
    """botocore's `.response` is a dict, so `_retry_wait`'s `.headers` attribute probe finds
    nothing — the Bedrock branch must read ResponseMetadata.HTTPHeaders instead."""
    exc = client_error("ThrottlingException", headers={"retry-after": "7"})
    assert _retry_wait(exc, 0) == pytest.approx(7.0)


def test_retry_wait_clamps_hostile_bedrock_header():
    exc = client_error("ThrottlingException", headers={"retry-after": "9000"})
    assert _retry_wait(exc, 0) <= 60.0


def test_retry_wait_falls_back_to_backoff_without_header():
    exc = client_error("ThrottlingException")
    assert _retry_wait(exc, 1) == pytest.approx(4.0)   # BACKOFF_BASE_SEC * 2**1


# ── Anthropic-on-Bedrock temperature/top_p exclusivity ───────────────────────────────────────
# Verified live 2026-08-04: ONLY Anthropic models reject the pair; Nova Lite/Pro, Gemma 3 27B,
# Mistral Large 3 and Kimi K2.5 all accept both in one request.

@pytest.mark.parametrize("model_id", [
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
])
def test_anthropic_models_omit_identity_top_p(model_id):
    cfg, note = bd.narrow_sampling_for_model(model_id, temperature=0.0, top_p=1.0)
    assert cfg == {"temperature": 0.0}, "top_p must not be sent alongside temperature"
    assert "identity" in note


@pytest.mark.parametrize("model_id", [
    "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0", "google.gemma-3-27b-it",
    "mistral.mistral-large-3-675b-instruct", "moonshotai.kimi-k2.5",
])
def test_non_anthropic_models_send_both(model_id):
    cfg, note = bd.narrow_sampling_for_model(model_id, temperature=0.0, top_p=1.0)
    assert cfg == {"temperature": 0.0, "topP": 1.0}
    assert note == "", "no narrowing happened, so nothing should be disclosed"


def test_anthropic_non_identity_top_p_raises_rather_than_dropping():
    """Dropping a real top_p would silently change the measurement while the condition hash kept
    claiming it was pinned. Refuse instead."""
    with pytest.raises(ValueError, match="not the identity value"):
        bd.narrow_sampling_for_model("us.anthropic.claude-haiku-4-5-20251001-v1:0",
                                     temperature=0.0, top_p=0.9)


def test_sampling_note_is_recorded_on_the_row(tmp_path, monkeypatch):
    """A narrowed request must be visible in the cache row, not silently equivalent to an
    un-narrowed one."""
    captured: dict = {}

    class AnthropicShapedClient(FakeBedrockClient):
        def converse(self, **kw):
            cfg, note = bd.narrow_sampling_for_model(
                "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                temperature=kw["temperature"], top_p=kw["top_p"])
            captured.update(cfg)
            text, usage, provider = super().converse(**kw)
            if note:
                usage["sampling_note"] = note
            return text, usage, provider

    fake = AnthropicShapedClient(['{"a": true}'])
    monkeypatch.setattr(bd, "BedrockConverseClient", lambda **_kw: fake)
    layout = make_run_dir(tmp_path)
    e = bedrock_entry(id="claude-haiku-4.5@bedrock",
                      model="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    run_direct(layout, [e], workers=1)
    assert "topP" not in captured
    row = json.loads(layout.cache_path(
        "q1", parser_key("claude-haiku-4.5@bedrock", STAGE1_CONDITION)).read_text())
    assert "sampling_note" in row["usage"]


def test_stage1_condition_top_p_is_the_identity_value():
    """The narrowing above is only sound because STAGE1_CONDITION pins top_p=1.0. If that ever
    changes, Anthropic-on-Bedrock entries must fail loudly (previous test) rather than quietly
    measure something else — this pins the premise itself."""
    assert STAGE1_CONDITION["sampling"]["top_p"] == 1.0


# ── run_direct integration ───────────────────────────────────────────────────────────────────

def test_bedrock_row_matches_openai_row_shape(tmp_path, patch_client):
    """The contract that makes this a transport and not a fork: a bedrock cache row carries every
    field downstream code reads, with resolved_provider pinning (modelId, region)."""
    layout = make_run_dir(tmp_path)
    patch_client(['{"a": true}'])
    summary = run_direct(layout, [bedrock_entry()], workers=1)
    assert summary["ok"] == 1 and summary["error"] == 0

    row = json.loads(layout.cache_path(
        "q1", parser_key("nova-lite@bedrock", STAGE1_CONDITION)).read_text())
    assert row["error_class"] == "none"
    assert row["answer"] == {"a": True}
    assert row["match"] is True
    assert row["resolved_provider"] == "bedrock:us-east-1"
    # usage remapped onto the names track() reads, native counts preserved alongside
    assert row["usage"]["prompt_tokens"] == 100
    assert row["usage"]["bedrock_usage"]["inputTokens"] == 100
    for field in ("qid", "parser", "source_file", "condition", "prompt_sha",
                  "image_sha", "image_px", "image_bytes", "latency_sec"):
        assert field in row, f"missing {field} — downstream readers expect it"


def test_bedrock_sends_raw_png_bytes_not_base64(tmp_path, patch_client):
    """Converse takes raw bytes; a base64 str is accepted as bytes and silently degrades the
    image, which would look like a model quality problem rather than a harness bug."""
    layout = make_run_dir(tmp_path)
    fake = patch_client(['{"a": true}'])
    run_direct(layout, [bedrock_entry()], workers=1)
    png = fake.calls[0]["png"]
    assert isinstance(png, bytes)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_bedrock_no_image_sends_none(tmp_path, patch_client):
    """The language-prior control must send no image part at all."""
    layout = make_run_dir(tmp_path)
    fake = patch_client(['{"a": true}'])
    run_direct(layout, [bedrock_entry()], workers=1, no_image=True)
    assert fake.calls[0]["png"] is None


def test_bedrock_markdown_fenced_json_is_parsed(tmp_path, patch_client):
    """Every Bedrock model probed live wrapped its JSON in a ```json fence — `_extract_json`'s
    brace-scan handles it, and this pins that it keeps doing so."""
    layout = make_run_dir(tmp_path)
    patch_client(['```json\n{"a": true}\n```'])
    summary = run_direct(layout, [bedrock_entry()], workers=1)
    assert summary["ok"] == 1
    row = json.loads(layout.cache_path(
        "q1", parser_key("nova-lite@bedrock", STAGE1_CONDITION)).read_text())
    assert row["answer"] == {"a": True} and row["match"] is True


def test_bedrock_throttling_is_retried_then_succeeds(tmp_path, patch_client, monkeypatch):
    monkeypatch.setattr("ocr_eval_ext.direct.time.sleep", lambda s: None)
    layout = make_run_dir(tmp_path)
    fake = patch_client([client_error("ThrottlingException"), '{"a": true}'])
    summary = run_direct(layout, [bedrock_entry()], workers=1)
    assert summary["ok"] == 1
    assert len(fake.calls) == 2


def test_bedrock_access_denied_fails_once_no_retry(tmp_path, patch_client, monkeypatch):
    """AccessDeniedException is the expected symptom of a listed-but-not-enabled model. Retrying
    it 4x per cell across 1,356 cells is pure waste."""
    monkeypatch.setattr("ocr_eval_ext.direct.time.sleep", lambda s: None)
    layout = make_run_dir(tmp_path)
    fake = patch_client([client_error("AccessDeniedException")])
    summary = run_direct(layout, [bedrock_entry()], workers=1)
    assert summary["error"] == 1 and summary["ok"] == 0
    assert len(fake.calls) == 1, "permanent error must not consume the retry budget"
    row = json.loads(layout.cache_path(
        "q1", parser_key("nova-lite@bedrock", STAGE1_CONDITION)).read_text())
    assert row["error_class"] == "api_error"
    assert "AccessDeniedException" in row["error"]


def test_bedrock_rejects_seeded_condition(tmp_path, patch_client):
    """Converse has no seed parameter. Dropping it silently would let the condition hash assert a
    reproducibility guarantee the wire never provided."""
    layout = make_run_dir(tmp_path)
    patch_client(['{"a": true}'])
    seeded = {**STAGE1_CONDITION,
              "sampling": {**STAGE1_CONDITION["sampling"], "seed": 42}}
    with pytest.raises(ValueError, match=r"cannot honor sampling\.seed"):
        run_direct(layout, [bedrock_entry()], condition=seeded, workers=1)


def test_bedrock_unpriced_cell_fails_closed_under_max_spend(tmp_path, patch_client):
    """Converse returns no usage.cost. With no registry pricing and an active cap, track() must
    raise rather than treat an unknowable cost as free."""
    layout = make_run_dir(tmp_path)
    patch_client(['{"a": true}'])
    with pytest.raises(RuntimeError, match="cannot enforce --max-spend"):
        run_direct(layout, [bedrock_entry()], workers=1, max_spend_usd=5.0)


def test_bedrock_priced_entry_tracks_spend_from_registry(tmp_path, patch_client):
    """With verified pricing added, the existing token-based fallback works unmodified."""
    layout = make_run_dir(tmp_path)
    patch_client(['{"a": true}'])
    e = bedrock_entry(pricing={"input_per_mtok": 1.0, "output_per_mtok": 2.0})
    summary = run_direct(layout, [e], workers=1, max_spend_usd=5.0)
    assert summary["ok"] == 1


def test_bedrock_dry_run_counts_cells_as_unpriced(tmp_path, patch_client):
    layout = make_run_dir(tmp_path)
    patch_client(['{"a": true}'])
    out = run_direct(layout, [bedrock_entry()], workers=1, dry_run=True)
    assert out["cells"] == 1
    assert out["unpriced_cells"] == 1
    assert out["estimated_usd"] == 0.0     # excluded, never folded in as $0


def test_bedrock_cache_hit_makes_no_second_call(tmp_path, patch_client):
    layout = make_run_dir(tmp_path)
    fake = patch_client(['{"a": true}'])
    run_direct(layout, [bedrock_entry()], workers=1)
    assert len(fake.calls) == 1
    summary = run_direct(layout, [bedrock_entry()], workers=1)
    assert summary == {"ok": 0, "error": 0, "cached": 1, "cached_ok": 1, "cached_error": 0}
    assert len(fake.calls) == 1, "cached cell must never reach the client"


def test_condition_hash_is_transport_independent(tmp_path):
    """A bedrock row and an openai row under the same condition share a condition hash — the
    condition describes the measurement, not the wire protocol. (The parser key still differs,
    since it embeds the registry id.)"""
    from ocr_eval_ext.direct import condition_hash
    assert condition_hash(STAGE1_CONDITION) == condition_hash(dict(STAGE1_CONDITION))
    assert parser_key("nova-lite@bedrock", STAGE1_CONDITION) != \
        parser_key("qwen3-vl-8b@openrouter", STAGE1_CONDITION)


def test_run_direct_rejects_unknown_transport(tmp_path):
    layout = make_run_dir(tmp_path)
    e = RegistryEntry(
        id="x@up", shape="vlm-chat", transport="upstream-parser", upstream_parser="gemini_3_5_flash",
        precision="provider-default", weights_licence="closed", provider_tos_commercial="ok",
        provenance="Google", release_date="2026-05-19")
    with pytest.raises(ValueError, match="supports transport="):
        run_direct(layout, [e], workers=1)
