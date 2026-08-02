from __future__ import annotations

import fitz  # pymupdf
import pytest

from ocr_eval_ext.config import RegistryEntry
from ocr_eval_ext.direct import condition_hash
from ocr_eval_ext.parsers_openai import (
    TRANSCRIBER_CONDITION,
    OpenAICompatVisionParser,
    preflight,
    register_openai_parsers,
    safe_name,
)
from tests_ext.mock_openai import MockOpenAI

_SUFFIX = condition_hash(TRANSCRIBER_CONDITION)


def _write_registry(path, entries: list[dict]) -> None:
    import yaml
    path.write_text(yaml.safe_dump(entries))


def test_safe_name():
    assert safe_name("glm-ocr@local-vllm") == "glm-ocr_local-vllm"


def test_call_page_uses_markdown_prompt_and_image(tmp_path):
    pdf = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    doc.save(pdf)
    with MockOpenAI(reply_text="## Page\n**Box:** ☒ Yes") as mock:
        p = OpenAICompatVisionParser(base_url=mock.base_url, model="org/m")
        result = p.parse(pdf)
        assert "☒" in result.markdown
        req = mock.requests[0]
        text_parts = [c["text"] for c in req["messages"][-1]["content"] if c.get("type") == "text"]
        assert any("☒ (checked)" in t for t in text_parts)     # upstream prompt in use
        assert any(c.get("type") == "image_url" for c in req["messages"][-1]["content"])
        assert req["temperature"] == 0.0
        assert len(mock.requests) == 1   # single page, no spurious retries on a clean 200


def test_register_from_registry(tmp_path):
    registry_yaml = tmp_path / "r.yaml"
    _write_registry(registry_yaml, [{
        "id": "t1@local", "shape": "transcriber", "transport": "openai-compat",
        "base_url": "http://localhost:9/v1", "model": "org/t1", "precision": "bf16",
        "weights_licence": "mit", "provider_tos_commercial": "ok",
        "provenance": "X", "release_date": "2025-01-01", "local": True,
    }])
    names = register_openai_parsers(registry_yaml)
    assert len(names) == 1
    assert names[0].startswith("t1_local__")            # safe_name + "__" + condition hash
    assert names[0] == f"t1_local__{_SUFFIX}"

    from realdoc_bench.evaluate.parsers.base import build
    assert build(names[0]).version == "org/t1"


def test_register_openai_parsers_is_idempotent(tmp_path):
    registry_yaml = tmp_path / "r.yaml"
    _write_registry(registry_yaml, [{
        "id": "t2@local", "shape": "transcriber", "transport": "openai-compat",
        "base_url": "http://localhost:9/v1", "model": "org/t2", "precision": "bf16",
        "weights_licence": "mit", "provider_tos_commercial": "ok",
        "provenance": "X", "release_date": "2025-01-01", "local": True,
    }])
    first = register_openai_parsers(registry_yaml)
    second = register_openai_parsers(registry_yaml)   # second call, same process: no re-register
    assert first == second == [f"t2_local__{_SUFFIX}"]


def test_register_openai_parsers_skips_non_transcriber_and_non_openai_compat(tmp_path):
    registry_yaml = tmp_path / "r.yaml"
    _write_registry(registry_yaml, [
        {
            "id": "vlm@openrouter", "shape": "vlm-chat", "transport": "openai-compat",
            "base_url": "https://openrouter.ai/api/v1", "model": "org/v",
            "api_key_env": "SOME_KEY", "precision": "provider-default",
            "weights_licence": "mit", "provider_tos_commercial": "ok",
            "provenance": "X", "release_date": "2025-01-01",
        },
        {
            "id": "upstream@x", "shape": "transcriber", "transport": "upstream-parser",
            "upstream_parser": "whatever", "precision": "bf16", "weights_licence": "mit",
            "provider_tos_commercial": "ok", "provenance": "X", "release_date": "2025-01-01",
        },
    ])
    assert register_openai_parsers(registry_yaml) == []


def _entry(base_url: str, model: str) -> RegistryEntry:
    return RegistryEntry(
        id="t@local", shape="transcriber", transport="openai-compat",
        base_url=base_url, model=model, precision="bf16", weights_licence="mit",
        provider_tos_commercial="ok", provenance="X", release_date="2025-01-01", local=True,
    )


def test_preflight_matches():
    with MockOpenAI(models=["org/served-model"]) as mock:
        served = preflight(_entry(mock.base_url, "org/served-model"))
        assert served == "org/served-model"


def test_preflight_rejects_mismatch():
    with MockOpenAI(models=["org/other-model"]) as mock, \
         pytest.raises(RuntimeError, match="do not include"):
        preflight(_entry(mock.base_url, "org/served-model"))


def test_call_page_retries_429_then_succeeds(tmp_path):
    pdf = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    doc.save(pdf)
    responses = [{"status": 429, "headers": {"Retry-After": "0"}}, {"status": 200}]
    with MockOpenAI(responses=responses) as mock:
        p = OpenAICompatVisionParser(base_url=mock.base_url, model="org/m")
        result = p.parse(pdf)
    assert len(mock.requests) == 2
    assert '{"a": true}' in result.markdown   # MockOpenAI's default reply_text


def test_call_page_does_not_retry_permanent_400(tmp_path):
    from openai import APIStatusError

    pdf = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    doc.save(pdf)
    responses = [{"status": 400, "message": "bad request"}]
    with MockOpenAI(responses=responses) as mock:
        p = OpenAICompatVisionParser(base_url=mock.base_url, model="org/m")
        with pytest.raises(APIStatusError):
            p.parse(pdf)
    assert len(mock.requests) == 1   # permanent 4xx — no retry burned on it
