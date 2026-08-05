from pathlib import Path

import pydantic
import pytest

from ocr_eval_ext.config import CONTAMINATION_CUTOFF, RegistryEntry, get_entry, load_registry

MINIMAL = """
- id: m1@host
  shape: vlm-chat
  transport: openai-compat
  base_url: https://example.com/v1
  model: org/m1
  api_key_env: EXAMPLE_KEY
  precision: provider-default
  weights_licence: apache-2.0
  provider_tos_commercial: ok
  provenance: Example Org
  release_date: "2026-01-01"
"""


def test_load_minimal(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(MINIMAL)
    entries = load_registry(p)
    assert entries[0].id == "m1@host"
    assert entries[0].local is False


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(MINIMAL + MINIMAL)
    with pytest.raises(ValueError, match="duplicate"):
        load_registry(p)


def test_transport_field_consistency(tmp_path):
    bad = MINIMAL.replace("transport: openai-compat", "transport: upstream-parser")
    p = tmp_path / "r.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError, match="upstream_parser"):
        load_registry(p)


def test_contamination_flag_logic():
    e = RegistryEntry(
        id="x@y", shape="vlm-chat", transport="openai-compat",
        base_url="u", model="m", precision="bf16", weights_licence="mit",
        provider_tos_commercial="ok", provenance="p", release_date="2026-07-01",
    )
    assert e.release_date > CONTAMINATION_CUTOFF   # ISO strings compare correctly


def test_get_entry_unknown_raises():
    with pytest.raises(KeyError):
        get_entry([], "nope")


def test_unknown_field_rejected(tmp_path):
    """Typo'd YAML keys (e.g. promptible/loacl/weights_license) must not validate silently."""
    bad = MINIMAL + "  promptible: false\n"
    p = tmp_path / "r.yaml"
    p.write_text(bad)
    with pytest.raises(pydantic.ValidationError):
        load_registry(p)


def test_registry_yaml_parses_and_satisfies_dod_categories():
    """configs/registry.yaml: unique ids + Stage 1 DoD category coverage."""
    entries = load_registry(Path(__file__).parent.parent / "configs" / "registry.yaml")

    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids))

    hosted_vlm_chat = [e for e in entries if e.shape == "vlm-chat" and not e.local]
    assert len(hosted_vlm_chat) >= 3, "need >=3 hosted vlm-chat entries"

    local_transcribers = [e for e in entries if e.shape == "transcriber" and e.local]
    assert len(local_transcribers) >= 1, "need >=1 local transcriber"

    upstream_parser_transcribers = [
        e for e in entries if e.shape == "transcriber" and e.transport == "upstream-parser"
    ]
    assert len(upstream_parser_transcribers) >= 1, "need >=1 upstream-parser transcriber"

    for shape in ("vlm-chat", "transcriber"):
        closed_anchor = [
            e for e in entries if e.shape == shape and e.weights_licence == "closed"
        ]
        assert closed_anchor, f"need >=1 closed-weights anchor for shape {shape!r}"

    # Calibration pair: at least one served `model` present under both shapes.
    models_by_shape: dict[str, set[str]] = {"vlm-chat": set(), "transcriber": set()}
    for e in entries:
        if e.model and e.shape in models_by_shape:
            models_by_shape[e.shape].add(e.model)
    assert models_by_shape["vlm-chat"] & models_by_shape["transcriber"], (
        "need >=1 model present under both vlm-chat and transcriber shapes (calibration pair)"
    )

    # `contaminated` property coverage: one flagged (post-cutoff), one not (pre-cutoff).
    assert get_entry(entries, "mistral-ocr@mistral").contaminated is True
    assert get_entry(entries, "qwen3-vl-8b@openrouter").contaminated is False


def test_docstrange_entry_points_at_the_registered_parser_and_declares_raster_input():
    """The DocStrange row is an upstream-parser entry whose adapter rasterizes. Both halves are
    load-bearing: `upstream_parser` is the EXACT string the report resolves rows by, and
    `input_mode` is what keeps it out of the pdf-direct free-ride caveat."""
    from realdoc_bench.evaluate.parsers.base import registry as parser_registry

    entries = load_registry(Path(__file__).parent.parent / "configs" / "registry.yaml")
    e = get_entry(entries, "docstrange@nanonets")

    assert e.shape == "transcriber"
    assert e.transport == "upstream-parser"
    assert e.upstream_parser in parser_registry
    assert e.input_mode == "raster-png"
    assert e.api_key_env == "DOCSTRANGE_API_KEY"
    assert e.contaminated is False       # 2025-07-31 public availability, pre-cutoff
