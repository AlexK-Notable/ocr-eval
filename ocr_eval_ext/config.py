"""Registry of (model, serving) pairs. One entry per pair — never merged."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

CONTAMINATION_CUTOFF = "2026-05-24"
# HF createdAt of Extend-AI/RealDoc-Bench (first public availability; spec rev 2 cited
# lastModified 2026-06-03 — reviewer verified createdAt 2026-05-24T19:26Z).


class RegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    shape: Literal["vlm-chat", "transcriber"]
    transport: Literal["openai-compat", "upstream-parser", "bedrock-converse", "gemini-native"]
    base_url: str | None = None
    model: str | None = None
    upstream_parser: str | None = None
    api_key_env: str | None = None
    region: str | None = None      # bedrock-converse only: part of serving identity, not optional there
    precision: Literal["bf16", "fp8-vllm", "q8-gguf", "provider-default"]
    weights_licence: str
    provider_tos_commercial: Literal["ok", "blocked", "conditional"]
    tos_note: str = ""
    provenance: str
    release_date: str
    provider_pin: dict | None = None
    reasoning: dict | None = None
    # Per-entry reasoning control, passed through to the provider in `extra_body` exactly like
    # `provider_pin` — and, like it, NOT folded into the condition hash, because it describes the
    # endpoint binding rather than the experimental condition. OpenRouter shape:
    # `{max_tokens: 4096}` caps thinking, `{enabled: false}` disables it. Only set it on entries
    # whose provider advertises `reasoning`/`include_reasoning` in supported_parameters; sending
    # it to a model that does not support it risks a 400. Exists because a thinking model with an
    # uncapped reasoning budget spends the whole completion allowance before writing an answer
    # and returns empty content.
    local: bool = False
    promptable: bool = True
    pricing: dict | None = None
    input_mode: Literal["raster-png", "pdf-direct"] | None = None
    # What the provider actually receives, for the report's `input` column. `None` keeps the
    # historical inference (openai-compat -> raster-png, anything else -> pdf-direct), which was
    # only ever a proxy: transport says how we talk to a provider, not what we hand it. An
    # upstream-parser entry whose adapter rasterizes (docstrange@nanonets) is raster-png despite
    # the transport, and inferring "pdf-direct" for it would attach the embedded-text-layer
    # free-ride caveat to a row that cannot free-ride. Set it explicitly to override.

    @model_validator(mode="after")
    def _check_transport_fields(self) -> RegistryEntry:
        if self.transport == "openai-compat" and not (self.base_url and self.model):
            raise ValueError(f"{self.id}: openai-compat requires base_url and model")
        if self.transport == "upstream-parser" and not self.upstream_parser:
            raise ValueError(f"{self.id}: upstream-parser transport requires upstream_parser name")
        if self.transport == "bedrock-converse":
            # `model` is the Bedrock modelId (or inference-profile id); `region` is REQUIRED rather
            # than defaulted from AWS_REGION/AWS_DEFAULT_REGION on purpose: region is part of the
            # serving identity stamped on every row, and an ambient env var would let the same
            # registry id silently produce rows from two different serving stacks across machines.
            if not self.model:
                raise ValueError(f"{self.id}: bedrock-converse requires model (the Bedrock modelId)")
            if not self.region:
                raise ValueError(
                    f"{self.id}: bedrock-converse requires an explicit region — it is part of the "
                    f"recorded serving identity and must never be inherited from the environment")
            if self.base_url:
                raise ValueError(
                    f"{self.id}: bedrock-converse takes no base_url (endpoint is derived from "
                    f"region by boto3); got {self.base_url!r}")
            if self.api_key_env:
                raise ValueError(
                    f"{self.id}: bedrock-converse authenticates via the AWS credential chain "
                    f"(SigV4), not an API-key env var; got api_key_env={self.api_key_env!r}")
            if self.provider_pin:
                raise ValueError(
                    f"{self.id}: provider_pin is an OpenRouter routing concept and has no meaning "
                    f"on Bedrock — serving identity is (modelId, region)")
        if self.transport == "gemini-native":
            # Exists only to reach `mediaResolution`, which Google's OpenAI-compat shim does not
            # expose (see ocr_eval_ext/gemini_native.py's module docstring for the live probe).
            if not self.model:
                raise ValueError(f"{self.id}: gemini-native requires model (the Gemini model id)")
            if not self.api_key_env:
                raise ValueError(
                    f"{self.id}: gemini-native requires api_key_env (e.g. GEMINI_API_KEY) — the "
                    f"native endpoint authenticates with an API key, not a credential chain")
            if self.base_url:
                raise ValueError(
                    f"{self.id}: gemini-native takes no base_url — the endpoint is fixed at "
                    f"generativelanguage.googleapis.com/v1beta; got {self.base_url!r}. A different "
                    f"base_url would be a different serving stack and belongs in its own transport")
            if self.region:
                raise ValueError(
                    f"{self.id}: gemini-native takes no region (Google does not expose a regional "
                    f"endpoint here); got {self.region!r}")
            if self.provider_pin:
                raise ValueError(
                    f"{self.id}: provider_pin is an OpenRouter routing concept — Gemini's native "
                    f"endpoint has one serving stack and nothing to pin")
        return self

    @property
    def contaminated(self) -> bool:
        return self.release_date > CONTAMINATION_CUTOFF


def load_registry(path: Path) -> list[RegistryEntry]:
    entries = [RegistryEntry(**raw) for raw in yaml.safe_load(path.read_text())]
    ids = [e.id for e in entries]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate registry ids: {sorted(dupes)}")
    return entries


def get_entry(entries: list[RegistryEntry], id: str) -> RegistryEntry:
    for e in entries:
        if e.id == id:
            return e
    raise KeyError(f"registry id not found: {id}")
