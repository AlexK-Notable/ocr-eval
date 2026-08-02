"""Registry of (model, serving) pairs. One entry per pair — never merged."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

CONTAMINATION_CUTOFF = "2026-06-03"  # RealDoc-Bench HF publication date


class RegistryEntry(BaseModel):
    id: str
    shape: Literal["vlm-chat", "transcriber"]
    transport: Literal["openai-compat", "upstream-parser"]
    base_url: str | None = None
    model: str | None = None
    upstream_parser: str | None = None
    api_key_env: str | None = None
    precision: Literal["bf16", "fp8-vllm", "q8-gguf", "provider-default"]
    weights_licence: str
    provider_tos_commercial: Literal["ok", "blocked", "conditional"]
    tos_note: str = ""
    provenance: str
    release_date: str
    provider_pin: dict | None = None
    local: bool = False
    promptable: bool = True
    pricing: dict | None = None

    @model_validator(mode="after")
    def _check_transport_fields(self) -> RegistryEntry:
        if self.transport == "openai-compat" and not (self.base_url and self.model):
            raise ValueError(f"{self.id}: openai-compat requires base_url and model")
        if self.transport == "upstream-parser" and not self.upstream_parser:
            raise ValueError(f"{self.id}: upstream-parser transport requires upstream_parser name")
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
