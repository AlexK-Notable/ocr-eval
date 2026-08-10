"""Mistral OCR 4 parse adapter -- PDF in, markdown out.

Uses Mistral's OCR endpoint directly instead of the Python SDK so the adapter
stays consistent with the repo's existing lightweight HTTP clients. Local PDFs
are sent as base64 data URLs, which the API documents as the supported path for
files that are not publicly hosted.

Auth: ``MISTRAL_API_KEY`` env var. Pricing: ``mistral_ocr_4`` in catalog.yaml
(per-page rate).
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from realdoc_bench.evaluate.parsers.base import (
    ParseProvider,
    ParseResult,
    register_parser,
)
from realdoc_bench.shared.io.cache import sha256_text
from realdoc_bench.shared.pricing.meter import parse_cost

DEFAULT_BASE_URL = "https://api.mistral.ai"


def _pdf_data_url(pdf_path: Path) -> str:
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _pages_to_markdown(response: dict[str, Any]) -> tuple[str, int]:
    pages = response.get("pages") or []
    chunks: list[str] = []
    for page in pages:
        markdown = (_get(page, "markdown", "") or "").strip()
        if markdown:
            chunks.append(markdown)
    return "\n\n---\n\n".join(chunks), len(pages)


@register_parser("mistral_ocr_4", version="mistral-ocr-4-1")
class MistralOCR4Parser(ParseProvider):
    """Mistral OCR 4 via ``POST /v1/ocr``.

    FORK CHANGES (2026-08-06), all four driven by one probe
    (``scripts/probe_mistral_ocr.py``) rather than by the docs:

    1. **``model`` is pinned to an explicit version, never ``mistral-ocr-latest``.**
       ``4-0`` and ``4-1`` are BOTH undeprecated, and on one identical page they
       returned DIFFERENT markdown (4,248 vs 4,608 chars; ``ACORD®`` vs
       ``ACORD`` — 4-1 strips the trademark glyph). So the two are not
       interchangeable, and an alias that silently re-points would change what a
       measured number means. OCR 1 and OCR 2 are already retired, so the
       aliases demonstrably move.

    2. **``include_blocks`` is sent explicitly.** The docs contradict themselves
       (``true`` in the API reference, ``false`` in the capabilities page). The
       probe settled it — blocks come back by default — but a default we cannot
       pin from documentation must not be left implicit in a benchmark.

    3. **``extract_header`` / ``extract_footer`` are exposed.** Both default
       ``false``; turning them on moves ~150 chars out of ``markdown`` into
       separate fields, so they change the transcript the grader sees.

    4. **``table_format`` carries a warning.** Setting ``html`` moved table
       content OUT of ``markdown`` and into a separate ``tables`` field, cutting
       the observed transcript from 4,608 to 289 chars. Since only ``markdown``
       reaches the extractor, ``html`` would hand the grader an almost empty
       document while looking like catastrophic OCR failure. Left ``None``.
    """

    # Explicit pin. `mistral-ocr-4-1` is what `mistral-ocr-latest` resolved to on 2026-08-06.
    model: str = "mistral-ocr-4-1"
    table_format: str | None = None
    confidence_scores_granularity: str | None = None
    include_blocks: bool | None = None
    extract_header: bool | None = None
    extract_footer: bool | None = None

    def __init__(self, **overrides: Any) -> None:
        """Accept per-instance overrides for the request-shaping fields.

        `ParseProvider` is a plain ABC, not a pydantic model, so class attributes alone are NOT
        settable through `build(name, **kwargs)` — the base `object.__init__` rejects arguments.
        Without this the new parameters would be reachable only by subclassing, which is how the
        model pin itself is done (`MistralOCR40Parser`) but is far too heavy for a table format.

        Unknown names are rejected rather than ignored: a silently-dropped `table_fmt=` typo would
        produce rows that claim a configuration they were never served at, and `config_hash` would
        agree with the claim.
        """
        allowed = {"model", "table_format", "confidence_scores_granularity",
                   "include_blocks", "extract_header", "extract_footer"}
        unknown = set(overrides) - allowed
        if unknown:
            raise TypeError(f"{type(self).__name__}: unknown parameter(s) {sorted(unknown)}; "
                            f"allowed: {sorted(allowed)}")
        for k, v in overrides.items():
            setattr(self, k, v)

    def _extra_payload(self) -> dict[str, Any]:
        """Request keys a subclass adds. Empty on the plain pins, so their payload — and therefore
        their cache — is byte-identical to before this hook existed."""
        return {}

    def config_hash(self) -> str:
        """EVERY request-shaping parameter must appear here.

        Two configurations that hash the same would share one cache key, so the second
        run would read the first's rows and silently report them as its own. `model` is
        the load-bearing one now that 4-0 and 4-1 are both live and demonstrably differ.
        """
        return sha256_text(
            f"{self.name}|{self.version}|model={self.model}"
            f"|table={self.table_format}|conf={self.confidence_scores_granularity}"
            f"|blocks={self.include_blocks}"
            f"|header={self.extract_header}|footer={self.extract_footer}"
        )[:7]

    def parse(self, pdf_path: Path, *, cache_dir: Path | None = None) -> ParseResult:
        del cache_dir
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY not set")

        base_url = os.environ.get("MISTRAL_OCR_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        payload: dict[str, Any] = {
            "model": self.model,
            "document": {
                "type": "document_url",
                "document_url": _pdf_data_url(pdf_path),
            },
        }
        # Only send what was explicitly configured: an omitted key takes Mistral's default, and
        # sending an explicit None is not the same thing (the API rejects some null values).
        for key, value in (
            ("table_format", self.table_format),
            ("confidence_scores_granularity", self.confidence_scores_granularity),
            ("include_blocks", self.include_blocks),
            ("extract_header", self.extract_header),
            ("extract_footer", self.extract_footer),
        ):
            if value is not None:
                payload[key] = value
        payload.update(self._extra_payload())

        t0 = time.perf_counter()
        with httpx.Client(timeout=None) as client:
            resp = client.post(
                f"{base_url}/v1/ocr",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            raw = resp.json()
        latency = time.perf_counter() - t0

        markdown, pages = _pages_to_markdown(raw)
        markdown += _annotation_markdown(raw.get("document_annotation"))
        return ParseResult(
            markdown=markdown,
            page_count=pages,
            latency_sec=latency,
            cost_estimate_usd=parse_cost(self.name, pages=pages) if pages else None,
            pages_processed=pages,
            provider=self.name,
            version=self.version,
            config_hash=self.config_hash(),
            # `model` is the SERVER-reported id, not `self.model` — the only way to catch an
            # alias resolving somewhere other than where we pinned it.
            raw={"model": raw.get("model"), "usage_info": raw.get("usage_info")},
        )


CHECKBOX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "checkbox_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string",
                              "description": "The visible text label beside the checkbox."},
                    "checked": {"type": "boolean",
                                "description": "True if the box is ticked/filled, false if empty."},
                },
                "required": ["label", "checked"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["checkbox_items"],
    "additionalProperties": False,
}

ANNOT_HEADING = "\n\n## Checkbox states (document_annotation)\n\n"


def _annotation_markdown(annotation: str | dict | None) -> str:
    """Render the annotation as glyph-prefixed lines APPENDED to the transcript.

    Glyph-before-label (`☑ Methane (CH4)`) is deliberate: it is the convention DocStrange already
    emits and the one the extractor demonstrably reads, so this block needs no prompt change to be
    understood. Rendering it as `checked: true` prose would make this leg a test of the extractor's
    JSON reading rather than of Mistral's checkbox recovery.
    """
    if annotation is None:
        return ""
    parsed: Any = annotation
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except (ValueError, TypeError):
            return ""      # a non-JSON annotation must not take the whole parse down
    if not isinstance(parsed, dict):
        return ""
    items = parsed.get("checkbox_items") or []
    lines = [f"{'☑' if it.get('checked') else '☐'} {it.get('label') or ''}".rstrip()
             for it in items if isinstance(it, dict)]
    return ANNOT_HEADING + "\n".join(lines) if lines else ""


@register_parser("mistral_ocr_4_0_annot", version="mistral-ocr-4-0+annot")
class MistralOCR40AnnotParser(MistralOCR4Parser):
    """`mistral-ocr-4-0` plus `document_annotation_format` carrying an explicit checkbox schema.

    WHY: plain OCR 4 recovers checkbox LABELS but silently drops the mark on some pages. On
    `finance_74` the markdown reads `Methane (CH4)` with no glyph while gold says checked; the
    annotation on the same call returns `{"label": "Methane (CH4)", "checked": true}`. The engine
    had the selection state all along — the markdown serialization loses it. Probed 2026-08-10
    against gold on the two worst pages: 6/6 correct, including 4 `false` values, on fields plain
    OCR got wrong.

    THE ANNOTATION IS APPENDED, NEVER SUBSTITUTED. Only `markdown` reaches the extractor, and 927
    of the bank's 1,356 questions are NOT checkbox questions — replacing the transcript with a
    checkbox list would blind those. The block is added under its own heading so the page text
    stays intact.

    NOT COMPARABLE WITH `mistral_ocr_4_0` AS A TRANSCRIBER PAIR. Mistral's docs describe
    annotations as a vision-capable LLM reading the OCR output, so this leg is
    (Mistral OCR + Mistral LLM + our gemini extractor) — three stages against the plain leg's two.
    A win here does not isolate OCR quality; it measures the product Mistral actually sells for
    form extraction. Its own parser name gives it its own transcripts and rows.

    UNDOCUMENTED REQUEST SHAPE. Mistral's annotations page documents only the bbox variant; the
    `document_annotation_format` wrapper below was derived by live probe (HTTP 200, 2026-08-10).
    """

    model: str = "mistral-ocr-4-0"

    def config_hash(self) -> str:
        # The schema shapes the request, so it belongs in the hash — two schemas under one parser
        # name would share a cache key and silently report each other's rows.
        return sha256_text(super().config_hash() + "|annot=" + json.dumps(
            CHECKBOX_SCHEMA, sort_keys=True))[:7]

    def _extra_payload(self) -> dict[str, Any]:
        return {"document_annotation_format": {
            "type": "json_schema",
            "json_schema": {"name": "CheckboxStates", "schema": CHECKBOX_SCHEMA, "strict": True},
        }}


@register_parser("mistral_ocr_4_0", version="mistral-ocr-4-0")
class MistralOCR40Parser(MistralOCR4Parser):
    """The other undeprecated 4.x pin, as a SEPARATE parser name.

    Not a config flag on one parser: upstream keys transcripts and cache rows by parser NAME
    (``parses/<parser>/``, ``eval/cache/<qid>__<parser>.json``), so two model versions under one
    name would overwrite each other's transcripts. Distinct names give each pin its own row set,
    which is what makes a 4-0 vs 4-1 comparison legible in the report.

    Measured difference on one page (2026-08-06): 4-0 produced 4,248 chars vs 4-1's 4,608, and
    kept ``ACORD®`` where 4-1 emitted ``ACORD``. Text normalization is exactly what this bench's
    string comparisons are sensitive to, so this is a real condition difference — not a label.
    """

    model: str = "mistral-ocr-4-0"
