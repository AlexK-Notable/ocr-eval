"""DocStrange sync-endpoint adapter.

The request assertions matter as much as the response ones: this provider would happily accept
a PDF upload, and the whole point of the adapter is that it never sends one (divergence D5 —
2 of 16 sampled corpus docs carry an embedded text layer a PDF-accepting parser would
free-ride on).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from realdoc_bench.evaluate.parsers import docstrange as ds
from realdoc_bench.evaluate.parsers.base import registry as parser_registry
from realdoc_bench.evaluate.parsers.docstrange import (
    DocStrangeSpendCapExceeded,
    DocStrangeSyncParser,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict[str, Any] | None = None,
                 text: str = "", headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._payload


def _ok(markdown: str = "# Hello\n\n☑ Consent given", pages: int | None = 1) -> _FakeResponse:
    """The shape the live API actually returns (verified 2026-08-04): `result.markdown` is an
    object with a `content` string, NOT the bare string the published OpenAPI schema shows."""
    return _FakeResponse(200, {
        "success": True,
        "record_id": "rec_1",
        "status": "completed",
        "result": {
            "markdown": {"content": markdown, "metadata": {}},
            "html": None, "json": None, "csv": None,
        },
        "processing_time": 1.5,
        "filename": "page.png",
        "output_format": "markdown",
        "pages_processed": pages,
    })


class _FakeClient:
    """Replays `script` in order; records every request it was handed."""

    script: ClassVar[list[Any]] = []
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def post(self, url: str, *, headers: dict[str, str], files: dict[str, Any],
             data: dict[str, str]) -> _FakeResponse:
        _FakeClient.calls.append({"url": url, "headers": headers, "files": files, "data": data})
        nxt = _FakeClient.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


@pytest.fixture(autouse=True)
def _harness(monkeypatch):
    """Real key in env, no real sleeping, and clean module-global spend state per test."""
    _FakeClient.script = []
    _FakeClient.calls = []
    monkeypatch.setenv("DOCSTRANGE_API_KEY", "test-key")
    monkeypatch.delenv("DOCSTRANGE_MAX_PAGES", raising=False)
    monkeypatch.setattr(ds.httpx, "Client", _FakeClient)
    monkeypatch.setattr(ds, "_sleep", lambda _s: None)
    ds.reset_billed_pages()
    yield
    ds.reset_billed_pages()


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    """A real single-page PDF — the adapter rasterizes with pymupdf, so a stub byte string
    would not exercise the code path under test."""
    import pymupdf

    path = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Consent form")
    doc.save(str(path))
    doc.close()
    return path


# ── registration ────────────────────────────────────────────────────────────────────────────


def test_registered_under_the_name_the_registry_entry_points_at():
    assert "docstrange_sync" in parser_registry
    assert parser_registry.get("docstrange_sync") is DocStrangeSyncParser


# ── request shape ───────────────────────────────────────────────────────────────────────────


def test_posts_a_rendered_png_to_the_sync_endpoint(pdf: Path):
    _FakeClient.script = [_ok()]

    result = DocStrangeSyncParser().parse(pdf)

    assert len(_FakeClient.calls) == 1
    call = _FakeClient.calls[0]
    assert call["url"] == "https://extraction-api.nanonets.com/api/v1/extract/sync"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["data"] == {"output_format": "markdown"}
    assert result.markdown == "## Page 1\n\n# Hello\n\n☑ Consent given"


def test_uploads_png_bytes_never_the_source_pdf(pdf: Path):
    """D5 raster-only. The uploaded payload must be a PNG render, not the PDF itself."""
    _FakeClient.script = [_ok()]

    DocStrangeSyncParser().parse(pdf)

    _name, blob, mime = _FakeClient.calls[0]["files"]["file"]
    assert blob.startswith(PNG_MAGIC)
    assert mime == "image/png"
    assert not blob.startswith(b"%PDF")
    assert blob != pdf.read_bytes()


def test_stock_run_sends_no_custom_instructions(pdf: Path):
    _FakeClient.script = [_ok()]

    DocStrangeSyncParser().parse(pdf)

    assert "custom_instructions" not in _FakeClient.calls[0]["data"]
    assert "prompt_mode" not in _FakeClient.calls[0]["data"]


def test_custom_instructions_variant_sends_prompt_and_changes_config_hash(pdf: Path):
    class _Contract(DocStrangeSyncParser):
        custom_instructions = "Render checkboxes as ☒ or ☐."

    _FakeClient.script = [_ok()]

    _Contract().parse(pdf)

    data = _FakeClient.calls[0]["data"]
    assert data["custom_instructions"] == "Render checkboxes as ☒ or ☐."
    assert data["prompt_mode"] == "append"
    assert _Contract().config_hash() != DocStrangeSyncParser().config_hash()


# ── billing ─────────────────────────────────────────────────────────────────────────────────


def test_cost_and_pages_come_from_the_api_not_from_our_render_count(pdf: Path):
    """One page rendered, three billed. The row must report what we were billed for — a paid
    run's spend audit is worthless if it reports our own page count back to us."""
    _FakeClient.script = [_ok(pages=3)]

    result = DocStrangeSyncParser().parse(pdf)

    assert result.page_count == 1
    assert result.pages_processed == 3
    assert result.cost_estimate_usd == pytest.approx(0.03)
    assert result.raw == {"billed_pages": 3, "billed_pages_source": "api",
                          "pages_rendered": 1, "requests": 1}


def test_null_pages_processed_falls_back_to_one_page_per_request(pdf: Path):
    """The live sync response returns `pages_processed: null` (verified 2026-08-04) even though
    the record fetched back later carries 1. Billing must not silently read as free, and the row
    must say the number was assumed rather than reported."""
    _FakeClient.script = [_ok(pages=None)]

    result = DocStrangeSyncParser().parse(pdf)

    assert result.pages_processed == 1
    assert result.cost_estimate_usd == pytest.approx(0.01)
    assert result.raw["billed_pages_source"] == "assumed-1-per-request"
    assert ds.billed_pages_total() == 1


def test_string_markdown_shape_still_accepted(pdf: Path):
    """The published OpenAPI schema documents `result.markdown` as a bare string. The live API
    returns an object instead; accept both rather than betting on one."""
    _FakeClient.script = [_FakeResponse(200, {
        "result": {"markdown": "# Schema shape"}, "pages_processed": 1,
    })]

    result = DocStrangeSyncParser().parse(pdf)

    assert result.markdown == "## Page 1\n\n# Schema shape"


def test_billed_page_counter_accumulates_across_documents(pdf: Path):
    _FakeClient.script = [_ok(pages=1), _ok(pages=2)]
    parser = DocStrangeSyncParser()

    parser.parse(pdf)
    parser.parse(pdf)

    assert ds.billed_pages_total() == 3


def test_spend_cap_refuses_further_calls_once_reached(pdf: Path, monkeypatch):
    monkeypatch.setenv("DOCSTRANGE_MAX_PAGES", "1")
    _FakeClient.script = [_ok(pages=1)]
    parser = DocStrangeSyncParser()

    parser.parse(pdf)
    with pytest.raises(DocStrangeSpendCapExceeded):
        parser.parse(pdf)

    assert len(_FakeClient.calls) == 1     # the refused call was never issued


def test_no_spend_cap_by_default(pdf: Path):
    _FakeClient.script = [_ok(pages=500), _ok(pages=1)]
    parser = DocStrangeSyncParser()

    parser.parse(pdf)
    parser.parse(pdf)

    assert ds.billed_pages_total() == 501


# ── failure handling ────────────────────────────────────────────────────────────────────────


def test_missing_api_key_fails_at_construction(monkeypatch):
    """run_parse builds every parser instance before parsing anything, so failing here fails the
    whole run up front instead of once per document."""
    monkeypatch.delenv("DOCSTRANGE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DOCSTRANGE_API_KEY"):
        DocStrangeSyncParser()


def test_200_without_markdown_raises_instead_of_writing_an_empty_transcript(pdf: Path):
    _FakeClient.script = [_FakeResponse(200, {"success": True, "result": {"json": {}}})]

    with pytest.raises(RuntimeError, match=r"no usable result\.markdown"):
        DocStrangeSyncParser().parse(pdf)


def test_markdown_object_without_content_raises(pdf: Path):
    """`result.markdown` present but shaped unexpectedly must fail loudly, not yield ""."""
    _FakeClient.script = [_FakeResponse(200, {"result": {"markdown": {"metadata": {}}}})]

    with pytest.raises(RuntimeError, match=r"no usable result\.markdown"):
        DocStrangeSyncParser().parse(pdf)


def test_transient_429_is_retried_then_succeeds(pdf: Path):
    _FakeClient.script = [
        _FakeResponse(429, {}, text="Too many requests", headers={"retry-after": "0"}),
        _ok(),
    ]

    result = DocStrangeSyncParser().parse(pdf)

    assert len(_FakeClient.calls) == 2
    assert "Hello" in result.markdown


def test_429_that_reads_as_exhausted_credits_fails_immediately(pdf: Path):
    """Retrying a credit-exhaustion 429 four times per page across 581 pages burns wall-clock
    and buries the real reason the run stopped."""
    _FakeClient.script = [_FakeResponse(429, {}, text="Insufficient credits remaining")]

    with pytest.raises(RuntimeError, match="exhausted credits"):
        DocStrangeSyncParser().parse(pdf)

    assert len(_FakeClient.calls) == 1


def test_500_is_retried(pdf: Path):
    _FakeClient.script = [_FakeResponse(503, {}, text="upstream unavailable"), _ok()]

    DocStrangeSyncParser().parse(pdf)

    assert len(_FakeClient.calls) == 2


def test_401_is_permanent_and_not_retried(pdf: Path):
    _FakeClient.script = [_FakeResponse(401, {}, text="invalid api key")]

    with pytest.raises(RuntimeError, match="DocStrange 401"):
        DocStrangeSyncParser().parse(pdf)

    assert len(_FakeClient.calls) == 1


def test_connection_errors_are_retried(pdf: Path):
    _FakeClient.script = [httpx.ConnectError("refused"), _ok()]

    DocStrangeSyncParser().parse(pdf)

    assert len(_FakeClient.calls) == 2


def test_retries_are_bounded(pdf: Path):
    _FakeClient.script = [_FakeResponse(503, {}, text="down")] * ds.MAX_RETRIES

    with pytest.raises(RuntimeError, match="DocStrange 503"):
        DocStrangeSyncParser().parse(pdf)

    assert len(_FakeClient.calls) == ds.MAX_RETRIES
