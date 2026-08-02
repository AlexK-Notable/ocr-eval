"""Task 10: end-to-end smoke test for the seam the whole Stage 1 pipeline is built on —
`run_direct` (spend a vlm-chat cell, write an upstream-compatible cache row) feeding straight into
`build_markdown_report` (a pure function reading that same cache off disk). No network beyond
`127.0.0.1` (`MockOpenAI`), no API keys, no `ocr-eval` CLI/`_preflight` involved — this is a
library-level proof that the two modules' on-disk contract (the `vlm__<id>__<cond>` cache-row
shape) actually holds, not a test of CLI wiring (that's `tests_ext/test_report_md.py`'s
`test_report_cli_writes_report_and_renames_dashboard`) or of `run_direct`'s own request/retry
mechanics (that's `tests_ext/test_direct.py`).

Synthetic corpus: 2 single-page PDFs + a 2-item checkbox-bucket bank (asymmetric class balance —
one true, one false gold — so "the mock model got everything right" is a real assertion, not an
artifact of a same-answer-always bank). `MockOpenAI` answers every request `{"checked": true}` OR
`{"checked": false}` matching each item's own gold via `responses=[...]` scripted per request
index — see `_scripted_replies_in_bank_order` below for why that's safe under `run_direct`'s
default 8-worker concurrent dispatch.

`iters=200` (never the CLI's default 2000, per Task 9's precedent for correctness-not-precision
tests). No `max_spend_usd` is set on `run_direct` — the registry entry below carries no `pricing`,
and `MockOpenAI`'s stock response has no `usage.cost` either, so passing a spend cap would trip
the C1 fail-closed rule in `direct.py`'s `track()` ("cannot enforce --max-spend ... provider
returned no usage.cost and registry has no pricing") for a reason that has nothing to do with what
this test is checking."""
from __future__ import annotations

import json
from pathlib import Path

import fitz  # pymupdf

from ocr_eval_ext.config import RegistryEntry
from ocr_eval_ext.direct import run_direct
from ocr_eval_ext.report_md import build_markdown_report
from realdoc_bench.evaluate.runs import RunLayout
from tests_ext.mock_openai import MockOpenAI

# Deliberately not 429/263/258/... (preconditions.EXPECTED) — this bank is synthetic and never
# goes near `check_bank`/`_preflight`; only `run_direct` and `build_markdown_report` see it.
_ITEMS = [
    {
        "question_id": "cb0", "source_file": "doc_0", "domain": "test",
        "question": "Is checkbox 0 checked?", "capabilities": ["checkbox_state"],
        "gold_dict": {"checked": True},
        "response_format": "Return exactly: checked=<boolean>",
    },
    {
        "question_id": "cb1", "source_file": "doc_1", "domain": "test",
        "question": "Is checkbox 1 checked?", "capabilities": ["checkbox_state"],
        "gold_dict": {"checked": False},
        "response_format": "Return exactly: checked=<boolean>",
    },
]


def _build_smoke_run_dir(tmp_path: Path) -> RunLayout:
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    for it in _ITEMS:
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 20), f"{it['source_file']}: is the box checked?")
        # non-trivial ink coverage — `_one`'s render path aborts a cell as "blank render" below
        # ink_coverage < 0.001, which would silently turn this into an error_class row instead of
        # a real answered one.
        page.draw_rect(fitz.Rect(20, 60, 120, 140), fill=(0, 0, 0))
        doc.save(layout.docs_dir / f"{it['source_file']}.pdf")
    layout.bank_path.write_text(json.dumps({"items": _ITEMS}))
    return layout


def _mock_entry(base_url: str) -> RegistryEntry:
    return RegistryEntry(
        id="mock-vlm@e2e-smoke", shape="vlm-chat", transport="openai-compat",
        base_url=base_url, model="org/e2e-smoke", api_key_env=None,
        precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
        provenance="Test", release_date="2025-01-01",
    )


def _scripted_replies_in_bank_order() -> list[dict]:
    """One scripted 200 response per bank item, each answering that item's OWN gold correctly.

    Correctness note: `run_direct`'s default `workers=8` dispatches both of this test's cells
    concurrently (see `_dispatch_bounded`), so the two HTTP requests can arrive at `MockOpenAI` in
    either order — the two cells are for different documents/questions with no ordering
    dependency between them, and `MockOpenAI._spec_for` just returns `responses[min(index,
    len-1)]` by REQUEST ARRIVAL ORDER, not by which bank item the request was for. To keep the
    test correct regardless of arrival order, every scripted item answers `{"checked": true}` —
    i.e. this only works because BOTH items are constructed with gold `checked=True`... which
    contradicts the module docstring's "asymmetric class balance" claim above.

    Resolved below: `workers=1` is passed to `run_direct` (single in-flight request), which makes
    `cells` dispatch strictly in the order `run_direct` built them (bank item order) — see the
    call site in `test_direct_to_report_seam_with_asymmetric_gold`. That makes request arrival
    order == bank order == this list's order, so item 0 (gold `True`) gets `responses[0]` and
    item 1 (gold `False`) gets `responses[1]`, both scored correct."""
    return [
        {"body": {"id": "cmpl-0", "model": "org/e2e-smoke", "provider": "MockProvider",
                  "choices": [{"message": {"role": "assistant", "content": '{"checked": true}'}}],
                  "usage": {"prompt_tokens": 100, "completion_tokens": 10}}},
        {"body": {"id": "cmpl-1", "model": "org/e2e-smoke", "provider": "MockProvider",
                  "choices": [{"message": {"role": "assistant", "content": '{"checked": false}'}}],
                  "usage": {"prompt_tokens": 100, "completion_tokens": 10}}},
    ]


def test_direct_to_report_seam_with_asymmetric_gold(tmp_path):
    """The seam under test: `run_direct` writes 2 real `vlm__mock-vlm_e2e-smoke__<cond>` cache
    rows (one per bank item, real rendered-page `image_sha`, real `score_typed` field_matches) →
    `build_markdown_report` reads them straight off disk (no re-scoring, no model calls) and
    renders a report whose Section A row shows the mock model at 100% checkbox accuracy, with the
    baseline rows present alongside it. No network beyond `127.0.0.1` (`MockOpenAI`'s
    `HTTPServer((\"127.0.0.1\", 0), ...)`), no API keys (`api_key_env=None` on the registry entry)."""
    layout = _build_smoke_run_dir(tmp_path)

    with MockOpenAI(responses=_scripted_replies_in_bank_order()) as mock:
        entry = _mock_entry(mock.base_url)
        # workers=1: see `_scripted_replies_in_bank_order`'s docstring — pins request-arrival
        # order to bank-item order so each scripted reply lands on the item it was written for.
        summary = run_direct(layout, [entry], workers=1)

    assert summary == {"ok": 2, "error": 0, "cached": 0}
    assert len(mock.requests) == 2       # exactly one call per bank item, no network beyond mock

    # Both cache rows landed, both correct, against their OWN (differing) gold — proves this
    # isn't a same-answer-always bank passing by coincidence.
    from ocr_eval_ext.direct import STAGE1_CONDITION, parser_key
    pk = parser_key(entry.id, STAGE1_CONDITION)
    rec0 = json.loads(layout.cache_path("cb0", pk).read_text())
    rec1 = json.loads(layout.cache_path("cb1", pk).read_text())
    assert rec0["answer"] == {"checked": True} and rec0["match"] is True
    assert rec1["answer"] == {"checked": False} and rec1["match"] is True

    md = build_markdown_report(layout, [entry], iters=200)

    # ── the mock model's row, at 100% checkbox accuracy ─────────────────────────────────────
    section_a = md.split("## Section A")[1].split("## Baseline rows")[0]
    assert entry.id in section_a
    assert "100.0%" in section_a          # checkbox acc-over-all: 2/2 correct
    assert "n: 2/2" in section_a          # both bank items answered, none missing

    # ── the baseline rows ────────────────────────────────────────────────────────────────────
    assert "## Baseline rows" in md
    baselines = md.split("## Baseline rows")[1].split("## Cross-shape comparison note")[0]
    assert "always-true:" in baselines
    assert "always-false:" in baselines
    assert "majority-class:" in baselines
    # asymmetric gold (1 true / 1 false of 2) — always-true and always-false must each land at
    # 50%, never both reporting the same number a symmetric bank would produce by accident.
    assert "always-true: 50.0%" in baselines
    assert "always-false: 50.0%" in baselines
