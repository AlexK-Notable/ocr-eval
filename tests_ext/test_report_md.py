"""Task 9: shape-segregated markdown report. Builds a synthetic run dir (two identical
vlm-chat entries + one local transcriber + parses/*.md) and asserts on the rendered markdown —
section order, baseline values, separability, ToS/CC-BY stamps, transcript-recall, INCOMPLETE
marking, the D3/D4 hard-fail guards, and the partial-corpus banner. iters=50-200 throughout
(never the CLI's default 2000) — these are correctness tests, not statistical-precision tests.

Also covers the fail-closed-guards review round: C1 (STALE-RENDER must genuinely re-render, not
read back the very cache that produced image_sha), C2 (an unrenderable doc is itself a failure,
never "presumably fine"), I1 (D3/D4 are keyed on the vlm__ prefix, not on registry resolution),
I2 (transcript-recall token matching needs word boundaries), I3 (INCOMPLETE counts rows ∩ bank
qids, never raw row count), I4 (Section B gets the same mixed-precision caveat as A), I5
(separability appendix marks incomplete row groups), and the M1/M2/M6/M7/M8 minors."""
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import fitz  # pymupdf
import pytest
import yaml
from typer.testing import CliRunner

import ocr_eval_ext.cli as cli_mod
from ocr_eval_ext.cli import app
from ocr_eval_ext.config import RegistryEntry
from ocr_eval_ext.direct import STAGE1_CONDITION, _render_page, condition_hash, parser_key
from ocr_eval_ext.metrics import FieldOutcome
from ocr_eval_ext.parsers_openai import TRANSCRIBER_CONDITION, safe_name
from ocr_eval_ext.report_md import (
    ServingIdentityError,
    StaleRenderError,
    _beats_majority,
    _pairwise_separability,
    _transcript_recall,
    build_markdown_report,
)
from realdoc_bench.evaluate.runs import RunLayout

runner = CliRunner()

N_CHECKBOX = 22       # >= MIN_CLUSTER_DOCS(20) so separability is actually EVALUATED, not floored
N_TRUE = 13           # asymmetric true/false split -> a distinctive, un-collidable class-balance string


def _write_pdf(layout: RunLayout, stem: str) -> None:
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((20, 20), stem)
    doc.save(layout.docs_dir / f"{stem}.pdf")


def _build_full_fixture(tmp_path: Path) -> tuple[RunLayout, list[RegistryEntry]]:
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()

    items = []
    for i in range(N_CHECKBOX):
        stem = f"doc_{i}"
        _write_pdf(layout, stem)
        items.append({
            "question_id": f"cb{i}", "source_file": stem, "domain": "test",
            "question": f"Is checkbox {i} checked?", "capabilities": ["checkbox_state"],
            "gold_dict": {"checked": i < N_TRUE},
            "response_format": "Return exactly: checked=<boolean>",
        })
    for j in range(3):     # blank-bucket items reuse the first 3 checkbox docs — no new PDFs
        items.append({
            "question_id": f"bl{j}", "source_file": f"doc_{j}", "domain": "test",
            "question": f"Any notes on doc_{j}?", "capabilities": ["blank_field"],
            "gold_dict": {"notes": None},
            "response_format": "Return exactly: notes=<string>",
        })
    layout.bank_path.write_text(json.dumps({"items": items}))

    now = dt.datetime.now(dt.UTC)
    png_cache = layout.root / "docs_png"

    def real_image_sha(stem: str, condition: dict) -> str:
        png = _render_page(layout, stem, condition, png_cache)
        return hashlib.sha256(png).hexdigest()

    def write_vlm_row(entry_id: str, condition: dict, qid: str, stem: str, gold_key: str,
                      gold_value, provider: str, *, no_image: bool = False) -> None:
        pk = parser_key(entry_id, condition)
        rec = {
            "qid": qid, "parser": pk, "source_file": stem, "domain": "test",
            "condition": condition, "retrieved_at": now.isoformat(),
            "prompt_sha": "deadbeef0000",
            "image_sha": None if no_image else real_image_sha(stem, condition),
            "image_px": None if no_image else [200, 200],
            "image_bytes": None if no_image else 123,
            "raw_response": json.dumps({gold_key: gold_value}),
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            "resolved_provider": provider, "latency_sec": 0.5,
            "answer": {gold_key: gold_value}, "field_matches": {gold_key: True}, "match": True,
            "error_class": "none",
        }
        cpath = layout.cache_path(qid, pk)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(rec))

    # vlmA@mock: every item (checkbox + blank), all correct, one provider -> complete, no D3/D4.
    for it in items:
        key = "checked" if "checked" in it["gold_dict"] else "notes"
        write_vlm_row("vlmA@mock", STAGE1_CONDITION, it["question_id"], it["source_file"],
                     key, it["gold_dict"][key], "ProviderX")

    # vlmB@mock: checkbox items ONLY (blank items deliberately skipped -> INCOMPLETE (22/25)),
    # but IDENTICAL checkbox correctness to vlmA -> checkbox outcomes are byte-identical, so the
    # pairwise appendix must call them "not separable".
    for it in items[:N_CHECKBOX]:
        write_vlm_row("vlmB@mock", STAGE1_CONDITION, it["question_id"], it["source_file"],
                     "checked", it["gold_dict"]["checked"], "ProviderY")

    # vlmA no-image control (language-prior-only) -> rendered in the Baselines block.
    no_image_cond = {**STAGE1_CONDITION, "no_image": True}
    for it in items[:N_CHECKBOX]:
        write_vlm_row("vlmA@mock", no_image_cond, it["question_id"], it["source_file"],
                     "checked", it["gold_dict"]["checked"], "ProviderX", no_image=True)

    # t1@local transcriber: correct on everything; transcript-recall demoed on the first 5 docs
    # only (glyph + "checked" token). doc_5 is the I2 trap: a glyph PLUS the substring "checked"
    # appearing only inside "Unchecked" — a naive substring-match implementation would wrongly
    # count this as recalled; the word-boundary-matched implementation must not. The remaining
    # 16 docs have a transcript but no recall pattern at all.
    t_pk = f"{safe_name('t1@local')}__{condition_hash(TRANSCRIBER_CONDITION)}"
    for it in items:
        key = "checked" if "checked" in it["gold_dict"] else "notes"
        rec = {"qid": it["question_id"], "parser": t_pk, "source_file": it["source_file"],
               "domain": "test", "answer": {key: it["gold_dict"][key]},
               "field_matches": {key: True}, "match": True}
        cpath = layout.cache_path(it["question_id"], t_pk)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(rec))
    parser_dir = layout.parser_dir(t_pk)
    parser_dir.mkdir(parents=True, exist_ok=True)
    for i in range(N_CHECKBOX):
        stem = f"doc_{i}"
        if i < 5:
            (parser_dir / f"{stem}.md").write_text("## Page 1\n\nChecked: [x]\n")
        elif i == 5:
            (parser_dir / f"{stem}.md").write_text("## Page 1\n\n[ ] Unchecked\n")
        else:
            (parser_dir / f"{stem}.md").write_text("## Page 1\n\nSome unrelated text.\n")

    meta = {
        "dataset_revision": "test-rev", "harness_commit": "test-commit",
        "extractor_id": "gemini-3-flash-preview", "renders_verified": True,
        "partial_corpus": True, "revision_source": "pins",
    }
    (layout.root / "run_meta.json").write_text(json.dumps(meta))

    registry = [
        RegistryEntry(id="vlmA@mock", shape="vlm-chat", transport="openai-compat",
                     base_url="http://vlmA.invalid/v1", model="org/vlmA", api_key_env=None,
                     precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
                     provenance="Test", release_date="2025-01-01"),
        RegistryEntry(id="vlmB@mock", shape="vlm-chat", transport="openai-compat",
                     base_url="http://vlmB.invalid/v1", model="org/vlmB", api_key_env=None,
                     precision="bf16", weights_licence="mit", provider_tos_commercial="blocked",
                     tos_note="test block reason", provenance="Test", release_date="2025-01-01"),
        RegistryEntry(id="t1@local", shape="transcriber", transport="openai-compat",
                     base_url="http://localhost:9/v1", model="org/t1", api_key_env=None,
                     precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
                     provenance="Test", release_date="2025-01-01", local=True),
    ]
    return layout, registry


def test_full_report_structure_and_content(tmp_path):
    layout, registry = _build_full_fixture(tmp_path)
    md = build_markdown_report(layout, registry, iters=200, seed=0)

    # ── section order ───────────────────────────────────────────────────────────────────────
    assert md.index("## Section A — Direct QA") < md.index("## Baseline rows")
    assert md.index("## Baseline rows") < md.index("## Section B — Transcribe-then-extract")
    assert md.index("## Section B — Transcribe-then-extract") < md.index("## Separability appendix")

    # ── header: CC-BY-4.0 + origin=None caveat ─────────────────────────────────────────────
    assert "CC-BY-4.0" in md
    assert "origin=None" in md

    # ── partial-corpus banner ──────────────────────────────────────────────────────────────
    assert "PARTIAL CORPUS" in md

    # ── baseline values (13 true / 9 false of 22) ──────────────────────────────────────────
    assert "majority-class" in md
    assert "always-true: 59.1%" in md
    assert "always-false: 40.9%" in md
    assert "true=13, false=9" in md.split("## Section B")[0]
    assert "true=13, false=9" not in md.split("## Section B")[1]
    assert "no-image control" in md   # vlmA's no-image row rendered in the baselines block

    # ── ToS stamp for the blocked entry ─────────────────────────────────────────────────────
    assert "⚠ ToS: blocked" in md

    # ── not separable: vlmA and vlmB have byte-identical checkbox outcomes ─────────────────
    assert "not separable" in md

    # ── INCOMPLETE marking: vlmB is missing the 3 blank items (22/25) ──────────────────────
    assert "**INCOMPLETE (22/25)**" in md

    # ── transcript-recall: 5 of 22 checkbox docs carry glyph+token; doc_5's "Unchecked" trap
    # (I2) must NOT inflate the count to 6/22 — word-boundary matching, not substring matching.
    assert "transcript-recall" in md.lower()
    assert "22.7%" in md   # 5/22 rounded to one decimal
    assert "27.3%" not in md   # what a substring-match bug would have produced (6/22)

    # ── Section B row label carries the extractor id ────────────────────────────────────────
    assert "t1@local + extractor gemini-3-flash-preview" in md

    # ── Section B provider column is never fabricated ───────────────────────────────────────
    assert "n/a (not persisted)" in md.split("## Section B")[1]


def test_beats_majority_respects_min_cluster_floor():
    """RULING: separable()/paired_delta_ci is never called below MIN_CLUSTER_DOCS(20) — below
    the floor this renders "insufficient clusters (n_docs=N)", never a spurious yes/no verdict."""
    outs_a = [FieldOutcome(f"q{i}", "k", f"d{i}", True, "correct") for i in range(5)]
    outs_b = [FieldOutcome(f"q{i}", "k", f"d{i}", True, "incorrect") for i in range(5)]
    assert _beats_majority(outs_a, outs_b, iters=50, seed=0, alpha=0.05) == "insufficient clusters (n_docs=5)"


def test_pairwise_separability_respects_min_cluster_floor():
    outs_a = [FieldOutcome(f"q{i}", "k", f"d{i}", True, "correct") for i in range(5)]
    outs_b = [FieldOutcome(f"q{i}", "k", f"d{i}", True, "incorrect") for i in range(5)]
    lines = _pairwise_separability([("A", outs_a), ("B", outs_b)], iters=50, seed=0, alpha=0.05)
    assert lines == ["- A vs B: insufficient clusters (n_docs=5)"]


def _minimal_layout(tmp_path: Path, *, with_pdf: bool = False) -> RunLayout:
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    if with_pdf:
        _write_pdf(layout, "doc_0")
    return layout


def test_serving_identity_guard_hard_fails_d4(tmp_path):
    """D4: >1 distinct resolved_provider under ONE parser key is an unconditional hard fail."""
    layout = _minimal_layout(tmp_path)
    items = [
        {"question_id": "cb0", "source_file": "doc_0", "domain": "test",
         "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": True}},
        {"question_id": "cb1", "source_file": "doc_1", "domain": "test",
         "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": False}},
    ]
    layout.bank_path.write_text(json.dumps({"items": items}))
    pk = parser_key("vlmD@mock", STAGE1_CONDITION)
    for qid, gold, provider in [("cb0", True, "ProviderA"), ("cb1", False, "ProviderB")]:
        rec = {"qid": qid, "parser": pk, "source_file": f"doc_{qid[-1]}", "domain": "test",
               "condition": STAGE1_CONDITION, "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
               "image_sha": None, "resolved_provider": provider,
               "answer": {"checked": gold}, "field_matches": {"checked": True}, "match": True,
               "error_class": "none"}
        cpath = layout.cache_path(qid, pk)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(rec))
    entry = RegistryEntry(id="vlmD@mock", shape="vlm-chat", transport="openai-compat",
                          base_url="http://vlmD.invalid/v1", model="org/vlmD", api_key_env=None,
                          precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
                          provenance="Test", release_date="2025-01-01")
    with pytest.raises(ServingIdentityError, match="vlm__vlmD@mock"):
        build_markdown_report(layout, [entry], iters=50)


def test_stale_render_hard_fails_without_flag_and_renders_with_flag(tmp_path):
    """D3: a stored image_sha that no longer matches the current render is a hard fail unless
    allow_stale_render=True — and even then, the STALE-RENDER section still appears in the text."""
    layout = _minimal_layout(tmp_path, with_pdf=True)
    items = [{"question_id": "cb0", "source_file": "doc_0", "domain": "test",
             "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": True}}]
    layout.bank_path.write_text(json.dumps({"items": items}))
    pk = parser_key("vlmE@mock", STAGE1_CONDITION)
    rec = {"qid": "cb0", "parser": pk, "source_file": "doc_0", "domain": "test",
           "condition": STAGE1_CONDITION, "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
           "image_sha": "0" * 64,          # deliberately wrong — never matches a real render
           "resolved_provider": "ProviderA",
           "answer": {"checked": True}, "field_matches": {"checked": True}, "match": True,
           "error_class": "none"}
    cpath = layout.cache_path("cb0", pk)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(rec))
    entry = RegistryEntry(id="vlmE@mock", shape="vlm-chat", transport="openai-compat",
                          base_url="http://vlmE.invalid/v1", model="org/vlmE", api_key_env=None,
                          precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
                          provenance="Test", release_date="2025-01-01")

    with pytest.raises(StaleRenderError, match="STALE-RENDER"):
        build_markdown_report(layout, [entry], iters=50)

    md = build_markdown_report(layout, [entry], iters=50, allow_stale_render=True)
    assert "## STALE-RENDER" in md
    assert "doc_0" in md.split("## STALE-RENDER")[1]
    # Round-2 disclosure (3b): the operator-facing STALE-RENDER section itself — not just the
    # developer-facing docstring — must say this check also fires on a different pymupdf
    # version/OS/machine than the one that originally scored the row, not only a swapped PDF.
    stale_section = md.split("## STALE-RENDER")[1]
    assert "pymupdf version" in stale_section
    assert "does not automatically mean the underlying PDF changed" in stale_section


def _write_single_vlm_registry(path: Path, entry_id: str, base_url: str) -> None:
    path.write_text(yaml.safe_dump([{
        "id": entry_id, "shape": "vlm-chat", "transport": "openai-compat",
        "base_url": base_url, "model": "org/m", "api_key_env": None,
        "precision": "bf16", "weights_licence": "mit", "provider_tos_commercial": "ok",
        "provenance": "Test", "release_date": "2025-01-01",
    }]))


def test_report_cli_writes_report_and_renames_dashboard(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _minimal_layout(tmp_path, with_pdf=True)
    items = [{"question_id": "cb0", "source_file": "doc_0", "domain": "test",
             "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": True}}]
    layout.bank_path.write_text(json.dumps({"items": items}))
    png = _render_page(layout, "doc_0", STAGE1_CONDITION, layout.root / "docs_png")
    image_sha = hashlib.sha256(png).hexdigest()
    pk = parser_key("vlmF@mock", STAGE1_CONDITION)
    rec = {"qid": "cb0", "parser": pk, "source_file": "doc_0", "domain": "test",
           "condition": STAGE1_CONDITION, "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
           "image_sha": image_sha, "resolved_provider": "ProviderA",
           "answer": {"checked": True}, "field_matches": {"checked": True}, "match": True,
           "error_class": "none"}
    cpath = layout.cache_path("cb0", pk)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(rec))
    registry_yaml = tmp_path / "registry.yaml"
    _write_single_vlm_registry(registry_yaml, "vlmF@mock", "http://vlmF.invalid/v1")
    layout.dashboard_path.write_text("<html>fake upstream shape-mixed dashboard</html>")

    result = runner.invoke(app, ["report", "--run-dir", str(layout.root),
                                 "--registry", str(registry_yaml), "--iters", "50"])
    assert result.exit_code == 0, result.output
    report_path = layout.root / "report.md"
    assert report_path.exists()
    assert "Section A" in report_path.read_text()
    assert not layout.dashboard_path.exists()
    assert (layout.root / "dashboard-upstream-UNSEGREGATED.html").exists()


def test_report_cli_exits_nonzero_on_stale_render_then_succeeds_with_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _minimal_layout(tmp_path, with_pdf=True)
    items = [{"question_id": "cb0", "source_file": "doc_0", "domain": "test",
             "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": True}}]
    layout.bank_path.write_text(json.dumps({"items": items}))
    pk = parser_key("vlmG@mock", STAGE1_CONDITION)
    rec = {"qid": "cb0", "parser": pk, "source_file": "doc_0", "domain": "test",
           "condition": STAGE1_CONDITION, "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
           "image_sha": "f" * 64, "resolved_provider": "ProviderA",
           "answer": {"checked": True}, "field_matches": {"checked": True}, "match": True,
           "error_class": "none"}
    cpath = layout.cache_path("cb0", pk)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(rec))
    registry_yaml = tmp_path / "registry.yaml"
    _write_single_vlm_registry(registry_yaml, "vlmG@mock", "http://vlmG.invalid/v1")

    result = runner.invoke(app, ["report", "--run-dir", str(layout.root),
                                 "--registry", str(registry_yaml), "--iters", "50"])
    assert result.exit_code == 1
    assert "STALE-RENDER" in result.output
    assert not (layout.root / "report.md").exists()

    result2 = runner.invoke(app, ["report", "--run-dir", str(layout.root),
                                  "--registry", str(registry_yaml), "--iters", "50",
                                  "--allow-stale-render"])
    assert result2.exit_code == 0, result2.output
    assert (layout.root / "report.md").exists()


# ── C1: STALE-RENDER must genuinely re-render, never read back the cache that produced image_sha ──

def test_c1_stale_render_positive_control_swapped_pdf_is_detected(tmp_path):
    """Reviewer-proved bug: pointing the re-render at the run's own `docs_png` cache makes the
    check tautological (it reads back the very PNG that produced `image_sha`, so a swapped PDF
    sails through). This is the positive control: warm the PRODUCTION docs_png cache exactly as
    a real scoring run would, THEN swap the PDF's content, leaving docs_png untouched. The fixed
    check must re-render into a fresh dir and catch the swap regardless of the stale cache."""
    layout = _minimal_layout(tmp_path)
    doc_path = layout.docs_dir / "doc_0.pdf"
    doc = fitz.open()
    doc.new_page(width=200, height=200).insert_text((20, 20), "original content")
    doc.save(doc_path)

    items = [{"question_id": "cb0", "source_file": "doc_0", "domain": "test",
             "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": True}}]
    layout.bank_path.write_text(json.dumps({"items": items}))

    docs_png = layout.root / "docs_png"
    original_png = _render_page(layout, "doc_0", STAGE1_CONDITION, docs_png)
    original_sha = hashlib.sha256(original_png).hexdigest()

    pk = parser_key("vlmSwap@mock", STAGE1_CONDITION)
    rec = {"qid": "cb0", "parser": pk, "source_file": "doc_0", "domain": "test",
           "condition": STAGE1_CONDITION, "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
           "image_sha": original_sha, "resolved_provider": "ProviderA",
           "answer": {"checked": True}, "field_matches": {"checked": True}, "match": True,
           "error_class": "none"}
    cpath = layout.cache_path("cb0", pk)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(rec))

    # Swap the PDF's content. docs_png (warmed above) is left completely untouched on disk.
    doc2 = fitz.open()
    page2 = doc2.new_page(width=200, height=200)
    page2.insert_text((20, 20), "SWAPPED content -- a totally different page")
    page2.draw_rect(fitz.Rect(10, 50, 190, 190), fill=(0, 0, 0))
    doc2.save(doc_path)
    assert docs_png.exists()   # sanity: the stale production cache genuinely still exists

    entry = RegistryEntry(id="vlmSwap@mock", shape="vlm-chat", transport="openai-compat",
                          base_url="http://vlmSwap.invalid/v1", model="org/vlmSwap",
                          api_key_env=None, precision="bf16", weights_licence="mit",
                          provider_tos_commercial="ok", provenance="Test", release_date="2025-01-01")
    with pytest.raises(StaleRenderError, match="STALE-RENDER"):
        build_markdown_report(layout, [entry], iters=50)


# ── C2: an unrenderable doc is a failure (RENDER-UNAVAILABLE), never treated as "not stale" ──────

def test_c2_render_unavailable_when_pdf_deleted_after_caching(tmp_path):
    layout = _minimal_layout(tmp_path, with_pdf=True)
    items = [{"question_id": "cb0", "source_file": "doc_0", "domain": "test",
             "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": True}}]
    layout.bank_path.write_text(json.dumps({"items": items}))
    pk = parser_key("vlmMissing@mock", STAGE1_CONDITION)
    rec = {"qid": "cb0", "parser": pk, "source_file": "doc_0", "domain": "test",
           "condition": STAGE1_CONDITION, "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
           "image_sha": "a" * 64,      # any value — the doc will be gone, so it can never be
                                        # confirmed as matching OR not matching
           "resolved_provider": "ProviderA",
           "answer": {"checked": True}, "field_matches": {"checked": True}, "match": True,
           "error_class": "none"}
    cpath = layout.cache_path("cb0", pk)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(rec))
    (layout.docs_dir / "doc_0.pdf").unlink()   # the doc a live row still references is GONE

    entry = RegistryEntry(id="vlmMissing@mock", shape="vlm-chat", transport="openai-compat",
                          base_url="http://vlmMissing.invalid/v1", model="org/vlmMissing",
                          api_key_env=None, precision="bf16", weights_licence="mit",
                          provider_tos_commercial="ok", provenance="Test", release_date="2025-01-01")

    with pytest.raises(StaleRenderError, match="RENDER-UNAVAILABLE"):
        build_markdown_report(layout, [entry], iters=50)

    # allow_stale_render=True still surfaces the RENDER-UNAVAILABLE reason in the text — it is
    # folded into the same failing set as STALE-RENDER, never silently dropped.
    md = build_markdown_report(layout, [entry], iters=50, allow_stale_render=True)
    assert "RENDER-UNAVAILABLE" in md


# ── I1: D3/D4 are keyed on the vlm__ prefix, not on registry resolution ──────────────────────────

def test_i1_unregistered_vlm_key_still_gets_d4_guard(tmp_path):
    layout = _minimal_layout(tmp_path)
    items = [
        {"question_id": "cb0", "source_file": "doc_0", "domain": "test",
         "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": True}},
        {"question_id": "cb1", "source_file": "doc_1", "domain": "test",
         "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": False}},
    ]
    layout.bank_path.write_text(json.dumps({"items": items}))
    pk = parser_key("ghost-model@nowhere", STAGE1_CONDITION)   # deliberately never registered
    for qid, gold, provider in [("cb0", True, "ProviderA"), ("cb1", False, "ProviderB")]:
        rec = {"qid": qid, "parser": pk, "source_file": f"doc_{qid[-1]}", "domain": "test",
               "condition": STAGE1_CONDITION, "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
               "image_sha": None, "resolved_provider": provider,
               "answer": {"checked": gold}, "field_matches": {"checked": True}, "match": True,
               "error_class": "none"}
        cpath = layout.cache_path(qid, pk)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(rec))

    with pytest.raises(ServingIdentityError, match=re.escape(pk)):
        build_markdown_report(layout, [], iters=50)   # empty registry -> definitely unregistered


def test_i1_unregistered_vlm_key_renders_in_section_a_never_section_b(tmp_path):
    layout = _minimal_layout(tmp_path)
    items = [{"question_id": "cb0", "source_file": "doc_0", "domain": "test",
             "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": True}}]
    layout.bank_path.write_text(json.dumps({"items": items}))
    pk = parser_key("ghost-model@nowhere", STAGE1_CONDITION)
    rec = {"qid": "cb0", "parser": pk, "source_file": "doc_0", "domain": "test",
           "condition": STAGE1_CONDITION, "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
           "image_sha": None, "resolved_provider": "ProviderA",
           "answer": {"checked": True}, "field_matches": {"checked": True}, "match": True,
           "error_class": "none"}
    cpath = layout.cache_path("cb0", pk)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(rec))

    md = build_markdown_report(layout, [], iters=50)
    section_a = md.split("## Section A")[1].split("## Baseline rows")[0]
    section_b = md.split("## Section B")[1]
    assert pk in section_a
    assert "unknown (unregistered)" in section_a
    assert pk not in section_b


# ── I2: transcript-recall requires a WORD-BOUNDARY token match, not a substring match ────────────

def test_i2_transcript_recall_word_boundary_rejects_unchecked_for_checked(tmp_path):
    layout = _minimal_layout(tmp_path)
    t_pk = "trap-parser"
    parser_dir = layout.parser_dir(t_pk)
    parser_dir.mkdir(parents=True, exist_ok=True)
    (parser_dir / "doc_0.md").write_text("## Page 1\n\n[ ] Unchecked\n")
    checkbox_items = [{"source_file": "doc_0", "gold_dict": {"checked": True}}]
    # glyph present ("[ ]") but "checked" only appears as a substring of "Unchecked" — must NOT
    # count as recalled (a substring-match bug would give 1/1 == 100%).
    assert _transcript_recall(layout, t_pk, checkbox_items) == 0.0


# ── I3: INCOMPLETE counts rows ∩ bank qids, never the raw row count (orphans must not suppress it) ──

def test_i3_incomplete_marking_ignores_orphan_rows_not_in_current_bank(tmp_path):
    layout = _minimal_layout(tmp_path)
    items = [
        {"question_id": f"q{i}", "source_file": f"doc_{i}", "domain": "test",
         "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": True}}
        for i in range(30)
    ]
    layout.bank_path.write_text(json.dumps({"items": items}))
    pk = parser_key("vlmH@mock", STAGE1_CONDITION)

    def _row(qid: str, stem: str) -> dict:
        return {"qid": qid, "parser": pk, "source_file": stem, "domain": "test",
                "condition": STAGE1_CONDITION, "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
                "image_sha": None, "resolved_provider": "ProviderA",
                "answer": {"checked": True}, "field_matches": {"checked": True}, "match": True,
                "error_class": "none"}

    # 25 rows against REAL current-bank qids (5 bank items — q25..q29 — are genuinely missing).
    for i in range(25):
        cpath = layout.cache_path(f"q{i}", pk)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(_row(f"q{i}", f"doc_{i}")))
    # 8 ORPHAN rows — qids that do not correspond to any item in the CURRENT bank (e.g. leftover
    # from a previous bank revision). A raw len(rows_by_qid) count would read 25+8=33 >= 30 and
    # wrongly suppress the INCOMPLETE marker.
    for i in range(8):
        cpath = layout.cache_path(f"orphan{i}", pk)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(_row(f"orphan{i}", f"doc_orphan{i}")))

    entry = RegistryEntry(id="vlmH@mock", shape="vlm-chat", transport="openai-compat",
                          base_url="http://vlmH.invalid/v1", model="org/vlmH", api_key_env=None,
                          precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
                          provenance="Test", release_date="2025-01-01")
    md = build_markdown_report(layout, [entry], iters=50)
    assert "**INCOMPLETE (25/30)**" in md


# ── I4: Section B gets the same mixed-precision caveat as Section A ───────────────────────────────

def test_i4_section_b_gets_mixed_precision_caveat(tmp_path):
    layout = _minimal_layout(tmp_path)
    items = [{"question_id": f"q{i}", "source_file": f"doc_{i}", "domain": "test",
             "question": "checked?", "capabilities": ["checkbox_state"],
             "gold_dict": {"checked": True}} for i in range(2)]
    layout.bank_path.write_text(json.dumps({"items": items}))

    t1_pk = f"{safe_name('t1prec@local')}__{condition_hash(TRANSCRIBER_CONDITION)}"
    t2_pk = f"{safe_name('t2prec@local')}__{condition_hash(TRANSCRIBER_CONDITION)}"
    for pk, qid in [(t1_pk, "q0"), (t2_pk, "q1")]:
        rec = {"qid": qid, "parser": pk, "source_file": qid.replace("q", "doc_"), "domain": "test",
               "answer": {"checked": True}, "field_matches": {"checked": True}, "match": True}
        cpath = layout.cache_path(qid, pk)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(rec))
    entries = [
        RegistryEntry(id="t1prec@local", shape="transcriber", transport="openai-compat",
                     base_url="http://t1prec.invalid/v1", model="org/t1", api_key_env=None,
                     precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
                     provenance="Test", release_date="2025-01-01", local=True),
        RegistryEntry(id="t2prec@local", shape="transcriber", transport="openai-compat",
                     base_url="http://t2prec.invalid/v1", model="org/t2", api_key_env=None,
                     precision="fp8-vllm", weights_licence="mit", provider_tos_commercial="ok",
                     provenance="Test", release_date="2025-01-01", local=True),
    ]
    md = build_markdown_report(layout, entries, iters=50)
    section_b = md.split("## Section B")[1]
    assert "mixed precision" in section_b


# ── I5: separability appendix flags incomplete row groups ────────────────────────────────────────

def test_i5_separability_appendix_flags_incomplete_rows(tmp_path):
    layout, registry = _build_full_fixture(tmp_path)
    md = build_markdown_report(layout, registry, iters=200, seed=0)
    appendix = md.split("## Separability appendix")[1]
    assert "vlmB@mock [INCOMPLETE 22/25]" in appendix


# ── M1: a 0.0 Section B cost coerced from None (unpriced) never renders as a fake $0.0000 ────────

def test_m1_section_b_zero_cost_from_none_coercion_renders_unpriced(tmp_path):
    layout = _minimal_layout(tmp_path)
    items = [{"question_id": "q0", "source_file": "doc_0", "domain": "test",
             "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": True}}]
    layout.bank_path.write_text(json.dumps({"items": items}))
    t_pk = f"{safe_name('tunpriced@hosted')}__{condition_hash(TRANSCRIBER_CONDITION)}"
    rec = {"qid": "q0", "parser": t_pk, "source_file": "doc_0", "domain": "test",
           "answer": {"checked": True}, "field_matches": {"checked": True}, "match": True}
    cpath = layout.cache_path("q0", t_pk)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(rec))
    parser_dir = layout.parser_dir(t_pk)
    parser_dir.mkdir(parents=True, exist_ok=True)
    (parser_dir / "doc_0.md").write_text("## Page 1\n\nirrelevant\n")
    (parser_dir / "doc_0.json").write_text(json.dumps({
        "parser": t_pk, "stem": "doc_0", "ok": True, "page_count": 1,
        "latency_sec": 1.2, "cost_usd": 0.0,   # parse.py's `cost_estimate_usd or 0.0` coercion,
                                                # with NO registry pricing entry below
        "md_length": 20, "elapsed_total": 1.3, "error": "",
    }))
    entry = RegistryEntry(id="tunpriced@hosted", shape="transcriber", transport="openai-compat",
                          base_url="http://tunpriced.invalid/v1", model="org/tu", api_key_env=None,
                          precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
                          provenance="Test", release_date="2025-01-01", local=False)

    md = build_markdown_report(layout, [entry], iters=50)
    assert "n/a (unpriced)" in md
    assert "$0.0000" not in md


# ── M2: an unregistered row's stamp columns are always "unknown (unregistered)", never blank ─────

def test_m2_section_b_unregistered_row_shows_unknown_stamp_never_blank(tmp_path):
    """Round-2 fix: the previous assertion (bare `"unknown (unregistered)"`) is non-discriminating
    — Section B's `input` column ALSO renders that exact fragment for an unresolved entry
    (`input_label = "unknown (unregistered)"`), so the old test passed even before `_stamp_columns`
    existed. Assert the full stamp-column fragment instead, which only `_stamp_columns(None)`
    produces."""
    layout = _minimal_layout(tmp_path)
    items = [{"question_id": "q0", "source_file": "doc_0", "domain": "test",
             "question": "value?", "capabilities": [], "gold_dict": {"a": "x"}}]
    layout.bank_path.write_text(json.dumps({"items": items}))
    pk = "totally-unknown-parser"   # not vlm__, not safe_name-prefix-matched, not upstream_parser
    rec = {"qid": "q0", "parser": pk, "source_file": "doc_0", "domain": "test",
           "answer": {"a": "x"}, "field_matches": {"a": True}, "match": True}
    cpath = layout.cache_path("q0", pk)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(rec))

    md = build_markdown_report(layout, [], iters=50)
    section_b = md.split("## Section B")[1]
    assert "precision: unknown (unregistered)" in section_b


# ── M6: the separability appendix states its sign convention exactly once ────────────────────────

def test_m6_appendix_states_sign_convention_once(tmp_path):
    layout, registry = _build_full_fixture(tmp_path)
    md = build_markdown_report(layout, registry, iters=200, seed=0)
    assert md.count("Δ = first label minus second label") == 1


# ── M7: a falsy/missing resolved_provider counts as its own distinct "(empty)" value for D4 ──────

def test_m7_empty_resolved_provider_counts_as_distinct_value_for_d4(tmp_path):
    layout = _minimal_layout(tmp_path)
    items = [
        {"question_id": "cb0", "source_file": "doc_0", "domain": "test",
         "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": True}},
        {"question_id": "cb1", "source_file": "doc_1", "domain": "test",
         "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": False}},
    ]
    layout.bank_path.write_text(json.dumps({"items": items}))
    pk = parser_key("vlmEmpty@mock", STAGE1_CONDITION)
    for qid, gold, provider in [("cb0", True, "ProviderA"), ("cb1", False, "")]:
        rec = {"qid": qid, "parser": pk, "source_file": f"doc_{qid[-1]}", "domain": "test",
               "condition": STAGE1_CONDITION, "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
               "image_sha": None, "resolved_provider": provider,
               "answer": {"checked": gold}, "field_matches": {"checked": True}, "match": True,
               "error_class": "none"}
        cpath = layout.cache_path(qid, pk)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(rec))
    entry = RegistryEntry(id="vlmEmpty@mock", shape="vlm-chat", transport="openai-compat",
                          base_url="http://vlmEmpty.invalid/v1", model="org/vlmEmpty",
                          api_key_env=None, precision="bf16", weights_licence="mit",
                          provider_tos_commercial="ok", provenance="Test", release_date="2025-01-01")
    with pytest.raises(ServingIdentityError, match=r"\(empty\)"):
        build_markdown_report(layout, [entry], iters=50)


# ── Round 2 HIGH regression: D4 must compare ANSWERED rows only, never errored cells ──────────────

def _d4_answered_row(pk: str, qid: str, stem: str, gold: bool, provider: str) -> dict:
    """Row shape copied from direct.py's `_one` success path — `common` dict + answer fields."""
    return {"qid": qid, "parser": pk, "source_file": stem, "domain": "test",
           "condition": STAGE1_CONDITION, "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
           "prompt_sha": "deadbeef0000", "image_sha": None, "image_px": None, "image_bytes": None,
           "raw_response": json.dumps({"checked": gold}),
           "usage": {"prompt_tokens": 100, "completion_tokens": 10},
           "resolved_provider": provider, "latency_sec": 0.5,
           "answer": {"checked": gold}, "field_matches": {"checked": True}, "match": True,
           "error_class": "none"}


def _d4_api_error_row(pk: str, qid: str, stem: str) -> dict:
    """Row shape copied from direct.py's `_one` except-branch: `{**base, "error": ..., "error_class":
    "api_error"}` — `base` NEVER carries "resolved_provider" (that key only exists on `common`,
    built AFTER a successful HTTP response), so a real api_error row has no such key at all."""
    return {"qid": qid, "parser": pk, "source_file": stem, "domain": "test",
           "condition": STAGE1_CONDITION, "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
           "prompt_sha": "deadbeef0000", "image_sha": None, "image_px": None, "image_bytes": None,
           "error": "Connection timeout", "error_class": "api_error"}


def test_d4_regression_transient_api_error_does_not_block_a_healthy_run(tmp_path):
    """[HIGH regression] Reviewer proved M7's unconditional `(r.get("resolved_provider") or
    "(empty)")` counted an api_error row's ABSENT resolved_provider as its own "(empty)" identity
    — one timeout in an otherwise single-provider (DeepInfra) run tripped the escape-hatch-free
    ServingIdentityError. Two healthy DeepInfra rows + one api_error row must build cleanly."""
    layout = _minimal_layout(tmp_path)
    items = [
        {"question_id": "cb0", "source_file": "doc_0", "domain": "test",
         "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": True}},
        {"question_id": "cb1", "source_file": "doc_1", "domain": "test",
         "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": False}},
        {"question_id": "cb2", "source_file": "doc_2", "domain": "test",
         "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": True}},
    ]
    layout.bank_path.write_text(json.dumps({"items": items}))
    pk = parser_key("vlmHealthy@mock", STAGE1_CONDITION)
    rows = [
        _d4_answered_row(pk, "cb0", "doc_0", True, "DeepInfra"),
        _d4_answered_row(pk, "cb1", "doc_1", False, "DeepInfra"),
        _d4_api_error_row(pk, "cb2", "doc_2"),
    ]
    for rec in rows:
        cpath = layout.cache_path(rec["qid"], pk)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(rec))

    entry = RegistryEntry(id="vlmHealthy@mock", shape="vlm-chat", transport="openai-compat",
                          base_url="http://vlmHealthy.invalid/v1", model="org/vlmHealthy",
                          api_key_env=None, precision="bf16", weights_licence="mit",
                          provider_tos_commercial="ok", provenance="Test", release_date="2025-01-01")
    md = build_markdown_report(layout, [entry], iters=50)   # must NOT raise
    assert "vlmHealthy@mock" in md


def test_d4_regression_still_fires_on_two_answered_rows_with_different_providers(tmp_path):
    """Companion to the regression fix above: the guard must still fire when TWO ANSWERED rows
    (both error_class == "none") disagree on provider — including the DeepInfra-vs-empty-string
    shape the reviewer specified."""
    layout = _minimal_layout(tmp_path)
    items = [
        {"question_id": "cb0", "source_file": "doc_0", "domain": "test",
         "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": True}},
        {"question_id": "cb1", "source_file": "doc_1", "domain": "test",
         "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": False}},
    ]
    layout.bank_path.write_text(json.dumps({"items": items}))
    pk = parser_key("vlmAmbig@mock", STAGE1_CONDITION)
    rows = [
        _d4_answered_row(pk, "cb0", "doc_0", True, "DeepInfra"),
        _d4_answered_row(pk, "cb1", "doc_1", False, ""),
    ]
    for rec in rows:
        cpath = layout.cache_path(rec["qid"], pk)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(rec))

    entry = RegistryEntry(id="vlmAmbig@mock", shape="vlm-chat", transport="openai-compat",
                          base_url="http://vlmAmbig.invalid/v1", model="org/vlmAmbig",
                          api_key_env=None, precision="bf16", weights_licence="mit",
                          provider_tos_commercial="ok", provenance="Test", release_date="2025-01-01")
    with pytest.raises(ServingIdentityError, match=r"\(empty\)"):
        build_markdown_report(layout, [entry], iters=50)


# ── M8: a parse dir with zero .md files renders transcript-recall n/a, never a fake 0.0 ──────────

def test_m8_transcript_recall_na_when_parse_dir_has_zero_md_files(tmp_path):
    layout = _minimal_layout(tmp_path)
    t_pk = "empty-parser"
    parser_dir = layout.parser_dir(t_pk)
    parser_dir.mkdir(parents=True, exist_ok=True)
    (parser_dir / "condition.json").write_text("{}")   # a sidecar exists, but zero transcripts
    checkbox_items = [{"source_file": "doc_0", "gold_dict": {"checked": True}}]
    assert _transcript_recall(layout, t_pk, checkbox_items) is None


# ── M4/M5/M9: glossary disclosures (CI-below-floor, Section B staleness, bank-wide null n) ───────

def test_m4_m5_m9_glossary_disclosures_present(tmp_path):
    layout, registry = _build_full_fixture(tmp_path)
    md = build_markdown_report(layout, registry, iters=200, seed=0)
    assert "CI-below-floor caveat" in md
    assert "Section B staleness is not assessed" in md
    assert "bank-wide" in md.lower()
    # Round-2 disclosure (3a): transcript-recall's snake_case/word-boundary caveat must be in the
    # operator-facing glossary (METRIC_DEFINITIONS), not just the internal _field_tokens docstring.
    assert "Snake_case field keys" in md
    assert "conservative undercount" in md.lower()
