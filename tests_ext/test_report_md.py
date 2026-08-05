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
    _direct_cost_latency,
    _general_and_strict,
    _input_label,
    _mixed_precision_note,
    _pairwise_separability,
    _section_mixed_precision_note,
    _stamp_columns,
    _tos_stamp,
    _transcript_recall,
    _transcription_cost_latency,
    _upstream_construction_metrics,
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


# ── R-c: report header renders pymupdf_version from run_meta ───────────────────────────────────

def test_header_renders_pymupdf_version_from_run_meta(tmp_path):
    layout = _minimal_layout(tmp_path)
    layout.bank_path.write_text(json.dumps({"items": []}))
    (layout.root / "run_meta.json").write_text(json.dumps({
        "dataset_revision": "test-rev", "harness_commit": "test-commit",
        "renders_verified": True, "pymupdf_version": "1.24.9",
    }))
    md = build_markdown_report(layout, [], iters=50)
    assert "pymupdf version: 1.24.9" in md


def test_header_pymupdf_version_falls_back_to_unknown(tmp_path):
    layout, registry = _build_full_fixture(tmp_path)   # meta fixture carries no pymupdf_version
    md = build_markdown_report(layout, registry, iters=200, seed=0)
    assert "pymupdf version: unknown" in md


# ── F2: reproduction-gate comparability — upstream construction vs the D7 ranking key ──────────

def test_f2_general_and_strict_vs_upstream_construction_diverge_on_null_key_absent():
    """The exact D7 divergence: a null-gold field whose answer key is ABSENT. Upstream's own
    `deep_equal(None, None)` already scored this correct at write time (baked into
    `field_matches`); the ranking-key `field_outcomes`/`_general_and_strict` (D7) overrides that
    specific case to incorrect. Same rows, same items — the two constructions must disagree."""
    items = [{"question_id": "q0", "source_file": "doc_0", "gold_dict": {"notes": None}}]
    all_fields = [("q0", "notes", None, "doc_0")]
    rows_by_qid = {
        # "answer" present (this is an OK row) but does NOT carry "notes" at all — upstream's
        # own score_typed would have written field_matches["notes"] = True for exactly this case
        # (deep_equal(None, None) treats key-absent-so-.get()-returns-None the same as explicit
        # None); D7 disagrees and scores it "incorrect" instead.
        "q0": {"answer": {}, "field_matches": {"notes": True}, "match": True},
    }
    d7_general, _d7_strict = _general_and_strict(rows_by_qid, items, all_fields)
    upstream_field_pct, _upstream_question_pct, n_ok = _upstream_construction_metrics(rows_by_qid)

    assert d7_general == 0.0                # D7 ranking key: null-gold key-absent -> incorrect
    assert upstream_field_pct == 1.0         # upstream construction: the same field -> correct
    assert n_ok == 1
    assert d7_general != upstream_field_pct  # the two constructions genuinely disagree


def test_f2_upstream_construction_metrics_excludes_error_rows():
    """"over ok rows only" — an error/missing row contributes neither fields nor a question to
    either denominator."""
    rows_by_qid = {
        "q0": {"answer": {"a": True}, "field_matches": {"a": True}, "match": True},
        "q1": {"error": "boom"},   # not ok — must be excluded entirely
    }
    field_pct, question_pct, n_ok = _upstream_construction_metrics(rows_by_qid)
    assert n_ok == 1
    assert field_pct == 1.0
    assert question_pct == 1.0


def test_f2_reproduction_gate_block_renders_with_both_diverging_numbers(tmp_path):
    """End-to-end: the new block exists, carries its caveat label verbatim, and — built from the
    same D7-divergent fixture as the unit test above — its field% number actually differs from
    the ranking-key `general/field` number in the Section B table above it."""
    layout = _minimal_layout(tmp_path)
    items = [{"question_id": "q0", "source_file": "doc_0", "domain": "test",
             "question": "any notes?", "capabilities": ["blank_field"],
             "gold_dict": {"notes": None}}]
    layout.bank_path.write_text(json.dumps({"items": items}))
    t_pk = f"{safe_name('reprogate@local')}__{condition_hash(TRANSCRIBER_CONDITION)}"
    rec = {"qid": "q0", "parser": t_pk, "source_file": "doc_0", "domain": "test",
           "answer": {}, "field_matches": {"notes": True}, "match": True}
    cpath = layout.cache_path("q0", t_pk)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(rec))
    entry = RegistryEntry(id="reprogate@local", shape="transcriber", transport="openai-compat",
                          base_url="http://reprogate.invalid/v1", model="org/reprogate",
                          api_key_env=None, precision="bf16", weights_licence="mit",
                          provider_tos_commercial="ok", provenance="Test",
                          release_date="2025-01-01", local=True)

    md = build_markdown_report(layout, [entry], iters=50)
    assert "## Reproduction gate (upstream construction)" in md
    assert ("upstream construction — for the reproduction gate only; not the ranking key" in md)

    gate_block = md.split("## Reproduction gate (upstream construction)")[1] \
                   .split("## Separability appendix")[0]
    section_b_block = md.split("## Section B")[1].split("## Reproduction gate")[0]

    # the SAME field: 100% under upstream construction, 0% under the D7 ranking key.
    assert "| reprogate@local | 100.0% | 100.0% | 1 |" in gate_block
    assert "| 0.0% | 100.0% |" in section_b_block   # general/field=0.0%, strict/question=100.0%


# ── F4: label disambiguation when >1 parser key resolves to one entry id ───────────────────────

def test_f4_two_condition_hashes_for_one_entry_get_disambiguated_labels(tmp_path):
    """F4 (live-validation finding): the SAME registry entry direct-answered under two different
    condition hashes (e.g. default max_tokens vs a widened one for a thinking model) must render
    with distinguishable labels in the table AND detail bullets AND the separability appendix —
    instead of both collapsing onto the bare entry.id."""
    layout = _minimal_layout(tmp_path)
    items = [
        {"question_id": "cb0", "source_file": "doc_0", "domain": "test",
         "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": True}},
        {"question_id": "cb1", "source_file": "doc_1", "domain": "test",
         "question": "checked?", "capabilities": ["checkbox_state"], "gold_dict": {"checked": False}},
    ]
    layout.bank_path.write_text(json.dumps({"items": items}))

    cond_a = STAGE1_CONDITION
    cond_b = {**STAGE1_CONDITION, "sampling": {**STAGE1_CONDITION["sampling"], "max_tokens": 8192}}
    pk_a, pk_b = parser_key("vlmDup@mock", cond_a), parser_key("vlmDup@mock", cond_b)
    assert pk_a != pk_b

    for pk, cond in [(pk_a, cond_a), (pk_b, cond_b)]:
        for qid, stem, gold in [("cb0", "doc_0", True), ("cb1", "doc_1", False)]:
            rec = {"qid": qid, "parser": pk, "source_file": stem, "domain": "test",
                   "condition": cond, "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
                   "image_sha": None, "resolved_provider": "ProviderA",
                   "answer": {"checked": gold}, "field_matches": {"checked": True}, "match": True,
                   "error_class": "none"}
            cpath = layout.cache_path(qid, pk)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps(rec))

    entry = RegistryEntry(id="vlmDup@mock", shape="vlm-chat", transport="openai-compat",
                          base_url="http://vlmDup.invalid/v1", model="org/vlmDup",
                          api_key_env=None, precision="bf16", weights_licence="mit",
                          provider_tos_commercial="ok", provenance="Test", release_date="2025-01-01")

    md = build_markdown_report(layout, [entry], iters=50)
    hash_a, hash_b = pk_a.rsplit("__", 1)[-1], pk_b.rsplit("__", 1)[-1]
    label_a, label_b = f"vlmDup@mock [cond {hash_a}]", f"vlmDup@mock [cond {hash_b}]"

    section_a = md.split("## Section A")[1].split("## Baseline rows")[0]
    assert label_a in section_a and label_b in section_a
    assert "vlmDup@mock |" not in section_a   # the bare (collision-prone) label never renders
    assert f"- **{label_a}**" in section_a    # detail bullet
    assert f"- **{label_b}**" in section_a

    appendix = md.split("## Separability appendix")[1]
    assert label_a in appendix and label_b in appendix


def test_f4_find_calibration_pairs_returns_both_condition_hash_pairs():
    """F4: two direct-QA condition-hash groups for one entry, both matching the same section_b
    transcriber's model, are TWO distinct legitimate calibration pairs — not something to
    collapse down to one via entry-object deduplication."""
    from ocr_eval_ext.report_md import _find_calibration_pairs, _Group

    v_entry = RegistryEntry(id="vDup@mock", shape="vlm-chat", transport="openai-compat",
                            base_url="http://v.invalid/v1", model="org/dup", api_key_env=None,
                            precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
                            provenance="Test", release_date="2025-01-01")
    t_entry = RegistryEntry(id="tDup@local", shape="transcriber", transport="openai-compat",
                            base_url="http://t.invalid/v1", model="org/dup", api_key_env=None,
                            precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
                            provenance="Test", release_date="2025-01-01", local=True)
    v_pk_a, v_pk_b = "vlm__vDup@mock__aaaaaaaaaaaa", "vlm__vDup@mock__bbbbbbbbbbbb"
    t_pk = "tDup_local__cccccccccccc"
    direct_groups = {v_pk_a: _Group(v_pk_a, v_entry, {}), v_pk_b: _Group(v_pk_b, v_entry, {})}
    section_b_groups = {t_pk: _Group(t_pk, t_entry, {})}

    pairs = _find_calibration_pairs(direct_groups, section_b_groups)
    assert sorted(pairs) == [(v_pk_a, t_pk), (v_pk_b, t_pk)]


def test_f4_calibration_pair_line_uses_disambiguated_labels():
    """The calibration-pair prose line itself (`_build_cross_shape_note`) must render each
    condition-hash pair with its OWN disambiguated label, not the shared bare entry.id twice."""
    from ocr_eval_ext.report_md import _build_cross_shape_note, _disambiguated_labels, _Group

    v_entry = RegistryEntry(id="vDup2@mock", shape="vlm-chat", transport="openai-compat",
                            base_url="http://v2.invalid/v1", model="org/dup2", api_key_env=None,
                            precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
                            provenance="Test", release_date="2025-01-01")
    t_entry = RegistryEntry(id="tDup2@local", shape="transcriber", transport="openai-compat",
                            base_url="http://t2.invalid/v1", model="org/dup2", api_key_env=None,
                            precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
                            provenance="Test", release_date="2025-01-01", local=True)
    v_pk_a, v_pk_b = "vlm__vDup2@mock__aaaaaaaaaaaa", "vlm__vDup2@mock__bbbbbbbbbbbb"
    t_pk = "tDup2_local__cccccccccccc"
    direct_groups = {v_pk_a: _Group(v_pk_a, v_entry, {}), v_pk_b: _Group(v_pk_b, v_entry, {})}
    section_b_groups = {t_pk: _Group(t_pk, t_entry, {})}
    direct_labels = _disambiguated_labels(direct_groups)
    section_b_labels = _disambiguated_labels(section_b_groups)

    lines = _build_cross_shape_note([], [], direct_groups, section_b_groups,
                                    direct_labels, section_b_labels)
    text = "\n".join(lines)
    assert text.count("calibration pair detected") == 2   # one line per condition-hash pair
    assert "vDup2@mock [cond aaaaaaaaaaaa]" in text
    assert "vDup2@mock [cond bbbbbbbbbbbb]" in text


# ── F5: contract stamp (promptable/not-promptable) + tos_note rendered for ok entries too ───────

def _stamp_entry(**overrides) -> RegistryEntry:
    base = dict(id="stampX@test", shape="vlm-chat", transport="openai-compat",
               base_url="http://stampX.invalid/v1", model="org/stampX", api_key_env=None,
               precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
               provenance="Test", release_date="2025-01-01")
    base.update(overrides)
    return RegistryEntry(**base)


def test_stamp_columns_renders_promptable_contract():
    e = _stamp_entry(promptable=True)
    assert "contract: promptable" in _stamp_columns(e)


def test_stamp_columns_renders_not_promptable_contract():
    e = _stamp_entry(promptable=False)
    assert "contract: not-promptable" in _stamp_columns(e)


def test_stamp_columns_unregistered_includes_contract_as_unknown():
    assert "contract: unknown (unregistered)" in _stamp_columns(None)


# ── the cross-shape warning must name the LIVE extractor, not a stale literal ─────────────────
# Regression guard: D11 moved score.py's DEFAULT_MODEL and this warning kept naming the old
# model, so one report contradicted itself — line 49 said gemini-3-flash-preview while the
# Section B header said gemini-3.1-flash-lite.

def test_cross_shape_warning_names_the_live_extractor_model():
    from realdoc_bench.evaluate.score import DEFAULT_MODEL

    from ocr_eval_ext.report_md import CROSS_SHAPE_WARNING

    assert f"`{DEFAULT_MODEL}`" in CROSS_SHAPE_WARNING


# ── Section B `input` column: explicit input_mode overrides the transport proxy ──────────────

def _transcriber_entry(**overrides) -> RegistryEntry:
    base = dict(id="inputX@test", shape="transcriber", transport="upstream-parser",
                upstream_parser="input_x", precision="provider-default", weights_licence="closed",
                provider_tos_commercial="ok", provenance="Test", release_date="2025-01-01")
    base.update(overrides)
    return RegistryEntry(**base)


def test_input_label_defaults_to_transport_proxy():
    """Unchanged behaviour for every pre-existing entry: openai-compat reads raster-png,
    anything else reads pdf-direct."""
    assert _input_label(_stamp_entry()) == "raster-png"
    assert _input_label(_transcriber_entry()) == "pdf-direct"


def test_input_label_honours_explicit_raster_png_on_an_upstream_parser_entry():
    """docstrange@nanonets is an upstream-parser entry whose adapter rasterizes. Inheriting the
    transport proxy would label it pdf-direct and hang the embedded-text-layer free-ride caveat
    on a row that only ever sees a PNG."""
    assert _input_label(_transcriber_entry(input_mode="raster-png")) == "raster-png"


def test_input_label_honours_explicit_pdf_direct_on_an_openai_compat_entry():
    assert _input_label(_stamp_entry(input_mode="pdf-direct")) == "pdf-direct"


def test_input_label_unregistered_entry():
    assert _input_label(None) == "unknown (unregistered)"


def test_tos_stamp_renders_note_for_ok_entries_too():
    """F5: previously only blocked/conditional entries surfaced tos_note — an `ok`-commercial
    entry with a real caveat (e.g. mistral-ocr@mistral's zero-retention-by-contract note) used to
    render a bare "ToS: ok" with the caveat silently dropped."""
    e = _stamp_entry(provider_tos_commercial="ok",
                     tos_note="zero-retention by contract only, not platform default")
    stamp = _tos_stamp(e)
    assert stamp.startswith("ToS: ok — ")
    assert "zero-retention by contract only" in stamp


def test_tos_stamp_ok_with_no_note_stays_bare():
    e = _stamp_entry(provider_tos_commercial="ok", tos_note="")
    assert _tos_stamp(e) == "ToS: ok"


def test_tos_stamp_truncates_note_to_60_chars():
    long_note = "x" * 200
    e = _stamp_entry(provider_tos_commercial="ok", tos_note=long_note)
    stamp = _tos_stamp(e)
    assert stamp == f"ToS: ok — {'x' * 60}"
    assert len(stamp) < len(long_note)


def test_registry_mistral_tos_note_carries_the_gate3_zero_retention_finding():
    """F5/ledger T2-N3: the registry's own tos_note (not just a test fixture) must state the real
    gate3 finding, not merely restate `promptable: false`."""
    from ocr_eval_ext.config import get_entry, load_registry

    entries = load_registry(Path(__file__).resolve().parents[1] / "configs" / "registry.yaml")
    mistral = get_entry(entries, "mistral-ocr@mistral")
    assert "zero-retention" in mistral.tos_note.lower()
    stamp = _tos_stamp(mistral)
    assert "zero-retention" in stamp.lower()   # survives the 60-char truncation


# ── F6: mixed-precision caveat treats "unknown (not asserted)" as its own distinct value ───────

def test_mixed_precision_fires_on_two_known_precisions():
    note = _mixed_precision_note({"bf16", "fp8-vllm"})
    assert note is not None and "mixed precision" in note


def test_mixed_precision_fires_when_known_mixes_with_unasserted():
    """F6: a known precision alongside an unasserted one is a real ambiguity — flagged as mixed,
    not silently treated as "only one known precision, so no caveat"."""
    note = _mixed_precision_note({"bf16", "unknown (not asserted)"})
    assert note is not None and "mixed precision" in note


def test_mixed_precision_all_unasserted_gets_the_distinct_note():
    note = _mixed_precision_note({"unknown (not asserted)"})
    assert note == "precision unasserted across all rows in this section"


def test_mixed_precision_single_known_precision_is_silent():
    assert _mixed_precision_note({"bf16"}) is None


def test_mixed_precision_empty_set_is_silent():
    assert _mixed_precision_note(set()) is None


def test_section_mixed_precision_note_uses_precision_label_not_raw_field():
    """F6: `_section_mixed_precision_note` must build its set from `_precision_label(e)`
    (provider-default -> "unknown (not asserted)"), not the raw `entry.precision` value that the
    pre-fix code filtered out entirely."""
    known = RegistryEntry(id="known@t", shape="vlm-chat", transport="openai-compat",
                          base_url="http://k.invalid/v1", model="org/k", api_key_env=None,
                          precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
                          provenance="Test", release_date="2025-01-01")
    unasserted = RegistryEntry(id="unasserted@t", shape="vlm-chat", transport="openai-compat",
                               base_url="http://u.invalid/v1", model="org/u", api_key_env=None,
                               precision="provider-default", weights_licence="mit",
                               provider_tos_commercial="ok", provenance="Test",
                               release_date="2025-01-01")
    assert _section_mixed_precision_note([known]) is None
    note = _section_mixed_precision_note([known, unasserted])
    assert note is not None and "mixed precision" in note
    all_unasserted_note = _section_mixed_precision_note([unasserted])
    assert all_unasserted_note == "precision unasserted across all rows in this section"


def test_full_report_all_unasserted_section_a_gets_distinct_note(tmp_path):
    """End-to-end: a Section A built entirely from provider-default entries renders the distinct
    all-unasserted note, not the "mixed precision" wording (which would misleadingly imply an
    actual precision DIFFERENCE exists)."""
    layout = _minimal_layout(tmp_path)
    items = [{"question_id": f"q{i}", "source_file": f"doc_{i}", "domain": "test",
             "question": "checked?", "capabilities": ["checkbox_state"],
             "gold_dict": {"checked": True}} for i in range(2)]
    layout.bank_path.write_text(json.dumps({"items": items}))
    entries = []
    for i, qid in enumerate(["q0", "q1"]):
        entry_id = f"unasserted{i}@mock"
        entries.append(RegistryEntry(id=entry_id, shape="vlm-chat", transport="openai-compat",
                                     base_url=f"http://u{i}.invalid/v1", model=f"org/u{i}",
                                     api_key_env=None, precision="provider-default",
                                     weights_licence="mit", provider_tos_commercial="ok",
                                     provenance="Test", release_date="2025-01-01"))
        pk = parser_key(entry_id, STAGE1_CONDITION)
        rec = {"qid": qid, "parser": pk, "source_file": qid.replace("q", "doc_"), "domain": "test",
               "condition": STAGE1_CONDITION, "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
               "image_sha": None, "resolved_provider": "ProviderA",
               "answer": {"checked": True}, "field_matches": {"checked": True}, "match": True,
               "error_class": "none"}
        cpath = layout.cache_path(qid, pk)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(rec))

    md = build_markdown_report(layout, entries, iters=50)
    section_a = md.split("## Section A")[1].split("## Baseline rows")[0]
    assert "precision unasserted across all rows in this section" in section_a
    assert "mixed precision" not in section_a


# ── F7: local rows still render a real measured-latency median; cost stays "n/a" ───────────────

def test_direct_cost_latency_local_row_renders_measured_median_latency():
    local_entry = RegistryEntry(id="loc@local", shape="vlm-chat", transport="openai-compat",
                                base_url="http://loc.invalid/v1", model="org/loc",
                                api_key_env=None, precision="bf16", weights_licence="mit",
                                provider_tos_commercial="ok", provenance="Test",
                                release_date="2025-01-01", local=True)
    rows = [{"latency_sec": 1.0}, {"latency_sec": 3.0}, {"latency_sec": 2.0}]
    lat, cost = _direct_cost_latency(local_entry, rows)
    assert lat == "2.00s"     # median of 1/2/3, not "n/a"
    assert cost == "n/a"      # cost is still never fabricated for a local row


def test_direct_cost_latency_local_row_with_no_latency_data_stays_na():
    local_entry = RegistryEntry(id="loc2@local", shape="vlm-chat", transport="openai-compat",
                                base_url="http://loc2.invalid/v1", model="org/loc2",
                                api_key_env=None, precision="bf16", weights_licence="mit",
                                provider_tos_commercial="ok", provenance="Test",
                                release_date="2025-01-01", local=True)
    lat, cost = _direct_cost_latency(local_entry, [])
    assert lat == "n/a"
    assert cost == "n/a"


def test_transcription_cost_latency_local_row_renders_measured_median_latency(tmp_path):
    layout = _minimal_layout(tmp_path)
    local_entry = RegistryEntry(id="tloc@local", shape="transcriber", transport="openai-compat",
                                base_url="http://tloc.invalid/v1", model="org/tloc",
                                api_key_env=None, precision="bf16", weights_licence="mit",
                                provider_tos_commercial="ok", provenance="Test",
                                release_date="2025-01-01", local=True)
    parser_dir = layout.parser_dir("tloc_local__cond")
    parser_dir.mkdir(parents=True, exist_ok=True)
    for stem, latency in [("doc_0", 1.0), ("doc_1", 3.0), ("doc_2", 2.0)]:
        (parser_dir / f"{stem}.json").write_text(json.dumps({
            "parser": "tloc_local__cond", "stem": stem, "ok": True, "page_count": 1,
            "latency_sec": latency, "cost_usd": 0.01, "md_length": 20,
            "elapsed_total": latency, "error": "",
        }))
    lat, cost = _transcription_cost_latency(layout, local_entry, "tloc_local__cond")
    assert lat == "2.00s"    # median of 1/2/3, not "n/a"
    assert cost == "n/a"     # cost is still never fabricated for a local row, even though
                              # cost_usd sidecars exist on disk


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
