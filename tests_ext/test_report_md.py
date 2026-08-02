"""Task 9: shape-segregated markdown report. Builds a synthetic run dir (two identical
vlm-chat entries + one local transcriber + parses/*.md) and asserts on the rendered markdown —
section order, baseline values, separability, ToS/CC-BY stamps, transcript-recall, INCOMPLETE
marking, the D3/D4 hard-fail guards, and the partial-corpus banner. iters=50-200 throughout
(never the CLI's default 2000) — these are correctness tests, not statistical-precision tests."""
import datetime as dt
import hashlib
import json
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
    # only (glyph + "checked" token) — the other 17 have a transcript but no recall pattern.
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

    # ── transcript-recall: 5 of 22 checkbox docs carry glyph+token ─────────────────────────
    assert "transcript-recall" in md.lower()
    assert "22.7%" in md   # 5/22 rounded to one decimal

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
