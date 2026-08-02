import json
import re
import shutil
import subprocess
from pathlib import Path

import fitz  # pymupdf
import pytest
import yaml
from typer.testing import CliRunner

import ocr_eval_ext.cli as cli_mod
import ocr_eval_ext.direct as direct_mod
from ocr_eval_ext.cli import PINS_PATH, REPO_ROOT, app, require_extractor_gate
from realdoc_bench.evaluate.runs import RunLayout

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_REAL_BANK_MISSING = not (REPO_ROOT / "runs" / "stage1" / "qa_bank.json").exists()
_UV_MISSING = shutil.which("uv") is None


def _plain(output: str) -> str:
    """Console(width=200) — the brief's exact construction — auto-detects a color system at
    import time from this session's real stdout and caches it, so Rich highlights numeric
    tokens with ANSI codes in `result.output` even though CliRunner captures a plain pipe.
    Strip them before substring-matching numeric content."""
    return _ANSI_RE.sub("", output)


def test_verify_fails_on_tiny_bank(tmp_path):
    run = tmp_path / "run"
    (run / "docs").mkdir(parents=True)
    (run / "qa_bank.json").write_text(json.dumps({"items": [
        {"question_id": "q1", "source_file": "d", "capabilities": [], "gold_dict": {"a": 1}}]}))
    result = runner.invoke(app, ["verify", "--run-dir", str(run)])
    assert result.exit_code == 1
    assert "cardinality mismatch" in result.output


def test_selftest_command_passes():
    result = runner.invoke(app, ["selftest"])
    assert result.exit_code == 0
    assert "offline scorer self-test: PASS" in result.output


@pytest.mark.skipif(_UV_MISSING, reason="uv not on PATH")
def test_ocr_eval_entrypoint_help_exits_zero():
    """Task 1's declared `ocr-eval` console_script entry point must actually resolve and run —
    not just the in-process typer `app` object exercised by CliRunner above."""
    result = subprocess.run(["uv", "run", "ocr-eval", "--help"],
                            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert "verify" in result.stdout and "selftest" in result.stdout


# ── REPO_ROOT / PINS_PATH sanity (item 4: package-relative pins path) ─────────────────────────

def test_pins_path_is_package_relative_and_exists():
    assert PINS_PATH == REPO_ROOT / "configs" / "pins.yaml"
    assert PINS_PATH.exists()


# ── verify against the REAL bank (item 2) ─────────────────────────────────────────────────────

@pytest.mark.skipif(_REAL_BANK_MISSING, reason="real 581-PDF corpus not present in this checkout")
def test_verify_skip_renders_passes_on_real_bank():
    """`runs/stage1/` ships the real 581-PDF corpus + bank at the pinned revision. --skip-renders
    keeps this test fast (pins + cardinality only, no render sweep) while still proving the real
    bank satisfies `check_bank`, the harness-commit pin resolves against actual HEAD, and (C1)
    every bank source_file has a matching PDF on disk."""
    real_run_dir = REPO_ROOT / "runs" / "stage1"
    result = runner.invoke(app, ["verify", "--run-dir", str(real_run_dir), "--skip-renders"])
    assert result.exit_code == 0, result.output
    assert "verify PASS" in result.output
    assert "renders skipped" in result.output


# ── C1: corpus/bank completeness ───────────────────────────────────────────────────────────────

def test_verify_fails_on_corpus_bank_mismatch(tmp_path, monkeypatch):
    """An empty (or `evaluate download --limit`-narrowed) docs/ against a real-shaped bank must
    fail loudly, not print 'all green' and stamp run_meta. Bank cardinality is bypassed here
    (monkeypatched) so the corpus-completeness gate is exercised in isolation."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((20, 20), "present")
    doc.save(layout.docs_dir / "present.pdf")
    bank = {"items": [
        {"question_id": "q1", "source_file": "present", "gold_dict": {"a": 1}},
        {"question_id": "q2", "source_file": "missing_doc", "gold_dict": {"a": 1}},
    ]}
    layout.bank_path.write_text(json.dumps(bank))

    result = runner.invoke(app, ["verify", "--run-dir", str(layout.root), "--skip-renders"])
    assert result.exit_code == 1
    assert "corpus incomplete" in _plain(result.output)
    assert "missing_doc" in result.output
    # _preflight (base dataset_revision/harness_commit identity) legitimately succeeded before
    # this run reached the corpus check — but it must never claim renders_verified.
    meta_path = layout.root / "run_meta.json"
    if meta_path.exists():
        assert "renders_verified" not in json.loads(meta_path.read_text())


# ── run_meta.json stamping (items 2 & 6) ────────────────────────────────────────────────────────

def _fake_layout_with_one_good_page(tmp_path: Path) -> RunLayout:
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "hello world")
    page.draw_rect(fitz.Rect(72, 100, 400, 400), fill=(0, 0, 0))  # well above the ink floor
    doc.save(layout.docs_dir / "doc_1.pdf")
    layout.bank_path.write_text(json.dumps({"items": []}))
    return layout


def test_verify_stamps_run_meta_on_first_success(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})   # bypass the 1356-item gate
    layout = _fake_layout_with_one_good_page(tmp_path)
    result = runner.invoke(app, ["verify", "--run-dir", str(layout.root)])
    assert result.exit_code == 0, result.output
    meta = json.loads((layout.root / "run_meta.json").read_text())
    pins = yaml.safe_load(PINS_PATH.read_text())
    assert meta["dataset_revision"] == pins["dataset_revision"]
    assert meta["harness_commit"] == pins["harness_commit"]
    assert meta["pymupdf_version"]   # non-empty, importlib.metadata resolved
    assert meta["renders_verified"] is True   # I2b: only the full sweep sets this
    assert meta["revision_source"] == "pins"  # no .cache/huggingface/ present in this fake layout


def test_verify_skip_renders_stamps_base_meta_but_not_renders_verified(tmp_path, monkeypatch):
    """I2a: `_preflight` stamps dataset_revision/harness_commit on ANY success, including
    --skip-renders — but --skip-renders never confirms per-page renders, so it must NOT set
    renders_verified (that's the full sweep's job alone, I2b)."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _fake_layout_with_one_good_page(tmp_path)
    result = runner.invoke(app, ["verify", "--run-dir", str(layout.root), "--skip-renders"])
    assert result.exit_code == 0, result.output
    meta = json.loads((layout.root / "run_meta.json").read_text())
    assert meta["dataset_revision"] and meta["harness_commit"]
    assert "renders_verified" not in meta


def test_verify_second_call_does_not_restamp(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _fake_layout_with_one_good_page(tmp_path)
    runner.invoke(app, ["verify", "--run-dir", str(layout.root)])
    first = (layout.root / "run_meta.json").read_text()
    result = runner.invoke(app, ["verify", "--run-dir", str(layout.root)])
    assert result.exit_code == 0
    assert (layout.root / "run_meta.json").read_text() == first


def test_preflight_fails_on_revision_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _fake_layout_with_one_good_page(tmp_path)
    (layout.root / "run_meta.json").write_text(json.dumps({"dataset_revision": "stale-rev"}))
    result = runner.invoke(app, ["verify", "--run-dir", str(layout.root), "--skip-renders"])
    assert result.exit_code == 1
    assert "stale-rev" in result.output


def test_preflight_fails_on_harness_commit_mismatch(tmp_path, monkeypatch):
    """M4 fix (via I2c): _preflight cross-checks the stamped harness_commit back against pins,
    not just dataset_revision."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _fake_layout_with_one_good_page(tmp_path)
    pins = yaml.safe_load(PINS_PATH.read_text())
    fake_commit = "0" * 40
    (layout.root / "run_meta.json").write_text(json.dumps({
        "dataset_revision": pins["dataset_revision"], "harness_commit": fake_commit}))
    result = runner.invoke(app, ["verify", "--run-dir", str(layout.root), "--skip-renders"])
    assert result.exit_code == 1
    assert fake_commit in result.output


def test_preflight_fails_when_observed_hf_revision_mismatches_pins(tmp_path, monkeypatch):
    """I2: the observed-revision check is a REAL check against on-disk HF snapshot metadata,
    independent of whatever run_meta.json claims."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _fake_layout_with_one_good_page(tmp_path)
    trees_dir = layout.root / ".cache" / "huggingface" / "trees"
    trees_dir.mkdir(parents=True)
    wrong_rev = "f" * 40
    (trees_dir / f"{wrong_rev}.json").write_text("{}")

    result = runner.invoke(app, ["verify", "--run-dir", str(layout.root), "--skip-renders"])
    assert result.exit_code == 1
    assert wrong_rev in result.output
    assert not (layout.root / "run_meta.json").exists()   # never stamped on a failed preflight


def test_stamp_uses_observed_hf_revision_via_trees_dir_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _fake_layout_with_one_good_page(tmp_path)
    pins = yaml.safe_load(PINS_PATH.read_text())
    trees_dir = layout.root / ".cache" / "huggingface" / "trees"
    trees_dir.mkdir(parents=True)
    (trees_dir / f"{pins['dataset_revision']}.json").write_text("{}")

    result = runner.invoke(app, ["verify", "--run-dir", str(layout.root), "--skip-renders"])
    assert result.exit_code == 0, result.output
    meta = json.loads((layout.root / "run_meta.json").read_text())
    assert meta["dataset_revision"] == pins["dataset_revision"]
    assert meta["revision_source"] == "hf_metadata"


def test_stamp_prefers_qa_bank_metadata_sidecar_over_trees_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _fake_layout_with_one_good_page(tmp_path)
    pins = yaml.safe_load(PINS_PATH.read_text())
    dl_dir = layout.root / ".cache" / "huggingface" / "download"
    dl_dir.mkdir(parents=True)
    (dl_dir / "qa_bank.json.metadata").write_text(f"{pins['dataset_revision']}\nsome-etag\n123.0\n")

    result = runner.invoke(app, ["verify", "--run-dir", str(layout.root), "--skip-renders"])
    assert result.exit_code == 0, result.output
    meta = json.loads((layout.root / "run_meta.json").read_text())
    assert meta["dataset_revision"] == pins["dataset_revision"]
    assert meta["revision_source"] == "hf_metadata"


# ── I1: warm PNG cache must not disarm the multi-page sweep ────────────────────────────────────

def test_verify_catches_pdf_swapped_to_multi_page_despite_warm_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _fake_layout_with_one_good_page(tmp_path)
    pdf_path = layout.docs_dir / "doc_1.pdf"

    first = runner.invoke(app, ["verify", "--run-dir", str(layout.root)])
    assert first.exit_code == 0, first.output
    assert (layout.root / "docs_png").exists()   # cache warmed

    doc = fitz.open()
    doc.new_page(width=612, height=792).insert_text((72, 72), "page one")
    doc.new_page(width=612, height=792).insert_text((72, 72), "page two")
    doc.save(pdf_path)   # overwrite in place — the warm PNG cache entry is now stale

    second = runner.invoke(app, ["verify", "--run-dir", str(layout.root)])
    assert second.exit_code == 1
    assert "multi-page" in second.output
    assert "doc_1" in second.output


# ── M8: informational dirty-tree warning ────────────────────────────────────────────────────────

def test_verify_warns_on_dirty_harness_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _fake_layout_with_one_good_page(tmp_path)
    real_run = cli_mod.subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=" M realdoc_bench/cli.py\n", stderr="")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    result = runner.invoke(app, ["verify", "--run-dir", str(layout.root), "--skip-renders"])
    assert result.exit_code == 0, result.output
    assert "uncommitted changes" in _plain(result.output)


# ── require_extractor_gate stamp/cache logic (item 7, + M5 stale-stamp pruning) ────────────────

def test_require_extractor_gate_stamps_and_caches(tmp_path, monkeypatch):
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    calls = {"n": 0}

    def fake_run_extractor():
        calls["n"] += 1
        return []

    monkeypatch.setattr(cli_mod.st, "run_extractor", fake_run_extractor)
    require_extractor_gate(layout)
    assert calls["n"] == 1
    stamps = list(layout.root.glob(".extractor_ok_*"))
    assert len(stamps) == 1
    require_extractor_gate(layout)                 # second call: cached, no re-run
    assert calls["n"] == 1


def test_require_extractor_gate_raises_and_does_not_stamp_on_failure(tmp_path, monkeypatch):
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    monkeypatch.setattr(cli_mod.st, "run_extractor",
                        lambda: ["extractor missed: 'q' -> None"])
    with pytest.raises(cli_mod.PreconditionError, match="extractor validation FAILED"):
        require_extractor_gate(layout)
    assert list(layout.root.glob(".extractor_ok_*")) == []


def test_require_extractor_gate_prunes_stale_stamps(tmp_path, monkeypatch):
    """M5: a stale `.extractor_ok_<model>_<oldhash>` from before EXTRACTOR_FIXTURES last changed
    must not survive alongside a freshly-written stamp."""
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    from realdoc_bench.evaluate.score import DEFAULT_MODEL
    stale = layout.root / f".extractor_ok_{DEFAULT_MODEL}_deadbeef"
    stale.touch()
    monkeypatch.setattr(cli_mod.st, "run_extractor", lambda: [])

    require_extractor_gate(layout)

    stamps = sorted(p.name for p in layout.root.glob(f".extractor_ok_{DEFAULT_MODEL}_*"))
    assert stale.name not in stamps
    assert len(stamps) == 1


# ── rescore ──────────────────────────────────────────────────────────────────────────────────

def _make_rescore_run_dir(tmp_path: Path) -> RunLayout:
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "irrelevant — rescore never renders")
    doc.save(layout.docs_dir / "doc_1.pdf")
    bank = {"items": [{
        "question_id": "q1", "source_file": "doc_1", "domain": "test",
        "question": "Is question 1 checkbox marked?", "capabilities": ["checkbox_state"],
        "gold_dict": {"a": True},
        "response_format": "Return exactly: a=<boolean>",
        "gold_answer": "a=true",
    }]}
    layout.bank_path.write_text(json.dumps(bank))
    return layout


def test_rescore_heals_corrupted_field_matches(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_rescore_run_dir(tmp_path)
    cache_path = layout.cache_path("q1", "vlm__m1@mock__deadbeef0000")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    corrupted = {"qid": "q1", "parser": "vlm__m1@mock__deadbeef0000", "answer": {"a": True},
                "field_matches": {"a": False}, "match": False}   # hand-corrupted: should be True
    cache_path.write_text(json.dumps(corrupted))

    result = runner.invoke(app, ["rescore", "--run-dir", str(layout.root)])
    assert result.exit_code == 0, result.output
    assert "changed=1" in _plain(result.output)
    healed = json.loads(cache_path.read_text())
    assert healed["field_matches"] == {"a": True}
    assert healed["match"] is True


def test_rescore_rescoring_covers_explicit_null_answer_rows(monkeypatch, tmp_path):
    """M2: upstream parity — `_worker`'s cache-hit branch gates on `'answer' in rec` alone (key
    presence), never on truthiness/nullness. A record with an explicit JSON `"answer": null`
    (key present, value None) must still be re-scored, not silently skipped."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_rescore_run_dir(tmp_path)
    cache_path = layout.cache_path("q1", "vlm__m1@mock__deadbeef0000")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"qid": "q1", "parser": "vlm__m1@mock__deadbeef0000", "answer": None,
           "field_matches": {"a": True}, "match": True}   # stale/wrong verdict for a null answer
    cache_path.write_text(json.dumps(rec))

    result = runner.invoke(app, ["rescore", "--run-dir", str(layout.root)])
    assert result.exit_code == 0, result.output
    assert "changed=1" in _plain(result.output)
    healed = json.loads(cache_path.read_text())
    assert healed["field_matches"] == {"a": False}
    assert healed["match"] is False


def test_rescore_prints_examined_skipped_changed_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_rescore_run_dir(tmp_path)
    p1 = layout.cache_path("q1", "vlm__m1@mock__deadbeef0000")
    p1.parent.mkdir(parents=True, exist_ok=True)
    p1.write_text(json.dumps({"qid": "q1", "answer": {"a": True},
                              "field_matches": {"a": False}, "match": False}))
    p2 = layout.cache_path("q_gone", "vlm__m1@mock__deadbeef0000")
    p2.write_text(json.dumps({"qid": "q_gone", "answer": {"a": True},
                              "field_matches": {}, "match": False}))

    result = runner.invoke(app, ["rescore", "--run-dir", str(layout.root)])
    assert result.exit_code == 0, result.output
    out = _plain(result.output)
    assert "examined=2" in out
    assert "skipped_not_in_bank=1" in out
    assert "changed=1" in out


def test_rescore_help_warns_about_upstream_force():
    result = runner.invoke(app, ["rescore", "--help"])
    assert result.exit_code == 0
    assert "vlm__" in result.output and "destroy" in result.output.lower()


# ── direct: shape/transport rejection, --dry-run passthrough, exit-2 contract (item I5) ───────

def _write_registry(path: Path, entries: list[dict]) -> None:
    path.write_text(yaml.safe_dump(entries))


def test_direct_rejects_wrong_shape_transport(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_rescore_run_dir(tmp_path)
    registry_yaml = tmp_path / "registry.yaml"
    _write_registry(registry_yaml, [{
        "id": "bad@transcriber", "shape": "transcriber", "transport": "upstream-parser",
        "upstream_parser": "whatever", "api_key_env": None, "precision": "bf16",
        "weights_licence": "mit", "provider_tos_commercial": "ok",
        "provenance": "Test", "release_date": "2025-01-01",
    }])
    result = runner.invoke(app, ["direct", "--run-dir", str(layout.root),
                                 "--registry", str(registry_yaml), "-m", "bad@transcriber"])
    assert result.exit_code == 1
    assert "not vlm-chat/openai-compat" in result.output


def _write_vlm_chat_registry(path: Path) -> None:
    _write_registry(path, [{
        "id": "m1@mock", "shape": "vlm-chat", "transport": "openai-compat",
        "base_url": "http://example.invalid/v1", "model": "org/m1", "api_key_env": None,
        "precision": "bf16", "weights_licence": "mit", "provider_tos_commercial": "ok",
        "provenance": "Test", "release_date": "2025-01-01",
    }])


def test_direct_dry_run_passthrough(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_rescore_run_dir(tmp_path)
    registry_yaml = tmp_path / "registry.yaml"
    _write_vlm_chat_registry(registry_yaml)
    captured = {}

    def fake_run_direct(layout_arg, entries, **kwargs):
        captured["dry_run"] = kwargs.get("dry_run")
        return {"cells": 1, "priced_cells": 0, "unpriced_cells": 1, "estimated_usd": 0.0,
                "per_entry": {}, "estimate_note": "test"}

    monkeypatch.setattr(direct_mod, "run_direct", fake_run_direct)
    result = runner.invoke(app, ["direct", "--run-dir", str(layout.root),
                                 "--registry", str(registry_yaml), "-m", "m1@mock", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert captured["dry_run"] is True


def test_direct_exit_code_2_on_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_rescore_run_dir(tmp_path)
    registry_yaml = tmp_path / "registry.yaml"
    _write_vlm_chat_registry(registry_yaml)
    monkeypatch.setattr(direct_mod, "run_direct",
                        lambda *a, **k: {"ok": 0, "error": 3, "cached": 0})
    result = runner.invoke(app, ["direct", "--run-dir", str(layout.root),
                                 "--registry", str(registry_yaml), "-m", "m1@mock"])
    assert result.exit_code == 2


# ── M3: --registry default is package-relative ─────────────────────────────────────────────────

def test_direct_registry_default_is_package_relative():
    import inspect

    sig = inspect.signature(cli_mod.direct)
    default = sig.parameters["registry"].default.default   # OptionInfo.default holds the value
    assert default == REPO_ROOT / "configs" / "registry.yaml"
