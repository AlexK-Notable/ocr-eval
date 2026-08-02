import json
import re
import subprocess
from pathlib import Path

import fitz  # pymupdf
import pytest
import yaml
from typer.testing import CliRunner

import ocr_eval_ext.cli as cli_mod
from ocr_eval_ext.cli import PINS_PATH, REPO_ROOT, app, require_extractor_gate
from realdoc_bench.evaluate.runs import RunLayout

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


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

def test_verify_skip_renders_passes_on_real_bank():
    """`runs/stage1/` ships the real 581-PDF corpus + bank at the pinned revision. --skip-renders
    keeps this test fast (pins + cardinality only, no render sweep) while still proving the real
    bank satisfies `check_bank` and the harness-commit pin resolves against actual HEAD."""
    real_run_dir = REPO_ROOT / "runs" / "stage1"
    result = runner.invoke(app, ["verify", "--run-dir", str(real_run_dir), "--skip-renders"])
    assert result.exit_code == 0, result.output
    assert "verify PASS" in result.output
    assert "renders skipped" in result.output


# ── run_meta.json stamping on first successful verify (item 6) ────────────────────────────────

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


def test_verify_skip_renders_does_not_stamp_run_meta(tmp_path, monkeypatch):
    """--skip-renders never warms the PNG cache or confirms per-page renders, so it must not
    make the 'first successful verify' claim that stamping represents."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _fake_layout_with_one_good_page(tmp_path)
    result = runner.invoke(app, ["verify", "--run-dir", str(layout.root), "--skip-renders"])
    assert result.exit_code == 0, result.output
    assert not (layout.root / "run_meta.json").exists()


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


# ── require_extractor_gate stamp/cache logic (item 7) ─────────────────────────────────────────

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


# ── rescore: heals a corrupted cache row with zero API calls (brief Step 5 test) ──────────────

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
    assert "1 row(s) changed" in _plain(result.output)
    healed = json.loads(cache_path.read_text())
    assert healed["field_matches"] == {"a": True}
    assert healed["match"] is True


def test_rescore_help_warns_about_upstream_force():
    result = runner.invoke(app, ["rescore", "--help"])
    assert result.exit_code == 0
    assert "vlm__" in result.output and "destroy" in result.output.lower()
