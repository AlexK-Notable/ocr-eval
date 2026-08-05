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
from ocr_eval_ext.parsers_openai import TRANSCRIBER_CONDITION
from realdoc_bench.evaluate.runs import RunLayout
from tests_ext.mock_openai import MockOpenAI

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# The guard must check for the CORPUS, not the bank. `runs/stage1/` now versions the run's
# results (parses/, eval/, qa_bank.json) while deliberately excluding the 1.2GB of source PDFs and
# renders, which are re-derivable from the pinned dataset revision. Keying the skip on
# qa_bank.json alone made this test FAIL rather than skip on a fresh clone: the bank is present,
# the PDFs it references are not, and `verify` correctly reports 581 missing source_files.
_REAL_CORPUS_MISSING = (
    not (REPO_ROOT / "runs" / "stage1" / "qa_bank.json").exists()
    or not any((REPO_ROOT / "runs" / "stage1" / "docs").glob("*.pdf"))
)
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

@pytest.mark.skipif(_REAL_CORPUS_MISSING,
                    reason="real 581-PDF corpus not present (re-download at the pinned revision)")
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


# ── F10: revision_source is backfilled on legacy meta that never got the key ───────────────────

def test_revision_source_backfilled_on_legacy_meta_missing_the_key(tmp_path, monkeypatch):
    """A run_meta.json written before `revision_source` existed (or otherwise hand-crafted
    without it) — `dataset_revision`/`harness_commit` present and matching pins, but no
    `revision_source` key at all — must gain the key on the NEXT successful preflight rather than
    carrying the gap forever (the original stamping branch only sets it the first time
    `dataset_revision` itself gets written)."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _fake_layout_with_one_good_page(tmp_path)
    pins = yaml.safe_load(PINS_PATH.read_text())
    (layout.root / "run_meta.json").write_text(json.dumps({
        "dataset_revision": pins["dataset_revision"], "harness_commit": pins["harness_commit"]}))

    result = runner.invoke(app, ["verify", "--run-dir", str(layout.root), "--skip-renders"])
    assert result.exit_code == 0, result.output
    meta = json.loads((layout.root / "run_meta.json").read_text())
    assert meta["revision_source"] == "pins"   # no .cache/huggingface/ present in this fake layout


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
    assert "not a vlm-chat entry" in result.output
    assert "bad@transcriber" in result.output


def test_direct_accepts_bedrock_converse_transport(tmp_path, monkeypatch):
    """The CLI gate must mirror `run_direct`'s transport precondition, not narrow it: a
    `bedrock-converse` vlm-chat entry belongs on this leg. Regression for a real defect — the gate
    hardcoded `openai-compat` after `run_direct` learned to drive Bedrock, so every Bedrock entry
    was rejected at the CLI with "use ocr-eval parse for those" (which would have been actively
    wrong advice — the parse leg cannot drive Bedrock at all).

    Asserts on `run_direct`'s ARGUMENTS rather than letting the real one run: reaching it at all is
    the whole claim, and stubbing keeps this test off the network.
    """
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_rescore_run_dir(tmp_path)
    registry_yaml = tmp_path / "registry.yaml"
    _write_registry(registry_yaml, [{
        "id": "haiku@bedrock", "shape": "vlm-chat", "transport": "bedrock-converse",
        "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "region": "us-east-1",
        "precision": "provider-default", "weights_licence": "closed",
        "provider_tos_commercial": "ok", "provenance": "Anthropic", "release_date": "2025-10-01",
    }])
    seen: dict = {}

    def fake_run_direct(layout, chosen, **kw):
        seen["ids"] = [e.id for e in chosen]
        return {"cells": 0, "ok": 0, "error": 0, "cached": 0}

    monkeypatch.setattr("ocr_eval_ext.direct.run_direct", fake_run_direct)
    result = runner.invoke(app, ["direct", "--run-dir", str(layout.root),
                                 "--registry", str(registry_yaml), "-m", "haiku@bedrock",
                                 "--dry-run"])
    assert result.exit_code == 0, result.output
    assert seen["ids"] == ["haiku@bedrock"]   # got past the gate into run_direct


def test_direct_gate_matches_run_direct_transports():
    """Pins the two to ONE name. A future third vlm-chat transport added to `run_direct` alone
    would otherwise reproduce exactly the drift this pair of tests exists to catch."""
    from ocr_eval_ext.direct import DIRECT_TRANSPORTS

    assert set(DIRECT_TRANSPORTS) == {"openai-compat", "bedrock-converse", "gemini-native"}


def _write_vlm_chat_registry(path: Path) -> None:
    _write_registry(path, [{
        "id": "m1@mock", "shape": "vlm-chat", "transport": "openai-compat",
        "base_url": "http://example.invalid/v1", "model": "org/m1", "api_key_env": None,
        "precision": "bf16", "weights_licence": "mit", "provider_tos_commercial": "ok",
        "provenance": "Test", "release_date": "2025-01-01",
    }])


def _make_direct_run_dir(tmp_path: Path) -> RunLayout:
    """Unlike `_make_rescore_run_dir` (plain text only — fine there, since `rescore` never
    renders), this PDF carries a filled rectangle so `ink_coverage` clears direct.py's blank-
    render floor — needed here because these tests exercise the real `ocr-eval direct` render
    path, not just cache-row rescoring."""
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Is question 1 checkbox marked?")
    page.draw_rect(fitz.Rect(72, 100, 400, 400), fill=(0, 0, 0))
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


def _write_vlm_chat_registry_with_url(path: Path, base_url: str) -> None:
    _write_registry(path, [{
        "id": "m1@mock", "shape": "vlm-chat", "transport": "openai-compat",
        "base_url": base_url, "model": "org/m1", "api_key_env": None,
        "precision": "bf16", "weights_licence": "mit", "provider_tos_commercial": "ok",
        "provenance": "Test", "release_date": "2025-01-01",
    }])


# ── F3: a rerun (no --force) that only ever revisits cached rows must not hide cached errors ───

def test_direct_cli_exits_2_on_rerun_over_cached_error_row(tmp_path, monkeypatch):
    """A rerun with no --force never re-attempts a cached cell — including one that recorded an
    error last time. Before F3, that error sat silently inside the aggregate `cached` count
    forever; now `run_direct` breaks it out as `cached_error` and the CLI fails visibly."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_direct_run_dir(tmp_path)
    registry_yaml = tmp_path / "registry.yaml"
    with MockOpenAI(responses=[{"status": 400, "message": "bad request"}]) as mock:
        _write_vlm_chat_registry_with_url(registry_yaml, mock.base_url)
        first = runner.invoke(app, ["direct", "--run-dir", str(layout.root),
                                    "--registry", str(registry_yaml), "-m", "m1@mock"])
        assert first.exit_code == 2, first.output   # fresh error this run

        second = runner.invoke(app, ["direct", "--run-dir", str(layout.root),
                                     "--registry", str(registry_yaml), "-m", "m1@mock"])
    assert second.exit_code == 2, second.output
    out = _plain(second.output).lower()
    assert "cached error" in out
    assert "--force" in out


def test_direct_cli_exits_0_on_rerun_over_cached_ok_row(tmp_path, monkeypatch):
    """Companion positive control: a rerun over a cleanly-cached ok row must stay exit 0 — F3's
    cached_error gate must not fire on cached_ok."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_direct_run_dir(tmp_path)
    registry_yaml = tmp_path / "registry.yaml"
    with MockOpenAI(reply_text='{"a": true}') as mock:
        _write_vlm_chat_registry_with_url(registry_yaml, mock.base_url)
        first = runner.invoke(app, ["direct", "--run-dir", str(layout.root),
                                    "--registry", str(registry_yaml), "-m", "m1@mock"])
        assert first.exit_code == 0, first.output

        second = runner.invoke(app, ["direct", "--run-dir", str(layout.root),
                                     "--registry", str(registry_yaml), "-m", "m1@mock"])
    assert second.exit_code == 0, second.output


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


# ── I6: direct also carries --allow-partial-corpus (structural — behavioral coverage lives on
# the parse/score tests below, which exercise the shared `_preflight` code path) ────────────────

def test_direct_has_allow_partial_corpus_flag():
    import inspect

    assert "allow_partial_corpus" in inspect.signature(cli_mod.direct).parameters


# ── preflight ────────────────────────────────────────────────────────────────────────────────

def _write_openai_compat_transcriber_registry(path: Path, base_url: str, model: str, *,
                                              entry_id: str = "t1@local",
                                              local: bool = True) -> None:
    """`entry_id` is parametrized (not hardcoded) because `register_openai_parsers` shares a
    single process-wide `parser_registry` across the whole pytest session — two tests that both
    register a transcriber with the same id but a DIFFERENT `base_url` now raise `ValueError`
    (I3's stale-binding guard) instead of one silently winning depending on run order. Every test
    below that actually calls `register_openai_parsers` (directly or via the `parse`/`score` CLI)
    uses its own unique id for exactly this reason."""
    _write_registry(path, [{
        "id": entry_id, "shape": "transcriber", "transport": "openai-compat",
        "base_url": base_url, "model": model, "api_key_env": None,
        "precision": "bf16", "weights_licence": "mit", "provider_tos_commercial": "ok",
        "provenance": "Test", "release_date": "2025-01-01", "local": local,
    }])


def test_preflight_cli_pass(tmp_path):
    registry_yaml = tmp_path / "registry.yaml"
    with MockOpenAI(models=["org/t1"]) as mock:
        _write_openai_compat_transcriber_registry(registry_yaml, mock.base_url, "org/t1")
        result = runner.invoke(app, ["preflight", "t1@local", "--registry", str(registry_yaml)])
        assert result.exit_code == 0, result.output
        assert "preflight PASS" in result.output


def test_preflight_cli_fails_on_model_mismatch(tmp_path):
    registry_yaml = tmp_path / "registry.yaml"
    with MockOpenAI(models=["org/other"]) as mock:
        _write_openai_compat_transcriber_registry(registry_yaml, mock.base_url, "org/t1")
        result = runner.invoke(app, ["preflight", "t1@local", "--registry", str(registry_yaml)])
        assert result.exit_code == 1
        assert "preflight FAILED" in result.output


def test_preflight_cli_url_build_tolerates_trailing_slash(tmp_path):
    """M9: a base_url with a trailing slash must not double-slash the /models URL."""
    registry_yaml = tmp_path / "registry.yaml"
    with MockOpenAI(models=["org/t1"]) as mock:
        _write_openai_compat_transcriber_registry(registry_yaml, mock.base_url + "/", "org/t1",
                                                  entry_id="trailingslash@local")
        result = runner.invoke(app, ["preflight", "trailingslash@local",
                                     "--registry", str(registry_yaml)])
        assert result.exit_code == 0, result.output
        assert "preflight PASS" in result.output


def test_preflight_cli_rejects_non_openai_compat_transport(tmp_path):
    registry_yaml = tmp_path / "registry.yaml"
    _write_registry(registry_yaml, [{
        "id": "g@google", "shape": "transcriber", "transport": "upstream-parser",
        "upstream_parser": "gemini_3_5_flash", "api_key_env": "GEMINI_API_KEY",
        "precision": "provider-default", "weights_licence": "closed",
        "provider_tos_commercial": "ok", "provenance": "Google", "release_date": "2025-01-01",
    }])
    result = runner.invoke(app, ["preflight", "g@google", "--registry", str(registry_yaml)])
    assert result.exit_code == 1
    assert "openai-compat" in result.output


# ── parse / score wrappers ───────────────────────────────────────────────────────────────────

def _make_transcriber_run_dir(tmp_path: Path) -> RunLayout:
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "irrelevant — the mock server ignores page content")
    doc.save(layout.docs_dir / "doc_1.pdf")
    bank = {"items": [{
        "question_id": "q1", "source_file": "doc_1", "domain": "test",
        "question": "What is the value?", "capabilities": [],
        "gold_dict": {"a": "42"},
        "response_format": "Return exactly: a=<string>",
        "gold_answer": "a=42",
    }]}
    layout.bank_path.write_text(json.dumps(bank))
    return layout


def test_parse_cli_produces_markdown_via_mock_server(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_transcriber_run_dir(tmp_path)
    registry_yaml = tmp_path / "registry.yaml"
    with MockOpenAI(reply_text="hello world") as mock:
        _write_openai_compat_transcriber_registry(registry_yaml, mock.base_url, "org/t1",
                                                  entry_id="parsecli@local")
        parser_name = f"parsecli_local__{direct_mod.condition_hash(TRANSCRIBER_CONDITION)}"
        result = runner.invoke(app, ["parse", "--run-dir", str(layout.root),
                                     "--registry", str(registry_yaml), "-p", parser_name])
    assert result.exit_code == 0, result.output
    md = (layout.parser_dir(parser_name) / "doc_1.md").read_text()
    assert "hello world" in md
    cond = json.loads((layout.parser_dir(parser_name) / "condition.json").read_text())
    assert cond == TRANSCRIBER_CONDITION


# ── F1: a parse whose "ok" transcripts are functionally empty must fail visibly, not silently ──

def test_parse_cli_fails_on_trivial_transcripts(tmp_path, monkeypatch):
    """A thinking-model-exhausted-its-budget transcript (or any other near-empty but `ok=true`
    parse) must not sail through the plain n_fail gate — scoring it would spend one extractor
    call per bank item against effectively no content. Simplest reproduction per the brief:
    monkeypatch `run_parse` to return `ParseRecord`-shaped fakes directly."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_transcriber_run_dir(tmp_path)
    registry_yaml = tmp_path / "registry.yaml"
    _write_openai_compat_transcriber_registry(registry_yaml, "http://localhost:9/v1", "org/t1",
                                              entry_id="trivialparse@local")
    parser_name = f"trivialparse_local__{direct_mod.condition_hash(TRANSCRIBER_CONDITION)}"

    import realdoc_bench.evaluate.parse as parse_mod
    from realdoc_bench.evaluate.parse import ParseRecord

    def fake_run_parse(layout_arg, parsers, **kwargs):
        return [ParseRecord(parser_name, "doc_1", True, False, page_count=1, md_length=9)]

    monkeypatch.setattr(parse_mod, "run_parse", fake_run_parse)
    result = runner.invoke(app, ["parse", "--run-dir", str(layout.root),
                                 "--registry", str(registry_yaml), "-p", parser_name])
    assert result.exit_code == 2
    out = _plain(result.output)
    assert "doc_1" in out
    assert "1 transcript" in out


def test_parse_cli_passes_on_transcripts_above_the_content_floor(tmp_path, monkeypatch):
    """Positive control for the F1 gate: a transcript comfortably above the floor must not trip
    it — exit 0, same as before F1 existed."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_transcriber_run_dir(tmp_path)
    registry_yaml = tmp_path / "registry.yaml"
    _write_openai_compat_transcriber_registry(registry_yaml, "http://localhost:9/v1", "org/t1",
                                              entry_id="realparse@local")
    parser_name = f"realparse_local__{direct_mod.condition_hash(TRANSCRIBER_CONDITION)}"

    import realdoc_bench.evaluate.parse as parse_mod
    from realdoc_bench.evaluate.parse import ParseRecord

    def fake_run_parse(layout_arg, parsers, **kwargs):
        return [ParseRecord(parser_name, "doc_1", True, False, page_count=1,
                            md_length=200)]

    monkeypatch.setattr(parse_mod, "run_parse", fake_run_parse)
    result = runner.invoke(app, ["parse", "--run-dir", str(layout.root),
                                 "--registry", str(registry_yaml), "-p", parser_name])
    assert result.exit_code == 0, result.output


def test_parse_cli_rejects_unknown_parser_name(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_transcriber_run_dir(tmp_path)
    registry_yaml = tmp_path / "registry.yaml"
    _write_openai_compat_transcriber_registry(registry_yaml, "http://localhost:9/v1", "org/t1",
                                              entry_id="parsecli-reject@local")
    result = runner.invoke(app, ["parse", "--run-dir", str(layout.root),
                                 "--registry", str(registry_yaml), "-p", "totally-unknown"])
    assert result.exit_code == 1
    assert "unknown parser name" in _plain(result.output)


# ── I5: parse workers default (1 for local:true, 8 otherwise; explicit --workers always wins) ──

def test_parse_defaults_workers_to_one_for_local_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_transcriber_run_dir(tmp_path)
    registry_yaml = tmp_path / "registry.yaml"
    _write_openai_compat_transcriber_registry(registry_yaml, "http://localhost:9/v1", "org/t1",
                                              entry_id="workerslocal@local", local=True)
    parser_name = f"workerslocal_local__{direct_mod.condition_hash(TRANSCRIBER_CONDITION)}"
    captured = {}

    def fake_run_parse(layout_arg, parsers, **kwargs):
        captured["workers"] = kwargs.get("workers")
        return []

    import realdoc_bench.evaluate.parse as parse_mod
    monkeypatch.setattr(parse_mod, "run_parse", fake_run_parse)
    result = runner.invoke(app, ["parse", "--run-dir", str(layout.root),
                                 "--registry", str(registry_yaml), "-p", parser_name])
    assert result.exit_code == 0, result.output
    assert captured["workers"] == 1
    assert "defaulting to 1" in _plain(result.output)


def test_parse_defaults_workers_to_eight_for_hosted_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_transcriber_run_dir(tmp_path)
    registry_yaml = tmp_path / "registry.yaml"
    _write_openai_compat_transcriber_registry(registry_yaml, "http://localhost:9/v1", "org/t1",
                                              entry_id="workershosted@openrouter", local=False)
    parser_name = f"workershosted_openrouter__{direct_mod.condition_hash(TRANSCRIBER_CONDITION)}"
    captured = {}

    def fake_run_parse(layout_arg, parsers, **kwargs):
        captured["workers"] = kwargs.get("workers")
        return []

    import realdoc_bench.evaluate.parse as parse_mod
    monkeypatch.setattr(parse_mod, "run_parse", fake_run_parse)
    result = runner.invoke(app, ["parse", "--run-dir", str(layout.root),
                                 "--registry", str(registry_yaml), "-p", parser_name])
    assert result.exit_code == 0, result.output
    assert captured["workers"] == 8
    assert "defaulting to 8" in _plain(result.output)


def test_parse_explicit_workers_overrides_local_default(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_transcriber_run_dir(tmp_path)
    registry_yaml = tmp_path / "registry.yaml"
    _write_openai_compat_transcriber_registry(registry_yaml, "http://localhost:9/v1", "org/t1",
                                              entry_id="workersoverride@local", local=True)
    parser_name = f"workersoverride_local__{direct_mod.condition_hash(TRANSCRIBER_CONDITION)}"
    captured = {}

    def fake_run_parse(layout_arg, parsers, **kwargs):
        captured["workers"] = kwargs.get("workers")
        return []

    import realdoc_bench.evaluate.parse as parse_mod
    monkeypatch.setattr(parse_mod, "run_parse", fake_run_parse)
    result = runner.invoke(app, ["parse", "--run-dir", str(layout.root),
                                 "--registry", str(registry_yaml), "-p", parser_name,
                                 "--workers", "4"])
    assert result.exit_code == 0, result.output
    assert captured["workers"] == 4    # explicit value wins over the local:true default of 1


# ── I6: --allow-partial-corpus on parse/score/direct, never on verify ───────────────────────────

def test_verify_has_no_allow_partial_corpus_flag(tmp_path):
    result = runner.invoke(app, ["verify", "--run-dir", str(tmp_path), "--allow-partial-corpus"])
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_parse_allow_partial_corpus_bypasses_corpus_check(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    doc = fitz.open()
    doc.new_page(width=612, height=792).insert_text((72, 72), "present")
    doc.save(layout.docs_dir / "present.pdf")
    layout.bank_path.write_text(json.dumps({"items": [
        {"question_id": "q1", "source_file": "present", "gold_dict": {"a": 1}},
        {"question_id": "q2", "source_file": "missing_doc", "gold_dict": {"a": 1}},
    ]}))
    registry_yaml = tmp_path / "registry.yaml"
    with MockOpenAI(reply_text="hello world") as mock:   # > F1's content floor (16 chars/page)
        _write_openai_compat_transcriber_registry(registry_yaml, mock.base_url, "org/t1",
                                                  entry_id="partialcorpus@local")
        parser_name = f"partialcorpus_local__{direct_mod.condition_hash(TRANSCRIBER_CONDITION)}"
        result = runner.invoke(app, ["parse", "--run-dir", str(layout.root),
                                     "--registry", str(registry_yaml), "-p", parser_name,
                                     "--allow-partial-corpus"])
    assert result.exit_code == 0, result.output
    out = _plain(result.output).lower()
    assert "allow-partial-corpus" in out and "skipped" in out
    meta = json.loads((layout.root / "run_meta.json").read_text())
    assert meta["partial_corpus"] is True
    assert (layout.parser_dir(parser_name) / "present.md").exists()


# ── N3 RULING: partial_corpus is sticky-true — only verify's full sweep clears it ──────────────

def test_partial_corpus_flag_is_sticky_across_a_later_full_preflight(tmp_path, monkeypatch):
    """A LATER, completely ordinary preflight success (full corpus-completeness check passes, no
    --allow-partial-corpus) must NOT flip the flag back to false — the ruling requires the more
    expensive verify full-sweep specifically, not just any passing preflight."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    doc = fitz.open()
    doc.new_page(width=612, height=792).insert_text((72, 72), "present")
    doc.save(layout.docs_dir / "present.pdf")
    layout.bank_path.write_text(json.dumps({"items": [
        {"question_id": "q1", "source_file": "present", "gold_dict": {"a": 1}},
    ]}))
    registry_yaml = tmp_path / "registry.yaml"
    with MockOpenAI(reply_text="hello world") as mock:   # > F1's content floor (16 chars/page)
        _write_openai_compat_transcriber_registry(registry_yaml, mock.base_url, "org/t1",
                                                  entry_id="stickytest@local")
        parser_name = f"stickytest_local__{direct_mod.condition_hash(TRANSCRIBER_CONDITION)}"

        first = runner.invoke(app, ["parse", "--run-dir", str(layout.root),
                                    "--registry", str(registry_yaml), "-p", parser_name,
                                    "--allow-partial-corpus"])
        assert first.exit_code == 0, first.output
        assert json.loads((layout.root / "run_meta.json").read_text())["partial_corpus"] is True

        # Completely ordinary follow-up call — corpus IS complete, no flag needed or passed.
        second = runner.invoke(app, ["parse", "--run-dir", str(layout.root),
                                     "--registry", str(registry_yaml), "-p", parser_name,
                                     "--force"])
        assert second.exit_code == 0, second.output

    meta2 = json.loads((layout.root / "run_meta.json").read_text())
    assert meta2["partial_corpus"] is True   # still sticky — NOT cleared by the plain success


def test_verify_full_sweep_clears_sticky_partial_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _fake_layout_with_one_good_page(tmp_path)
    runner.invoke(app, ["verify", "--run-dir", str(layout.root), "--skip-renders"])
    meta_p = layout.root / "run_meta.json"
    meta = json.loads(meta_p.read_text())
    meta["partial_corpus"] = True   # simulate an earlier --allow-partial-corpus call
    meta_p.write_text(json.dumps(meta))

    result = runner.invoke(app, ["verify", "--run-dir", str(layout.root)])   # full sweep
    assert result.exit_code == 0, result.output
    assert json.loads(meta_p.read_text())["partial_corpus"] is False


def test_verify_skip_renders_does_not_clear_sticky_partial_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _fake_layout_with_one_good_page(tmp_path)
    runner.invoke(app, ["verify", "--run-dir", str(layout.root), "--skip-renders"])
    meta_p = layout.root / "run_meta.json"
    meta = json.loads(meta_p.read_text())
    meta["partial_corpus"] = True
    meta_p.write_text(json.dumps(meta))

    result = runner.invoke(app, ["verify", "--run-dir", str(layout.root), "--skip-renders"])
    assert result.exit_code == 0, result.output
    assert json.loads(meta_p.read_text())["partial_corpus"] is True   # sticky: not the full sweep


# ── N4 RULING: rescore skips the corpus-completeness check entirely ────────────────────────────

def test_rescore_succeeds_on_partial_corpus_dir(tmp_path, monkeypatch):
    """rescore only ever reads eval/cache/ and the bank, never docs/ — a bank item referencing a
    source_file with no matching PDF must not block it, and no --allow-partial-corpus flag is
    even needed (rescore has none)."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "irrelevant — rescore never renders")
    doc.save(layout.docs_dir / "doc_1.pdf")
    bank = {"items": [
        {"question_id": "q1", "source_file": "doc_1", "domain": "test",
         "question": "Is question 1 checkbox marked?", "capabilities": ["checkbox_state"],
         "gold_dict": {"a": True}, "response_format": "Return exactly: a=<boolean>",
         "gold_answer": "a=true"},
        # references a PDF that does NOT exist under docs/ — verify/parse/score (without
        # --allow-partial-corpus) would refuse on this alone; rescore must not care.
        {"question_id": "q2", "source_file": "missing_doc", "domain": "test",
         "question": "irrelevant", "capabilities": [], "gold_dict": {"a": True},
         "response_format": "Return exactly: a=<boolean>", "gold_answer": "a=true"},
    ]}
    layout.bank_path.write_text(json.dumps(bank))
    cache_path = layout.cache_path("q1", "vlm__m1@mock__deadbeef0000")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "qid": "q1", "parser": "vlm__m1@mock__deadbeef0000", "answer": {"a": True},
        "field_matches": {"a": False}, "match": False}))   # hand-corrupted: should heal to True

    result = runner.invoke(app, ["rescore", "--run-dir", str(layout.root)])
    assert result.exit_code == 0, result.output
    assert "corpus incomplete" not in _plain(result.output)
    healed = json.loads(cache_path.read_text())
    assert healed["match"] is True


def test_parse_wrapper_enforces_corpus_completeness(tmp_path, monkeypatch):
    """Carried finding from Task 7 review: the corpus-completeness check must also protect the
    parse/score wrappers, not just `verify`."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    layout.bank_path.write_text(json.dumps(
        {"items": [{"question_id": "q1", "source_file": "missing_doc", "gold_dict": {"a": 1}}]}))
    registry_yaml = tmp_path / "registry.yaml"
    _write_registry(registry_yaml, [])
    result = runner.invoke(app, ["parse", "--run-dir", str(layout.root),
                                 "--registry", str(registry_yaml), "-p", "whatever"])
    assert result.exit_code == 1
    assert "corpus incomplete" in _plain(result.output)


def test_score_wrapper_enforces_corpus_completeness(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    layout.bank_path.write_text(json.dumps(
        {"items": [{"question_id": "q1", "source_file": "missing_doc", "gold_dict": {"a": 1}}]}))
    registry_yaml = tmp_path / "registry.yaml"
    _write_registry(registry_yaml, [])
    result = runner.invoke(app, ["score", "--run-dir", str(layout.root),
                                 "--registry", str(registry_yaml), "-p", "whatever"])
    assert result.exit_code == 1
    assert "corpus incomplete" in _plain(result.output)


def _run_score_cli(layout: RunLayout, registry_yaml: Path, parser_name: str,
                   extra: list[str] | None = None):
    return runner.invoke(app, ["score", "--run-dir", str(layout.root),
                               "--registry", str(registry_yaml), "-p", parser_name,
                               *(extra or [])])


def test_score_cli_writes_results_and_records_extractor_id(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(cli_mod.st, "run_extractor", lambda: [])
    layout = _make_transcriber_run_dir(tmp_path)
    parser_name = "t1_local__abc123"
    parser_dir = layout.parser_dir(parser_name)
    parser_dir.mkdir(parents=True)
    (parser_dir / "doc_1.md").write_text("## Page 1\n\n**Value:** 42")
    registry_yaml = tmp_path / "registry.yaml"
    _write_registry(registry_yaml, [])

    import realdoc_bench.evaluate.score as score_mod
    monkeypatch.setattr(score_mod, "gemini_extract", lambda *a, **k: {"a": "42"})

    result = _run_score_cli(layout, registry_yaml, parser_name)
    assert result.exit_code == 0, result.output
    results = json.loads(layout.results_path.read_text())
    assert len(results) == 1
    assert results[0]["match"] is True
    meta = json.loads((layout.root / "run_meta.json").read_text())
    assert meta["extractor_id"] == score_mod.DEFAULT_MODEL


def test_scored_row_stamps_the_extractor_that_produced_it(tmp_path, monkeypatch):
    """Provenance belongs in the ROW, not only in run_meta.json.

    Regression for a real ambiguity: two people disagreed about which extractor had scored 4,068
    existing transcriber rows, and the rows themselves could not settle it — they carried only
    qid/parser/source_file/domain/answer/field_matches/match. Cache rows are keyed per
    (question, parser) and outlive any single run's metadata, so the grader's identity has to live
    with the answer it produced."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(cli_mod.st, "run_extractor", lambda: [])
    layout = _make_transcriber_run_dir(tmp_path)
    parser_name = "t1_local__abc123"
    parser_dir = layout.parser_dir(parser_name)
    parser_dir.mkdir(parents=True)
    (parser_dir / "doc_1.md").write_text("## Page 1\n\n**Value:** 42")
    registry_yaml = tmp_path / "registry.yaml"
    _write_registry(registry_yaml, [])

    import realdoc_bench.evaluate.score as score_mod
    monkeypatch.setattr(score_mod, "gemini_extract", lambda *a, **k: {"a": "42"})

    result = _run_score_cli(layout, registry_yaml, parser_name)
    assert result.exit_code == 0, result.output
    rec = json.loads(layout.cache_path("q1", parser_name).read_text())
    assert rec["extractor"] == score_mod.DEFAULT_MODEL


def test_cache_hit_does_not_backfill_the_extractor_stamp(tmp_path, monkeypatch):
    """The complement, and the part that makes the stamp trustworthy: an already-answered row must
    NOT acquire today's pin on a rescore. Stamping it would assert precisely the false provenance
    this field exists to prevent — absence means "scored before the stamp existed", never "scored
    by whatever is pinned now"."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(cli_mod.st, "run_extractor", lambda: [])
    layout = _make_transcriber_run_dir(tmp_path)
    parser_name = "t1_local__abc123"
    parser_dir = layout.parser_dir(parser_name)
    parser_dir.mkdir(parents=True)
    (parser_dir / "doc_1.md").write_text("## Page 1\n\n**Value:** 42")
    registry_yaml = tmp_path / "registry.yaml"
    _write_registry(registry_yaml, [])

    # A pre-existing row from an unknown extractor generation — no `extractor` key.
    legacy = {"qid": "q1", "parser": parser_name, "source_file": "doc_1", "domain": "test",
              "answer": {"a": "42"}, "field_matches": {"a": True}, "match": True}
    cpath = layout.cache_path("q1", parser_name)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(legacy))

    import realdoc_bench.evaluate.score as score_mod
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        return {"a": "42"}

    monkeypatch.setattr(score_mod, "gemini_extract", _boom)
    result = _run_score_cli(layout, registry_yaml, parser_name)
    assert result.exit_code == 0, result.output
    assert called["n"] == 0                       # cache hit: no extractor call
    rec = json.loads(cpath.read_text())
    assert "extractor" not in rec                  # provenance not invented


def test_score_cli_refuses_to_mix_extractor_generations(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(cli_mod.st, "run_extractor", lambda: [])
    layout = _make_transcriber_run_dir(tmp_path)
    parser_name = "t1_local__abc123"
    parser_dir = layout.parser_dir(parser_name)
    parser_dir.mkdir(parents=True)
    (parser_dir / "doc_1.md").write_text("## Page 1\n\n**Value:** 42")
    registry_yaml = tmp_path / "registry.yaml"
    _write_registry(registry_yaml, [])

    import realdoc_bench.evaluate.score as score_mod
    monkeypatch.setattr(score_mod, "gemini_extract", lambda *a, **k: {"a": "42"})

    first = _run_score_cli(layout, registry_yaml, parser_name)
    assert first.exit_code == 0, first.output
    meta_p = layout.root / "run_meta.json"
    meta = json.loads(meta_p.read_text())
    meta["extractor_id"] = "old-extractor-id"
    meta_p.write_text(json.dumps(meta))

    second = _run_score_cli(layout, registry_yaml, parser_name)
    assert second.exit_code == 1
    assert "old-extractor-id" in second.output
    assert "--new-extractor-generation" in second.output
    # N5: `require_extractor_gate` now runs BEFORE the generation-mismatch refusal (M5/M6), so
    # this single stamp is NOT evidence that the refusal short-circuits the gate — it's evidence
    # that the gate is cached per `DEFAULT_MODEL` (a real Python constant), which this test never
    # changes between the two invocations; only `run_meta.json`'s "extractor_id" is hand-edited
    # to simulate a mismatch. Since `DEFAULT_MODEL` is unchanged, the SAME stamp file satisfies
    # both calls regardless of ordering. Had `DEFAULT_MODEL` genuinely changed between the two
    # calls, the gate (running first) would spend fresh Gemini calls before the refusal ever
    # fired — the refusal would still happen afterward, just not for free.
    assert len(list(layout.root.glob(".extractor_ok_*"))) == 1


def test_score_cli_new_extractor_generation_end_to_end(tmp_path, monkeypatch):
    """Smoke test: the full `score` command with --new-extractor-generation still completes
    (gate -> archive -> re-run_score all cooperate under the M5/M6-reordered pipeline). Exact
    row-level selectivity (I4: vlm__* stays live, everything else archives) is pinned precisely,
    without `run_score`'s own re-population muddying the assertions, by
    `test_enforce_extractor_generation_archives_only_non_vlm_rows` below."""
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(cli_mod.st, "run_extractor", lambda: [])
    layout = _make_transcriber_run_dir(tmp_path)
    parser_name = "t1_local__abc123"
    parser_dir = layout.parser_dir(parser_name)
    parser_dir.mkdir(parents=True)
    (parser_dir / "doc_1.md").write_text("## Page 1\n\n**Value:** 42")
    registry_yaml = tmp_path / "registry.yaml"
    _write_registry(registry_yaml, [])

    import realdoc_bench.evaluate.score as score_mod
    monkeypatch.setattr(score_mod, "gemini_extract", lambda *a, **k: {"a": "42"})

    first = _run_score_cli(layout, registry_yaml, parser_name)
    assert first.exit_code == 0, first.output
    meta_p = layout.root / "run_meta.json"
    meta = json.loads(meta_p.read_text())
    meta["extractor_id"] = "old-extractor-id"
    meta_p.write_text(json.dumps(meta))

    second = _run_score_cli(layout, registry_yaml, parser_name, ["--new-extractor-generation"])
    assert second.exit_code == 0, second.output
    assert (layout.root / "eval" / "cache@old-extractor-id").exists()
    meta2 = json.loads(meta_p.read_text())
    assert meta2["extractor_id"] == score_mod.DEFAULT_MODEL


# ── I4: --new-extractor-generation archives ONLY extractor-dependent (non-vlm__) rows ───────────

def test_enforce_extractor_generation_archives_only_non_vlm_rows(tmp_path):
    """Unit-level (not through the full `score` command, which would immediately re-populate an
    archived row via `run_score` in the very same invocation — see the end-to-end smoke test
    above): calls `_enforce_extractor_generation` directly against a pre-seeded eval/cache/ to
    pin the exact selectivity rule without that confound."""
    import realdoc_bench.evaluate.score as score_mod

    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()

    vlm_row = layout.cache_path("q1", "vlm__m1@mock__deadbeef0000")
    vlm_content = {"qid": "q1", "parser": "vlm__m1@mock__deadbeef0000",
                   "answer": {"a": True}, "field_matches": {"a": True}, "match": True}
    vlm_row.parent.mkdir(parents=True, exist_ok=True)
    vlm_row.write_text(json.dumps(vlm_content))

    scored_row = layout.cache_path("q2", "t1_local__abc123")
    scored_content = {"qid": "q2", "parser": "t1_local__abc123",
                      "answer": {"a": "42"}, "field_matches": {"a": True}, "match": True}
    scored_row.write_text(json.dumps(scored_content))

    (layout.root / "run_meta.json").write_text(json.dumps({"extractor_id": "old-id"}))

    cli_mod._enforce_extractor_generation(layout, True)

    archive = layout.root / "eval" / "cache@old-id"
    assert vlm_row.exists()                              # direct row: left live, untouched
    assert json.loads(vlm_row.read_text()) == vlm_content
    assert not (archive / vlm_row.name).exists()

    assert not scored_row.exists()                        # extractor-dependent row: moved out...
    assert json.loads((archive / scored_row.name).read_text()) == scored_content   # ...intact

    meta = json.loads((layout.root / "run_meta.json").read_text())
    assert meta["extractor_id"] == score_mod.DEFAULT_MODEL


def test_enforce_extractor_generation_treats_corrupt_row_as_extractor_dependent(tmp_path):
    """An unreadable/corrupt cache row can't be confirmed as a vlm__ direct row, so the safe
    default is to archive it rather than silently leave a row of unknown provenance mixed into
    the new generation's live eval/cache/."""
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    corrupt = layout.cache_dir / "q_bad__unknown.json"
    layout.cache_dir.mkdir(parents=True, exist_ok=True)
    corrupt.write_text("{not valid json")
    (layout.root / "run_meta.json").write_text(json.dumps({"extractor_id": "old-id"}))

    cli_mod._enforce_extractor_generation(layout, True)

    assert not corrupt.exists()
    assert (layout.root / "eval" / "cache@old-id" / corrupt.name).exists()


# ── N2: archive pre-scans ALL destination collisions before moving ANY file ─────────────────────

def test_enforce_extractor_generation_prescans_collisions_moves_nothing_on_conflict(tmp_path):
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()

    row1 = layout.cache_path("q1", "t1_local__abc123")
    row1_content = {"qid": "q1", "parser": "t1_local__abc123", "answer": {"a": "1"}}
    row1.parent.mkdir(parents=True, exist_ok=True)
    row1.write_text(json.dumps(row1_content))

    row2 = layout.cache_path("q2", "t1_local__abc123")
    row2_content = {"qid": "q2", "parser": "t1_local__abc123", "answer": {"a": "2"}}
    row2.write_text(json.dumps(row2_content))

    # Pre-create a COLLIDING file at row2's would-be archive destination — row1's destination is
    # clear. Before N2, a file-by-file loop would have already moved row1 by the time it hit
    # row2's collision and bailed; the pre-scan must catch this BEFORE touching row1 at all.
    archive = layout.root / "eval" / "cache@old-id"
    archive.mkdir(parents=True)
    (archive / row2.name).write_text(json.dumps({"pre-existing": "conflicting content"}))

    (layout.root / "run_meta.json").write_text(json.dumps({"extractor_id": "old-id"}))

    with pytest.raises(cli_mod.typer.Exit):
        cli_mod._enforce_extractor_generation(layout, True)

    # Nothing moved: row1 is still live (would have been moved first under the old file-by-file
    # design), row2 is still live, and the pre-existing archive file is untouched.
    assert row1.exists()
    assert json.loads(row1.read_text()) == row1_content
    assert row2.exists()
    assert json.loads(row2.read_text()) == row2_content
    assert json.loads((archive / row2.name).read_text()) == {"pre-existing": "conflicting content"}
    # extractor_id must not have been updated either — the whole operation aborted cleanly.
    meta = json.loads((layout.root / "run_meta.json").read_text())
    assert meta["extractor_id"] == "old-id"


# ── M1: score refuses an unknown -p name BEFORE writing sticky "markdown missing" rows ──────────

def test_score_cli_rejects_unknown_parser_name_before_run_score(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(cli_mod.st, "run_extractor", lambda: [])
    layout = _make_transcriber_run_dir(tmp_path)
    registry_yaml = tmp_path / "registry.yaml"
    _write_registry(registry_yaml, [])

    result = runner.invoke(app, ["score", "--run-dir", str(layout.root),
                                 "--registry", str(registry_yaml), "-p", "totally-bogus-typo"])
    assert result.exit_code == 1
    assert "unknown parser name" in _plain(result.output).lower()
    assert list(layout.cache_dir.glob("*.json")) == []   # no sticky "markdown missing" rows


# ── M5/M6: extractor gate runs BEFORE any stamping/archiving; its failure is a clean exit ──────

def test_score_cli_extractor_gate_failure_is_clean_exit_no_traceback(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(cli_mod.st, "run_extractor",
                        lambda: ["extractor missed: 'q' -> None"])
    layout = _make_transcriber_run_dir(tmp_path)
    parser_name = "t1_local__abc123"
    parser_dir = layout.parser_dir(parser_name)
    parser_dir.mkdir(parents=True)
    (parser_dir / "doc_1.md").write_text("## Page 1\n\n**Value:** 42")
    registry_yaml = tmp_path / "registry.yaml"
    _write_registry(registry_yaml, [])

    result = _run_score_cli(layout, registry_yaml, parser_name)
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "extractor validation FAILED" in _plain(result.output)
    # never stamped on a failed gate — no extractor_id recorded, nothing archived
    meta = json.loads((layout.root / "run_meta.json").read_text())
    assert "extractor_id" not in meta
    assert not any((layout.root / "eval").glob("cache@*"))


# ── I3 (CLI level): a stale binding surfaces as a clean red message, not a traceback ────────────

def test_parse_cli_stale_binding_mismatch_is_clean_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "check_bank", lambda items: {})
    layout = _make_transcriber_run_dir(tmp_path)
    registry_a = tmp_path / "a.yaml"
    registry_b = tmp_path / "b.yaml"
    _write_openai_compat_transcriber_registry(registry_a, "http://localhost:9/v1", "org/t1",
                                              entry_id="staleclitest@local")
    _write_openai_compat_transcriber_registry(registry_b, "http://localhost:10/v1", "org/t1",
                                              entry_id="staleclitest@local")
    parser_name = f"staleclitest_local__{direct_mod.condition_hash(TRANSCRIBER_CONDITION)}"

    import realdoc_bench.evaluate.parse as parse_mod
    monkeypatch.setattr(parse_mod, "run_parse", lambda *a, **k: [])   # never actually connects

    first = runner.invoke(app, ["parse", "--run-dir", str(layout.root),
                                "--registry", str(registry_a), "-p", parser_name])
    assert first.exit_code == 0, first.output

    second = runner.invoke(app, ["parse", "--run-dir", str(layout.root),
                                 "--registry", str(registry_b), "-p", parser_name])
    assert second.exit_code == 1
    assert "Traceback" not in second.output
    assert "already registered" in _plain(second.output)
