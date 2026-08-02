"""ocr-eval — Stage 1 commands. Wraps upstream for download/parse/score; adds
verify (fail-closed preconditions), direct (vlm-chat), selftest, report."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import typer
import yaml
from rich.console import Console

from ocr_eval_ext import selftest as st
from ocr_eval_ext.config import load_registry
from ocr_eval_ext.preconditions import PreconditionError, check_bank
from realdoc_bench.evaluate.runs import RunLayout

app = typer.Typer(no_args_is_help=True)
console = Console(width=200)          # fixed width — CI COLUMNS must not split assertion tokens

REPO_ROOT = Path(__file__).resolve().parents[1]
PINS_PATH = REPO_ROOT / "configs" / "pins.yaml"


def _run_meta_path(layout: RunLayout) -> Path:
    return layout.root / "run_meta.json"


def _preflight(layout: RunLayout) -> None:
    """The gate every spending/reporting command calls first. Fail-closed."""
    if st.run_offline():
        console.print("[red]scorer self-test failed — refusing to proceed[/red]")
        raise typer.Exit(1)
    pins = yaml.safe_load(PINS_PATH.read_text())
    head = subprocess.run(["git", "merge-base", "HEAD", pins["harness_commit"]],
                          capture_output=True, text=True, cwd=REPO_ROOT)
    if head.returncode != 0 or pins["harness_commit"] not in head.stdout:
        console.print(f"[red]harness pin {pins['harness_commit'][:10]} is not an ancestor of HEAD[/red]")
        raise typer.Exit(1)
    meta_p = _run_meta_path(layout)
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("dataset_revision") != pins["dataset_revision"]:
            console.print(f"[red]run dir was downloaded at revision "
                          f"{meta.get('dataset_revision')!r}, pins say "
                          f"{pins['dataset_revision']!r}[/red]")
            raise typer.Exit(1)
    try:
        items = json.loads(layout.bank_path.read_text())["items"]
        check_bank(items)
    except (FileNotFoundError, KeyError, PreconditionError) as e:
        console.print(f"[red]preflight FAILED: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def verify(
    run_dir: Path = typer.Option(..., "--run-dir"),
    skip_renders: bool = typer.Option(False, "--skip-renders",
                                      help="Skip the per-PDF single-page + non-blank render "
                                           "sweep — pins + bank cardinality only. Fast path for "
                                           "CI / iteration; the full sweep (minutes, over the "
                                           "whole corpus) is still required at least once."),
) -> None:
    """Full fail-closed sweep: pins, cardinalities, every PDF single-page + non-blank render.
    Warms the PNG cache as a side effect. Run once after download, before any spend."""
    from ocr_eval_ext.direct import STAGE1_CONDITION, _render_page
    from ocr_eval_ext.preconditions import ink_coverage

    layout = RunLayout.at(run_dir)
    _preflight(layout)
    if skip_renders:
        console.print("[yellow]--skip-renders: pins + cardinality checked, "
                      "pages/renders NOT swept[/yellow]")
        console.print("[green]verify PASS[/green] — pins, cardinalities green (renders skipped)")
        return
    png_cache = layout.root / "docs_png"
    blank, multi = [], []
    for pdf in sorted(layout.docs_dir.glob("*.pdf")):
        try:
            png = _render_page(layout, pdf.stem, STAGE1_CONDITION, png_cache)
        except PreconditionError:
            multi.append(pdf.stem)
            continue
        if ink_coverage(png) < 0.001:
            blank.append(pdf.stem)
    if blank or multi:
        console.print(f"[red]verify FAILED — multi-page: {multi} blank-render: {blank}[/red]")
        raise typer.Exit(1)
    meta_p = _run_meta_path(layout)
    if not meta_p.exists():                # stamp pins + renderer version on first successful verify
        from importlib.metadata import version as _v
        pins = yaml.safe_load(PINS_PATH.read_text())
        meta_p.write_text(json.dumps({
            "dataset_revision": pins["dataset_revision"],
            "harness_commit": pins["harness_commit"],
            "pymupdf_version": _v("pymupdf"),
            # vLLM runs in its own serving env, not this one — its version is recorded per model
            # in docs/local-serving.md, which is the operative record for local rows.
        }, indent=2))
    console.print("[green]verify PASS[/green] — pins, cardinalities, pages, renders all green")


@app.command()
def selftest(extractor: bool = typer.Option(False, "--extractor")) -> None:
    fails = st.run_offline()
    if fails:
        console.print("[red]offline scorer self-test: FAIL[/red]")
        for f in fails:
            console.print(f"  - {f}")
        raise typer.Exit(1)
    console.print("offline scorer self-test: PASS")
    if extractor:
        efails = st.run_extractor()
        if efails:
            console.print("[red]extractor validation: FAIL[/red]")
            for f in efails:
                console.print(f"  - {f}")
            raise typer.Exit(1)
        console.print("extractor validation: PASS (5/5)")


def _fixture_hash() -> str:
    return hashlib.sha256(json.dumps(st.EXTRACTOR_FIXTURES, sort_keys=True).encode()).hexdigest()


def require_extractor_gate(layout: RunLayout) -> None:
    """BLOCKING extractor-validation gate for `ocr-eval score` (wired in Task 8): 5 Gemini calls
    against `st.EXTRACTOR_FIXTURES`, cached per (run dir, DEFAULT_MODEL, fixture-set) via a stamp
    file (`.extractor_ok_<model>_<fixturehash>`) so it never re-runs once passed for that
    combination, and never skippable when the stamp is absent — a scorer that cannot validate its
    own extractor must not be allowed to spend."""
    from realdoc_bench.evaluate.score import DEFAULT_MODEL

    stamp = layout.root / f".extractor_ok_{DEFAULT_MODEL}_{_fixture_hash()[:8]}"
    if stamp.exists():
        return
    fails = st.run_extractor()
    if fails:
        raise PreconditionError(f"extractor validation FAILED for {DEFAULT_MODEL}: {fails}")
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.touch()


@app.command()
def direct(
    run_dir: Path = typer.Option(..., "--run-dir"),
    registry: Path = typer.Option(Path("configs/registry.yaml"), "--registry"),
    model: list[str] = typer.Option(..., "--model", "-m"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    max_spend: float | None = typer.Option(None, "--max-spend"),
    limit: int | None = typer.Option(None, "--limit"),
    no_image: bool = typer.Option(False, "--no-image"),
    workers: int = typer.Option(8, "--workers"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    from ocr_eval_ext.config import get_entry
    from ocr_eval_ext.direct import run_direct

    layout = RunLayout.at(run_dir)
    _preflight(layout)                     # cardinalities + pins + scorer self-test, fail-closed
    entries = load_registry(registry)
    chosen = [get_entry(entries, m) for m in model]
    bad_shape = [e.id for e in chosen if e.shape != "vlm-chat" or e.transport != "openai-compat"]
    if bad_shape:
        console.print(f"[red]not vlm-chat/openai-compat: {bad_shape} — use ocr-eval parse for those[/red]")
        raise typer.Exit(1)
    summary = run_direct(layout, chosen, dry_run=dry_run, max_spend_usd=max_spend,
                         limit=limit, no_image=no_image, workers=workers, force=force)
    console.print(summary)
    if not dry_run and summary.get("error"):
        raise typer.Exit(2)   # fail-visible: errors occurred, report will mark them


# Additional commands added in Tasks 8-9 (parse/score/preflight/report) all call _preflight(layout)
# first. `ocr-eval score` additionally runs the BLOCKING extractor-validation gate via
# `require_extractor_gate(layout)` above — 5 Gemini calls per (extractor, fixture-set), cached
# per run dir via a `.extractor_ok_<model>_<fixturehash>` stamp file, never skippable when the
# stamp is absent.
# `ocr-eval score` also records DEFAULT_MODEL into run_meta.json as extractor id; if a later
# invocation sees a different extractor id there, it refuses to mix generations unless
# --new-extractor-generation is passed, which archives eval/cache/ to eval/cache@<old-id>/ first.


@app.command()
def rescore(run_dir: Path = typer.Option(..., "--run-dir")) -> None:
    """Recompute field_matches/match from stored answers with CURRENT templates — both shapes,
    zero API calls. (Never point upstream `evaluate score --force` at vlm__* keys: it would
    overwrite them with 'markdown missing' and destroy the paid-for answers.)"""
    from realdoc_bench.evaluate.score import _ensure_template, score_typed

    layout = RunLayout.at(run_dir)
    _preflight(layout)
    items = {i["question_id"]: i for i in json.loads(layout.bank_path.read_text())["items"]}
    changed = 0
    for f in sorted(layout.cache_dir.glob("*.json")):
        rec = json.loads(f.read_text())
        item = items.get(rec.get("qid"))
        if item is None or "answer" not in rec or rec["answer"] is None:
            continue
        _ensure_template(item)
        fm, allc = score_typed(rec["answer"], item["gold_dict"], item["str_keys"])
        if fm != rec.get("field_matches") or allc != rec.get("match"):
            rec["field_matches"], rec["match"] = fm, allc
            f.write_text(json.dumps(rec, ensure_ascii=False))
            changed += 1
    console.print(f"rescore: {changed} row(s) changed")
