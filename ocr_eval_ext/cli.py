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


def _observed_dataset_revision(layout: RunLayout) -> str | None:
    """huggingface_hub's `local_dir=...` download mode (used by `evaluate download`) writes
    `.cache/huggingface/download/<file>.metadata` sidecars whose first line is the resolved
    commit hash for that file, and a `.cache/huggingface/trees/<resolved revision>.json` tree
    snapshot named after that same hash — both confirmed on disk under `runs/stage1/.cache/`.
    Prefer the `qa_bank.json` sidecar (the bank is the file whose revision `check_bank`'s
    cardinality contract actually depends on); fall back to the sole `trees/*.json` filename if
    the sidecar is missing or the trees dir doesn't resolve unambiguously. Returns None if
    neither is present — e.g. the corpus wasn't fetched via `evaluate download`, or an older
    huggingface_hub cache-mode layout is in use. Callers must treat None as 'unknown', never as
    'matches the pin'."""
    sidecar = layout.root / ".cache" / "huggingface" / "download" / "qa_bank.json.metadata"
    if sidecar.exists():
        lines = sidecar.read_text().splitlines()
        if lines and lines[0].strip():
            return lines[0].strip()
    trees_dir = layout.root / ".cache" / "huggingface" / "trees"
    if trees_dir.is_dir():
        revs = [p.stem for p in trees_dir.glob("*.json")]
        if len(revs) == 1:
            return revs[0]
    return None


def _stamp_base_meta(layout: RunLayout, pins: dict, *, mark_partial_corpus: bool = False) -> None:
    """Called at the end of ANY successful `_preflight` — verify, direct, rescore, parse, and
    score. Fills in `dataset_revision`/`harness_commit` once, the first time a run dir passes
    preflight; never overwrites them afterwards (a run dir's identity is fixed at first success,
    not re-derived on every command). Prefers the OBSERVED on-disk HF revision over the pins
    value when available (the observed-vs-pins comparison in `_preflight`, just above the call
    site, has already confirmed it matches the pin, so this is never a silent divergence),
    recording which source was used so a run stamped before `evaluate download` ever ran is
    distinguishable from one whose revision was actually cross-checked against disk.

    N3 RULING (supersedes the round-1 "overwritten every call" design): `partial_corpus` is
    STICKY-true. `mark_partial_corpus=True` (i.e. this preflight used `--allow-partial-corpus`)
    sets it to `true`. Every OTHER call — including a normal preflight that just re-confirmed
    full corpus completeness — leaves an existing `true` alone; a plain `_preflight` success is
    NOT considered strong enough evidence to clear the flag (it doesn't confirm renders, and
    doesn't want the flag to flicker back to `false` via routine low-cost calls). The ONLY
    sanctioned way to clear it is `verify`'s own full-sweep success path (renders + complete
    corpus), which writes `partial_corpus = False` directly — see `verify()`."""
    meta_p = _run_meta_path(layout)
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    if "dataset_revision" not in meta:
        observed = _observed_dataset_revision(layout)
        if observed is not None:
            meta["dataset_revision"] = observed
            meta["revision_source"] = "hf_metadata"
        else:
            meta["dataset_revision"] = pins["dataset_revision"]
            meta["revision_source"] = "pins"
    # F10: backfill for a run_meta.json stamped before `revision_source` existed (or otherwise
    # hand-crafted without it) — the branch above only ever sets the key the FIRST time
    # `dataset_revision` itself gets written, so a legacy meta that already has `dataset_revision`
    # but never got `revision_source` would carry the gap forever. `setdefault` makes this a
    # no-op on every meta that already has the key (including the branch above, same call).
    meta.setdefault("revision_source", "hf_metadata"
                    if _observed_dataset_revision(layout) == pins["dataset_revision"] else "pins")
    meta.setdefault("harness_commit", pins["harness_commit"])
    if mark_partial_corpus:
        meta["partial_corpus"] = True
    else:
        meta.setdefault("partial_corpus", False)   # sticky: never flips true -> false here
    meta_p.write_text(json.dumps(meta, indent=2))


def _check_corpus_completeness(layout: RunLayout, items: list[dict]) -> None:
    """C1 (flagged in Task 7 review, carried here): docs/ must be a superset of every
    source_file the bank references. Lives inside `_preflight` itself — rather than duplicated
    per-command (verify used to run this check standalone, after its own `_preflight` call) — so
    EVERY command that calls `_preflight` (verify, direct, parse, score) is protected for free,
    unless the caller explicitly opted out via `--allow-partial-corpus` (I6; `verify` never
    exposes that flag) or is `rescore` (N4; never applicable — see `_preflight`'s
    `skip_corpus_check`). Cheap (path existence only, no rendering). Without this, an empty (or
    `evaluate download --limit`-narrowed) docs/ against a full, real-shaped bank would
    sweep/parse/score zero PDFs and look like a clean pass — a false pass reachable in normal
    operation, not just a contrived test.

    M3: a bank item missing the `source_file` key raises a plain `KeyError` from the set
    comprehension below — converted to `PreconditionError` so `_preflight`'s existing except
    clause (which already catches `PreconditionError`) reports it as a clean red message, never a
    raw traceback."""
    docs_stems = {p.stem for p in layout.docs_dir.glob("*.pdf")}
    try:
        bank_stems = {i["source_file"] for i in items}
    except KeyError as e:
        raise PreconditionError(f"bank item missing required key {e} — malformed bank") from e
    missing = sorted(bank_stems - docs_stems)
    if missing:
        console.print(f"[red]preflight FAILED — corpus incomplete: {len(missing)} bank "
                      f"source_file(s) have no PDF under docs/ (first 10): {missing[:10]}[/red]")
        raise typer.Exit(1)


def _preflight(layout: RunLayout, *, allow_partial_corpus: bool = False,
              skip_corpus_check: bool = False) -> None:
    """The gate every spending/reporting command calls first. Fail-closed.

    `allow_partial_corpus` (I6): when true, skips `_check_corpus_completeness` with a yellow
    warning instead of failing closed, and marks `partial_corpus: true` in `run_meta.json` (see
    `_stamp_base_meta`'s N3 docstring — this is a sticky, user-visible flag, not a per-call
    toggle) so the bypass is visible on disk, not just in a log line. Default is `False`
    everywhere; `verify` never passes `True` — it has no `--allow-partial-corpus` flag at all.

    `skip_corpus_check` (N4 RULING): used ONLY by `rescore`, which recomputes scores from
    already-cached answers and never reads `docs/` at all — the corpus-completeness question
    ("does every bank source_file have a PDF on disk?") simply doesn't apply to it, so it's
    skipped silently (no warning banner, unlike `allow_partial_corpus`, since nothing is actually
    being bypassed) and `partial_corpus` is left completely untouched either way. Mutually
    exclusive in practice with `allow_partial_corpus` (no command needs both; `skip_corpus_check`
    takes precedence if somehow both were set)."""
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
        if meta.get("harness_commit") != pins["harness_commit"]:
            console.print(f"[red]run dir was stamped with harness_commit "
                          f"{meta.get('harness_commit')!r}, pins say "
                          f"{pins['harness_commit']!r}[/red]")
            raise typer.Exit(1)
    observed = _observed_dataset_revision(layout)
    if observed is not None and observed != pins["dataset_revision"]:
        # A REAL check, independent of whatever run_meta.json claims: the actual on-disk HF
        # snapshot metadata is ground truth for what was downloaded, and must match the pin too.
        console.print(f"[red]on-disk HF snapshot metadata shows dataset_revision {observed!r}, "
                      f"pins say {pins['dataset_revision']!r} — the downloaded corpus does not "
                      f"match the pin[/red]")
        raise typer.Exit(1)
    try:
        items = json.loads(layout.bank_path.read_text())["items"]
        check_bank(items)
        if skip_corpus_check:
            pass    # rescore: never reads docs/, corpus completeness is not applicable
        elif allow_partial_corpus:
            console.print("[yellow]--allow-partial-corpus: corpus-completeness check "
                          "skipped[/yellow]")
        else:
            _check_corpus_completeness(layout, items)
    except (FileNotFoundError, KeyError, PreconditionError) as e:
        console.print(f"[red]preflight FAILED: {e}[/red]")
        raise typer.Exit(1) from e
    _stamp_base_meta(layout, pins, mark_partial_corpus=allow_partial_corpus)


def _png_cache_path(png_cache: Path, stem: str, condition: dict) -> Path:
    """Must stay in sync with `ocr_eval_ext.direct._render_page`'s internal naming convention
    (`{stem}@{dpi}@{preprocess}.png`) — duplicated here (rather than importing a private helper
    that doesn't expose its path) so `verify` can check the cached file's mtime *before* deciding
    whether to trust it, which `_render_page` itself never does (see I1 below)."""
    dpi = condition["render"]["dpi"]
    pre = condition["preprocess"]
    return png_cache / f"{stem}@{dpi}@{pre}.png"


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
    from ocr_eval_ext.preconditions import assert_single_page, ink_coverage

    layout = RunLayout.at(run_dir)
    _preflight(layout)

    # M8: the harness-commit pin only proves ancestry (some commit reachable from HEAD), never
    # working-tree purity — uncommitted local edits to the harness code are invisible to it.
    # Informational only; never blocks the run.
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "realdoc_bench", "ocr_eval_ext"],
                           capture_output=True, text=True, cwd=REPO_ROOT)
    if dirty.returncode == 0 and dirty.stdout.strip():
        console.print("[yellow]warning: realdoc_bench/ or ocr_eval_ext/ has uncommitted "
                      "changes — the harness-commit pin proves ancestry, not working-tree "
                      "purity[/yellow]")

    # C1 (Task 8): the bank-stems-subset-of-docs-stems check now lives in `_check_corpus_
    # completeness`, called from `_preflight` above — so it protects every command that calls
    # `_preflight`, not just `verify`. `pdfs` is still needed here for the render sweep below.
    pdfs = sorted(layout.docs_dir.glob("*.pdf"))

    if skip_renders:
        console.print("[yellow]--skip-renders: pins + cardinality checked, "
                      "pages/renders NOT swept[/yellow]")
        console.print("[green]verify PASS[/green] — pins, cardinalities green (renders skipped)")
        return

    png_cache = layout.root / "docs_png"
    blank, multi = [], []
    for pdf in pdfs:
        # I1: check the page count DIRECTLY on the PDF every time, regardless of cache state —
        # cheap (fitz page_count, no render) and never disarmable by a warm PNG cache. Before
        # this fix, a swapped-in 2-page PDF whose PNG was already cached would sail through
        # unnoticed, since `_render_page`'s cache-hit path never re-checks page count.
        try:
            assert_single_page(pdf)
        except PreconditionError:
            multi.append(pdf.stem)
            continue
        cache_p = _png_cache_path(png_cache, pdf.stem, STAGE1_CONDITION)
        if cache_p.exists() and cache_p.stat().st_mtime >= pdf.stat().st_mtime:
            png = cache_p.read_bytes()
        else:
            if cache_p.exists():
                cache_p.unlink()        # stale — a PDF-newer-than-cache entry must never be trusted
            png = _render_page(layout, pdf.stem, STAGE1_CONDITION, png_cache)
        if ink_coverage(png) < 0.001:
            blank.append(pdf.stem)
    if blank or multi:
        console.print(f"[red]verify FAILED — multi-page: {multi} blank-render: {blank}[/red]")
        raise typer.Exit(1)

    # I2b: the full sweep additionally affirms renders_verified — distinct from the base
    # dataset_revision/harness_commit stamp that `_preflight` (I2a) already wrote above, since
    # only THIS path actually confirms every page renders and is non-blank.
    from importlib.metadata import version as _v

    meta_p = _run_meta_path(layout)
    meta = json.loads(meta_p.read_text())     # guaranteed to exist: _preflight always stamps on success
    meta.setdefault("pymupdf_version", _v("pymupdf"))
    # vLLM runs in its own serving env, not this one — its version is recorded per model in
    # docs/local-serving.md (Task 8), which is the operative record for local rows.
    meta["renders_verified"] = True
    # N3 RULING: this full sweep (renders + complete corpus, never --skip-renders, never
    # --allow-partial-corpus — verify has no such flag) is the ONLY event allowed to clear a
    # sticky `partial_corpus` flag back to `false`. Every other successful preflight (including
    # this same command's own `--skip-renders` path) leaves an existing `true` alone.
    meta["partial_corpus"] = False
    meta_p.write_text(json.dumps(meta, indent=2))
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
    own extractor must not be allowed to spend.

    Raises `PreconditionError` when the extractor demonstrably gets a known-answer fixture wrong
    (a validation FAILURE — no stamp is written). If `GEMINI_API_KEY`/`GOOGLE_API_KEY` is unset,
    `run_extractor` (via `gemini_extract` -> `require_api_key`) raises `RuntimeError` instead —
    that's a configuration problem, not a validation failure, and is left to propagate unmodified
    rather than being swallowed into a misleading `PreconditionError`.

    Writing a fresh stamp also prunes every OTHER `.extractor_ok_<DEFAULT_MODEL>_*` file for this
    model — a stale fixture-hash left over from before `st.EXTRACTOR_FIXTURES` last changed —
    so an old stamp can never be mistaken for a currently-valid pass."""
    from realdoc_bench.evaluate.score import DEFAULT_MODEL

    stamp = layout.root / f".extractor_ok_{DEFAULT_MODEL}_{_fixture_hash()[:8]}"
    if stamp.exists():
        return
    fails = st.run_extractor()
    if fails:
        raise PreconditionError(f"extractor validation FAILED for {DEFAULT_MODEL}: {fails}")
    for stale in layout.root.glob(f".extractor_ok_{DEFAULT_MODEL}_*"):
        stale.unlink()
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.touch()


@app.command()
def direct(
    run_dir: Path = typer.Option(..., "--run-dir"),
    registry: Path = typer.Option(REPO_ROOT / "configs" / "registry.yaml", "--registry"),
    model: list[str] = typer.Option(..., "--model", "-m"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    max_spend: float | None = typer.Option(None, "--max-spend"),
    limit: int | None = typer.Option(None, "--limit"),
    no_image: bool = typer.Option(False, "--no-image"),
    workers: int = typer.Option(8, "--workers"),
    force: bool = typer.Option(False, "--force"),
    allow_partial_corpus: bool = typer.Option(
        False, "--allow-partial-corpus",
        help="Skip the corpus-completeness check (docs/ ⊇ bank source_files) with a warning, "
             "and record partial_corpus: true in run_meta.json. Default stays fail-closed."),
) -> None:
    from ocr_eval_ext.config import get_entry
    from ocr_eval_ext.direct import DIRECT_TRANSPORTS, run_direct

    layout = RunLayout.at(run_dir)
    _preflight(layout, allow_partial_corpus=allow_partial_corpus)   # cardinalities + pins +
                                                                      # scorer self-test, fail-closed
    entries = load_registry(registry)
    chosen = [get_entry(entries, m) for m in model]
    # Mirrors `run_direct`'s own transport precondition rather than narrowing it: both of the
    # vlm-chat transports belong on this leg. (This check predated `bedrock-converse` and still
    # said `openai-compat` only, which rejected every Bedrock entry at the CLI even though
    # `run_direct` drives them — caught running the runbook's Bedrock path 2026-08-04.)
    bad_shape = [e.id for e in chosen
                 if e.shape != "vlm-chat" or e.transport not in DIRECT_TRANSPORTS]
    if bad_shape:
        console.print(f"[red]not a vlm-chat entry on {'/'.join(sorted(DIRECT_TRANSPORTS))}: "
                      f"{bad_shape} — use ocr-eval parse for those[/red]")
        raise typer.Exit(1)
    summary = run_direct(layout, chosen, dry_run=dry_run, max_spend_usd=max_spend,
                         limit=limit, no_image=no_image, workers=workers, force=force)
    console.print(summary)
    if dry_run:
        return
    # F3: a rerun that never re-attempts a prior error cell (no --force) still counts that error
    # under "cached" — surface it explicitly and fail-visible, rather than a summary reading
    # "cached: N, error: 0" that hides N-of-those-cached being unresolved failures.
    if summary.get("cached_error"):
        console.print(f"[red]{summary['cached_error']} cached error cell(s) present — cheapest remedy: "
                      f"delete just those rows from eval/cache/ and rerun (only they re-attempt); "
                      f"--force re-runs and RE-BILLS the entire matrix[/red]")
        raise typer.Exit(2)
    if summary.get("error"):
        raise typer.Exit(2)   # fail-visible: errors occurred, report will mark them


@app.command()
def rescore(run_dir: Path = typer.Option(..., "--run-dir")) -> None:
    """Recompute field_matches/match from stored answers with CURRENT templates — both shapes,
    zero API calls. (Never point upstream `evaluate score --force` at vlm__* keys: it would
    overwrite them with 'markdown missing' and destroy the paid-for answers.)

    A row is rescored whenever its cache JSON has an `"answer"` KEY at all — including an
    explicit `"answer": null` — matching upstream `_worker`'s own cache-hit gate (`"answer" in
    rec`), which never additionally checks truthiness. Rows with no `answer` key (no qid match in
    the current bank, or a terminal error record with nothing to rescore) are counted under
    `skipped_not_in_bank` and left untouched. Writes are atomic (`_atomic_write_json`, the same
    tmp-file + os.replace pattern `direct.py` uses) so a killed rescore can never leave a torn
    cache row.

    N4 RULING: skips the corpus-completeness check entirely (`_preflight(..., skip_corpus_check=
    True)`) — rescore only ever reads `eval/cache/` and the bank, never `docs/`, so "does every
    bank source_file have a PDF on disk" is not a question this command can be blocked by. All
    other preflight gates (pins, harness-commit ancestry, dataset-revision cross-check, bank
    cardinality) still apply in full."""
    from ocr_eval_ext.direct import _atomic_write_json
    from realdoc_bench.evaluate.score import _ensure_template, score_typed

    layout = RunLayout.at(run_dir)
    _preflight(layout, skip_corpus_check=True)
    items = {i["question_id"]: i for i in json.loads(layout.bank_path.read_text())["items"]}
    examined = skipped_not_in_bank = changed = 0
    for f in sorted(layout.cache_dir.glob("*.json")):
        examined += 1
        rec = json.loads(f.read_text())
        item = items.get(rec.get("qid"))
        if item is None or "answer" not in rec:
            skipped_not_in_bank += 1
            continue
        _ensure_template(item)
        fm, allc = score_typed(rec["answer"], item["gold_dict"], item["str_keys"])
        if fm != rec.get("field_matches") or allc != rec.get("match"):
            rec["field_matches"], rec["match"] = fm, allc
            _atomic_write_json(f, rec)
            changed += 1
    console.print(f"rescore: examined={examined}, skipped_not_in_bank={skipped_not_in_bank}, "
                 f"changed={changed}")


@app.command()
def preflight(
    entry_id: str = typer.Argument(..., help="Registry id, e.g. glm-ocr@local-vllm"),
    registry: Path = typer.Option(REPO_ROOT / "configs" / "registry.yaml", "--registry"),
) -> None:
    """Confirm the entry's `model` is actually reachable and serving before any spend.

    - **openai-compat:** `GET {base_url}/models` and check the entry's `model` is in the served
      list. Run before any local vLLM spend (`direct`/`parse`/`score` against a `local: true`
      entry) — a served-model mismatch (wrong checkpoint resident, server not restarted after a
      config change) is caught here rather than silently transcribing against the wrong weights.
    - **bedrock-converse:** one minimal real `Converse` call (a few tokens). Bedrock has no
      served-model listing that answers the question that matters: on a probed account 119 models
      were listed and only 7 were invokable by the calling role, so `AccessDeniedException` on
      invoke is the normal state for a listed-but-not-enabled model. Only an actual call
      distinguishes them.

    upstream-parser entries (Gemini, Mistral) have no server to preflight."""
    from ocr_eval_ext.bedrock import preflight_bedrock
    from ocr_eval_ext.config import get_entry
    from ocr_eval_ext.parsers_openai import preflight as _served_model

    entries = load_registry(registry)
    entry = get_entry(entries, entry_id)
    if entry.transport not in ("openai-compat", "bedrock-converse"):
        console.print(f"[red]{entry_id}: transport={entry.transport!r} — preflight only applies "
                      f"to openai-compat and bedrock-converse entries[/red]")
        raise typer.Exit(1)
    try:
        served = (preflight_bedrock(entry) if entry.transport == "bedrock-converse"
                  else _served_model(entry))
    except Exception as e:
        console.print(f"[red]preflight FAILED for {entry_id}: {e}[/red]")
        raise typer.Exit(1) from e
    target = f"bedrock:{entry.region}" if entry.transport == "bedrock-converse" else entry.base_url
    console.print(f"[green]preflight PASS[/green] — {target} serves {served}")


def _default_parse_workers(registry_path: Path, requested: list[str]) -> int:
    """I5: `page_concurrency=1` on `OpenAICompatVisionParser` only serializes requests WITHIN one
    document's `parse()` call — it does nothing to stop upstream `run_parse`'s own cross-document
    `ThreadPoolExecutor(max_workers=workers)` from firing `workers` PDFs at the same local vLLM
    server concurrently, each from a different document's worker thread. So the local-GPU
    serialization invariant only actually holds if `--workers` is ALSO 1 whenever a requested
    parser is `local: true`. Re-derives each requested name's registry entry the same way
    `register_openai_parsers` does (`safe_name(id) + "__" + condition_hash(...)`) rather than
    threading extra state through that function, since its return contract (`list[str]`) is
    established by Task 8's brief and callers besides this one rely on it staying that way."""
    from ocr_eval_ext.config import load_registry
    from ocr_eval_ext.direct import condition_hash
    from ocr_eval_ext.parsers_openai import TRANSCRIBER_CONDITION, safe_name

    suffix = condition_hash(TRANSCRIBER_CONDITION)
    requested_set = set(requested)
    for e in load_registry(registry_path):
        if (e.shape == "transcriber" and e.transport == "openai-compat" and e.local
                and f"{safe_name(e.id)}__{suffix}" in requested_set):
            return 1
    return 8


@app.command()
def parse(
    run_dir: Path = typer.Option(..., "--run-dir"),
    registry: Path = typer.Option(REPO_ROOT / "configs" / "registry.yaml", "--registry"),
    parser: list[str] = typer.Option(..., "--parser", "-p"),
    workers: int | None = typer.Option(
        None, "--workers",
        help="Default: 1 if any requested parser is a local:true registry entry (serializes "
             "cross-document calls against the single resident vLLM server), else 8. An "
             "explicit value here always wins over that default. See also RDB_PAGE_CONCURRENCY "
             "(docs/local-serving.md), which controls PAGE concurrency within one document, a "
             "separate knob from this cross-DOCUMENT concurrency."),
    force: bool = typer.Option(False, "--force"),
    limit: int | None = typer.Option(None, "--limit"),
    allow_partial_corpus: bool = typer.Option(
        False, "--allow-partial-corpus",
        help="Skip the corpus-completeness check (docs/ ⊇ bank source_files) with a warning, "
             "and record partial_corpus: true in run_meta.json. Default stays fail-closed."),
) -> None:
    """Register this run's dynamically-built openai-compat transcriber parsers — upstream's own
    `realdoc-bench` CLI never imports `ocr_eval_ext`, so these parsers don't exist in that
    process; this wrapper makes them reachable in-process — then delegate to upstream
    `run_parse`. Also writes `parses/<parser>/condition.json` (the TRANSCRIBER_CONDITION dict,
    verbatim) as a sidecar per parser we registered, so transcriber rows are self-describing;
    Stage 2's deskew registers under a different condition hash (a different parser NAME)
    instead of overwriting these."""
    from ocr_eval_ext.direct import _atomic_write_json
    from ocr_eval_ext.parsers_openai import TRANSCRIBER_CONDITION, register_openai_parsers
    from realdoc_bench.evaluate.parse import run_parse
    from realdoc_bench.evaluate.parsers.base import registry as parser_registry

    layout = RunLayout.at(run_dir)
    _preflight(layout, allow_partial_corpus=allow_partial_corpus)
    try:
        registered = register_openai_parsers(registry)
    except ValueError as e:      # I3: stale binding under an already-registered parser name
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    unknown = [p for p in parser if p not in parser_registry]
    if unknown:
        console.print(f"[red]unknown parser name(s): {unknown}. registered this run: "
                      f"{registered}. available: {parser_registry.names()}[/red]")
        raise typer.Exit(1)
    if workers is None:
        workers = _default_parse_workers(registry, parser)
        console.print(f"[cyan]--workers not given — defaulting to {workers} "
                      f"({'a local:true parser was requested' if workers == 1 else 'no local:true parser requested'})[/cyan]")
    records = run_parse(layout, parser, force=force, workers=workers, limit=limit)
    for p in parser:
        if p in registered:      # only our openai-compat transcribers carry a condition —
                                  # upstream-parser names (gemini_3_5_flash, mistral_ocr_4) never
                                  # vary conditions and have none to write
            _atomic_write_json(layout.parser_dir(p) / "condition.json", TRANSCRIBER_CONDITION)
    n_ok = sum(1 for r in records if r.ok)
    n_fail = len(records) - n_ok
    console.print(f"parse: {n_ok} ok, {n_fail} fail")
    if n_fail:
        raise typer.Exit(2)
    # F1: an `ok=true` transcript that is still essentially EMPTY (e.g. a thinking model that
    # exhausted its completion budget on reasoning and left "## Page 1\n\n" or a few stray
    # characters — see progress.md's Task 10 local-validation finding) sails through the n_fail
    # gate above with a false-green ok/fail split. Scoring would then spend one Gemini extractor
    # call per bank item transcribed by this parser against effectively no content. Floor:
    # md_length must clear 16 chars per page — generous (well under even the shortest genuine
    # per-page markdown observed), so this only trips on transcripts that are functionally empty.
    trivial = [r for r in records if r.ok and r.md_length <= 16 * max(1, r.page_count)]
    if trivial:
        stems = [r.stem for r in trivial]
        console.print(
            f"[red]parse FAILED — {len(trivial)} transcript(s) too short to score "
            f"(md_length <= 16 chars/page) (first 10): {stems[:10]} — scoring would spend one "
            f"extractor call per bank item against no content[/red]")
        raise typer.Exit(2)


def _enforce_extractor_generation(layout: RunLayout, new_extractor_generation: bool) -> None:
    """`ocr-eval score` records the Gemini extractor id (`score.DEFAULT_MODEL`) into
    `run_meta.json` the first time it scores a run dir. A LATER invocation whose current
    `DEFAULT_MODEL` differs from what's recorded there refuses to mix generations under the same
    `eval/cache/` — verdicts from two different extractor judges are not directly comparable —
    unless `--new-extractor-generation` is passed.

    I4 ruling: archiving is SELECTIVE, not whole-directory. Only extractor-dependent rows (parser
    key does NOT start with `vlm__`) move to `eval/cache@<old-id>/`; `vlm__*` rows (direct
    vlm-chat answers) stay live in `eval/cache/` — they carry no Gemini-extractor dependency at
    all, so re-archiving them would force needless re-spend on the NEXT direct/score run for no
    correctness benefit. The parser is read from each cache row's own `"parser"` field (not
    parsed out of the filename) since that's the single source of truth `_worker`/`run_direct`
    both write.

    N2: collisions are checked for ALL candidate rows BEFORE any file is moved — never
    file-by-file (which would leave a partial, already-mutated archive if row N of M collided).
    A single colliding destination aborts the whole operation with `exit 1`, naming every
    colliding path, and moves nothing.

    M5 (N5-round-3 corrected — the ordering below is the reality, this comment previously claimed
    the opposite): called AFTER `require_extractor_gate` in `score` — never stamp/archive on the
    strength of an extractor generation that hasn't yet proven it can pass its own fixtures; a
    failed gate raises before this function is ever reached, leaving `run_meta.json`/
    `eval/cache/` exactly as they were. Writes `run_meta.json` atomically (`_atomic_write_json`)
    so a killed run can never leave a torn file."""
    from ocr_eval_ext.direct import _atomic_write_json
    from realdoc_bench.evaluate.score import DEFAULT_MODEL

    meta_p = _run_meta_path(layout)
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    prev = meta.get("extractor_id")
    if prev is not None and prev != DEFAULT_MODEL:
        if not new_extractor_generation:
            console.print(f"[red]run dir was scored with extractor {prev!r}; current extractor "
                          f"is {DEFAULT_MODEL!r} — pass --new-extractor-generation to start a "
                          f"new generation (archives extractor-dependent rows in eval/cache/ to "
                          f"eval/cache@{prev}/ first)[/red]")
            raise typer.Exit(1)
        archive = layout.root / "eval" / f"cache@{prev}"

        to_move: list[Path] = []
        if layout.cache_dir.exists():
            for f in sorted(layout.cache_dir.glob("*.json")):
                try:
                    rec = json.loads(f.read_text())
                except Exception:
                    rec = {}
                if not str(rec.get("parser", "")).startswith("vlm__"):
                    to_move.append(f)   # extractor-dependent (incl. unreadable/unknown) rows only

        collisions = [f for f in to_move if (archive / f.name).exists()]
        if collisions:
            console.print(f"[red]archive target(s) already exist under {archive} — refusing to "
                          f"overwrite; move or remove them before retrying: "
                          f"{[str(archive / f.name) for f in collisions]}[/red]")
            raise typer.Exit(1)     # nothing moved — the scan above was read-only

        if to_move:
            archive.mkdir(parents=True, exist_ok=True)
            for f in to_move:
                f.rename(archive / f.name)
        console.print(f"[yellow]--new-extractor-generation: archived {len(to_move)} "
                      f"extractor-dependent row(s) to {archive}; vlm__* direct rows left "
                      f"live[/yellow]")
    meta["extractor_id"] = DEFAULT_MODEL
    _atomic_write_json(meta_p, meta)


@app.command()
def score(
    run_dir: Path = typer.Option(..., "--run-dir"),
    registry: Path = typer.Option(REPO_ROOT / "configs" / "registry.yaml", "--registry"),
    parser: list[str] = typer.Option([], "--parser", "-p"),
    force: bool = typer.Option(False, "--force"),
    workers: int = typer.Option(16, "--workers"),
    limit: int | None = typer.Option(None, "--limit"),
    new_extractor_generation: bool = typer.Option(
        False, "--new-extractor-generation",
        help="Confirm switching Gemini extractor generations for this run dir — archives "
             "extractor-dependent rows in eval/cache/ to eval/cache@<old-id>/ first (vlm__* "
             "direct rows stay live) so scores from two different judges are never silently "
             "mixed."),
    allow_partial_corpus: bool = typer.Option(
        False, "--allow-partial-corpus",
        help="Skip the corpus-completeness check (docs/ ⊇ bank source_files) with a warning, "
             "and record partial_corpus: true in run_meta.json. Default stays fail-closed."),
) -> None:
    """Register this run's openai-compat transcriber parsers, then delegate to upstream
    `run_score` + `aggregate_results`. Requires GEMINI_API_KEY/GOOGLE_API_KEY (`require_api_key`
    fails fast, before any Gemini spend) and passes the BLOCKING extractor-validation gate
    (`require_extractor_gate` — 5 Gemini calls against known-answer fixtures, cached per
    (run dir, extractor, fixture-set)) BEFORE any generation bookkeeping (M5/M6) — a broken
    extractor must never get to stamp itself in or archive anything."""
    from ocr_eval_ext.parsers_openai import register_openai_parsers
    from realdoc_bench.evaluate.parsers.base import registry as parser_registry
    from realdoc_bench.evaluate.score import aggregate_results, require_api_key, run_score

    layout = RunLayout.at(run_dir)
    _preflight(layout, allow_partial_corpus=allow_partial_corpus)
    try:
        register_openai_parsers(registry)
    except ValueError as e:      # I3: stale binding under an already-registered parser name
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    require_api_key()                              # fail fast, before any Gemini spend

    # M1: a typo'd -p name that matches neither a registered parser class nor an on-disk parsed
    # dir would otherwise sail into `run_score`, which treats it as a legitimate-but-unparsed
    # parser and writes "markdown missing" for every single bank item (1356 sticky rows for the
    # real corpus) — refuse before any of that gets written.
    if parser:
        known = set(parser_registry.names()) | set(layout.parsed_parsers())
        unknown = [p for p in parser if p not in known]
        if unknown:
            console.print(f"[red]unknown parser name(s) for score: {unknown}. registered: "
                          f"{sorted(parser_registry.names())}. parsed on disk: "
                          f"{sorted(layout.parsed_parsers())}[/red]")
            raise typer.Exit(1)

    try:
        require_extractor_gate(layout)              # BLOCKING — 5 Gemini calls, cached
    except PreconditionError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    _enforce_extractor_generation(layout, new_extractor_generation)
    records = run_score(layout, parser or None, force=force, workers=workers, limit=limit)
    agg = aggregate_results(layout)
    n_ok = sum(1 for r in records if r.ok)
    n_match = sum(1 for r in records if r.match)
    console.print(f"score: {len(records)} cells, {n_ok} answered, {n_match} fully correct — "
                 f"results written to {agg['path']}")
    if len(records) - n_ok:
        raise typer.Exit(2)


@app.command()
def report(
    run_dir: Path = typer.Option(..., "--run-dir"),
    registry: list[Path] = typer.Option(
        [REPO_ROOT / "configs" / "registry.yaml"], "--registry",
        help="Repeatable. A run dir can legitimately hold rows from more than one registry "
             "(e.g. configs/registry.yaml + configs/registry-bedrock.yaml); pass each one so "
             "every row resolves its stamp columns instead of rendering '(unregistered)'."),
    allow_stale_render: bool = typer.Option(
        False, "--allow-stale-render",
        help="D3: proceed even when a vlm__ row's stored image_sha no longer matches a "
             "freshly recomputed render of the same doc (STALE-RENDER). Default stays "
             "fail-closed — the report still shows the STALE-RENDER section either way."),
    iters: int = typer.Option(2000, "--iters", help="Bootstrap resamples per CI/paired delta."),
) -> None:
    """Build the shape-segregated markdown report (`<run-dir>/report.md`). Runs `_preflight`
    against the bank first — a report over the wrong bank (wrong revision, wrong cardinalities)
    must not render. Any pre-existing `dashboard.html` (upstream's shape-mixed HTML report) is
    renamed to `dashboard-upstream-UNSEGREGATED.html` so it can never be mistaken for the
    authoritative output."""
    from ocr_eval_ext.report_md import ReportError, build_markdown_report

    layout = RunLayout.at(run_dir)
    _preflight(layout)
    # Concatenate, then re-check ids ACROSS files: `load_registry` only rejects duplicates within
    # one file, and two registries silently sharing an id would make `_resolve_entry` pick by list
    # order — a stamp column quietly attributed to the wrong entry. Fail closed instead.
    entries = [e for path in registry for e in load_registry(path)]
    ids = [e.id for e in entries]
    cross_dupes = sorted({i for i in ids if ids.count(i) > 1})
    if cross_dupes:
        console.print(f"[red]duplicate registry ids across --registry files: {cross_dupes}[/red]")
        raise typer.Exit(1)
    try:
        md = build_markdown_report(layout, entries, allow_stale_render=allow_stale_render,
                                   iters=iters)
    except ReportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    out_path = layout.root / "report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    dash = layout.dashboard_path
    if dash.exists():
        dash.rename(dash.with_name("dashboard-upstream-UNSEGREGATED.html"))
        console.print("[yellow]renamed pre-existing dashboard.html -> "
                      "dashboard-upstream-UNSEGREGATED.html (shape-mixed, not authoritative)[/yellow]")
    console.print(f"[green]report written[/green] -> {out_path}")
