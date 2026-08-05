"""Shape-segregated markdown report — `vlm-chat` (Section A) and transcribe-then-extract
(Section B) NEVER share a leaderboard table (spec's "report segregation" hard rule): a
transcriber's score already reflects two stages (transcription + a separate Gemini extractor
call), so a Section A vs Section B number for the "same" model is not the same measurement.

Pure-function core: `build_markdown_report(layout, entries, ...)` only reads
`eval/cache/*.json`, `qa_bank.json`, `run_meta.json`, and `parses/<parser>/*` off disk — it never
calls a model, so it is fully testable without API access. All spend/rendering happens upstream
of this module (`direct.py`/`parse.py`/`score.py`); this module only aggregates what's already on
disk.

Hard-fail rulings this module enforces (never silently degrade to a warning):
  - D3 STALE-RENDER: a `vlm__*` row's stored `image_sha` must still match a freshly recomputed
    render of the same doc under the row's own `condition`. The re-render is done into a FRESH,
    empty temp directory for the duration of one `build_markdown_report` call — NEVER the run's
    own `docs_png` cache, which is cache-first and would just read back the very PNG that
    produced `image_sha` in the first place, making the check tautological (a swapped PDF would
    sail through unnoticed — review finding C1). A doc that cannot be re-rendered at all (missing
    file, corrupt, page count changed, ...) while a row still references its `image_sha` is
    itself a failure — RENDER-UNAVAILABLE — folded into the same failing set as a hash mismatch,
    never silently treated as "probably fine" (review finding C2). `StaleRenderError` unless the
    caller passes `allow_stale_render=True`. Applies to every parser key with the `vlm__` prefix,
    REGISTERED OR NOT — an unresolvable `vlm__` key is still a direct-QA row with a real
    `image_sha`, not a Section B row (review finding I1).
  - D4 serving identity: `vlm__*` cache keys do NOT pin the resolved serving provider (an
    OpenRouter route can differ per request). If two rows filed under the SAME parser key
    resolved to different providers, the key is comparing answers from two different serving
    stacks under one label — `ServingIdentityError`, unconditionally (no escape hatch: unlike
    staleness this is not something a caller should be able to wave through). Also shape-keyed,
    not registration-keyed (I1): applies to every `vlm__` parser key. Only ANSWERED rows
    (`error_class == "none"`) enter the comparison (round-2 regression fix — see
    `_check_serving_identity`'s docstring: a transient `api_error`/`render_error` row never
    reached any provider and must not be compared as a distinct identity). Among answered rows, a
    missing/falsy `resolved_provider` counts as its own distinct value, `"(empty)"`, rather than
    being dropped from the comparison — a mix of "resolved to ProviderA" and "nothing recorded"
    on two rows that BOTH actually answered is still an identity inconsistency (review finding
    M7).

D9 ruling (metrics.py) carried into this report: the headline blank-field number is
`null_metrics()["overall"].acc_over_all`, never `hallucination_rate` alone — every render of
`hallucination_rate` here is accompanied by `n_answered` and `error_rate` in the same string.

CI ruling (stats.py review): CIs render ONLY on `acc_over_all` metrics, always with `n_docs`
alongside, and `separable()`/paired deltas are never computed below `MIN_CLUSTER_DOCS` — below
the floor this renders "insufficient clusters (n_docs=N)" instead of a possibly-spurious verdict.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from ocr_eval_ext.config import RegistryEntry
from ocr_eval_ext.metrics import FieldOutcome, baseline_rows, checkbox_metrics, field_outcomes, null_metrics
from ocr_eval_ext.parsers_openai import safe_name
from ocr_eval_ext.preconditions import BLANK_TAGS, CHECKBOX_TAGS, boolean_fields, items_with_tags, null_fields
from ocr_eval_ext.stats import cluster_bootstrap_ci, paired_delta_ci, separable
from realdoc_bench.evaluate.runs import RunLayout

# Never call separable()/paired_delta_ci below this many distinct docs feeding the comparison —
# ruling carried from Task 5's stats.py review (a handful of clustered docs makes "separable"
# meaningless, not just noisy).
MIN_CLUSTER_DOCS = 20

# retrieved_at span beyond which a parser key's rows are flagged stale-serving-window (distinct
# from D3 STALE-RENDER — this is about WHEN cells were collected, not whether the page changed).
STALENESS_SPAN = dt.timedelta(days=7)

# The extractor (score.py's DEFAULT_MODEL, "gemini-3-flash-preview") is a Google model — a
# transcriber whose OWN provenance is also Google is judged by a same-family extractor.
EXTRACTOR_FAMILY_PROVENANCE = "google"

# ☒ ☐ ☑ (Unicode ballot-box glyphs) or ASCII "[x]"/"[ ]" — case-insensitive on the ASCII form.
_CHECKBOX_GLYPH_RE = re.compile(r"[☒☐☑]|\[x\]|\[ \]", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

# Sentinel for a doc whose current render could not be recomputed at all (C2). Never collides
# with a real sha256 hexdigest, which is exactly 64 lowercase hex characters.
_RENDER_FAILED = "RENDER_FAILED"


class ReportError(RuntimeError):
    """Base for hard-fail conditions build_markdown_report refuses to silently paper over."""


class StaleRenderError(ReportError):
    """D3: a vlm__ row's image_sha no longer matches the current render, or the current render
    could not be recomputed at all (RENDER-UNAVAILABLE). `.report_md` carries the full report
    text (STALE-RENDER section included) for a caller that wants to inspect it before deciding
    whether to retry with --allow-stale-render."""

    def __init__(self, message: str, report_md: str | None = None):
        super().__init__(message)
        self.report_md = report_md


class ServingIdentityError(ReportError):
    """D4: one vlm__ parser key's rows resolved to more than one distinct provider. No escape
    hatch — unlike staleness, this is a cache-key design invariant, not a freshness judgment
    call."""


# ── FieldOutcome tuple builders ─────────────────────────────────────────────────────────────

def _all_fields(items: list[dict]) -> list[tuple]:
    """Every (qid, key, gold, doc) across every gold_dict key of every bank item — used for the
    "general per-field" column, which (unlike checkbox/null) is not restricted to one bucket."""
    return [(it["question_id"], k, v, it["source_file"])
            for it in items for k, v in (it.get("gold_dict") or {}).items()]


# ── Cache loading + parser-key -> registry-entry resolution ────────────────────────────────

def _load_cache_rows(layout: RunLayout) -> dict[str, dict[str, dict]]:
    """parser -> {qid: row}. Grouped by each row's OWN `"parser"` field (the single source of
    truth `_worker`/`run_direct` both write), never by filename — robust to any future filename
    convention change. Unreadable/corrupt files are skipped, matching `aggregate_results`."""
    by_parser: dict[str, dict[str, dict]] = defaultdict(dict)
    if not layout.cache_dir.exists():
        return {}
    for f in sorted(layout.cache_dir.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        pk, qid = rec.get("parser"), rec.get("qid")
        if not pk or not qid:
            continue
        by_parser[pk][qid] = rec
    return dict(by_parser)


def _resolve_entry(parser_key: str, entries: list[RegistryEntry]) -> RegistryEntry | None:
    """vlm__<id>__<hash> by entry-id segment; openai-compat transcribers by
    safe_name(entry.id) + "__" prefix; upstream-parser entries by e.upstream_parser EXACTLY
    (safe_name(e.id) is the WRONG string for these — see module docstring / task brief).
    Returns None for anything that doesn't resolve — callers must not assume a resolved entry."""
    if parser_key.startswith("vlm__"):
        rest = parser_key[len("vlm__"):]
        if "__" not in rest:
            return None
        entry_id = rest.rsplit("__", 1)[0]
        for e in entries:
            if e.shape == "vlm-chat" and e.id == entry_id:
                return e
        return None
    for e in entries:
        if e.shape == "transcriber" and e.transport == "openai-compat" \
                and parser_key.startswith(safe_name(e.id) + "__"):
            return e
    for e in entries:
        if e.transport == "upstream-parser" and e.upstream_parser == parser_key:
            return e
    return None


def _input_label(entry: RegistryEntry | None) -> str:
    """What Section B's `input` column says the provider actually received.

    An explicit `input_mode` on the entry wins. Otherwise fall back to the historical transport
    proxy (openai-compat -> raster-png, else pdf-direct), which is only ever a proxy: transport
    says how we talk to a provider, not what we hand it. An upstream-parser adapter that
    rasterizes (docstrange@nanonets) is raster-png despite the transport, and calling it
    pdf-direct would attach the embedded-text-layer free-ride caveat to a row that cannot
    free-ride on one."""
    if entry is None:
        return "unknown (unregistered)"
    if entry.input_mode is not None:
        return entry.input_mode
    return "raster-png" if entry.transport == "openai-compat" else "pdf-direct"


def _row_label(pk: str, entry: RegistryEntry | None) -> str:
    return entry.id if entry is not None else f"{pk} (unregistered)"


def _disambiguated_labels(groups: dict[str, _Group]) -> dict[str, str]:
    """F4 (live-validation finding): when more than one parser key in ONE groups dict (Section A's
    direct_groups, or Section B's section_b_groups) resolves to the SAME registry entry id —
    e.g. the same vlm-chat entry direct-answered under two different condition hashes, the
    live-reproduced case — plain `_row_label` (`entry.id`) collides across both rows and a reader
    can no longer tell them apart in the table, detail bullets, or appendix.

    Disambiguates by appending ` [cond <hash>]`, where `<hash>` is the parser key's own trailing
    condition-hash segment (`vlm__<id>__<hash>` / `<safe_name>__<hash>` both end in exactly the
    12-hex-char `condition_hash` output) — only for entries whose id appears more than once in
    THIS dict. Unregistered rows (`entry is None`) already render a unique `{pk} (unregistered)`
    label per key and never need disambiguation."""
    id_counts: dict[str, int] = defaultdict(int)
    for g in groups.values():
        if g.entry is not None:
            id_counts[g.entry.id] += 1
    labels: dict[str, str] = {}
    for pk, g in groups.items():
        label = _row_label(pk, g.entry)
        if g.entry is not None and id_counts[g.entry.id] > 1:
            label = f"{label} [cond {pk.rsplit('__', 1)[-1]}]"
        labels[pk] = label
    return labels


def _stamp_columns(entry: RegistryEntry | None) -> str:
    """precision/licence/ToS/contaminated/contract in one string. M2: a row whose parser key
    never resolved to a registry entry renders "unknown (unregistered)" across all five — never
    blank, never silently omitted.

    F5: `contract` is `promptable`/`not-promptable` from `entry.promptable` — a pdf-direct
    upload endpoint (e.g. `mistral-ocr@mistral`) accepts no prompt at all, which is a materially
    different contract from a chat-style endpoint and deserves its own stamp rather than being
    buried in free-text `tos_note`."""
    if entry is None:
        return ("precision: unknown (unregistered) · licence: unknown (unregistered) "
                "· ToS: unknown (unregistered) · contaminated: unknown (unregistered) "
                "· contract: unknown (unregistered)")
    contract = "promptable" if entry.promptable else "not-promptable"
    return (f"precision: {_precision_label(entry)} · licence: {entry.weights_licence} "
           f"· {_tos_stamp(entry)} · contaminated: {'yes' if entry.contaminated else 'no'} "
           f"· contract: {contract}")


@dataclass
class _Group:
    parser_key: str
    entry: RegistryEntry | None
    rows_by_qid: dict[str, dict]


def _categorize(by_parser: dict[str, dict[str, dict]],
                 entries: list[RegistryEntry]) -> tuple[dict[str, _Group], dict[str, _Group], dict[str, _Group]]:
    """Splits every parser key into: direct (vlm__ prefix, real image), no_image (vlm__ prefix,
    the no-image control condition — rendered in the Baselines block, not as a ranked Section A
    row), and section_b (every non-vlm__ key: transcriber/upstream-parser/unregistered-non-vlm).

    I1 ruling: routing is keyed on the `vlm__` PREFIX itself, not on whether `_resolve_entry`
    found a registry match. An unresolved `vlm__` key is still structurally a direct-QA row (it
    carries `image_sha`/`resolved_provider` the same as any registered one) and must get the same
    D3/D4 guards and render in Section A marked `(unregistered)` — routing it into Section B would
    both skip those guards (fail-open) and mislabel it as a two-stage row it never was.

    A parser key's condition is baked into the key itself (that's WHY the no-image variant gets a
    different key), so checking one row's condition is representative of the whole group."""
    direct: dict[str, _Group] = {}
    no_image: dict[str, _Group] = {}
    section_b: dict[str, _Group] = {}
    for pk, rows_by_qid in by_parser.items():
        entry = _resolve_entry(pk, entries)
        if pk.startswith("vlm__"):
            sample = next(iter(rows_by_qid.values()))
            is_no_image = bool((sample.get("condition") or {}).get("no_image"))
            (no_image if is_no_image else direct)[pk] = _Group(pk, entry, rows_by_qid)
        else:
            section_b[pk] = _Group(pk, entry, rows_by_qid)
    return direct, no_image, section_b


# ── Per-row metric computation (shared by Section A and Section B — the metric definitions
#    don't change with shape, only the extra columns rendered alongside them do) ────────────

def _n_docs(outcomes: list[FieldOutcome]) -> int:
    return len({o.doc for o in outcomes})


def _error_class_of(row: dict | None) -> str:
    if row is None:
        return "missing"                 # no cache row at all for this qid under this parser
    if "error_class" in row:
        return row["error_class"]        # vlm__ rows always carry this (direct.py's `_one`)
    if "error" in row:
        return "error"                   # score.py rows: plain "error" string, no error_class
    if "answer" in row:
        return "none"                    # success
    return "unknown"                     # malformed row — neither answer nor error present


def _error_class_counts(rows_by_qid: dict[str, dict], items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for it in items:
        counts[_error_class_of(rows_by_qid.get(it["question_id"]))] += 1
    return dict(counts)


def _general_and_strict(rows_by_qid: dict[str, dict], items: list[dict],
                        all_fields: list[tuple]) -> tuple[float, float]:
    """general per-field: acc_over_all across EVERY gold field of EVERY bank item (not just the
    checkbox/null buckets), via `field_outcomes` so D7/MINOR-1 null-gold and error-precedence
    semantics stay identical to every other metric in this report. strict per-question: fraction
    of ALL bank items whose row is fully correct (`match is True`) — denominator is every item,
    not just answered ones (acc-over-all style: a missing/errored row counts as wrong, not
    excluded)."""
    outcomes = field_outcomes(rows_by_qid, all_fields)
    n = len(outcomes)
    correct = sum(1 for o in outcomes if o.status == "correct")
    general = correct / n if n else 0.0
    n_items = len(items)
    q_correct = sum(1 for it in items
                    if (rows_by_qid.get(it["question_id"]) or {}).get("match") is True)
    strict = q_correct / n_items if n_items else 0.0
    return general, strict


def _upstream_construction_metrics(rows_by_qid: dict[str, dict]) -> tuple[float, float, int]:
    """F2 (MERGE GATE): field%/question% using UPSTREAM's OWN scoring semantics — read straight
    from each row's stored `field_matches`/`match` (upstream's `score_typed`/`deep_equal`
    output, written at score time), with NO D7 re-scoring layered on top and NO acc-over-all-style
    denominator expansion. Restricted to "ok" rows only (`_error_class_of(row) == "none"`) — an
    error/missing row carries no `field_matches` to read.

    THIS is the number the reproduction gate (`docs/superpowers/specs/table3-snapshot.md`) must
    compare against the paper/README targets — never `_general_and_strict`'s D7-corrected
    `general/field`/`strict/question` columns in the main Section B table, which is the RANKING
    KEY for this report and diverges from upstream specifically on null-gold fields: upstream's
    `deep_equal(None, None)` already scores a key-absent answer as correct (baked into
    `field_matches` at write time — nothing to recompute here), while `field_outcomes` (D7)
    overrides exactly that case to "incorrect" for the ranking key. On the full RealDoc-Bench bank
    that's 188/3742 = 5.02% of all fields — comparing the WRONG columns against upstream's
    published numbers can look like up to a ~5pp systematic gap that has nothing to do with the
    model or harness under test."""
    total_fields = correct_fields = 0
    n_ok = question_correct = 0
    for row in rows_by_qid.values():
        if _error_class_of(row) != "none":
            continue
        fm = row.get("field_matches") or {}
        total_fields += len(fm)
        correct_fields += sum(1 for v in fm.values() if v)
        n_ok += 1
        if row.get("match") is True:
            question_correct += 1
    field_pct = correct_fields / total_fields if total_fields else 0.0
    question_pct = question_correct / n_ok if n_ok else 0.0
    return field_pct, question_pct, n_ok


def _beats_majority(model_outcomes: list[FieldOutcome], majority_outcomes: list[FieldOutcome],
                    *, iters: int, seed: int, alpha: float) -> str:
    """yes / no / not separable — and, below the cluster floor, "insufficient clusters (n_docs=N)"
    instead of calling separable() at all. A model is NEVER labelled above-baseline when the
    delta isn't separable from zero."""
    docs = len({o.doc for o in model_outcomes} | {o.doc for o in majority_outcomes})
    if docs < MIN_CLUSTER_DOCS:
        return f"insufficient clusters (n_docs={docs})"
    ci = paired_delta_ci(model_outcomes, majority_outcomes, iters=iters, seed=seed, alpha=alpha)
    if not separable(ci):
        return "not separable"
    return "yes" if ci[0] > 0 else "no"


def _pairwise_separability(label_outcomes: list[tuple[str, list[FieldOutcome]]],
                           *, iters: int, seed: int, alpha: float) -> list[str]:
    lines = []
    for (la, oa), (lb, ob) in combinations(label_outcomes, 2):
        docs = len({o.doc for o in oa} | {o.doc for o in ob})
        if docs < MIN_CLUSTER_DOCS:
            lines.append(f"- {la} vs {lb}: insufficient clusters (n_docs={docs})")
            continue
        ci = paired_delta_ci(oa, ob, iters=iters, seed=seed, alpha=alpha)
        if separable(ci):
            better = la if ci[0] > 0 else lb
            lines.append(f"- {la} vs {lb}: separable — {better} ahead, "
                        f"Δ=[{ci[0]*100:.1f}pp, {ci[1]*100:.1f}pp]")
        else:
            lines.append(f"- {la} vs {lb}: not separable — Δ CI [{ci[0]*100:.1f}pp, "
                        f"{ci[1]*100:.1f}pp] spans 0")
    return lines


@dataclass
class RowMetrics:
    n_expected: int
    n_seen: int
    checkbox_outcomes: list[FieldOutcome]
    null_outcomes: list[FieldOutcome]
    checkbox: dict
    null: dict
    checkbox_ci: tuple[float, float] | None
    checkbox_docs: int
    null_ci: tuple[float, float] | None
    null_docs: int
    general_field_acc: float
    strict_question_acc: float
    error_classes: dict[str, int]
    beats_majority: str


def _compute_row_metrics(rows_by_qid: dict[str, dict], items: list[dict], all_fields: list[tuple],
                         checkbox_fields: list[tuple], null_fields_all: list[tuple],
                         majority_outcomes: list[FieldOutcome], bank_qids: set[str], *,
                         iters: int, seed: int, alpha: float) -> RowMetrics:
    checkbox_outcomes = field_outcomes(rows_by_qid, checkbox_fields)
    null_outcomes = field_outcomes(rows_by_qid, null_fields_all)
    cb = checkbox_metrics(checkbox_outcomes)
    nm = null_metrics(null_outcomes)
    general, strict = _general_and_strict(rows_by_qid, items, all_fields)
    cb_docs = _n_docs(checkbox_outcomes)
    cb_ci = cluster_bootstrap_ci(checkbox_outcomes, iters=iters, seed=seed, alpha=alpha) if cb_docs else None
    null_docs = _n_docs(null_outcomes)
    null_ci = cluster_bootstrap_ci(null_outcomes, iters=iters, seed=seed, alpha=alpha) if null_docs else None
    beats = _beats_majority(checkbox_outcomes, majority_outcomes, iters=iters, seed=seed, alpha=alpha)
    # I3: n_seen is rows-that-match-a-CURRENT-bank-qid, never the raw row count. A parser dir can
    # accumulate "orphan" rows (a qid from a previous bank revision, a stale/renamed question)
    # that inflate len(rows_by_qid) without representing real current-bank coverage — counting
    # them would silently suppress the INCOMPLETE marker for a parser that is, in fact, missing
    # real bank items.
    n_seen = len(set(rows_by_qid) & bank_qids)
    return RowMetrics(
        n_expected=len(items), n_seen=n_seen,
        checkbox_outcomes=checkbox_outcomes, null_outcomes=null_outcomes,
        checkbox=cb, null=nm, checkbox_ci=cb_ci, checkbox_docs=cb_docs,
        null_ci=null_ci, null_docs=null_docs,
        general_field_acc=general, strict_question_acc=strict,
        error_classes=_error_class_counts(rows_by_qid, items), beats_majority=beats,
    )


# ── D3 STALE-RENDER + D4 serving-identity guards (vlm__ rows only) ─────────────────────────

def _check_serving_identity(parser_key: str, rows: list[dict]) -> None:
    """M7: a falsy/missing `resolved_provider` on an ANSWERED row is NOT dropped from the
    comparison — it counts as its own distinct value `"(empty)"`, so a mix of "resolved to
    ProviderA" and "nothing recorded" on rows that both actually got an answer is still flagged
    as a serving-identity inconsistency, not silently ignored.

    Round-2 regression fix: the comparison set is restricted to ANSWERED rows
    (`error_class == "none"`) BEFORE the "(empty)" fallback is applied. `direct.py`'s `_one` only
    ever sets `resolved_provider` on the `common` dict it builds AFTER a successful HTTP response
    — an `api_error`/`render_error`/etc. row (a transient timeout, a blank render, ...) carries NO
    such key at all and never reached any provider. Treating that absence as its own distinct
    "(empty)" identity — as the M7 fix originally did, unconditionally — meant one ordinary
    transient failure in an otherwise single-provider run would trip this unconditional,
    escape-hatch-free guard. A row that never got an answer has nothing to say about WHICH
    provider served it, so it must not enter the comparison at all."""
    answered = [r for r in rows if r.get("error_class") == "none"]
    providers = {(r.get("resolved_provider") or "(empty)") for r in answered}
    if len(providers) > 1:
        raise ServingIdentityError(
            f"D4 serving-identity violation: parser key {parser_key!r} has ANSWERED rows "
            f"resolved to >1 distinct provider ({sorted(providers)}) — the cache key does not "
            f"pin serving identity, so rows filed under one key must all share it. Split the run "
            f"or re-pin provider_pin before reporting.")


def _check_stale_renders(layout: RunLayout, rows: list[dict], render_cache: dict[tuple, str],
                         render_dir: Path) -> dict[str, str]:
    """Recomputes the CURRENT render hash per referenced doc (once per (stem, condition) pair,
    memoized in `render_cache` across the whole report) and compares it against each row's own
    stored `image_sha`.

    C1: `render_dir` MUST be a fresh, empty directory scoped to one `build_markdown_report` call
    — NEVER the run's own `docs_png` cache. `direct._render_page` is cache-first (it returns an
    existing PNG under that path without re-reading the PDF at all), so pointing this check at the
    warm production cache would just read back the very PNG that produced `image_sha` in the first
    place: the check would ALWAYS "pass" regardless of what the PDF on disk currently contains. An
    empty `render_dir` forces a genuine re-render from the current PDF bytes every time.

    C2: a doc that CANNOT be re-rendered (missing/corrupt PDF, page count changed, ...) while a
    row still references its `image_sha` is itself a failure — recorded as RENDER-UNAVAILABLE and
    returned in the same failing dict as a hash mismatch (STALE-RENDER), never silently treated as
    "presumably fine". Rows with no image at all (the no-image control) carry `image_sha: None`
    and are skipped — there is nothing to compare."""
    from ocr_eval_ext.direct import _render_page, condition_hash

    stale: dict[str, str] = {}
    for r in rows:
        sha = r.get("image_sha")
        if not sha:
            continue
        stem = r.get("source_file")
        condition = r.get("condition") or {}
        key = (stem, condition_hash(condition))
        if key not in render_cache:
            try:
                png = _render_page(layout, stem, condition, render_dir)
                render_cache[key] = hashlib.sha256(png).hexdigest()
            except Exception:
                render_cache[key] = _RENDER_FAILED
        current = render_cache[key]
        if current == _RENDER_FAILED:
            stale[stem] = "RENDER-UNAVAILABLE"
        elif current != sha:
            stale[stem] = "STALE-RENDER"
    return stale


# ── Cost / latency ──────────────────────────────────────────────────────────────────────────

def _direct_cost_latency(entry: RegistryEntry | None, rows: list[dict]) -> tuple[str, str]:
    """Section A: sourced straight from the vlm__ cache rows' own usage/latency_sec (the same
    fields `direct.py`'s `track()` reads).

    F7: `local: true` still always renders "n/a" for COST — never a fake $0.00 (rule carried from
    Task 6's max-spend review) — but LATENCY is real, locally-measured wall-clock time regardless
    of serving cost, so it now renders the same median-of-`latency_sec` every other row gets."""
    latencies = [r["latency_sec"] for r in rows if isinstance(r.get("latency_sec"), int | float)]
    lat = f"{statistics.median(latencies):.2f}s" if latencies else "n/a"
    if entry is not None and entry.local:
        return lat, "n/a"
    total_cost, any_cost = 0.0, False
    for r in rows:
        usage = r.get("usage") or {}
        cost = usage.get("cost")
        if cost is None and entry is not None and entry.pricing:
            cost = ((usage.get("prompt_tokens", 0) / 1e6) * entry.pricing["input_per_mtok"]
                    + (usage.get("completion_tokens", 0) / 1e6) * entry.pricing["output_per_mtok"])
        if cost is not None:
            total_cost += cost
            any_cost = True
    cost = f"${total_cost:.4f}" if any_cost else "n/a"
    return lat, cost


def _transcription_cost_latency(layout: RunLayout, entry: RegistryEntry | None,
                                parser_key: str) -> tuple[str, str]:
    """Section B: `eval/cache/` rows written by `score.py` carry NO token/latency fields at all
    (only `qid`/`parser`/`source_file`/`domain`/`answer`/`field_matches`/`match`/`error`) — the
    transcription leg's own cost/latency lives in `parses/<parser>/<stem>.json` (written by
    `run_parse`'s `_parse_one`), which is why these columns are scoped to "the transcription leg
    only" in the Section B header. `local: true` still renders "n/a" for both, unconditionally.

    M1: `_parse_one` writes `"cost_usd": result.cost_estimate_usd or 0.0` — an UNPRICED parser (no
    catalog rate for its pricing_key) coerces a genuine "unknown" (None) into a literal `0.0` that
    is indistinguishable on disk from "confirmed free". Heuristically detected here: every sidecar
    read back is exactly `0.0` AND the registry entry (if resolved) carries no `pricing` — render
    "n/a (unpriced)", never a fabricated "$0.0000" that would misrepresent an unknown cost as a
    confirmed one.

    F7: `local: true` still always renders "n/a" for COST, but LATENCY is measured wall-clock time
    from the same per-stem `latency_sec` sidecar field every other row reads — read it BEFORE the
    local short-circuit so a local transcriber's median latency renders instead of a blanket
    "n/a"."""
    parser_dir = layout.parser_dir(parser_key)
    latencies: list[float] = []
    costs: list[float] = []
    if parser_dir.exists():
        for meta_path in parser_dir.glob("*.json"):
            if meta_path.name == "condition.json":     # sidecar, not a per-stem parse record
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            if not meta.get("ok"):
                continue
            if isinstance(meta.get("latency_sec"), int | float):
                latencies.append(meta["latency_sec"])
            if isinstance(meta.get("cost_usd"), int | float):
                costs.append(meta["cost_usd"])
    lat = f"{statistics.median(latencies):.2f}s" if latencies else "n/a"
    if entry is not None and entry.local:
        return lat, "n/a"
    if not costs:
        cost = "n/a"
    elif sum(costs) == 0.0 and (entry is None or not entry.pricing):
        cost = "n/a (unpriced)"
    else:
        cost = f"${sum(costs):.4f}"
    return lat, cost


# ── Transcript-recall diagnostic (Section B only) ───────────────────────────────────────────

def _field_tokens(key: str) -> list[str]:
    """DISCLOSURE (round-2 review, no machinery change): `_TOKEN_RE` splits a snake_case key
    like `"signature_present"` into separate tokens (`"signature"`, `"present"`), but the
    word-boundary regex `\\b` this feeds into (see `_transcript_recall`) treats `_` as a WORD
    character (Python's `\\w` == `[a-zA-Z0-9_]`), not a boundary. So a transcript that emits the
    field's full snake_case identifier verbatim — e.g. literally the text "signature_present" —
    will satisfy NEITHER `\\bsignature\\b` NOR `\\bpresent\\b`, because neither sub-token has an
    actual word boundary against the underscore it's glued to. Net effect: `_TOKEN_RE`'s split
    on underscores buys nothing for THIS purpose — the boundary matcher effectively treats the
    whole snake_case string as a single indivisible word. This is a CONSERVATIVE UNDERCOUNT (a
    real semantic match can be missed), never an overcount — deliberately left as-is rather than
    "fixed" by also matching the joined identifier, since introducing that match rule changes the
    field's recall semantics rather than merely disclosing them."""
    return [t for t in _TOKEN_RE.findall(key) if len(t) >= 3]


def _transcript_recall(layout: RunLayout, parser_key: str, checkbox_items: list[dict]) -> float | None:
    """Fraction of checkbox-bucket docs whose `parses/<parser>/<stem>.md` contains BOTH a
    checkbox glyph AND a WORD-BOUNDARY-MATCHED token of one of that doc's boolean gold field keys.

    I2: matching must respect word boundaries — a bare substring check would let field key
    "checked" match inside "unchecked", counting a transcript that emitted the OPPOSITE state as
    if it had recalled the field.

    Snake_case caveat (see `_field_tokens`): a transcript that emits a compound identifier like
    "signature_present" verbatim will NOT be recalled via either of its split sub-tokens — `\\b`
    treats the underscore as a word character, not a boundary. Conservative undercount by design.

    M8: requires the parse dir to contain at least one `.md` file — a dir that exists but holds
    zero transcripts (e.g. only a `condition.json` sidecar, or a parser that was registered but
    never actually run) has no data to compute a fraction FROM, and must render "n/a" (None), not
    a misleadingly-confident `0.0`."""
    parser_dir = layout.parser_dir(parser_key)
    if not parser_dir.exists() or not any(parser_dir.glob("*.md")):
        return None
    doc_tokens: dict[str, set[str]] = defaultdict(set)
    for it in checkbox_items:
        doc = it["source_file"]
        for k, v in (it.get("gold_dict") or {}).items():
            if isinstance(v, bool):
                doc_tokens[doc].update(_field_tokens(k))
    if not doc_tokens:
        return None
    recalled = 0
    for doc, tokens in doc_tokens.items():
        md_path = layout.parse_md(parser_key, doc)
        if not md_path.exists():
            continue
        text = md_path.read_text(errors="ignore")
        if not _CHECKBOX_GLYPH_RE.search(text):
            continue
        text_lower = text.lower()
        if any(re.search(rf"\b{re.escape(tok.lower())}\b", text_lower) for tok in tokens):
            recalled += 1
    return recalled / len(doc_tokens)


# ── Formatting helpers ──────────────────────────────────────────────────────────────────────

def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_ci_metric(value: float, ci: tuple[float, float] | None, n_docs: int) -> str:
    if ci is None:
        return f"{_pct(value)} (n_docs={n_docs})"
    return f"{_pct(value)} [{_pct(ci[0])}, {_pct(ci[1])}] (n_docs={n_docs})"


def _fmt_hallucination(null_block: dict) -> str:
    """D9 ruling: never render hallucination_rate without n_answered and error_rate beside it."""
    ov = null_block["overall"]
    return (f"{_pct(null_block['hallucination_rate'])} "
            f"(n_answered={ov.n_answered}, error_rate={_pct(ov.error_rate)})")


def _tos_stamp(entry: RegistryEntry) -> str:
    """F5: an `ok`-commercial entry's `tos_note` is now rendered too (e.g. mistral-ocr@mistral's
    "zero-retention by contract only" — a real caveat that doesn't rise to blocked/conditional
    but is still worth surfacing, not just restated-and-buried in the registry comments). Every
    branch truncates to the first 60 chars so one long note can't blow out a table row."""
    note = f" — {entry.tos_note[:60]}" if entry.tos_note else ""
    if entry.provider_tos_commercial == "blocked":
        return f"⚠ ToS: blocked{note}"
    if entry.provider_tos_commercial == "conditional":
        return f"⚠ ToS: conditional{note}"
    return f"ToS: ok{note}"


def _precision_label(entry: RegistryEntry) -> str:
    return "unknown (not asserted)" if entry.precision == "provider-default" else entry.precision


def _mixed_precision_note(precision_labels: set[str]) -> str | None:
    """F6: `precision_labels` is a set of DISPLAY labels (`_precision_label` output), so
    "unknown (not asserted)" is its own distinct value here — mixing a known precision with an
    unasserted one is flagged as mixed too (a real ambiguity: the unasserted row's actual serving
    precision could match or differ from the known ones, and nothing here can tell which). The
    all-unasserted case is deliberately NOT folded into "mixed precision" — it renders a separate,
    less alarming note instead (there is no known precision difference to warn about, just an
    absence of any assertion at all)."""
    if len(precision_labels) > 1:
        return (f"**mixed precision** in this section: models are served at different pinned "
                f"precisions ({', '.join(sorted(precision_labels))}) — cross-model deltas may "
                f"reflect serving precision, not model quality alone.")
    if precision_labels == {"unknown (not asserted)"}:
        return "precision unasserted across all rows in this section"
    return None


def _section_mixed_precision_note(entries: list[RegistryEntry | None]) -> str | None:
    """I4: the SAME mixed-precision caveat Section A gets, extracted so Section B carries it too
    — a transcriber section mixing e.g. bf16 and fp8-vllm rows deserves the identical warning.

    F6: the set is built from `_precision_label(e)`, not the raw `entry.precision` field, so a
    provider-default entry contributes "unknown (not asserted)" as its own value instead of being
    silently dropped from the comparison."""
    labels = {_precision_label(e) for e in entries if e is not None}
    return _mixed_precision_note(labels)


def _error_classes_str(counts: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def _incomplete_head(label: str, n_seen: int, n_expected: int) -> str:
    if n_seen < n_expected:
        return f"**INCOMPLETE ({n_seen}/{n_expected})** {label}"
    return label


CROSS_SHAPE_WARNING = (
    "**Cross-shape comparability warning:** Section A (`vlm-chat`) rows answer the question "
    "directly from the rendered page image in one model call. Section B "
    "(transcribe-then-extract) rows first transcribe the page to markdown, then a SEPARATE "
    "`gemini-3-flash-preview` extractor call answers the question from that markdown. A Section "
    "B score reflects BOTH stages at once — a low score can come from a bad transcription OR a "
    "bad extraction — so a Section A and a Section B number for the \"same\" underlying model "
    "are not directly comparable rankings. Use the direct-vs-two-stage calibration pair (below, "
    "when both rows exist) to gauge how much of a two-stage gap is pipeline overhead versus "
    "genuine transcription quality."
)

METRIC_DEFINITIONS = (
    "- **checkbox acc-over-all**: correct / all boolean-typed checkbox-bucket fields (errors "
    "count as wrong, never excluded from the denominator).\n"
    "- **blank/null acc-over-all**: correct-null / all null-gold fields bank-wide — the headline "
    "blank-field number (D9 ruling); a collapsed extractor (answer: null) and an inventive one "
    "both drive this DOWN, never up.\n"
    "- **hallucination_rate**: incorrect / n_answered on null-gold fields ONLY — the propensity "
    "to invent a value given that the extractor answered at all. Meaningless read alone; always "
    "shown with n_answered and error_rate.\n"
    "- **general per-field**: acc-over-all across every gold field of every bank item.\n"
    "- **strict per-question**: fraction of ALL bank items fully correct (row `match is True`).\n"
    "- **beats majority**: paired-bootstrap delta vs the majority-class predictor on checkbox "
    "acc-over-all — \"not separable\" (or \"insufficient clusters\" below n_docs=20) never "
    "counts as \"yes\".\n"
    "- **CI-below-floor caveat**: a cluster-bootstrap CI is rendered whenever n_docs≥1, but "
    "MIN_CLUSTER_DOCS=20 only gates PAIRED comparisons (beats-majority, the separability "
    "appendix) — a single-row CI shown beside a small n_docs is not equally trustworthy just "
    "because it renders; read n_docs before trusting the interval.\n"
    "- **Section B staleness is not assessed**: the D3 STALE-RENDER check only applies to "
    "vlm__ rows (the only rows carrying `image_sha`) — a swapped/stale PDF behind a transcriber "
    "or upstream-parser row is NOT detected by this report.\n"
    "- the blank/null column's n is BANK-WIDE (every null-gold field in the whole corpus, not "
    "just the blank-field-tagged bucket) — e.g. n=188 on the full RealDoc-Bench corpus.\n"
    "- **transcript-recall**: fraction of checkbox-bucket docs whose transcript contains both a "
    "checkbox glyph and a word-boundary match of a gold field key token. Snake_case field keys "
    "(e.g. `signature_present`) are split into separate tokens, but the boundary matcher treats "
    "underscore as a word character, not a boundary — a transcript that emits the compound "
    "identifier verbatim will NOT be recalled via either half. This is a conservative "
    "undercount, never an overcount."
)


# ── Main entry point ─────────────────────────────────────────────────────────────────────────

def build_markdown_report(layout: RunLayout, entries: list[RegistryEntry], *,
                          bank_path: Path | None = None, allow_stale_render: bool = False,
                          iters: int = 2000, seed: int = 0, alpha: float = 0.05) -> str:
    """Pure function: reads cache/bank/run_meta/parses off disk, returns the report body as a
    markdown string. Never calls a model. Raises `ServingIdentityError` (D4, unconditional) or
    `StaleRenderError` (D3, unless `allow_stale_render=True`) rather than silently rendering a
    misleading report."""
    bank_path = bank_path or layout.bank_path
    items = json.loads(bank_path.read_text())["items"]
    # Matches ocr_eval_ext.cli._run_meta_path(layout) exactly (layout.root / "run_meta.json") —
    # inlined rather than imported to keep this module import-cycle-free of cli.py, which lazily
    # imports report_md.py from inside its own `report` command body.
    meta_path = layout.root / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    bank_qids = {it["question_id"] for it in items}
    checkbox_items = items_with_tags(items, CHECKBOX_TAGS)
    blank_items = items_with_tags(items, BLANK_TAGS)
    checkbox_fields = boolean_fields(checkbox_items)
    null_fields_all = null_fields(items)          # whole-bank nulls — the D9 headline denominator
    all_fields = _all_fields(items)
    baselines = baseline_rows(checkbox_fields)
    majority_true = baselines["always_true"] >= baselines["always_false"]
    majority_outcomes = [
        FieldOutcome(qid, key, doc, gold, "correct" if gold is majority_true else "incorrect")
        for qid, key, gold, doc in checkbox_fields
    ]

    by_parser = _load_cache_rows(layout)
    direct_groups, no_image_groups, section_b_groups = _categorize(by_parser, entries)

    # ── vlm__ pass: D4 guard, D3 staleness, row metrics — shared by direct + no-image groups.
    # C1: render_dir is a FRESH temp dir for this call only — never the run's own docs_png cache
    # (see _check_stale_renders' docstring for why that would make the check tautological).
    render_cache: dict[tuple, str] = {}
    stale_report: dict[str, dict[str, str]] = {}
    staleness_warnings: list[str] = []
    vlm_data: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="ocr_eval_report_render_") as _render_dir_str:
        render_dir = Path(_render_dir_str)
        for pk, g in {**direct_groups, **no_image_groups}.items():
            rows = list(g.rows_by_qid.values())
            _check_serving_identity(pk, rows)                # D4 — raises immediately, no escape
            stale = _check_stale_renders(layout, rows, render_cache, render_dir)
            if stale:
                stale_report[pk] = stale
            metrics = _compute_row_metrics(g.rows_by_qid, items, all_fields, checkbox_fields,
                                           null_fields_all, majority_outcomes, bank_qids,
                                           iters=iters, seed=seed, alpha=alpha)
            lat, cost = _direct_cost_latency(g.entry, rows)
            providers = sorted({r["resolved_provider"] for r in rows if r.get("resolved_provider")})
            retrieved_ats = [r["retrieved_at"] for r in rows if r.get("retrieved_at")]
            span = None
            if len(retrieved_ats) >= 2:
                try:
                    parsed = [dt.datetime.fromisoformat(ts) for ts in retrieved_ats]
                    span = max(parsed) - min(parsed)
                except ValueError:
                    span = None
            if span is not None and span > STALENESS_SPAN:
                staleness_warnings.append(
                    f"- **{pk}**: retrieved_at spans {span.days} day(s) "
                    f"({min(parsed).date()} .. {max(parsed).date()}) — exceeds the 7-day "
                    f"freshness window.")
            vlm_data[pk] = {"latency": lat, "cost": cost, "providers": providers, "metrics": metrics}

    # ── Section B pass ──────────────────────────────────────────────────────────────────────
    section_b_data: dict[str, dict] = {}
    for pk, g in section_b_groups.items():
        metrics = _compute_row_metrics(g.rows_by_qid, items, all_fields, checkbox_fields,
                                       null_fields_all, majority_outcomes, bank_qids,
                                       iters=iters, seed=seed, alpha=alpha)
        lat, cost = _transcription_cost_latency(layout, g.entry, pk)
        recall = _transcript_recall(layout, pk, checkbox_items)
        input_label = _input_label(g.entry)
        same_family = g.entry is not None and g.entry.provenance.strip().lower() == EXTRACTOR_FAMILY_PROVENANCE
        section_b_data[pk] = {
            "metrics": metrics, "latency": lat, "cost": cost, "recall": recall,
            "input": input_label, "same_family": same_family,
        }

    extractor_id = meta.get("extractor_id", "not yet scored")

    # F4: precompute the disambiguated label for every row ONCE per section, so a live-
    # reproduced entry (same registry id, two different condition hashes) renders distinguishable
    # labels everywhere it appears — section tables, detail bullets, the appendix, and the
    # calibration-pair lines below — instead of colliding on the bare `entry.id`.
    direct_labels = _disambiguated_labels(direct_groups)
    section_b_labels = _disambiguated_labels(section_b_groups)

    lines: list[str] = []
    lines += _build_header(meta, items)
    lines += _build_section_a(direct_groups, vlm_data, stale_report, direct_labels)
    lines += _build_baselines(baselines, checkbox_fields, no_image_groups, vlm_data)
    lines += _build_cross_shape_note(checkbox_items, blank_items, direct_groups, section_b_groups,
                                     direct_labels, section_b_labels)
    lines += _build_section_b(section_b_groups, section_b_data, extractor_id, layout,
                              section_b_labels)
    lines += _build_reproduction_gate(section_b_groups, section_b_labels)
    lines += _build_separability_appendix(direct_groups, section_b_groups, vlm_data,
                                          section_b_data, direct_labels, section_b_labels,
                                          iters=iters, seed=seed, alpha=alpha)
    lines += _build_staleness_section(staleness_warnings)
    lines += _build_stale_render_section(stale_report)

    md = "\n".join(lines) + "\n"
    if stale_report and not allow_stale_render:
        affected = sorted(stale_report)
        raise StaleRenderError(
            f"STALE-RENDER: {len(stale_report)} parser key(s) have rows that are stale or whose "
            f"current render could not be confirmed (RENDER-UNAVAILABLE) — {affected}. "
            f"Re-render/re-score, or pass --allow-stale-render to proceed anyway.", report_md=md)
    return md


# ── Section builders ─────────────────────────────────────────────────────────────────────────

def _build_header(meta: dict, items: list[dict]) -> list[str]:
    lines = [
        "# Stage 1 OCR Evaluation Report", "",
        f"Generated: {dt.datetime.now(dt.UTC).isoformat()}",
        f"Dataset revision: {meta.get('dataset_revision', 'unknown')} "
        f"(revision_source: {meta.get('revision_source', 'unknown')})",
        f"Harness commit: {meta.get('harness_commit', 'unknown')}",
        f"Extractor: {meta.get('extractor_id', 'not yet scored')}",
        f"Renders verified: {meta.get('renders_verified', False)}",
        # R-c: pymupdf renders the PDF pages this whole report's D3 STALE-RENDER check and every
        # image_sha are computed against — stamped by `verify`'s full sweep (I2b), never
        # re-derived here (this module never imports pymupdf directly).
        f"pymupdf version: {meta.get('pymupdf_version', 'unknown')}",
        f"Bank items in this report: {len(items)}",
        "",
        "**Dataset licence: CC-BY-4.0** — Extend-AI/RealDoc-Bench. Figures reproduced/derived "
        "from this dataset in this report are attributed to Extend-AI/RealDoc-Bench under "
        "CC-BY-4.0.",
        "",
    ]
    if meta.get("partial_corpus"):
        lines += ["> **⚠ PARTIAL CORPUS** — this run was scored with `--allow-partial-corpus`; "
                 "results below are over a PARTIAL corpus, not the full bank.", ""]
    lines += [
        "## Caveats",
        "",
        "- `origin=None` for every bank item: real vs synthetic (filled-in) documents cannot be "
        "distinguished from the released bank — provenance is mixed and unrecoverable per-item.",
        "- Composition is an estimated ~18% born-digital (gate2's 40-doc stratified sample: "
        "74.4% scanned / 17.9% born-digital) — per-class filtering is not possible from the "
        "released bank alone.",
        "- Provider-side image downscaling is UNCONTROLLED: hosted providers may resize or cap "
        "the rendered page before inference; per-provider caps are not independently verified "
        "per request.",
        "",
    ]
    return lines


def _build_section_a(direct_groups: dict[str, _Group], vlm_data: dict[str, dict],
                     stale_report: dict[str, dict[str, str]],
                     labels: dict[str, str]) -> list[str]:
    lines = ["## Section A — Direct QA (vlm-chat)", ""]
    if not direct_groups:
        return [*lines, "_no vlm-chat rows in this run_", ""]
    note = _section_mixed_precision_note([g.entry for g in direct_groups.values()])
    if note:
        lines += [note, ""]
    lines += ["| model | checkbox acc-over-all | polarity checked/unchecked | blank/null "
             "acc-over-all | hallucination_rate | general/field | strict/question | beats "
             "majority |",
             "|---|---|---|---|---|---|---|---|"]
    ordered = sorted(direct_groups, key=lambda k: labels[k])
    for pk in ordered:
        g = direct_groups[pk]
        m: RowMetrics = vlm_data[pk]["metrics"]
        head = _incomplete_head(labels[pk], m.n_seen, m.n_expected)
        pol = m.checkbox["polarity"]
        polarity = f"checked {_pct(pol['checked'].acc_over_all)} / unchecked {_pct(pol['unchecked'].acc_over_all)}"
        if pk in stale_report:
            head += " ⚠STALE-RENDER"
        lines.append(
            f"| {head} | {_fmt_ci_metric(m.checkbox['overall'].acc_over_all, m.checkbox_ci, m.checkbox_docs)} "
            f"| {polarity} "
            f"| {_fmt_ci_metric(m.null['overall'].acc_over_all, m.null_ci, m.null_docs)} "
            f"| {_fmt_hallucination(m.null)} "
            f"| {_pct(m.general_field_acc)} | {_pct(m.strict_question_acc)} "
            f"| {m.beats_majority} |")
    lines.append("")
    for pk in ordered:
        g = direct_groups[pk]
        m = vlm_data[pk]["metrics"]
        lines.append(
            f"- **{labels[pk]}** — {_stamp_columns(g.entry)} "
            f"· providers seen: {', '.join(vlm_data[pk]['providers']) or '(none recorded)'} "
            f"· median latency: {vlm_data[pk]['latency']} "
            f"· realized cost: {vlm_data[pk]['cost']} "
            f"· n: {m.n_seen}/{m.n_expected} "
            f"· error classes: {_error_classes_str(m.error_classes)}")
    lines.append("")
    return lines


def _build_baselines(baselines: dict, checkbox_fields: list[tuple],
                     no_image_groups: dict[str, _Group], vlm_data: dict[str, dict]) -> list[str]:
    cb = baselines["class_balance"]
    lines = [
        "## Baseline rows", "",
        f"- always-true: {_pct(baselines['always_true'])}",
        f"- always-false: {_pct(baselines['always_false'])}",
        f"- majority-class: {_pct(baselines['majority'])} "
        f"(class balance: true={cb['true']}, false={cb['false']}, n={len(checkbox_fields)})",
    ]
    for pk, g in no_image_groups.items():
        m: RowMetrics = vlm_data[pk]["metrics"]
        label = _row_label(pk, g.entry)
        lines.append(
            f"- no-image control — **{label}**: checkbox acc-over-all "
            f"{_fmt_ci_metric(m.checkbox['overall'].acc_over_all, m.checkbox_ci, m.checkbox_docs)} "
            f"(question text only, language-prior guessing)")
    lines.append("")
    return lines


def _find_calibration_pairs(direct_groups: dict[str, _Group],
                            section_b_groups: dict[str, _Group]) -> list[tuple[str, str]]:
    """The registered direct-vs-two-stage calibration cell: one model registered under BOTH
    shapes (matched by the underlying `model` string — the same weights served the same way,
    just prompted differently). Unregistered vlm__ groups (entry=None) have no `.model` to match
    against and are correctly excluded.

    F4: returns parser-KEY pairs `(v_pk, t_pk)`, not entry pairs — condition-hash disambiguation
    means the SAME registry entry can now legitimately back more than one group in a dict (two
    condition hashes for one entry), so matching/rendering must stay keyed on the parser key, not
    collapse back onto the shared entry object. Deduped on `(v_pk, t_pk)` — a defensive, explicit
    guard against ever rendering the exact same pair twice (dict keys are already unique, so this
    is a no-op in practice, but a cheap one worth keeping now that the matching loop is written by
    hand instead of a list comprehension over already-deduplicated entry lists)."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for v_pk, vg in direct_groups.items():
        v = vg.entry
        if v is None or not v.model:
            continue
        for t_pk, tg in section_b_groups.items():
            t = tg.entry
            if t is None or t.transport != "openai-compat" or not t.model:
                continue
            if v.model != t.model:
                continue
            key = (v_pk, t_pk)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


def _build_cross_shape_note(checkbox_items: list[dict], blank_items: list[dict],
                            direct_groups: dict[str, _Group],
                            section_b_groups: dict[str, _Group],
                            direct_labels: dict[str, str],
                            section_b_labels: dict[str, str]) -> list[str]:
    overlap = len({i["question_id"] for i in checkbox_items} & {i["question_id"] for i in blank_items})
    lines = [
        "## Cross-shape comparison note", "",
        CROSS_SHAPE_WARNING, "",
        f"{overlap} questions appear in both the checkbox and blank-field buckets "
        f"(of {len(checkbox_items)} checkbox-bucket and {len(blank_items)} blank-field-bucket "
        f"questions) — a field counted in both buckets contributes to both metrics "
        f"independently.",
        "",
        "**Metric definitions:**", "", METRIC_DEFINITIONS, "",
    ]
    pairs = _find_calibration_pairs(direct_groups, section_b_groups)
    for v_pk, t_pk in pairs:
        v = direct_groups[v_pk].entry
        lines.append(f"**Direct-vs-two-stage calibration pair detected:** "
                     f"`{direct_labels[v_pk]}` (Section A, direct) vs `{section_b_labels[t_pk]}` "
                     f"(Section B, transcribe-then-extract) — same underlying model "
                     f"(`{v.model}`); see the separability appendix for the paired delta between "
                     f"their rows.")
        lines.append("")
    return lines


def _build_section_b(section_b_groups: dict[str, _Group], section_b_data: dict[str, dict],
                     extractor_id: str, layout: RunLayout,
                     labels: dict[str, str]) -> list[str]:
    lines = ["## Section B — Transcribe-then-extract", "",
             f"_Cost/latency below describe the TRANSCRIPTION LEG only — the extractor "
             f"(`{extractor_id}`) call is a separate, shared cost not attributed per-row._", ""]
    if not section_b_groups:
        return [*lines, "_no transcriber rows in this run_", ""]
    note = _section_mixed_precision_note([g.entry for g in section_b_groups.values()])
    if note:
        lines += [note, ""]
    any_pdf_direct = any(section_b_data[pk]["input"] == "pdf-direct" for pk in section_b_groups)
    lines += ["| row | checkbox acc-over-all | blank/null acc-over-all | hallucination_rate | "
             "general/field | strict/question | transcript-recall | input | beats majority |",
             "|---|---|---|---|---|---|---|---|---|"]
    ordered = sorted(section_b_groups,
                     key=lambda pk: (section_b_groups[pk].entry is None, labels[pk]))
    for pk in ordered:
        g = section_b_groups[pk]
        d = section_b_data[pk]
        m: RowMetrics = d["metrics"]
        label = f"{labels[pk]} + extractor {extractor_id}"
        head = _incomplete_head(label, m.n_seen, m.n_expected)
        if d["same_family"]:
            head += " (same-family scoring)"
        recall = "n/a" if d["recall"] is None else _pct(d["recall"])
        lines.append(
            f"| {head} "
            f"| {_fmt_ci_metric(m.checkbox['overall'].acc_over_all, m.checkbox_ci, m.checkbox_docs)} "
            f"| {_fmt_ci_metric(m.null['overall'].acc_over_all, m.null_ci, m.null_docs)} "
            f"| {_fmt_hallucination(m.null)} "
            f"| {_pct(m.general_field_acc)} | {_pct(m.strict_question_acc)} "
            f"| {recall} | {d['input']} | {m.beats_majority} |")
    lines.append("")
    for pk in ordered:
        g = section_b_groups[pk]
        d = section_b_data[pk]
        m = d["metrics"]
        detail = [
            f"- **{labels[pk]}**",
            _stamp_columns(g.entry),
            "provider: n/a (not persisted)",     # D4/Task-8 note: upstream never persists
                                                  # ParseResult.raw
            f"median latency: {d['latency']} · realized cost: {d['cost']}",
            f"n: {m.n_seen}/{m.n_expected} · error classes: {_error_classes_str(m.error_classes)}",
        ]
        lines.append(" · ".join(detail))
    lines.append("")
    if any_pdf_direct:
        lines += ["_pdf-direct rows upload the PDF to the provider directly (no rendered PNG "
                 "leg); such rows can free-ride on an embedded text layer already in the PDF "
                 "(measured: 2 of 16 sampled docs carry one) — a pdf-direct score is not purely "
                 "an OCR/vision measurement._", ""]
    return lines


def _build_reproduction_gate(section_b_groups: dict[str, _Group],
                             labels: dict[str, str]) -> list[str]:
    """F2 (MERGE GATE): the reproduction gate (paper Table 3 / README targets,
    `docs/superpowers/specs/table3-snapshot.md`) must compare against THIS block, never the
    ranking-key `general/field`/`strict/question` columns in the Section B table above — those
    two constructions diverge on null-gold fields (D7 ruling; see `_upstream_construction_metrics`
    docstring for the exact size of the gap)."""
    lines = ["## Reproduction gate (upstream construction)", "",
             "_upstream construction — for the reproduction gate only; not the ranking key._ "
             "Recomputed straight from each row's stored `field_matches`/`match` (upstream's own "
             "`score_typed`/`deep_equal` semantics — a null-gold key-absent answer counts as "
             "upstream scored it), over ok rows only, with NO D7 re-scoring applied. Point the "
             "paper/README reproduction-target comparison at THIS table, never at the "
             "`general/field`/`strict/question` columns above.", ""]
    if not section_b_groups:
        return [*lines, "_no transcriber rows in this run_", ""]
    lines += ["| row | field% (upstream) | question% (upstream) | n (ok rows) |",
             "|---|---|---|---|"]
    ordered = sorted(section_b_groups,
                     key=lambda pk: (section_b_groups[pk].entry is None, labels[pk]))
    for pk in ordered:
        g = section_b_groups[pk]
        field_pct, question_pct, n_ok = _upstream_construction_metrics(g.rows_by_qid)
        lines.append(f"| {labels[pk]} | {_pct(field_pct)} | {_pct(question_pct)} | {n_ok} |")
    lines.append("")
    return lines


def _appendix_label(label: str, m: RowMetrics) -> str:
    """I5: any row group whose n_seen falls short of n_expected carries the SAME `[INCOMPLETE
    k/N]` marker in the separability appendix that its leaderboard row carries — a paired delta
    against a partial matrix must be visibly partial too, not just quietly discounted.

    F4: takes the already-disambiguated label (from `_disambiguated_labels`) rather than
    re-deriving it from `_row_label(pk, entry)` — so a live-reproduced entry's `[cond <hash>]`
    suffix carries into the appendix instead of colliding back down to the bare `entry.id`."""
    if m.n_seen < m.n_expected:
        return f"{label} [INCOMPLETE {m.n_seen}/{m.n_expected}]"
    return label


def _build_separability_appendix(direct_groups: dict[str, _Group], section_b_groups: dict[str, _Group],
                                 vlm_data: dict[str, dict], section_b_data: dict[str, dict],
                                 direct_labels: dict[str, str], section_b_labels: dict[str, str],
                                 *, iters: int, seed: int, alpha: float) -> list[str]:
    lines = ["## Separability appendix", "",
             "Pairwise paired-bootstrap deltas on checkbox acc-over-all, WITHIN each section "
             "only (Section A vs Section B pipelines are not directly comparable — see the "
             "cross-shape note above). Sign convention: Δ = first label minus second label (a "
             "positive Δ favors the FIRST name listed in each comparison).", ""]
    section_a_items = [
        (_appendix_label(direct_labels[pk], vlm_data[pk]["metrics"]),
         vlm_data[pk]["metrics"].checkbox_outcomes)
        for pk in direct_groups
    ]
    lines.append("### Section A")
    lines += _pairwise_separability(section_a_items, iters=iters, seed=seed, alpha=alpha) or ["_fewer than 2 rows_"]
    lines.append("")
    section_b_items = [
        (_appendix_label(section_b_labels[pk], section_b_data[pk]["metrics"]),
         section_b_data[pk]["metrics"].checkbox_outcomes)
        for pk in section_b_groups
    ]
    lines.append("### Section B")
    lines += _pairwise_separability(section_b_items, iters=iters, seed=seed, alpha=alpha) or ["_fewer than 2 rows_"]
    lines.append("")
    return lines


def _build_staleness_section(warnings: list[str]) -> list[str]:
    if not warnings:
        return []
    return ["## Staleness warnings (retrieved_at span > 7 days)", "", *warnings, ""]


def _build_stale_render_section(stale_report: dict[str, dict[str, str]]) -> list[str]:
    if not stale_report:
        return []
    lines = ["## STALE-RENDER", "",
             "The following parser keys have rows that are STALE-RENDER (stored `image_sha` no "
             "longer matches a freshly recomputed render) or RENDER-UNAVAILABLE (the referenced "
             "document could not be re-rendered at all — treated as a failure, not a pass, since "
             "the row's basis can no longer be confirmed).",
             "",
             "This check is intentionally sensitive to MORE than a swapped PDF: a different "
             "pymupdf version, OS/font stack, or machine than the one that originally scored "
             "these rows can also change the rendered pixels enough to change the hash. A "
             "STALE-RENDER here does not automatically mean the underlying PDF changed — it "
             "means THIS environment cannot currently confirm the row still matches what it "
             "claims to; re-render on the original scoring machine/pymupdf version before "
             "assuming the worse explanation.", ""]
    for pk, docs in sorted(stale_report.items()):
        entries_str = [f"{stem} ({reason})" for stem, reason in sorted(docs.items())]
        shown = entries_str[:10]
        suffix = f" (+{len(docs) - 10} more)" if len(docs) > 10 else ""
        lines.append(f"- **{pk}**: {len(docs)} doc(s) — {shown}{suffix}")
    lines.append("")
    return lines
