"""Score stage — one Gemini call per (question, parser); typed deep_equal.

For every (item, parser) pair: load `<run-dir>/parses/<parser>/<stem>.md`,
send Gemini the question + the item's typed template + the markdown, parse the
typed JSON answer back, score per-field with `deep_equal` (with `fuzzy_equal`
on plain `<string>` fields), write
`<run-dir>/eval/cache/<qid>__<parser>.json`.

Cache key: `(qid, parser)`. On hit, the cached answer is **re-scored** with
the current `score_typed` and the cache file is rewritten if the verdict
changed — so editing scoring code + re-running `score` updates verdicts for
free (no model calls). `--force` re-calls Gemini.

When `parsers` is empty, defaults to every parser that has parses under
`<run-dir>/parses/` (the natural chain after `parse`).
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from rapidfuzz import fuzz

from realdoc_bench.evaluate.jsonify import (
    _placeholder_to_type_hint,
    render_template,
    widen_types_from_gold,
)
from realdoc_bench.evaluate.runs import RunLayout


# ── Constants — tunable scoring knobs ──────────────────────────────────────
DEFAULT_MODEL = "gemini-3.1-flash-lite"
# FORK DIVERGENCE (D11, 2026-08-04) — upstream and this project's own spec both pin
# `gemini-3-flash-preview` here, ratified "exactly as upstream — required for published-number
# comparability" (specs/2026-08-01-ocr-eval-pipeline-design.md:79). Overridden by user decision
# on two grounds:
#   1. DURABILITY. The instrument must outlive the project. `gemini-3-flash-preview` is a PREVIEW
#      endpoint; preview/older models get retired, demonstrated the same day when
#      `gemini-2.0-flash-lite` returned HTTP 404 "no longer available". An extractor that
#      disappears mid-project destroys reproducibility far more completely than a judge swap.
#      `gemini-3.1-flash-lite` is GA.
#   2. EQUIVALENCE IS MEASURED, NOT ASSUMED. Paired A/B over 300 real bank items on real
#      DocStrange transcripts, identical items for every model (McNemar on discordant pairs):
#        gemini-3.1-flash-lite   244/300 (81.3%)  +14/-11 vs incumbent, p=0.690
#        gemini-3-flash-preview  241/300 (80.3%)  — incumbent
#        gemini-3.5-flash-lite   240/300 (80.0%)  +13/-14, p=1.000
#        gemini-2.5-flash        235/300 (78.3%)  +12/-18, p=0.362
#      Nothing separates them. Cost falls $1.83 -> $0.91 per full-corpus transcriber row and
#      per-call latency drops roughly an order of magnitude.
# NOTE the 5-fixture `selftest --extractor` gate does NOT discriminate between these models
# (all score 5/5) — it is a floor, not evidence of equivalence. The n=300 paired run is.
# COST OF THE CHANGE: DoD #2 compares our absolute numbers against upstream's published Table 3,
# which upstream produced with `gemini-3-flash-preview`. A repro gap now carries one extra
# uncontrolled variable. Re-pin the constant above to reproduce upstream exactly.
FUZZ_THRESHOLD = 92      # rapidfuzz ratio above which strings count as equal
FUZZ_MIN_WORDS = 5       # min word count on EITHER side for fuzzy to apply

# ── Extractor system prompt — the canonical typed-extraction prompt ────────
SYSTEM_PROMPT = (
    "You are extracting structured answers from document-parser markdown. "
    "Read the markdown only -- never use outside knowledge. "
    "Return ONE JSON object whose keys and value types exactly match the answer "
    "template provided. The TYPE HINT inside `< >` is the source of truth for "
    "each field; the KEY NAME does not change typing or formatting. Rules: "
    "(1) numbers must be JSON numbers (no commas, no '$', no '%', no quotes); "
    "(2) only fields whose template says `<boolean>` may be JSON true / false; "
    "(3) fields whose template says `<one of: a | b | ...>` MUST return one of "
    "the listed tokens as a JSON string, copied verbatim including case; "
    "(4) ONLY fields whose type hint is `<date string YYYY-MM-DD>` get "
    "normalized to ISO YYYY-MM-DD; "
    "(5) all other fields -- including any `<string>` -- copy the page text "
    "verbatim, trimmed (do not normalize dates, numbers, capitalization or "
    "separators), EXCEPT collapse spurious OCR character-spacing: if a value "
    "is printed with stray spaces between its characters (e.g. '0 . 0 0', "
    "'1 2 3 4', 'C A T'), return its natural un-spaced form; "
    "(6) for any field whose value is absent, blank, illegible, or genuinely "
    "ambiguous in the markdown, use JSON null; "
    "(6a) treat redaction markers like `[Redacted]` as ABSENT -- return JSON "
    "null if a cell holds only a marker; drop the marker if mixed with real "
    "content. "
    "Do not include keys outside the template. Do not wrap in an 'answer' "
    "parent key beyond what the template shows. Output the JSON object only, "
    "no prose or fences."
)


# ── String normalization for deep_equal ────────────────────────────────────
_WS_AROUND_SEP = re.compile(r"\s*([/\-,:])\s*")
_WS_RUN = re.compile(r"\s+")
_DASH_RE = re.compile(r"[‐‑‒–—―−]")
_SMART_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"'})
_TAG_RE = re.compile(r"<br\s*/?>|</?[a-z][^>]*>", re.I)
_STYLE_RE = re.compile(r"\*\*|__|`")


def _norm_str(s: str) -> str:
    s = s.translate(_SMART_QUOTES)
    s = _DASH_RE.sub("-", s)
    s = _TAG_RE.sub(" ", s)
    s = _STYLE_RE.sub("", s)
    s = _WS_AROUND_SEP.sub(r"\1", s)
    return _WS_RUN.sub(" ", s).strip()


def deep_equal(a: Any, b: Any) -> bool:
    """Type-tolerant equality used as the base comparison for every key.

    Tolerates: parser-style 1-element list wrapping a dict, smart-quote /
    dash variants, parser markdown styling (`<br>`, `**`, ``), OCR
    character-spacing on numeric strings, and number-vs-numeric-string with
    currency / percent / comma formatting (`"$1,234" == 1234`).
    """
    if isinstance(a, list) and len(a) == 1 and isinstance(b, dict) and isinstance(a[0], dict):
        return deep_equal(a[0], b)
    if isinstance(b, list) and len(b) == 1 and isinstance(a, dict) and isinstance(b[0], dict):
        return deep_equal(a, b[0])
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, str) and isinstance(b, str):
        na, nb = _norm_str(a), _norm_str(b)
        if na == nb:
            return True
        # OCR character-spacing — equal once ALL whitespace removed.
        return re.sub(r"\s+", "", na) == re.sub(r"\s+", "", nb)
    if isinstance(a, (int, float)) and isinstance(b, str):
        num, s = a, b
    elif isinstance(b, (int, float)) and isinstance(a, str):
        num, s = b, a
    else:
        return a == b
    try:
        return float(re.sub(r"[,$%\s]", "", s)) == float(num)
    except ValueError:
        return False


def fuzzy_equal(a: Any, b: Any) -> bool:
    """Conservative rapidfuzz comparison — ONLY for plain <string> fields.
    Falls back to deep_equal first; then punctuation-insensitive equality;
    then rapidfuzz ratio above FUZZ_THRESHOLD, but only when both sides are
    multi-word (short strings are IDs/amounts where 90+ ratio would forgive
    a single wrong digit)."""
    if deep_equal(a, b):
        return True
    if not (isinstance(a, str) and isinstance(b, str)):
        return False
    ca = re.sub(r"[^0-9a-z]+", " ", a.lower()).strip()
    cb = re.sub(r"[^0-9a-z]+", " ", b.lower()).strip()
    if ca and ca == cb:
        return True
    if min(len(ca.split()), len(cb.split())) < FUZZ_MIN_WORDS:
        return False
    return bool(ca) and fuzz.ratio(ca, cb) >= FUZZ_THRESHOLD


_TYPE_RE = re.compile(r'"([^"]+)":\s*<([^>]+)>')


def string_keys(template: str) -> set[str]:
    """Keys whose template type is plain <string> — the fuzzy-eligible ones.
    Enums, numbers, booleans and dates stay exact."""
    return {k for k, h in _TYPE_RE.findall(template)
            if h.replace("| null", "").strip() == "string"}


def score_typed(answer: Any, gold_dict: dict,
                str_keys: set[str]) -> tuple[dict, bool]:
    """Per-field score: fuzzy_equal for <string> fields, deep_equal otherwise."""
    fm = {}
    for k, gv in gold_dict.items():
        av = answer.get(k) if isinstance(answer, dict) else None
        fm[k] = fuzzy_equal(av, gv) if k in str_keys else deep_equal(av, gv)
    return fm, (bool(fm) and all(fm.values()))


# ── Template building ──────────────────────────────────────────────────────
_RF_PREFIX = re.compile(r"^\s*Return exactly[^:]*:\s*")
_PAIR_RE = re.compile(r"([^=<>;|]+)=<([^>]+)>")


def build_template(rf: str, gold_dict: dict) -> str:
    """Turn a freeform `response_format` into a typed JSON template.

    Handles three dialects observed in the bank:
      'Return exactly: k=<tok>; k2=<tok>'            semicolon-separated
      'Return exactly (... \" || \"): k=<tok> || ...'   pipe-pair-separated
      'Return ... separated by semicolon-space'      list-style; gold is list

    If neither pattern matches, derives the shape from the gold (list → list of
    strings, scalar → string).
    """
    body = _RF_PREFIX.sub("", rf or "")
    pairs = [(k.strip(), tok) for k, tok in _PAIR_RE.findall(body)]
    if not pairs:
        kt = {k: ("list of strings" if isinstance(v, list) else "string")
              for k, v in (gold_dict or {}).items()}
        return render_template(kt or {"answer": "string"})
    kt = {k: _placeholder_to_type_hint(tok) for k, tok in pairs}
    try:
        widen_types_from_gold(kt, gold_dict or {})
    except Exception:  # noqa: BLE001 — keep going on a bad widen
        pass
    return render_template(kt)


# ── Gemini ─────────────────────────────────────────────────────────────────
def require_api_key() -> str:
    k = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not k:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set")
    return k


# Retry policy for the scoring leg (fork addition — see `gemini_extract`'s docstring). Kept
# deliberately small and local rather than importing `ocr_eval_ext.direct`: upstream modules must
# not depend on the fork's package (the dependency runs one way only, see architecture.md's fork
# boundaries), and httpx exceptions need different handling from the openai client's anyway.
_EXTRACT_MAX_ATTEMPTS = 4
_EXTRACT_BACKOFF_BASE_SEC = 2.0
_EXTRACT_MAX_WAIT_SEC = 60.0


def _extract_retry_wait(resp: httpx.Response | None, attempt: int) -> float:
    """`Retry-After` (numeric seconds only — Gemini sends seconds, and an HTTP-date here would be
    honoured as a fallback rather than misparsed) clamped to [0, 60]. A hostile or buggy header can
    never cost more than a minute of wall-clock per attempt."""
    fallback = _EXTRACT_BACKOFF_BASE_SEC * (2 ** attempt)
    wait = fallback
    if resp is not None:
        raw = resp.headers.get("retry-after")
        if raw is not None:
            try:
                wait = float(raw)
            except (TypeError, ValueError):
                wait = fallback
    return max(0.0, min(_EXTRACT_MAX_WAIT_SEC, wait))


def _status_error_with_body(e: httpx.HTTPStatusError) -> httpx.HTTPStatusError:
    """Re-wrap an `HTTPStatusError` so its message includes the response body.

    Google puts the machine-readable reason in the body (`error.message`/`error.status`), while
    httpx's message has only the status line and URL. Without this, a revoked key, an unknown model
    id, and a malformed request are indistinguishable `400 Bad Request` tracebacks.

    Safe to surface: the key travels in a header, never the URL, and Google does not echo the key
    back in the error body. Truncated so a large HTML error page cannot flood a log.
    """
    try:
        detail = e.response.text[:800]
    except Exception:                      # response not readable (streamed/closed) — keep original
        return e
    if not detail:
        return e
    return httpx.HTTPStatusError(f"{e}\nresponse body: {detail}",
                                 request=e.request, response=e.response)


def _post_with_retry(c: httpx.Client, url: str, body: dict, headers: dict) -> dict:
    """POST with bounded retry on 408/429/5xx and connection-level failures.

    Raises the last error once the budget is exhausted — a genuinely unavailable extractor must
    still fail the run (the gate is fail-closed by design), just not on the first transient blip.
    `httpx.HTTPStatusError` messages are safe to propagate now that the key travels in a header
    rather than the URL.

    The raised message is ENRICHED WITH THE RESPONSE BODY (`_status_error_with_body`). httpx's own
    message carries only the status and URL, but Google returns the actual reason in the body — a
    revoked key, an unknown model id, and a malformed request are all bare `400 Bad Request`
    otherwise, which makes a real failure undiagnosable from the traceback alone. Encountered for
    real: a 400 on `:generateContent` whose cause was invisible until the body was read.
    """
    last_exc: Exception | None = None
    for attempt in range(_EXTRACT_MAX_ATTEMPTS):
        try:
            r = c.post(url, json=body, headers=headers)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            retryable = code in (408, 429) or 500 <= code < 600
            if not retryable or attempt == _EXTRACT_MAX_ATTEMPTS - 1:
                raise _status_error_with_body(e) from e
            last_exc = e
            time.sleep(_extract_retry_wait(e.response, attempt))
        except (httpx.TransportError, httpx.StreamError) as e:
            # No HTTP response obtained (DNS, refused connection, read timeout) — nothing has been
            # established about permanence, so this is retryable for the same reason a 5xx is.
            if attempt == _EXTRACT_MAX_ATTEMPTS - 1:
                raise
            last_exc = e
            time.sleep(_extract_retry_wait(None, attempt))
    raise last_exc or RuntimeError("unreachable")   # loop always returns or raises


def gemini_extract(question: str, template: str, markdown: str,
                   *, model: str = DEFAULT_MODEL,
                   client: httpx.Client | None = None) -> Any:
    """One Gemini call. Returns the parsed JSON answer or None on failure.

    FORK CHANGE (2026-08-04) — two upstream behaviours corrected here:

    1. **The API key is sent as an `x-goog-api-key` HEADER, never a URL query parameter.**
       Upstream passed `params={"key": ...}`, which puts the secret in the request URL — and httpx
       embeds the full URL in `HTTPStatusError`, so any non-2xx response prints the live key into
       the traceback (observed for real: a transient 503 leaked a working key into a session
       transcript). URLs also reach proxy and server access logs. Headers do neither.

    2. **Transient failures are retried.** Upstream called `raise_for_status()` once, so a single
       503 aborted the whole extractor gate (and, mid-run, `ocr-eval score`) with an
       `HTTPStatusError` that is neither a `PreconditionError` nor the `RuntimeError` the callers
       expect. 429/5xx/408 and connection-level failures now get bounded exponential backoff,
       honouring `Retry-After` when present. Permanent failures (400/401/403/404/...) still fail on
       the first attempt — retrying a bad key or a malformed request is pure waste. This mirrors
       the retry discipline `ocr_eval_ext/direct.py` already owns for the candidate legs; the
       scoring leg previously had none.
    """
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    prompt = (f"Question:\n{question}\n\n"
              f"Answer template (return JSON with exactly these keys and value "
              f"types):\n{template}\n\n"
              f"Parser markdown:\n```markdown\n{markdown}\n```")
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8192,
                             "responseMimeType": "application/json"},
    }
    headers = {"x-goog-api-key": require_api_key()}
    own_client = client is None
    c = client or httpx.Client(timeout=300)
    try:
        raw = _post_with_retry(c, url, body, headers)
    finally:
        if own_client:
            c.close()
    cands = raw.get("candidates") or []
    parts = ((cands[0].get("content") or {}).get("parts")) if cands else []
    text = "\n".join(p.get("text", "") for p in (parts or []) if p.get("text")).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if 0 <= s < e:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                return None
    return None


# ── Worker ─────────────────────────────────────────────────────────────────
@dataclass
class ScoreRecord:
    qid: str
    parser: str
    source_file: str
    domain: str
    cached: bool
    ok: bool                  # answer is present (no error)
    match: bool               # all fields correct
    field_matches: dict
    error: str = ""


def _ensure_template(item: dict) -> None:
    if "template" not in item:
        item["template"] = build_template(item.get("response_format", ""),
                                          item.get("gold_dict") or {})
    if "str_keys" not in item:
        item["str_keys"] = string_keys(item["template"])


def _worker(item: dict, parser: str, layout: RunLayout, *,
            force: bool) -> ScoreRecord:
    qid = item["question_id"]
    sf = item["source_file"]
    cache = layout.cache_path(qid, parser)
    base = ScoreRecord(qid, parser, sf, item.get("domain", ""),
                       False, False, False, {})

    if cache.exists() and not force:
        try:
            rec = json.loads(cache.read_text())
        except Exception:
            rec = None
        if rec is not None:
            if "answer" in rec:
                fm, allc = score_typed(rec["answer"], item["gold_dict"],
                                       item["str_keys"])
                changed = (fm != rec.get("field_matches")
                           or allc != rec.get("match"))
                if changed:
                    rec["field_matches"], rec["match"] = fm, allc
                    cache.write_text(json.dumps(rec, ensure_ascii=False))
                return ScoreRecord(qid, parser, sf, item.get("domain", ""),
                                   True, True, allc, fm, rec.get("error", ""))
            if "error" in rec and "no answer" not in rec["error"]:
                # cached terminal error (e.g. "markdown missing") — re-attempt
                # only when the markdown later appears, never automatically
                return ScoreRecord(qid, parser, sf, item.get("domain", ""),
                                   True, False, False, {}, rec["error"])

    md_path = layout.parse_md(parser, sf)
    if not md_path.exists():
        rec = {"qid": qid, "parser": parser, "source_file": sf,
               "domain": item.get("domain", ""),
               "error": "markdown missing"}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rec, ensure_ascii=False))
        base.error = "markdown missing"
        return base

    md = md_path.read_text()
    for attempt in range(3):
        try:
            ans = gemini_extract(item["question"], item["template"], md)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                rec = {"qid": qid, "parser": parser, "source_file": sf,
                       "domain": item.get("domain", ""),
                       "error": str(e)[:200]}
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(rec, ensure_ascii=False))
                base.error = str(e)[:200]
                return base
            time.sleep(2 + attempt * 3)
    fm, allc = score_typed(ans, item["gold_dict"], item["str_keys"])
    rec = {"qid": qid, "parser": parser, "source_file": sf,
           "domain": item.get("domain", ""), "answer": ans,
           "field_matches": fm, "match": allc}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rec, ensure_ascii=False))
    return ScoreRecord(qid, parser, sf, item.get("domain", ""),
                       False, True, allc, fm)


# ── Public entry ───────────────────────────────────────────────────────────
def load_bank(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    items = data["items"] if isinstance(data, dict) else data
    for it in items:
        if "gold_dict" not in it:
            raise ValueError(
                f"bank item {it.get('question_id', '?')} missing gold_dict — "
                f"run jsonify on the bank first")
        _ensure_template(it)
    return items


def docs_for_items(items: list[dict]) -> list[str]:
    """Ordered unique `source_file` stems referenced by these bank items.

    All `--limit N` flows ultimately resolve to "the source_files referenced
    by the first N bank items" — this helper is the single source of truth
    so download, parse, and score stay aligned for smoke runs.
    """
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        sf = it.get("source_file")
        if sf and sf not in seen:
            seen.add(sf)
            out.append(sf)
    return out


def run_score(layout: RunLayout, parsers: list[str] | None = None, *,
              bank_path: Path | None = None, force: bool = False,
              workers: int = 16, limit: int | None = None,
              progress: bool = True) -> list[ScoreRecord]:
    """Score every (item × parser) pair against the bank. When `parsers` is
    falsy, scores every parser found under `<run-dir>/parses/`.

    Tests stub the model call by monkeypatching :func:`gemini_extract`
    (and setting ``GEMINI_API_KEY`` in the test env to satisfy the preflight).
    """
    layout.ensure_dirs()
    require_api_key()  # fail fast on missing GEMINI_API_KEY before spinning up the worker pool
    bank_path = bank_path or layout.bank_path
    if not bank_path.exists():
        raise FileNotFoundError(f"no bank at {bank_path}")
    items = load_bank(bank_path)
    if limit:
        items = items[:limit]

    parsers = list(parsers or layout.parsed_parsers())
    if not parsers:
        raise ValueError(
            f"no parsers given and no parses under {layout.parses_root} — "
            f"run `evaluate parse` first or pass -p")

    jobs = [(it, p) for it in items for p in parsers]
    if progress:
        print(f"{len(items)} questions × {len(parsers)} parsers = "
              f"{len(jobs)} (question, parser) cells", flush=True)

    records: list[ScoreRecord] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, it, p, layout,
                          force=force) for it, p in jobs]
        n_cached = 0
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            records.append(rec)
            if rec.cached:
                n_cached += 1
            if progress and (i % 500 == 0 or i == len(jobs)):
                print(f"  [{i}/{len(jobs)}]  cached={n_cached}  "
                      f"({(time.time()-t0)/60:.1f}min)", flush=True)

    if progress:
        n_ok = sum(1 for r in records if r.ok)
        n_match = sum(1 for r in records if r.match)
        print(f"score complete — {n_ok} answered, {n_match} fully correct, "
              f"{len(records) - n_ok} errors", flush=True)
    return records


def aggregate_results(layout: RunLayout) -> dict:
    """Flatten the cache dir into `eval/results.json` for downstream tools."""
    out: list[dict] = []
    for f in sorted(layout.cache_dir.glob("*__*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:  # noqa: BLE001
            continue
    layout.results_path.parent.mkdir(parents=True, exist_ok=True)
    layout.results_path.write_text(
        json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    return {"n_results": len(out), "path": str(layout.results_path)}


def summarize(records: list[ScoreRecord], parsers: list[str]) -> dict:
    """Per-parser per-field + per-question accuracy."""
    out = {}
    for p in parsers:
        rs = [r for r in records if r.parser == p and r.ok]
        f_ok = sum(sum(1 for v in r.field_matches.values() if v) for r in rs)
        f_tot = sum(len(r.field_matches) for r in rs)
        q_ok = sum(1 for r in rs if r.match)
        out[p] = {
            "n_results": len(rs),
            "field_pct": (100 * f_ok / f_tot) if f_tot else 0.0,
            "field_ok": f_ok, "field_tot": f_tot,
            "q_pct": (100 * q_ok / len(rs)) if rs else 0.0,
            "q_ok": q_ok, "q_tot": len(rs),
        }
    return out
