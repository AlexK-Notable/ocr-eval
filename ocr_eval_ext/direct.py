"""Direct QA (vlm-chat): page image + question + typed template → JSON answer.
Writes upstream-score-cache-compatible records under parser key vlm__<id>__<cond>,
so upstream rescoring/aggregation work unchanged on our rows."""
from __future__ import annotations

import base64
import concurrent.futures
import datetime as dt
import email.utils
import hashlib
import io
import json
import os
import time
import uuid
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI
from PIL import Image

from ocr_eval_ext.config import RegistryEntry
from ocr_eval_ext.preconditions import assert_single_page, ink_coverage
from realdoc_bench.evaluate.parsers._vision_base import _render_pdf_pages
from realdoc_bench.evaluate.runs import RunLayout
from realdoc_bench.evaluate.score import _ensure_template, score_typed

STAGE1_CONDITION = {
    "preprocess": "raw",
    "output_contract": "schema_prompted",
    "render": {"engine": "pymupdf", "dpi": 150},
    "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024, "seed": None},
    "sample_index": 0,
    "no_image": False,     # in the dict from commit one — flipping a VALUE, never adding a key
}

SYSTEM = ("You answer questions about a scanned document page. "
          "Return JSON with exactly the keys and value types of the template. "
          "Booleans must be true/false. If a field is empty or not filled in, return null for it. "
          "Do not guess values that are not visible on the page.")

REFUSAL_MARKERS = ("i cannot", "i can't", "i'm unable", "i am unable", "cannot assist",
                   "can't help", "against my", "i won't")

MAX_RETRIES = 4
BACKOFF_BASE_SEC = 2.0
MAX_RETRY_WAIT_SEC = 60.0


def condition_hash(condition: dict) -> str:
    return hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest()[:12]


def parser_key(entry_id: str, condition: dict) -> str:
    return f"vlm__{entry_id}__{condition_hash(condition)}"


def direct_prompt(question: str, template: str) -> str:
    return (f"Question:\n{question}\n\n"
            f"Answer template (return JSON with exactly these keys and value types):\n{template}")


def _extract_json(text: str):
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


def _is_retryable(exc: Exception) -> bool:
    """408/429/5xx and connection-level failures (no HTTP response at all — DNS, refused
    connection, timeout) are transient: retry them. Everything else (400/401/403/404/409/422/...)
    is permanent: fail fast after exactly one attempt rather than burning the retry budget on a
    request that will never succeed."""
    if isinstance(exc, APIConnectionError):     # covers APITimeoutError (subclass)
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 408 or exc.status_code == 429 or 500 <= exc.status_code < 600
    return False


def _retry_wait(exc: Exception, attempt: int) -> float:
    """Never raises. Prefers the server's Retry-After (numeric seconds, or an HTTP-date per
    RFC 7231), falls back to exponential backoff on any parse failure, and always clamps to
    [0, MAX_RETRY_WAIT_SEC] — a misbehaving or hostile header (e.g. Retry-After: 900) must never
    cost more than a minute of real wall-clock time per attempt."""
    fallback = BACKOFF_BASE_SEC * (2 ** attempt)
    try:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        value = headers.get("retry-after") if headers else None
        if value is None:
            wait = fallback
        else:
            try:
                wait = float(value)
            except (TypeError, ValueError):
                parsed = email.utils.parsedate_to_datetime(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=dt.UTC)
                wait = (parsed - dt.datetime.now(dt.UTC)).total_seconds()
    except Exception:
        wait = fallback
    return max(0.0, min(MAX_RETRY_WAIT_SEC, wait))


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Same tmp-file + os.replace pattern as PNG caching — a torn cache row (killed mid
    write_text) is permanent silent cell loss: it never re-runs (the cache file exists) and
    never scores (the JSON is unparseable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)


def _render_page(layout: RunLayout, stem: str, condition: dict, png_cache: Path) -> bytes:
    """Atomic write (os.replace) — write_bytes truncates first, and a concurrent reader seeing a
    half-written file kills the run (reproduced under 8-thread stress during plan review)."""
    dpi = condition["render"]["dpi"]
    pre = condition["preprocess"]
    png_cache.mkdir(parents=True, exist_ok=True)
    p = png_cache / f"{stem}@{dpi}@{pre}.png"     # preprocess in the name — Stage 2 deskew must not
    if p.exists():                                 # silently reuse raw renders
        data = p.read_bytes()
        if data:
            return data
    pdf = layout.docs_dir / f"{stem}.pdf"
    assert_single_page(pdf)
    pages = _render_pdf_pages(pdf, dpi)
    tmp = p.with_name(p.name + f".tmp.{uuid.uuid4().hex}")
    tmp.write_bytes(pages[0])
    os.replace(tmp, p)
    return pages[0]


def _one(client: OpenAI, entry: RegistryEntry, item: dict, png: bytes | None,
         condition: dict, base: dict, prompt: str) -> dict:
    """`base` already carries qid/parser/source_file/domain/condition/retrieved_at/prompt_sha
    and the image_sha/image_px/image_bytes fields (null unless a render already succeeded) — see
    `do()` in `run_direct`. This function only adds the outcome (success or one of the
    error_class buckets)."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    if not condition.get("no_image"):
        b64 = base64.b64encode(png).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    extra_body = {}
    if entry.provider_pin:
        extra_body["provider"] = entry.provider_pin
    sampling = condition["sampling"]
    kwargs: dict = {
        "model": entry.model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": content}],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "max_tokens": sampling["max_tokens"],
        "extra_body": extra_body or None,
    }
    if sampling.get("seed") is not None:          # Stage 1 default (None) stays unsent on the wire
        kwargs["seed"] = sampling["seed"]
    t0 = time.perf_counter()
    resp = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except Exception as e:  # classified by _is_retryable; per-cell isolation either way
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                return {**base, "error": str(e)[:300], "error_class": "api_error"}
            time.sleep(_retry_wait(e, attempt))
    raw = resp.model_dump()
    text = (resp.choices[0].message.content or "").strip()
    usage = raw.get("usage") or {}    # OpenRouter now always includes usage details (incl. cost)
    common = {**base, "raw_response": text, "usage": usage,
              "resolved_provider": raw.get("provider") or "",
              "latency_sec": time.perf_counter() - t0}
    if not text:
        return {**common, "error": "empty response", "error_class": "empty"}
    ans = _extract_json(text)
    if ans is None:
        cls = "refusal" if any(m in text.lower() for m in REFUSAL_MARKERS) else "parse_error"
        return {**common, "error": f"unparseable response ({cls})", "error_class": cls}
    fm, allc = score_typed(ans, item["gold_dict"], item["str_keys"])
    return {**common, "answer": ans, "field_matches": fm, "match": allc, "error_class": "none"}


# Deliberate divergence from spec (documented): the rendered-image hash lives in the ROW
# (image_sha), not the cache key — the render is deterministic given (pinned dataset revision,
# pymupdf version, dpi), so drift is detectable rather than auto-invalidating: `rescore` and
# `report` warn when a row's image_sha no longer matches the current render.


def run_direct(layout: RunLayout, entries: list[RegistryEntry], *, bank_path: Path | None = None,
               condition: dict = STAGE1_CONDITION, workers: int = 8, force: bool = False,
               dry_run: bool = False, max_spend_usd: float | None = None,
               limit: int | None = None, no_image: bool = False) -> dict:
    for e in entries:
        if e.shape != "vlm-chat" or e.transport != "openai-compat" or not e.base_url:
            raise ValueError(
                f"{e.id}: run_direct requires shape='vlm-chat', transport='openai-compat', and a "
                f"base_url (got shape={e.shape!r}, transport={e.transport!r}, "
                f"base_url={e.base_url!r}) — OpenAI(base_url=None) would silently target "
                f"api.openai.com")

    bank = json.loads((bank_path or layout.bank_path).read_text())
    items = bank["items"][:limit] if limit else bank["items"]
    for it in items:
        _ensure_template(it)
    cond = dict(condition)
    if no_image:
        cond = {**cond, "no_image": True}

    cells = []
    for e in entries:
        pk = parser_key(e.id, cond)
        for it in items:
            cpath = layout.cache_path(it["question_id"], pk)
            if force or not cpath.exists():
                cells.append((e, it, cpath))

    if dry_run:
        # ESTIMATE (labelled so in output): ~1,600 image tokens/page + prompt ~400 in, ~120 out.
        per_entry: dict[str, dict] = {}
        priced_cells = unpriced_cells = 0
        total_est = 0.0
        for e in entries:
            n = sum(1 for c in cells if c[0].id == e.id)
            if e.pricing:
                est = n * ((2000 / 1e6) * e.pricing["input_per_mtok"]
                           + (120 / 1e6) * e.pricing["output_per_mtok"])
                per_entry[e.id] = {"cells": n, "est_usd": round(est, 2)}
                priced_cells += n
                total_est += est
            else:
                per_entry[e.id] = {"cells": n, "est_usd": None}
                unpriced_cells += n
        return {"cells": len(cells), "priced_cells": priced_cells, "unpriced_cells": unpriced_cells,
                "estimated_usd": round(total_est, 2), "per_entry": per_entry,
                "estimate_note": "±2x — token counts are guesses; unpriced_cells "
                                 "(no registry pricing) are excluded from estimated_usd, "
                                 "never silently folded in as $0"}

    png_cache = layout.root / "docs_png"
    summary = {"ok": 0, "error": 0, "cached": len(entries) * len(items) - len(cells)}
    spend = 0.0
    clients = {}
    for e in entries:
        key = os.environ.get(e.api_key_env) if e.api_key_env else "local"
        if e.api_key_env and not key:
            raise RuntimeError(f"{e.id}: env var {e.api_key_env} not set")
        # max_retries=0: this module owns retries end-to-end (_is_retryable / _retry_wait). The
        # SDK's own default retry-on-408/429/5xx would otherwise nest under ours — up to
        # MAX_RETRIES outer attempts x the SDK's own default retries each, observed as 12 real
        # HTTP attempts for what should have been at most 4.
        clients[e.id] = OpenAI(base_url=e.base_url, api_key=key or "none", max_retries=0)

    local_cells = [c for c in cells if c[0].local]
    hosted_cells = [c for c in cells if not c[0].local]

    def do(cell) -> tuple[RegistryEntry, dict]:
        e, it, cpath = cell
        base = {
            "qid": it.get("question_id"), "parser": parser_key(e.id, cond),
            "source_file": it.get("source_file"), "domain": it.get("domain", ""),
            "condition": cond, "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
            "prompt_sha": None, "image_sha": None, "image_px": None, "image_bytes": None,
        }
        try:
            prompt = direct_prompt(it["question"], it["template"])
            base["prompt_sha"] = hashlib.sha256((SYSTEM + "\x00" + prompt).encode()).hexdigest()[:12]
            if cond.get("no_image"):
                # nothing is rendered or sent — image_sha/image_px/image_bytes stay null, matching
                # "nothing was sent" rather than reporting the hash of an image the model never saw
                rec = _one(clients[e.id], e, it, None, cond, base, prompt)
            else:
                png = _render_page(layout, it["source_file"], cond, png_cache)
                if ink_coverage(png) < 0.001:
                    rec = {**base, "error": "blank render", "error_class": "render_error"}
                else:
                    dims = Image.open(io.BytesIO(png)).size
                    base = {**base, "image_sha": hashlib.sha256(png).hexdigest(),
                            "image_px": list(dims), "image_bytes": len(png)}
                    rec = _one(clients[e.id], e, it, png, cond, base, prompt)
        except Exception as exc:  # one bad doc costs one cell, never the run
            rec = {**base, "error": str(exc)[:300], "error_class": "render_error"}
        _atomic_write_json(cpath, rec)
        return e, rec

    def track(entry: RegistryEntry, rec: dict) -> None:
        nonlocal spend
        summary["ok" if rec.get("error_class") == "none" else "error"] += 1
        u = rec.get("usage") or {}
        cost = u.get("cost")
        if cost is None and entry.pricing:                 # vLLM/local/no-cost-field fallback
            cost = (u.get("prompt_tokens", 0) / 1e6) * entry.pricing["input_per_mtok"] \
                 + (u.get("completion_tokens", 0) / 1e6) * entry.pricing["output_per_mtok"]
        if cost is None:
            if max_spend_usd is not None:
                # FAIL CLOSED: never treat an unknown cost as free. A cell whose true cost cannot
                # be established (no usage.cost from the provider, no registry pricing to fall
                # back on) must stop the run, not silently pass as a $0 contribution to spend.
                raise RuntimeError(
                    f"cannot enforce --max-spend for {entry.id}: provider returned no "
                    f"usage.cost and registry has no pricing")
            return                       # no budget enforced — unpriced cell, spend left untouched
        spend += cost
        if max_spend_usd is not None and spend > max_spend_usd:
            raise RuntimeError(f"--max-spend {max_spend_usd} exceeded (realized {spend:.2f})")

    def _dispatch_bounded(pool: concurrent.futures.ThreadPoolExecutor, cells: list) -> None:
        """At most `workers` cells in flight at any time (submit/wait, never Executor.map's eager
        internal queue). `track()` may raise (max-spend exceeded) — once it does, no further
        cells are submitted (spend is effectively checked before every submit, since `error`
        below is set exactly when track() raises); the <= workers cells already in flight are
        let finish (their rows are still written by `do()`; only their spend/summary bookkeeping
        is skipped) and then the recorded error is re-raised. Overshoot is therefore bounded
        exactly by the window size, not by however many cells the pool had already queued."""
        pending = iter(cells)
        in_flight: dict[concurrent.futures.Future, None] = {}
        error: BaseException | None = None

        def _fill() -> None:
            while error is None and len(in_flight) < workers:
                cell = next(pending, None)
                if cell is None:
                    return
                in_flight[pool.submit(do, cell)] = None

        _fill()
        while in_flight:
            done, _pending_futs = concurrent.futures.wait(
                in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                del in_flight[fut]
                entry_result, rec = fut.result()
                if error is None:
                    try:
                        track(entry_result, rec)
                    except Exception as exc:  # captured, re-raised once the window is drained
                        error = exc
            if error is None:
                _fill()
        if error is not None:
            raise error

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        _dispatch_bounded(pool, hosted_cells)
    for cell in local_cells:                   # one local model resident at a time — serialize
        track(*do(cell))
    return summary
