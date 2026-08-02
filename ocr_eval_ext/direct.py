"""Direct QA (vlm-chat): page image + question + typed template → JSON answer.
Writes upstream-score-cache-compatible records under parser key vlm__<id>__<cond>,
so upstream rescoring/aggregation work unchanged on our rows."""
from __future__ import annotations

import base64
import concurrent.futures
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

from openai import OpenAI

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


def _render_page(layout: RunLayout, stem: str, condition: dict, png_cache: Path) -> bytes:
    """Atomic write (os.replace) — write_bytes truncates first, and a concurrent reader seeing a
    half-written file kills the run (reproduced under 8-thread stress during plan review)."""
    import os
    import uuid

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


def _one(client: OpenAI, entry: RegistryEntry, item: dict, png: bytes,
         condition: dict, png_dims: tuple[int, int]) -> dict:
    prompt = direct_prompt(item["question"], item["template"])
    base = {"qid": item["question_id"], "parser": parser_key(entry.id, condition),
            "source_file": item["source_file"], "domain": item.get("domain", ""),
            "condition": condition, "image_sha": hashlib.sha256(png).hexdigest(),
            "image_px": list(png_dims), "image_bytes": len(png),
            "prompt_sha": hashlib.sha256((SYSTEM + "\x00" + prompt).encode()).hexdigest()[:12],
            "retrieved_at": dt.datetime.now(dt.UTC).isoformat()}
    content: list[dict] = [{"type": "text", "text": prompt}]
    if not condition.get("no_image"):
        b64 = base64.b64encode(png).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    extra_body = {}
    if entry.provider_pin:
        extra_body["provider"] = entry.provider_pin
    t0 = time.perf_counter()
    resp = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=entry.model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": content}],
                temperature=condition["sampling"]["temperature"],
                top_p=condition["sampling"]["top_p"],
                max_tokens=condition["sampling"]["max_tokens"],
                extra_body=extra_body or None,
            )
            break
        except Exception as e:
            retry_after = getattr(getattr(e, "response", None), "headers", {}) or {}
            wait = float(retry_after.get("retry-after") or BACKOFF_BASE_SEC * (2 ** attempt))
            if attempt == MAX_RETRIES - 1:
                return {**base, "error": str(e)[:300], "error_class": "api_error"}
            time.sleep(wait)
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
    import os

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
        est = 0.0
        for e, _it, _c in cells:
            if e.pricing:
                est += (2000 / 1e6) * e.pricing["input_per_mtok"] \
                     + (120 / 1e6) * e.pricing["output_per_mtok"]
        return {"cells": len(cells), "entries": len(entries), "items": len(items),
                "estimated_usd": round(est, 2), "estimate_note": "±2x — token counts are guesses"}

    png_cache = layout.root / "docs_png"
    summary = {"ok": 0, "error": 0, "cached": len(entries) * len(items) - len(cells)}
    spend = 0.0
    clients = {}
    for e in entries:
        key = os.environ.get(e.api_key_env) if e.api_key_env else "local"
        if e.api_key_env and not key:
            raise RuntimeError(f"{e.id}: env var {e.api_key_env} not set")
        clients[e.id] = OpenAI(base_url=e.base_url, api_key=key or "none")

    local_cells = [c for c in cells if c[0].local]
    hosted_cells = [c for c in cells if not c[0].local]

    def do(cell):
        e, it, cpath = cell
        try:
            png = _render_page(layout, it["source_file"], cond, png_cache)
            if ink_coverage(png) < 0.001:
                rec = {"qid": it["question_id"], "parser": parser_key(e.id, cond),
                       "source_file": it["source_file"], "domain": it.get("domain", ""),
                       "error": "blank render", "error_class": "render_error"}
            else:
                import io as _io

                from PIL import Image as _Img
                dims = _Img.open(_io.BytesIO(png)).size
                rec = _one(clients[e.id], e, it, png, cond, dims)
        except Exception as exc:
            rec = {"qid": it["question_id"], "parser": parser_key(e.id, cond),
                   "source_file": it["source_file"], "domain": it.get("domain", ""),
                   "error": str(exc)[:300], "error_class": "render_error"}
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(rec, ensure_ascii=False))
        return rec

    def track(rec):
        # Runs single-threaded in the consumer loop — no lock needed. Raising here cancels
        # pending futures via the pool's generator cleanup; overshoot is bounded by ≤ workers
        # in-flight calls (measured 12/50 cells at workers=8 during plan review).
        nonlocal spend
        summary["ok" if rec.get("error_class") == "none" else "error"] += 1
        u = rec.get("usage") or {}
        cost = u.get("cost")
        if cost is None:                       # vLLM/local: no cost field — token x rate fallback
            e = next((x for x in entries if rec["parser"].startswith(f"vlm__{x.id}__")), None)
            if e and e.pricing:
                cost = (u.get("prompt_tokens", 0) / 1e6) * e.pricing["input_per_mtok"] \
                     + (u.get("completion_tokens", 0) / 1e6) * e.pricing["output_per_mtok"]
        spend += cost or 0.0
        if max_spend_usd is not None and spend > max_spend_usd:
            raise RuntimeError(f"--max-spend {max_spend_usd} exceeded (realized {spend:.2f})")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for rec in pool.map(do, hosted_cells):
            track(rec)
    for cell in local_cells:                   # one local model resident at a time — serialize
        track(do(cell))
    return summary
