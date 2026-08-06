#!/usr/bin/env python
"""Probe Mistral's `POST /v1/ocr` — which models are real, and what the new parameters actually do.

WHY THIS EXISTS. `mistral-ocr-4-1` is undeprecated and is what `mistral-ocr-latest` resolves to
today, while `mistral-ocr-4-0` is a separate undeprecated pin. Both being live means picking one is
a CHOICE, not an upgrade, and this repo pins explicit versions everywhere — an alias can re-point
under a measurement and silently change what a number means (OCR 1 and OCR 2 have already been
retired, so the aliases demonstrably move).

Listing a model is not the same as being able to invoke it (learned on `gemini-3.1-pro-preview`,
which listed fine and returned empty content at a starved token budget). So this sends ONE real page
to each model and reports what came back, including whether the two versions differ on the same
input — which is the thing that decides whether the choice is material.

Also probes the parameters the current adapter does not implement (`include_blocks`,
`extract_header`/`extract_footer`, `confidence_scores_granularity`, `table_format`), because the
docs disagree with themselves on `include_blocks`' default (`true` in the API reference, `false` in
the capabilities page) and a default we cannot pin from docs has to be measured.

Costs real money: $4/1,000 pages, so each single-page call is ~$0.004. The full matrix below is
about 6 calls ~ $0.024.

Usage:
    gemkey && uv run python scripts/probe_mistral_ocr.py
"""
from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

import httpx

BASE_URL = os.environ.get("MISTRAL_OCR_BASE_URL", "https://api.mistral.ai").rstrip("/")
PDF = Path("runs/stage1/docs/finance_17.pdf")

# Both are undeprecated as of 2026-08-06 (verified against GET /v1/models, not recalled).
MODELS = ["mistral-ocr-4-0", "mistral-ocr-4-1"]


def _key() -> str:
    k = os.environ.get("MISTRAL_API_KEY")
    if not k:
        sys.exit("MISTRAL_API_KEY not set — run `gemkey` first")
    return k


def _post(client: httpx.Client, payload: dict) -> tuple[int, dict]:
    """Credential in a HEADER, never the URL: httpx embeds the full URL in HTTPStatusError, so a
    non-2xx on a query-param key prints the live secret into the traceback."""
    r = client.post(f"{BASE_URL}/v1/ocr",
                    headers={"Authorization": f"Bearer {_key()}"},
                    json=payload)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"_raw": r.text[:400]}


def _doc() -> dict:
    b64 = base64.b64encode(PDF.read_bytes()).decode("ascii")
    return {"type": "document_url", "document_url": f"data:application/pdf;base64,{b64}"}


def _summarize(tag: str, status: int, raw: dict, elapsed: float) -> dict | None:
    if status != 200:
        err = raw.get("message") or raw.get("detail") or raw.get("_raw") or raw
        print(f"  {tag:52s} HTTP {status}  {str(err)[:150]}")
        return None
    pages = raw.get("pages") or []
    p0 = pages[0] if pages else {}
    md = (p0.get("markdown") or "")
    # Keys present on the page object tell us which features actually returned data, rather than
    # trusting the docs' stated defaults.
    feats = {
        "blocks": p0.get("blocks") is not None,
        "tables": bool(p0.get("tables")),
        "hyperlinks": bool(p0.get("hyperlinks")),
        "header": p0.get("header") is not None,
        "footer": p0.get("footer") is not None,
        "confidence": p0.get("confidence_scores") is not None,
    }
    on = ",".join(k for k, v in feats.items() if v) or "none"
    print(f"  {tag:52s} {elapsed:5.1f}s  pages={len(pages):2d}  md={len(md):5d}ch  "
          f"model={raw.get('model')}  present: {on}")
    return {"md": md, "pages": len(pages), "feats": feats,
            "model": raw.get("model"), "usage": raw.get("usage_info")}


def main() -> int:
    if not PDF.exists():
        sys.exit(f"missing probe PDF: {PDF}")
    print(f"probe document: {PDF}  ({PDF.stat().st_size/1024:.0f} KB)")
    print(f"endpoint: {BASE_URL}/v1/ocr\n")

    results: dict[str, dict] = {}
    with httpx.Client(timeout=300) as c:
        print("── baseline: identical request to each undeprecated 4.x pin ──")
        for m in MODELS:
            t0 = time.perf_counter()
            st, raw = _post(c, {"model": m, "document": _doc()})
            r = _summarize(f"{m} (bare)", st, raw, time.perf_counter() - t0)
            if r:
                results[m] = r

        print("\n── parameters the current adapter does NOT implement (on 4-1) ──")
        variants = [
            ("include_blocks=True", {"include_blocks": True}),
            ("include_blocks=False", {"include_blocks": False}),
            ("table_format=html", {"table_format": "html"}),
            ("extract_header+footer", {"extract_header": True, "extract_footer": True}),
            ("confidence=word", {"confidence_scores_granularity": "word"}),
        ]
        for tag, extra in variants:
            t0 = time.perf_counter()
            st, raw = _post(c, {"model": "mistral-ocr-4-1", "document": _doc(), **extra})
            _summarize(tag, st, raw, time.perf_counter() - t0)

    # The decision this probe exists to inform: do the two pins differ on identical input?
    print("\n── 4-0 vs 4-1 on identical input ──")
    if len(results) == 2:
        a, b = results["mistral-ocr-4-0"], results["mistral-ocr-4-1"]
        same = a["md"] == b["md"]
        print(f"  markdown identical: {same}")
        print(f"  length 4-0={len(a['md'])}  4-1={len(b['md'])}  delta={len(b['md'])-len(a['md']):+d}")
        print(f"  server-reported model: 4-0->{a['model']}  4-1->{b['model']}")
        if not same:
            print("\n  NOT interchangeable — the pin is a measurable condition, not a label.")
            # strict=False on purpose: the two outputs have DIFFERENT line counts (that is the
            # finding), and we only want the first divergence within the common prefix.
            for i, (x, y) in enumerate(zip(a["md"].splitlines(), b["md"].splitlines(),
                                           strict=False)):
                if x != y:
                    print(f"  first divergence at line {i}:\n    4-0: {x[:110]}\n    4-1: {y[:110]}")
                    break
    else:
        print("  inconclusive — at least one model did not return 200 (see above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
