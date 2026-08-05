#!/usr/bin/env python
"""Diagnose a Gemini extractor failure by showing what the API actually said.

`gemini_extract` raises `httpx.HTTPStatusError`, whose message carries the status and URL but NOT
the response body — and the body is where Google puts the reason (bad key vs unknown model vs
malformed request all surface as 4xx). This probe prints the body, lists the model ids the key can
actually see, and confirms whether the pinned DEFAULT_MODEL is among them.

    uv run python scripts/probe_gemini_extractor.py

Reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment (run `gemkey` first). The key is sent
as an `x-goog-api-key` header and is NEVER printed, and the output is scrubbed for it before
display, so this is safe to paste into a transcript.
"""
from __future__ import annotations

import json
import os
import sys

import httpx

from realdoc_bench.evaluate.score import DEFAULT_MODEL

BASE = "https://generativelanguage.googleapis.com/v1beta"


def _key() -> str:
    k = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not k:
        sys.exit("GEMINI_API_KEY / GOOGLE_API_KEY not set — run `gemkey` first")
    return k


def _scrub(text: str, key: str) -> str:
    """Defense in depth: the key should never reach the output, but if an error echoes it back,
    redact rather than print it."""
    return text.replace(key, "<REDACTED-KEY>") if key else text


def main() -> None:
    key = _key()
    headers = {"x-goog-api-key": key}
    print(f"pinned DEFAULT_MODEL: {DEFAULT_MODEL}\n")

    with httpx.Client(timeout=60) as c:
        # 1. What can this key see? A 400 here (rather than on generateContent) points at the key
        #    itself; a clean list with DEFAULT_MODEL absent points at the model id.
        print("── models visible to this key (generateContent-capable) ──")
        try:
            r = c.get(f"{BASE}/models", headers=headers, params={"pageSize": 200})
            if r.status_code != 200:
                print(f"  list FAILED {r.status_code}: {_scrub(r.text, key)[:600]}")
                print("\n  A 4xx on plain model listing means the KEY is the problem, not the model id.")
                return
            models = r.json().get("models", [])
            names = sorted(
                m["name"].removeprefix("models/") for m in models
                if "generateContent" in (m.get("supportedGenerationMethods") or [])
            )
            for n in names:
                mark = "  <-- pinned" if n == DEFAULT_MODEL else ""
                print(f"  {n}{mark}")
            print(f"\n  total: {len(names)}")
            print(f"  pinned id present: {DEFAULT_MODEL in names}")
            if DEFAULT_MODEL not in names:
                near = [n for n in names if "flash" in n and ("lite" in n or "3" in n)]
                print(f"  closest flash/lite candidates: {near}")
        except httpx.HTTPError as e:
            print(f"  transport error: {type(e).__name__}: {e}")
            return

        # 2. Reproduce the real call and show the BODY, which the harness's exception omits.
        print(f"\n── minimal generateContent against {DEFAULT_MODEL} ──")
        body = {
            "contents": [{"role": "user", "parts": [{"text": "Return {\"ok\": true}"}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 64,
                                 "responseMimeType": "application/json"},
        }
        r = c.post(f"{BASE}/models/{DEFAULT_MODEL}:generateContent",
                   json=body, headers=headers)
        print(f"  status: {r.status_code}")
        if r.status_code == 200:
            print(f"  body: {json.dumps(r.json())[:400]}")
            print("\n  This model works for a minimal call — if the gate still fails, the problem is")
            print("  in the FULL request (systemInstruction / maxOutputTokens 8192), not the id.")
        else:
            print(f"  body: {_scrub(r.text, key)[:900]}")


if __name__ == "__main__":
    main()
