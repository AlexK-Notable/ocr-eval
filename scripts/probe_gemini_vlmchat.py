#!/usr/bin/env python
"""Confirm candidate Gemini models work on the endpoint `direct.py` ACTUALLY uses.

Why this is not redundant with `probe_gemini_extractor.py`: that one hits Google's NATIVE API
(`/v1beta/models/<id>:generateContent`). The vlm-chat leg goes through the OPENAI-COMPAT shim
(`base_url=.../v1beta/openai`, `openai.OpenAI().chat.completions.create`). They are different
surfaces with different failure modes — a model can be listed and work natively while the compat
shim rejects it, or silently ignores a parameter the condition dict claims to pin.

Checks per model, using the same request shape as a real cell (system message + image_url part +
temperature/top_p/max_tokens from STAGE1_CONDITION):
  * does it answer at all through the shim
  * does it return PARSEABLE JSON matching the template
  * does it read a checkbox correctly (a synthetic page whose answer is unambiguous)
  * how long, and how many tokens (for cost projection over 1,356 cells)

    uv run python scripts/probe_gemini_vlmchat.py
    uv run python scripts/probe_gemini_vlmchat.py --models gemini-2.5-flash,gemini-2.5-pro
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time

import fitz
from openai import OpenAI

from ocr_eval_ext.direct import STAGE1_CONDITION, SYSTEM, _extract_json, direct_prompt

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
]
QUESTION = "Is the 'Yes' checkbox for AUTHORIZED marked, and what is the applicant name?"
TEMPLATE = '{"authorized_yes_marked": <boolean>, "applicant_name": <string>}'
GOLD = {"authorized_yes_marked": True, "applicant_name": "Dana Whitfield"}


def synthetic_page_png() -> bytes:
    """A page whose answer is visually unambiguous, so a wrong answer means the model could not read
    it — never that the fixture was ambiguous."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "APPLICATION FOR SERVICE", fontsize=14)
    page.insert_text((72, 140), "Applicant name: Dana Whitfield", fontsize=11)
    page.insert_text((72, 180), "AUTHORIZED:", fontsize=11)
    page.insert_text((200, 180), "Yes", fontsize=11)
    page.insert_text((260, 180), "No", fontsize=11)
    # checked box next to Yes, empty box next to No
    page.draw_rect(fitz.Rect(170, 170, 184, 184), width=1.0)
    page.draw_line(fitz.Point(170, 170), fitz.Point(184, 184))
    page.draw_line(fitz.Point(184, 170), fitz.Point(170, 184))
    page.draw_rect(fitz.Rect(232, 170, 246, 184), width=1.0)
    pix = page.get_pixmap(dpi=STAGE1_CONDITION["render"]["dpi"])
    return pix.tobytes("png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY / GOOGLE_API_KEY not set — run `gemkey` first")

    png = synthetic_page_png()
    b64 = base64.b64encode(png).decode()
    s = STAGE1_CONDITION["sampling"]
    client = OpenAI(base_url=BASE_URL, api_key=key, max_retries=0)

    print(f"endpoint: {BASE_URL}")
    print(f"image: {len(png)} bytes @ {STAGE1_CONDITION['render']['dpi']} dpi")
    print(f"sampling: temperature={s['temperature']} top_p={s['top_p']} max_tokens={s['max_tokens']}\n")
    print(f"{'model':26s} {'ok':>4s} {'json':>5s} {'right':>6s} {'sec':>6s} {'in':>7s} {'out':>5s}  note")

    for model in args.models.split(","):
        model = model.strip()
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": direct_prompt(QUESTION, TEMPLATE)},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ]},
                ],
                temperature=s["temperature"], top_p=s["top_p"], max_tokens=s["max_tokens"],
            )
        except Exception as e:
            print(f"{model:26s} {'FAIL':>4s} {'-':>5s} {'-':>6s} {time.time()-t0:6.1f} "
                  f"{'-':>7s} {'-':>5s}  {type(e).__name__}: {str(e)[:90]}")
            continue
        dt = time.time() - t0
        text = (resp.choices[0].message.content or "").strip()
        ans = _extract_json(text)
        u = resp.model_dump().get("usage") or {}
        right = ans == GOLD if isinstance(ans, dict) else False
        note = "" if ans is not None else f"unparseable: {text[:60]!r}"
        if isinstance(ans, dict) and not right:
            note = f"read wrong: {json.dumps(ans)[:70]}"
        print(f"{model:26s} {'yes':>4s} {'yes' if ans is not None else 'NO':>5s} "
              f"{'yes' if right else 'NO':>6s} {dt:6.1f} "
              f"{u.get('prompt_tokens', '?'):>7} {u.get('completion_tokens', '?'):>5}  {note}")


if __name__ == "__main__":
    main()
