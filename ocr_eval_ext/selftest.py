"""Fail-closed scorer self-test. A scorer that cannot flag a known-wrong answer
is worse than no scorer — any failure here must abort runs before spend."""
from __future__ import annotations

from realdoc_bench.evaluate.score import build_template, score_typed, string_keys

FIXTURES: list[dict] = [
    {"kind": "correct",
     "rf": "Return exactly: box_marked=<boolean>", "gold": {"box_marked": True},
     "answer": {"box_marked": True}, "expect_match": True},
    {"kind": "polarity_inverted",
     "rf": "Return exactly: box_marked=<boolean>", "gold": {"box_marked": True},
     "answer": {"box_marked": False}, "expect_match": False},
    {"kind": "missing_field",
     "rf": "Return exactly: box_marked=<boolean>; name=<text>",
     "gold": {"box_marked": True, "name": "Ann"},
     "answer": {"box_marked": True}, "expect_match": False},
    {"kind": "null_correct",
     "rf": "Return exactly: state=<text|null>", "gold": {"state": None},
     "answer": {"state": None}, "expect_match": True},
    {"kind": "null_hallucinated",
     "rf": "Return exactly: state=<text|null>", "gold": {"state": None},
     "answer": {"state": "California"}, "expect_match": False},
    # Typed-contract semantics: a textual "blank" is NOT a null — must score wrong.
    {"kind": "null_textual_blank_is_wrong",
     "rf": "Return exactly: state=<text|null>", "gold": {"state": None},
     "answer": {"state": "blank"}, "expect_match": False},
    # A refusal never reaches score_typed (no JSON) — encode as answer None ⇒ no fields match.
    {"kind": "refusal_is_error",
     "rf": "Return exactly: box_marked=<boolean>", "gold": {"box_marked": True},
     "answer": None, "expect_match": False},
]


def run_offline() -> list[str]:
    failures = []
    for f in FIXTURES:
        template = build_template(f["rf"], f["gold"])
        sk = string_keys(template)
        _, match = score_typed(f["answer"] or {}, f["gold"], sk)
        if match is not f["expect_match"]:
            failures.append(f"{f['kind']}: expected match={f['expect_match']}, got {match}")
    return failures


EXTRACTOR_FIXTURES = [
    {"question": "Is the applicant a US citizen?",
     "rf": "Return exactly: us_citizen=<boolean>", "gold": {"us_citizen": True},
     "markdown": "**US Citizen:** ☒ Yes ☐ No\n**Name:** John Example"},
    {"question": "What is the policy state, if filled in?",
     "rf": "Return exactly: state=<text|null>", "gold": {"state": None},
     "markdown": "**State:** <blank>\n**Policy #:** 12345"},
    {"question": "Is the smoke detector box checked?",
     "rf": "Return exactly: smoke_detector=<boolean>", "gold": {"smoke_detector": False},
     "markdown": "Safety checklist:\n☐ Smoke detector\n☒ Fire extinguisher"},
    # Gold must be the full verbatim span: the extractor prompt says "copy the page text
    # verbatim", and rapidfuzz fallback needs ≥5 words — "Rivera" vs "Maria Rivera" would
    # false-fail this fail-closed gate on a CORRECT extraction.
    {"question": "What name is on the claimant line?",
     "rf": "Return exactly: claimant_name=<text>", "gold": {"claimant_name": "Maria Rivera"},
     "markdown": "**Claimant:** Maria Rivera\n**Date:** 01/02/2026"},
    {"question": "Is renewal requested?",
     "rf": "Return exactly: renewal=<boolean>", "gold": {"renewal": True},
     "markdown": "**Renewal requested:** ☒ Yes ☐ No"},
]


def run_extractor() -> list[str]:
    from realdoc_bench.evaluate.score import gemini_extract

    failures = []
    for f in EXTRACTOR_FIXTURES:
        template = build_template(f["rf"], f["gold"])
        ans = gemini_extract(f["question"], template, f["markdown"])
        _, match = score_typed(ans or {}, f["gold"], string_keys(template))
        if not match:
            failures.append(f"extractor missed: {f['question']!r} -> {ans!r}")
    return failures   # caller requires [] — near-ceiling means 5/5 on these
