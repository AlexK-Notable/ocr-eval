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

    # ── Comparator-selection layer ──────────────────────────────────────────────────────────
    # The six fixtures above never exercise `string_keys`' fuzzy/exact split: none of them use a
    # plain <string> field with a fuzzy-only variant. These three do — see task-7-report.md's
    # fix-round section for the empirical mutation-kill check behind each construction (each was
    # actually run through a monkeypatched string_keys/FUZZ_THRESHOLD and confirmed to flip).
    #
    # A note on a fixture NOT included here: an earlier draft tried "wrong boolean given as a
    # typo'd string ('truee'), plus correct text, expect_match=False" as a way to probe whether
    # string_keys wrongly treats a boolean field as fuzzy-eligible. Verified empirically that
    # construction can't discriminate anything: score_typed's `match = all(fm.values())` is
    # already pinned to False by the deliberately-wrong boolean field regardless of how that
    # field is scored (fuzzy_equal(str, bool) is False unconditionally, since its type guard
    # requires BOTH sides to be str — gold here is a real Python bool), so no aggregate-level
    # assertion built around "one deliberately-wrong field ⇒ expect_match=False" can ever catch
    # a comparator-selection bug. The three fixtures below instead make the CORRECT case
    # expect_match=True, so a wrongly-excluded (or wrongly-included) string field actually flips
    # the aggregate result.
    {"kind": "fuzzy_string_punctuation_variant",
     "rf": "Return exactly: notes=<text>", "gold": {"notes": "The quick brown fox jumped"},
     "answer": {"notes": "The quick brown fox jumped."}, "expect_match": True},
    {"kind": "fuzzy_string_same_wordcount_wrong",
     "rf": "Return exactly: notes=<text>", "gold": {"notes": "The quick brown fox jumped"},
     "answer": {"notes": "The slow green fox walked"}, "expect_match": False},
    # Mixed boolean+text template: confirms build_template/widen_types_from_gold keep per-field
    # typing distinct (the boolean field renders <boolean>, not <string>) and string_keys returns
    # exactly {"name"} — not "{}" (collapsed template). NOTE (correcting an earlier overclaim):
    # this fixture does NOT discriminate a string_keys over-inclusion bug (e.g. {"active",
    # "name"}) — fuzzy_equal() calls deep_equal() first and short-circuits True on two equal
    # Python booleans regardless of whether "active" wrongly ended up fuzzy-eligible, so an
    # over-included "active" key is silently absorbed and expect_match stays True either way.
    # Over-inclusion specifically is NOT killed by any fixture here.
    {"kind": "mixed_boolean_text_template_keeps_types_distinct",
     "rf": "Return exactly: active=<boolean>; name=<text>",
     "gold": {"active": True, "name": "Maria Isabella Rodriguez Torres"},
     "answer": {"active": True, "name": "Maria Isabella Rodriguez Torres."},
     "expect_match": True},
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
