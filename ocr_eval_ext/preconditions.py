"""Fail-closed gates. A run that cannot see its target must not look like a pass
(capability_buckets.yaml silently skips unknown tags — these assertions are the guard)."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

CHECKBOX_TAGS = ("checkbox_state", "handdrawn_check", "form_checkbox_grid")
BLANK_TAGS = ("blank_field",)

EXPECTED = {
    "bank_items": 1356, "bank_fields": 3742, "bank_nulls": 188,
    "checkbox_questions": 429, "checkbox_docs": 263,
    "checkbox_fields": 1117, "checkbox_booleans": 258,
    "checkbox_true": 165, "checkbox_false": 93,
    "blank_questions": 122, "blank_nulls": 34,
    "bucket_overlap": 40,
}


class PreconditionError(RuntimeError):
    pass


def items_with_tags(items: list[dict], tags: tuple[str, ...]) -> list[dict]:
    ts = set(tags)
    return [i for i in items if ts & set(i.get("capabilities") or [])]


def boolean_fields(items: list[dict]) -> list[tuple[str, str, bool, str]]:
    out = []
    for i in items:
        for k, v in (i.get("gold_dict") or {}).items():
            if isinstance(v, bool):
                out.append((i["question_id"], k, v, i["source_file"]))
    return out


def null_fields(items: list[dict]) -> list[tuple[str, str, None, str]]:
    out = []
    for i in items:
        for k, v in (i.get("gold_dict") or {}).items():
            if v is None:
                out.append((i["question_id"], k, None, i["source_file"]))
    return out


def check_bank(items: list[dict]) -> dict:
    cb = items_with_tags(items, CHECKBOX_TAGS)
    bf = items_with_tags(items, BLANK_TAGS)
    bools = boolean_fields(cb)
    measured = {
        "bank_items": len(items),
        "bank_fields": sum(len(i.get("gold_dict") or {}) for i in items),
        "bank_nulls": len(null_fields(items)),
        "checkbox_questions": len(cb),
        "checkbox_docs": len({i["source_file"] for i in cb}),
        "checkbox_fields": sum(len(i.get("gold_dict") or {}) for i in cb),
        "checkbox_booleans": len(bools),
        "checkbox_true": sum(1 for _q, _k, g, _sf in bools if g),
        "checkbox_false": sum(1 for _q, _k, g, _sf in bools if not g),
        "blank_questions": len(bf),
        "blank_nulls": len(null_fields(bf)),
        "bucket_overlap": len({i["question_id"] for i in cb} & {i["question_id"] for i in bf}),
    }
    mismatches = {k: (measured[k], EXPECTED[k]) for k in EXPECTED if measured[k] != EXPECTED[k]}
    if mismatches:
        detail = ", ".join(f"{k}: measured {m} != expected {e}" for k, (m, e) in mismatches.items())
        raise PreconditionError(f"bank cardinality mismatch — {detail}")
    return measured

# GUARD: a mismatch here is STOP-AND-INVESTIGATE, never "fix the constant". Any change to
# EXPECTED must be re-derived from the pinned dataset revision, with the derivation command
# recorded in the commit message — otherwise this gate degrades into decoration.


def assert_single_page(pdf_path: Path) -> int:
    import fitz  # pymupdf

    with fitz.open(pdf_path) as doc:
        n = doc.page_count
    if n != 1:
        raise PreconditionError(f"{pdf_path.name}: page_count == {n}, expected 1 "
                                "(the bank has no page index — multi-page breaks the design)")
    return n


def ink_coverage(png_bytes: bytes) -> float:
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    px = list(img.getdata())
    dark = sum(1 for p in px if p < 200)
    return dark / len(px)
