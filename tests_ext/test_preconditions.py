import io

import fitz  # pymupdf
import pytest
from PIL import Image

from ocr_eval_ext.preconditions import (
    CHECKBOX_TAGS,
    PreconditionError,
    assert_single_page,
    boolean_fields,
    check_bank,
    ink_coverage,
    items_with_tags,
    null_fields,
)


def item(qid, tags, gold, sf="doc_1"):
    return {"question_id": qid, "source_file": sf, "capabilities": tags, "gold_dict": gold}


def test_boolean_fields_extracts_only_bools():
    items = [item("q1", ["checkbox_state"], {"a": True, "b": "text", "c": None})]
    assert boolean_fields(items) == [("q1", "a", True, "doc_1")]


def test_null_fields_extracts_only_nulls_same_shape_as_boolean_fields():
    items = [item("q1", ["blank_field"], {"a": True, "c": None})]
    assert null_fields(items) == [("q1", "c", None, "doc_1")]


def test_items_with_tags_matches_any():
    items = [item("q1", ["checkbox_state"], {}), item("q2", ["tables"], {})]
    assert [i["question_id"] for i in items_with_tags(items, CHECKBOX_TAGS)] == ["q1"]


def test_check_bank_fails_closed_on_wrong_counts():
    with pytest.raises(PreconditionError, match="cardinality mismatch"):
        check_bank([item("q1", [], {"a": 1})])       # 1 item ≠ 1356


def test_ink_coverage_blank_vs_marked():
    import io

    from PIL import Image

    blank = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(blank, "PNG")
    marked_img = Image.new("RGB", (100, 100), "white")
    for x in range(50):
        for y in range(50):
            marked_img.putpixel((x, y), (0, 0, 0))
    marked = io.BytesIO()
    marked_img.save(marked, "PNG")
    assert ink_coverage(blank.getvalue()) < 0.001
    assert ink_coverage(marked.getvalue()) > 0.2


def _uniform_gray_png(value: int) -> bytes:
    """A flat 10x10 grayscale PNG at exactly `value` — pixels this uniform sidestep any
    L-mode conversion rounding, isolating the `< 200` boundary itself."""
    img = Image.new("L", (10, 10), value)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_ink_coverage_threshold_boundary_199_200_201():
    """`ink_coverage` counts a pixel as 'ink' iff its grayscale value is STRICTLY < 200 — so a
    uniform image at 199 must read as 100% ink, and uniform images at 200 and 201 (the boundary
    and just above it) must both read as 0% ink."""
    assert ink_coverage(_uniform_gray_png(199)) == 1.0
    assert ink_coverage(_uniform_gray_png(200)) == 0.0
    assert ink_coverage(_uniform_gray_png(201)) == 0.0


def _fitz_pdf_bytes(n_pages: int) -> bytes:
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 20), "x")
    return doc.tobytes()


def test_assert_single_page_accepts_one_page_pdf(tmp_path):
    pdf_path = tmp_path / "one.pdf"
    pdf_path.write_bytes(_fitz_pdf_bytes(1))
    assert assert_single_page(pdf_path) == 1


def test_assert_single_page_rejects_two_page_pdf(tmp_path):
    pdf_path = tmp_path / "two.pdf"
    pdf_path.write_bytes(_fitz_pdf_bytes(2))
    with pytest.raises(PreconditionError, match="page_count == 2"):
        assert_single_page(pdf_path)
