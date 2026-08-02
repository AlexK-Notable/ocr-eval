import pytest

from ocr_eval_ext.preconditions import (
    CHECKBOX_TAGS, PreconditionError, boolean_fields, check_bank, ink_coverage,
    items_with_tags, null_fields,
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
    blank = io.BytesIO(); Image.new("RGB", (100, 100), "white").save(blank, "PNG")
    marked_img = Image.new("RGB", (100, 100), "white")
    for x in range(50):
        for y in range(50):
            marked_img.putpixel((x, y), (0, 0, 0))
    marked = io.BytesIO(); marked_img.save(marked, "PNG")
    assert ink_coverage(blank.getvalue()) < 0.001
    assert ink_coverage(marked.getvalue()) > 0.2
