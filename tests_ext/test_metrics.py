from ocr_eval_ext.metrics import baseline_rows, checkbox_metrics, field_outcomes, null_metrics

BOOL_FIELDS = [("q1", "a", True, "d1"), ("q2", "b", False, "d1"), ("q3", "c", True, "d2")]


def test_field_outcomes_statuses():
    records = {
        "q1": {"answer": {"a": True}, "field_matches": {"a": True}, "match": True},
        "q2": {"answer": {"b": True}, "field_matches": {"b": False}, "match": False},
        # q3 absent → error
    }
    out = field_outcomes(records, BOOL_FIELDS)
    assert [o.status for o in out] == ["correct", "incorrect", "error"]


def test_acc_over_all_counts_errors_as_wrong():
    records = {"q1": {"answer": {"a": True}, "field_matches": {"a": True}, "match": True}}
    m = checkbox_metrics(field_outcomes(records, BOOL_FIELDS))
    assert m["overall"].n == 3
    assert m["overall"].acc_over_all == 1 / 3
    assert m["overall"].acc_over_answered == 1.0   # only q1 answered
    assert m["confusion"]["err"] == 2


def test_polarity_split():
    records = {
        "q1": {"field_matches": {"a": True}, "answer": {}, "match": True},    # gold True, correct
        "q2": {"field_matches": {"b": False}, "answer": {}, "match": False},  # gold False, wrong
        "q3": {"field_matches": {"c": False}, "answer": {}, "match": False},  # gold True, wrong
    }
    m = checkbox_metrics(field_outcomes(records, BOOL_FIELDS))
    assert m["polarity"]["checked"].acc_over_all == 0.5       # q1 right, q3 wrong
    assert m["polarity"]["unchecked"].acc_over_all == 0.0


def test_baselines():
    import pytest as _pytest
    b = baseline_rows(BOOL_FIELDS)
    assert b["always_true"] == _pytest.approx(2 / 3)
    assert b["always_false"] == _pytest.approx(1 / 3)
    assert b["majority"] == _pytest.approx(2 / 3)
    assert b["class_balance"] == {"true": 2, "false": 1}


def test_null_hallucination():
    from ocr_eval_ext.preconditions import null_fields
    items = [{"question_id": "q9", "source_file": "d3", "capabilities": ["blank_field"],
              "gold_dict": {"z": None}}]
    nulls = null_fields(items)                       # exercise the REAL seam, not a hand-built tuple
    records = {"q9": {"answer": {"z": "invented value"}, "field_matches": {"z": False}, "match": False}}
    m = null_metrics(field_outcomes(records, nulls))
    assert m["hallucination_rate"] == 1.0


def test_null_gold_answer_none_is_error_not_correct():
    # D7: a collapsed extractor (answer=None) must NOT score as "did not hallucinate"
    nulls = [("q9", "z", None, "d3")]
    records = {"q9": {"answer": None, "field_matches": {"z": True}, "match": True}}
    out = field_outcomes(records, nulls)
    assert out[0].status == "error"


def test_null_gold_key_absent_is_incorrect_not_correct():
    nulls = [("q9", "z", None, "d3")]
    records = {"q9": {"answer": {"other": 1}, "field_matches": {"z": True}, "match": True}}
    out = field_outcomes(records, nulls)
    assert out[0].status == "incorrect"
