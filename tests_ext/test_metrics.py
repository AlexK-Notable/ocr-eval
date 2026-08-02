import pytest

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
    b = baseline_rows(BOOL_FIELDS)
    assert b["always_true"] == pytest.approx(2 / 3)
    assert b["always_false"] == 1 / 3          # EXACT: f/n must not be computed as 1 - t/n
    assert b["majority"] == pytest.approx(2 / 3)
    assert b["class_balance"] == {"true": 2, "false": 1}


def test_baselines_false_majority():
    # Distinguishes majority from "always defer to always_true" — a mutant that hardcodes
    # majority = always_true would pass test_baselines (2/3 == 2/3 coincidentally) but fail
    # here, where false is the majority class.
    fields = [("q1", "a", False, "d1"), ("q2", "b", False, "d1"), ("q3", "c", True, "d1")]
    b = baseline_rows(fields)
    assert b["always_true"] == 1 / 3
    assert b["always_false"] == 2 / 3
    assert b["majority"] == b["always_false"]
    assert b["majority"] != b["always_true"]
    assert b["class_balance"] == {"true": 1, "false": 2}


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


# --- C1: null_metrics collapse loophole -------------------------------------------------

def test_null_metrics_collapse_is_not_a_perfect_score():
    # All-answer-None collapse over null golds: the old hall/n definition gave this a
    # perfect 0.0 hallucination_rate. C1 ruling: overall.acc_over_all is now the headline
    # number and it must reflect the collapse as a failure; hallucination_rate must clamp
    # to 0.0 (not "perfect") specifically because n_answered == 0.
    nulls = [("q9", "z", None, "d3"), ("q10", "y", None, "d4")]
    records = {
        "q9": {"answer": None, "field_matches": {}, "match": False},
        "q10": {"answer": None, "field_matches": {}, "match": False},
    }
    m = null_metrics(field_outcomes(records, nulls))
    assert m["overall"].acc_over_all == 0.0
    assert m["overall"].error_rate == 1.0
    assert m["overall"].n_answered == 0
    assert m["hallucination_rate"] == 0.0


def test_null_metrics_all_invented_hallucination_rate_one():
    nulls = [("q9", "z", None, "d3"), ("q10", "y", None, "d4")]
    records = {
        "q9": {"answer": {"z": "invented"}, "field_matches": {"z": False}, "match": False},
        "q10": {"answer": {"y": "also invented"}, "field_matches": {"y": False}, "match": False},
    }
    m = null_metrics(field_outcomes(records, nulls))
    assert m["hallucination_rate"] == 1.0
    assert m["overall"].acc_over_all == 0.0


# --- I3: non-dict answers are errors; empty dict stays answered -------------------------

def test_field_outcomes_non_dict_answer_is_error():
    nulls = [("q9", "z", None, "d3")]
    records = {"q9": {"answer": "some string", "field_matches": {"z": True}, "match": True}}
    out = field_outcomes(records, nulls)
    assert out[0].status == "error"


def test_field_outcomes_empty_dict_answer_is_answered_not_error():
    nulls = [("q9", "z", None, "d3")]
    records = {"q9": {"answer": {}, "field_matches": {}, "match": False}}
    out = field_outcomes(records, nulls)
    assert out[0].status == "incorrect"   # key absent under D7, but NOT "error"


# --- MINOR-1: "error" key forces error even when answer is present ----------------------

def test_field_outcomes_error_key_forces_error_even_with_answer():
    fields = [("q9", "a", True, "d1")]
    records = {"q9": {"error": "boom", "answer": {"a": True}, "field_matches": {"a": True}, "match": True}}
    out = field_outcomes(records, fields)
    assert out[0].status == "error"


# --- I6: gold-type guard ------------------------------------------------------------------

def test_checkbox_metrics_rejects_non_boolean_gold():
    bad = field_outcomes({}, [("q1", "a", None, "d1")])   # null gold, not True/False
    with pytest.raises(ValueError, match="non-boolean gold"):
        checkbox_metrics(bad)


# --- Mutation-killing: confusion matrix, invariants --------------------------------------

_CONFUSION_FIELDS = [
    ("q1", "a", True, "d1"),
    ("q2", "b", True, "d1"),
    ("q3", "c", True, "d1"),
    ("q4", "d", False, "d1"),
    ("q5", "e", False, "d1"),
    ("q6", "f", False, "d1"),
    ("q7", "g", False, "d1"),
    ("q8", "h", True, "d1"),
]
_CONFUSION_RECORDS = {
    "q1": {"answer": {"a": True}, "field_matches": {"a": True}, "match": True},     # tt
    "q2": {"answer": {"b": False}, "field_matches": {"b": False}, "match": False},  # tf
    "q3": {"answer": {"c": False}, "field_matches": {"c": False}, "match": False},  # tf
    "q4": {"answer": {"d": False}, "field_matches": {"d": True}, "match": True},    # ff
    "q5": {"answer": {"e": True}, "field_matches": {"e": False}, "match": False},   # ft
    "q6": {"answer": {"f": True}, "field_matches": {"f": False}, "match": False},   # ft
    "q7": {"answer": {"g": True}, "field_matches": {"g": False}, "match": False},   # ft
    # q8 absent → err
}


def test_confusion_matrix_exact_cells():
    # Distinct counts per cell (tt=1, tf=2, ff=1, ft=3, err=1) so a tf<->ft polarity-inversion
    # mutant (the classic failure this project exists to catch) cannot slip through by
    # coincidence of equal counts.
    m = checkbox_metrics(field_outcomes(_CONFUSION_RECORDS, _CONFUSION_FIELDS))
    assert m["confusion"] == {"tt": 1, "tf": 2, "ff": 1, "ft": 3, "err": 1}
    assert m["overall"].n == 8
    assert m["overall"].n_answered == 7
    assert m["overall"].error_rate == 0.125


def test_confusion_and_polarity_invariants():
    m = checkbox_metrics(field_outcomes(_CONFUSION_RECORDS, _CONFUSION_FIELDS))
    conf = m["confusion"]
    assert conf["tt"] + conf["tf"] + conf["ft"] + conf["ff"] + conf["err"] == m["overall"].n
    assert m["polarity"]["checked"].n + m["polarity"]["unchecked"].n == m["overall"].n
