from ocr_eval_ext.selftest import FIXTURES, run_offline


def test_offline_selftest_passes():
    assert run_offline() == []


def test_fixture_coverage():
    kinds = {f["kind"] for f in FIXTURES}
    assert {"correct", "polarity_inverted", "missing_field",
            "null_correct", "null_hallucinated", "refusal_is_error"} <= kinds
