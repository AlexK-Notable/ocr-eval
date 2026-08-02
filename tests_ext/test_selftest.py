import ocr_eval_ext.selftest as st_mod
from ocr_eval_ext.selftest import FIXTURES, run_offline


def test_offline_selftest_passes():
    assert run_offline() == []


def test_fixture_coverage():
    kinds = {f["kind"] for f in FIXTURES}
    assert {"correct", "polarity_inverted", "missing_field",
            "null_correct", "null_hallucinated", "null_textual_blank_is_wrong",
            "refusal_is_error"} <= kinds


def test_comparator_layer_fixture_coverage():
    """I3: the comparator-selection layer (string_keys' fuzzy/exact split, and per-field typing
    in a mixed-type template) has dedicated fixtures — the six baseline fixtures never exercise
    a plain <string> field with a fuzzy-only variant."""
    kinds = {f["kind"] for f in FIXTURES}
    assert {"fuzzy_string_punctuation_variant", "fuzzy_string_same_wordcount_wrong",
            "mixed_boolean_text_template_keeps_types_distinct"} <= kinds


def test_fuzzy_inclusion_mutation_is_caught(monkeypatch):
    """Mutation-kill check for `fuzzy_string_punctuation_variant` and
    `mixed_boolean_text_template_keeps_types_distinct`: both fixtures require their plain
    <string> field to be fuzzy-matched (a trailing-period variant that `deep_equal` alone would
    reject). If `string_keys` regressed to wrongly excluding a real <string> field (simulated
    here by forcing it to always return the empty set), `score_typed` would fall back to
    `deep_equal` for that field and both fixtures would flip from PASS to FAIL — `run_offline()`
    must catch it, not silently keep returning []."""
    monkeypatch.setattr(st_mod, "string_keys", lambda template: set())
    failures = run_offline()
    assert failures != []
    failure_kinds = {f.split(":")[0] for f in failures}
    assert "fuzzy_string_punctuation_variant" in failure_kinds
    assert "mixed_boolean_text_template_keeps_types_distinct" in failure_kinds


def test_fuzz_threshold_mutation_is_caught(monkeypatch):
    """Mutation-kill check for `fuzzy_string_same_wordcount_wrong`: it depends on
    FUZZ_THRESHOLD being a real similarity floor, not a rubber stamp. If the threshold
    regressed to accept-everything (simulated by monkeypatching it to 0 on the module that
    actually reads it — `fuzzy_equal` looks up `FUZZ_THRESHOLD` as a global in
    `realdoc_bench.evaluate.score`, not via a name bound into `ocr_eval_ext.selftest`), the
    same-word-count-but-wrong-content answer would incorrectly pass."""
    import realdoc_bench.evaluate.score as score_mod

    monkeypatch.setattr(score_mod, "FUZZ_THRESHOLD", 0)
    failures = run_offline()
    assert any(f.startswith("fuzzy_string_same_wordcount_wrong") for f in failures)
