import email.utils
import json
from datetime import UTC, datetime
from pathlib import Path

import fitz  # pymupdf
import pytest

from ocr_eval_ext.config import RegistryEntry
from ocr_eval_ext.direct import STAGE1_CONDITION, condition_hash, parser_key, run_direct
from realdoc_bench.evaluate.runs import RunLayout
from tests_ext.mock_openai import MockOpenAI


def make_run_dir(tmp_path: Path) -> RunLayout:
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Is the box checked?  [X] Yes   [ ] No")
    page.draw_rect(fitz.Rect(72, 100, 220, 160), fill=(0, 0, 0))  # keep ink_coverage safely > floor
    doc.save(layout.docs_dir / "doc_1.pdf")
    bank = {"items": [{
        "question_id": "q1", "source_file": "doc_1", "domain": "test",
        "question": "Is question 1 checkbox marked?", "capabilities": ["checkbox_state"],
        "gold_dict": {"a": True},
        "response_format": "Return exactly: a=<boolean>",
        "gold_answer": "a=true",
    }]}
    layout.bank_path.write_text(json.dumps(bank))
    return layout


def make_multi_item_run_dir(tmp_path: Path, n: int) -> RunLayout:
    """n independent single-page docs/questions — used to exercise the bounded-dispatch
    submit/wait/refill loop with real concurrency (every other fixture in this file uses a
    single-item bank, which never reaches the refill path)."""
    layout = RunLayout.at(tmp_path / "run")
    layout.ensure_dirs()
    items = []
    for i in range(n):
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), f"Is box {i} checked?  [X] Yes   [ ] No")
        page.draw_rect(fitz.Rect(72, 100, 220, 160), fill=(0, 0, 0))
        doc.save(layout.docs_dir / f"doc_{i}.pdf")
        items.append({
            "question_id": f"q{i}", "source_file": f"doc_{i}", "domain": "test",
            "question": f"Is question {i} checkbox marked?", "capabilities": ["checkbox_state"],
            "gold_dict": {"a": True},
            "response_format": "Return exactly: a=<boolean>",
            "gold_answer": "a=true",
        })
    layout.bank_path.write_text(json.dumps({"items": items}))
    return layout


def entry(base_url: str) -> RegistryEntry:
    return RegistryEntry(
        id="m1@mock", shape="vlm-chat", transport="openai-compat",
        base_url=base_url, model="org/m1", api_key_env=None,
        precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
        provenance="Test", release_date="2025-01-01",
    )


# ── truncated response bodies are transient, not permanent ────────────────────────────────────
# A body cut off mid-stream reaches us as a bare json.JSONDecodeError: the SDK wraps transport
# failures during the REQUEST into APIConnectionError, but a body that arrives and then stops is
# parsed outside that wrapping. It used to be classified permanent, so one truncated body burned
# the whole document after a single attempt — measured live 2026-08-04 on a qwen3.5-9b page that
# died after 335s and then succeeded in 145s when re-issued by hand.

_TRUNCATED = '{"id":"cmpl-1","choices":[{"message":{"role":"assistant","content":"partial'


def test_is_retryable_treats_truncated_body_as_transient():
    from ocr_eval_ext.direct import _is_retryable
    try:
        json.loads(_TRUNCATED)
    except json.JSONDecodeError as e:
        assert _is_retryable(e) is True
    else:
        pytest.fail("fixture is supposed to be malformed JSON")


def test_is_retryable_stays_narrow_and_ignores_plain_value_errors():
    """JSONDecodeError subclasses ValueError. Retrying every ValueError would silently retry
    ordinary programming errors, so the check must be on the concrete type."""
    from ocr_eval_ext.direct import _is_retryable
    assert _is_retryable(ValueError("not a transport symptom")) is False
    assert _is_retryable(RuntimeError("nope")) is False


def test_run_direct_retries_a_truncated_body_and_then_succeeds(tmp_path):
    layout = make_run_dir(tmp_path)
    responses = [{"raw": _TRUNCATED}, {"status": 200, "content": '{"a": true}'}]
    with MockOpenAI(responses=responses) as mock:
        summary = run_direct(layout, [entry(mock.base_url)])
    assert len(mock.requests) == 2      # retried rather than failing the cell outright
    assert summary["ok"] == 1
    pk = parser_key("m1@mock", STAGE1_CONDITION)
    rec = json.loads(layout.cache_path("q1", pk).read_text())
    assert rec["error_class"] == "none"
    assert rec["answer"] == {"a": True}


# ── reasoning cap reaches the wire on the direct leg too ───────────────────────────────────────
# The 2026-08-04 finding was not transcriber-only: uncapped thinking made ~15% of qwen3.5-9b's
# direct cells come back error_class "empty" against 3 parse_errors across qwen3-vl-8b's full
# 1,356. The vlm-chat path needs the same guard as the transcriber path.

def test_run_direct_sends_reasoning_cap_when_entry_sets_one(tmp_path):
    layout = make_run_dir(tmp_path)
    e = entry("placeholder")
    with MockOpenAI(reply_text='{"a": true}') as mock:
        e = e.model_copy(update={"base_url": mock.base_url,
                                 "reasoning": {"max_tokens": 4096}})
        run_direct(layout, [e])
    assert mock.requests[0]["reasoning"] == {"max_tokens": 4096}


def test_run_direct_omits_reasoning_when_entry_sets_none(tmp_path):
    """Endpoints that do not advertise reasoning must never receive the field — it risks a 400."""
    layout = make_run_dir(tmp_path)
    with MockOpenAI(reply_text='{"a": true}') as mock:
        run_direct(layout, [entry(mock.base_url)])
    assert "reasoning" not in mock.requests[0]


def test_stage1_condition_completion_budget_is_pinned():
    """Guards the budget against silent drift. 1024 -> 12288 on 2026-08-04 because a thinking
    model spends the allowance before writing an answer; raising it costs nothing for
    non-thinking models, since max_tokens is a cap rather than a target."""
    assert STAGE1_CONDITION["sampling"]["max_tokens"] == 12288


def test_condition_hash_stable_and_order_free():
    a = condition_hash({"x": 1, "y": {"z": 2}})
    b = condition_hash({"y": {"z": 2}, "x": 1})
    assert a == b and len(a) == 12


def test_run_direct_writes_upstream_compatible_record(tmp_path):
    layout = make_run_dir(tmp_path)
    with MockOpenAI(reply_text='{"a": true}') as mock:
        summary = run_direct(layout, [entry(mock.base_url)])
    pk = parser_key("m1@mock", STAGE1_CONDITION)
    rec = json.loads(layout.cache_path("q1", pk).read_text())
    assert rec["match"] is True and rec["field_matches"] == {"a": True}
    assert rec["answer"] == {"a": True}
    assert rec["resolved_provider"] == "MockProvider"
    assert rec["error_class"] == "none"
    assert rec["condition"]["sampling"]["temperature"] == 0.0
    assert summary["ok"] == 1
    # request assertions: image attached, temperature 0, template in prompt
    req = mock.requests[0]
    assert req["temperature"] == 0.0
    content = req["messages"][-1]["content"]
    assert any(part.get("type") == "image_url" for part in content)
    assert any("boolean" in str(part.get("text", "")) for part in content)


def test_run_direct_cache_hit_makes_no_calls(tmp_path):
    layout = make_run_dir(tmp_path)
    with MockOpenAI() as mock:
        run_direct(layout, [entry(mock.base_url)])
        n1 = len(mock.requests)
        run_direct(layout, [entry(mock.base_url)])
        assert len(mock.requests) == n1        # second pass: 100% cache hit


def test_unparseable_reply_is_parse_error_row(tmp_path):
    layout = make_run_dir(tmp_path)
    # Deliberately not refusal-flavored text: it must miss every REFUSAL_MARKERS substring
    # (the brief's original fixture "I cannot help with that" collides with "i cannot" and
    # gets classified "refusal" instead of "parse_error" — see task-6-report.md for detail).
    with MockOpenAI(reply_text="The requested field is not legible in this scan.") as mock:
        run_direct(layout, [entry(mock.base_url)])
    pk = parser_key("m1@mock", STAGE1_CONDITION)
    rec = json.loads(layout.cache_path("q1", pk).read_text())
    assert rec["error_class"] == "parse_error" and "answer" not in rec


def test_dry_run_prices_without_calls(tmp_path):
    layout = make_run_dir(tmp_path)
    with MockOpenAI() as mock:
        s = run_direct(layout, [entry(mock.base_url)], dry_run=True)
        assert mock.requests == [] and s["cells"] == 1


def _success_body(usage: dict) -> dict:
    return {"id": "cmpl-1", "model": "org/m1", "provider": "MockProvider",
            "choices": [{"message": {"role": "assistant", "content": '{"a": true}'}}],
            "usage": usage}


def test_max_spend_aborts_on_priced_mock(tmp_path):
    """usage.cost drives the abort directly — no registry pricing needed."""
    layout = make_run_dir(tmp_path)
    responses = [{"body": _success_body({"prompt_tokens": 100, "completion_tokens": 10,
                                         "cost": 5.0})}]
    with MockOpenAI(responses=responses) as mock, \
         pytest.raises(RuntimeError, match=r"--max-spend 0\.01 exceeded"):
        run_direct(layout, [entry(mock.base_url)], max_spend_usd=0.01)


def test_max_spend_fails_closed_without_cost_or_pricing(tmp_path):
    """C1 ruling: an unresolvable cost (no usage.cost, no registry pricing) must abort the run
    rather than being treated as free — this is the mutant `spend += cost or 0.0` would pass."""
    layout = make_run_dir(tmp_path)
    with MockOpenAI() as mock, \
         pytest.raises(RuntimeError, match=r"cannot enforce --max-spend for m1@mock"):
        run_direct(layout, [entry(mock.base_url)], max_spend_usd=1.0)


def test_retry_after_numeric_header_then_success(tmp_path):
    layout = make_run_dir(tmp_path)
    responses = [{"status": 429, "headers": {"Retry-After": "0"}}, {"status": 200}]
    with MockOpenAI(responses=responses) as mock:
        run_direct(layout, [entry(mock.base_url)])
    pk = parser_key("m1@mock", STAGE1_CONDITION)
    rec = json.loads(layout.cache_path("q1", pk).read_text())
    assert rec["error_class"] == "none" and rec["match"] is True
    assert len(mock.requests) == 2   # one 429, one retry that succeeded


def test_retry_after_http_date_header_then_success(tmp_path):
    layout = make_run_dir(tmp_path)
    retry_at = email.utils.format_datetime(datetime.now(UTC))  # "now" -> wait clamps to ~0
    responses = [{"status": 429, "headers": {"Retry-After": retry_at}}, {"status": 200}]
    with MockOpenAI(responses=responses) as mock:
        run_direct(layout, [entry(mock.base_url)])
    pk = parser_key("m1@mock", STAGE1_CONDITION)
    rec = json.loads(layout.cache_path("q1", pk).read_text())
    assert rec["error_class"] == "none" and rec["match"] is True
    assert len(mock.requests) == 2


def test_permanent_400_fails_after_one_attempt(tmp_path):
    layout = make_run_dir(tmp_path)
    responses = [{"status": 400, "message": "bad request"}]
    with MockOpenAI(responses=responses) as mock:
        run_direct(layout, [entry(mock.base_url)])
    pk = parser_key("m1@mock", STAGE1_CONDITION)
    rec = json.loads(layout.cache_path("q1", pk).read_text())
    assert rec["error_class"] == "api_error"
    assert len(mock.requests) == 1   # permanent 4xx — no retry burned on it


def test_refusal_reply_is_refusal_row(tmp_path):
    """Companion to the parse_error fixture fix: this is the text that SHOULD classify as
    refusal (contains the REFUSAL_MARKERS substring "i cannot")."""
    layout = make_run_dir(tmp_path)
    with MockOpenAI(reply_text="I cannot help with that") as mock:
        run_direct(layout, [entry(mock.base_url)])
    pk = parser_key("m1@mock", STAGE1_CONDITION)
    rec = json.loads(layout.cache_path("q1", pk).read_text())
    assert rec["error_class"] == "refusal" and "answer" not in rec


def test_bounded_dispatch_handles_multiple_cells(tmp_path):
    """Exercises the submit/wait/refill loop past its first window — every other test here uses
    a single-item bank and never reaches this path."""
    n = 6
    layout = make_multi_item_run_dir(tmp_path, n)
    with MockOpenAI() as mock:
        summary = run_direct(layout, [entry(mock.base_url)], workers=2)
    assert summary["ok"] == n
    assert len(mock.requests) == n
    pk = parser_key("m1@mock", STAGE1_CONDITION)
    for i in range(n):
        rec = json.loads(layout.cache_path(f"q{i}", pk).read_text())
        assert rec["error_class"] == "none"


def test_max_spend_bounds_overshoot_to_window_size(tmp_path):
    """I2: overshoot past the budget must be bounded by `workers`, not by however many cells a
    naive Executor.map would have already queued. 10 cells, workers=2, each cell costs enough to
    blow a tiny budget on the very first result — assert the abort happens well short of
    processing all 10 cells."""
    n = 10
    workers = 2
    layout = make_multi_item_run_dir(tmp_path, n)
    responses = [{"body": _success_body({"prompt_tokens": 100, "completion_tokens": 10,
                                         "cost": 5.0})}]
    with MockOpenAI(responses=responses) as mock, pytest.raises(RuntimeError, match=r"--max-spend"):
        run_direct(layout, [entry(mock.base_url)], workers=workers, max_spend_usd=0.01)
    # Generous bound: the window can have up to `workers` cells in flight when the first result
    # trips the budget, plus the one that tripped it — never anywhere near all 10 cells.
    assert len(mock.requests) <= workers + 1
    assert len(mock.requests) < n


def test_run_direct_rejects_entry_missing_base_url(tmp_path):
    """I8: OpenAI(base_url=None) would silently target api.openai.com — must never reach that
    constructor call for a misconfigured entry."""
    layout = make_run_dir(tmp_path)
    bad = RegistryEntry(
        id="m1@nobaseurl", shape="vlm-chat", transport="upstream-parser",
        upstream_parser="whatever", api_key_env=None,
        precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
        provenance="Test", release_date="2025-01-01",
    )
    with pytest.raises(ValueError, match="base_url"):
        run_direct(layout, [bad])


# ── F3: cached rows are tallied into cached_ok/cached_error while building the cell list ───────

def test_run_direct_tallies_cached_ok_and_cached_error(tmp_path):
    layout = make_multi_item_run_dir(tmp_path, 2)
    pk = parser_key("m1@mock", STAGE1_CONDITION)
    ok_row = {"qid": "q0", "parser": pk, "answer": {"a": True},
              "field_matches": {"a": True}, "match": True, "error_class": "none"}
    err_row = {"qid": "q1", "parser": pk, "error": "boom", "error_class": "api_error"}
    for qid, rec in [("q0", ok_row), ("q1", err_row)]:
        cpath = layout.cache_path(qid, pk)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(rec))

    with MockOpenAI() as mock:
        summary = run_direct(layout, [entry(mock.base_url)])   # no --force: both cells are cached

    assert mock.requests == []                # neither cell was re-attempted
    assert summary["cached_ok"] == 1
    assert summary["cached_error"] == 1
    assert summary["cached"] == 2              # kept as the sum, for callers reading it alone


def test_run_direct_cached_error_tally_treats_corrupt_row_as_error_not_ok(tmp_path):
    layout = make_run_dir(tmp_path)
    pk = parser_key("m1@mock", STAGE1_CONDITION)
    cpath = layout.cache_path("q1", pk)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text("{not valid json")

    with MockOpenAI() as mock:
        summary = run_direct(layout, [entry(mock.base_url)])

    assert mock.requests == []
    assert summary["cached_ok"] == 0
    assert summary["cached_error"] == 1


# ── F11: catch-all split — only _render_page/ink_coverage failures are render_error ────────────

def test_harness_error_when_scoring_raises_after_a_good_render(tmp_path, monkeypatch):
    """F11: a failure AFTER a perfectly good render/ink-coverage check (here, `score_typed` inside
    `_one` raising) is a harness bug, not a document/render problem — it must land as
    `error_class: "harness_error"`, never the `render_error` bucket that a genuinely bad scan
    gets."""
    import ocr_eval_ext.direct as direct_mod

    layout = make_run_dir(tmp_path)
    monkeypatch.setattr(direct_mod, "score_typed",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("scorer exploded")))
    with MockOpenAI(reply_text='{"a": true}') as mock:
        run_direct(layout, [entry(mock.base_url)])
    pk = parser_key("m1@mock", STAGE1_CONDITION)
    rec = json.loads(layout.cache_path("q1", pk).read_text())
    assert rec["error_class"] == "harness_error"
    assert "scorer exploded" in rec["error"]


def test_render_error_still_used_for_a_genuine_render_failure(tmp_path):
    """Positive control for the F11 split: a document whose PDF doesn't exist at all (the
    original catch-all's canonical case) must still land as `error_class: "render_error"`, not
    the new `harness_error` bucket."""
    layout = make_run_dir(tmp_path)
    (layout.docs_dir / "doc_1.pdf").unlink()   # _render_page can no longer find the source PDF
    with MockOpenAI(reply_text='{"a": true}') as mock:
        run_direct(layout, [entry(mock.base_url)])
    pk = parser_key("m1@mock", STAGE1_CONDITION)
    rec = json.loads(layout.cache_path("q1", pk).read_text())
    assert rec["error_class"] == "render_error"


def test_no_image_row_has_null_image_fields_and_distinct_cache_key(tmp_path):
    layout = make_run_dir(tmp_path)
    with MockOpenAI() as mock:
        run_direct(layout, [entry(mock.base_url)], no_image=True)
    no_image_cond = {**STAGE1_CONDITION, "no_image": True}
    pk_no_image = parser_key("m1@mock", no_image_cond)
    pk_default = parser_key("m1@mock", STAGE1_CONDITION)
    assert pk_no_image != pk_default                        # distinct cache key
    rec = json.loads(layout.cache_path("q1", pk_no_image).read_text())
    assert rec["image_sha"] is None
    assert rec["image_px"] is None
    assert rec["image_bytes"] is None
    assert rec["error_class"] == "none"                      # the call itself still succeeded
    # request never carried an image part
    content = mock.requests[0]["messages"][-1]["content"]
    assert not any(part.get("type") == "image_url" for part in content)
