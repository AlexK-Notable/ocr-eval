import json
from pathlib import Path

import fitz  # pymupdf

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


def entry(base_url: str) -> RegistryEntry:
    return RegistryEntry(
        id="m1@mock", shape="vlm-chat", transport="openai-compat",
        base_url=base_url, model="org/m1", api_key_env=None,
        precision="bf16", weights_licence="mit", provider_tos_commercial="ok",
        provenance="Test", release_date="2025-01-01",
    )


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
