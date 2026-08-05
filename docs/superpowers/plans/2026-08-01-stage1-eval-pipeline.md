# Stage 1 OCR Eval Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fork `extend-hq/realdoc-bench` and extend it to produce checkbox-state, blank-field, and general-extraction numbers — with CIs, baselines, and fail-closed gates — for VLMs via direct QA (`vlm-chat`) and OCR systems via transcribe-then-extract (`transcriber`), hosted or on the local GPU.

**Architecture:** Upstream stays intact (parse → score → report over `RunLayout` run dirs; per-doc transcript files are the transcript cache, per-(qid,parser) JSON files are the extraction cache). All new code lives in a sibling package `ocr_eval_ext/` with its own `ocr-eval` CLI: a registry config, fail-closed preconditions, a direct-QA runner that writes score-cache-compatible records under parser key `vlm__<id>`, boolean/null-restricted metrics, clustered-bootstrap CIs, trivial baselines, and a shape-segregated markdown report. The only upstream file modified is `realdoc_bench/cli.py` (disable `.env` loading).

**Tech Stack:** Python ≥3.11, uv, pytest, typer, pydantic v2, `openai` client (OpenAI-compatible endpoints incl. OpenRouter + local vLLM), pymupdf @ 150 DPI, numpy (bootstrap), upstream's `score_typed`/`build_template`/`gemini_extract` for scoring.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-01-ocr-eval-pipeline-design.md` (rev 2). On conflict, the spec wins; flag the conflict.
- Pins: `harness_commit: fb26a6876481de76dc293f722ab4efa71279904d` · `dataset_revision: 906170ab201d7b8238a32a9115fc66b4b72e0710` (HF `Extend-AI/RealDoc-Bench`).
- API keys via environment only (`bws run` injection). Never in configs, code, JSONL, or git. Upstream `.env` loading must be disabled (Task 1). **[Superseded 2026-08-03 — injection mechanism only:** this plan was authored against a host with Bitwarden Secrets Manager; the current host exports keys in the shell profile instead. The *environment-only* constraint is unchanged and was never implemented in code (`bws` appears in zero lines of code — every key is a plain `os.environ.get()`), so no task output is affected. Operative doc: [`docs/api.md`](../../api.md#keys).**]
- Rendering: pymupdf @ 150 DPI (upstream `DEFAULT_DPI`), PNG, raster-only to every model we control. **Exception (ratified divergence D5):** upstream adapters that upload the PDF (`mistral_ocr_4`) run as-is and are labelled `input: pdf-direct` in the report.
- Stage 1 sampling: `temperature 0.0, top_p 1.0, max_tokens 1024, sample_index 0` for all **vlm-chat** cells. Transcription cells use upstream's `VisionParserBase.max_tokens = 12000` (a full page of markdown does not fit in 1024 — divergence D2).
- Stage 1 output contract: the bank's typed template via `build_template` in the prompt; **no** provider-native structured-output params (mechanism = `schema_prompted`, uniform across providers; `schema_native` is Stage 2).

**Divergence ledger (each ratified against the spec — mirrored in the spec's rev 2.1 appendix):**
- **D1** Per-cell JSON records (upstream's cache shape), not JSONL; upstream `aggregate_results` flattens to `results.json`.
- **D2** `max_tokens 1024` applies to vlm-chat cells only.
- **D3** Rendered-image hash lives in the row (`image_sha`), not the cache key; `report` marks rows `STALE-RENDER` when the current render hash differs and fails without `--allow-stale-render`.
- **D4** Resolved serving identity is not in the cache key; instead the report **hard-fails** if one parser key's rows span more than one `resolved_provider`.
- **D5** Raster-only is enforced for all parsers we implement; upstream pdf-uploading adapters are labelled `pdf-direct` with the text-layer caveat (measured: 2/16 sampled docs carry a text layer).
- **D6** Cost preview: `--dry-run` prices the direct leg as an **estimate** (cells × per-entry token estimate × registry `pricing` rates); the transcriber scoring leg has no automated preview (upstream has no hook) and is budgeted in the runbook. In-run control is `--max-spend` (realized cost; overshoot bounded by ≤ `workers` in-flight calls).
- **D7** Null-gold scoring is **stricter than upstream**: upstream's `score_typed` treats a missing/None answer as matching a null gold (`deep_equal(None, None) → True`). Our `field_outcomes` overrides this at the metrics layer — see Task 4. Published-number comparability is unaffected (upstream rows are compared on upstream's own per-question `match`, which we do not alter).
- **D8** The local-extractor sensitivity check from the ratified extractor decision is deferred to Stage 3 (roadmap item 4); Stage 1's mitigation is the blocking extractor-validation fixture.
- **D9** (Task 4 review, C1 ruling) The headline blank-field number is `acc_over_all` over null-gold fields (fail-safe: collapse and invention both score badly). `hallucination_rate` is redefined as incorrect/n_answered — propensity to invent WHEN answering — and is structurally grouped with `n_answered`/`error_rate` in `null_metrics`'s return; report code consumes the dict whole and never renders the rate alone. Additionally: an answer that is present but not a dict is an error row, and a record bearing an `"error"` key is an error row even if `answer` is present (matches upstream report.py's own predicate).
- **D10** (F11, final review wave) `error_class` gains two more values: `render_error` (a genuine `_render_page`/`ink_coverage` failure on the document itself — missing/corrupt PDF, multi-page swap, blank scan) and `harness_error` (anything else that goes wrong inside `direct.py`'s `do()`, e.g. `_one`'s own request/scoring plumbing raising) — the prior single catch-all mislabelled every non-render failure as a document problem.
- Cardinality preconditions (fail-closed, before any spend): bank == 1,356 items / 3,742 fields / 188 nulls; checkbox bucket (tags `checkbox_state, handdrawn_check, form_checkbox_grid`) == 429 q / 263 docs / 1,117 fields / 258 boolean (165 True / 93 False); `blank_field` == 122 q / 34 nulls; bucket overlap == 40.
- Accuracy-over-all (errors incorrect) is the ranking key; every reported number carries its n.
- Upstream tests must keep passing after every task (`uv run pytest`).
- Python style: ruff, line length 110 (upstream config). Match upstream idiom in `ocr_eval_ext/`.
- Commit after every task (small, descriptive; no co-author lines required by upstream).

**Verified upstream interfaces used throughout** (at the pinned commit — re-verify in Task 1):

```python
# realdoc_bench/evaluate/runs.py
RunLayout.at(run_dir); layout.docs_dir; layout.bank_path; layout.parse_md(parser, stem)
layout.cache_dir; layout.cache_path(qid, parser); layout.results_path; layout.ensure_dirs()

# realdoc_bench/evaluate/score.py
build_template(rf: str, gold_dict: dict) -> str
string_keys(template: str) -> set[str]
score_typed(answer, gold_dict, str_keys) -> tuple[dict, bool]   # (field_matches, all_correct)
gemini_extract(question, template, markdown, *, model=DEFAULT_MODEL) -> Any|None
DEFAULT_MODEL = "gemini-3-flash-preview"; require_api_key()
_ensure_template(item)  # adds item["template"], item["str_keys"]
run_score(layout, parsers, ...); aggregate_results(layout); summarize(records, parsers)

# realdoc_bench/evaluate/parsers/base.py
class ParseResult(BaseModel): markdown, page_count, latency_sec, cost_estimate_usd,
                              pages_processed, provider, version, config_hash, raw
class ParseProvider(ABC): name, version, config_hash(), parse(pdf_path, *, cache_dir=None)
register_parser(name, *, version) ; build(name, **kwargs) ; registry

# realdoc_bench/evaluate/parsers/_vision_base.py
DEFAULT_DPI = 150
_render_pdf_pages(pdf_path, dpi) -> list[bytes]          # PNG bytes per page (pymupdf)
class VisionParserBase(ParseProvider):                    # subclass contract:
    prompt: str; dpi: int; pricing_key: str; max_tokens: int
    def _call_page(self, png_bytes) -> tuple[str, int, int]   # (text, in_tok, out_tok)

# realdoc_bench/evaluate/parsers/cloud_vlm.py
_MARKDOWN_PROMPT  # the transcription contract (☒/☐, **<label>:** <blank>) — import, don't copy

# realdoc_bench/evaluate/download.py
download_dataset(layout, *, repo_id, revision=None, force=False, limit=None)

# Score-cache record shape (evaluate/score.py _worker) — our direct runner MUST write this
# shape (plus extras) so upstream rescoring and aggregation work on our rows:
{"qid": ..., "parser": ..., "source_file": ..., "domain": ...,
 "answer": {...}, "field_matches": {...}, "match": bool}          # success
{"qid": ..., "parser": ..., "source_file": ..., "domain": ..., "error": "..."}  # failure
```

---

### Task 1: Fork bring-up

**Files:**
- Modify: repo root (merge upstream history), `pyproject.toml`, `realdoc_bench/cli.py:246-249`
- Create: `configs/pins.yaml`, `ocr_eval_ext/__init__.py`, `tests_ext/__init__.py`, `tests_ext/test_dotenv_disabled.py`, `.gitignore` (extend upstream's if present)

**Interfaces:**
- Produces: a repo where `uv run pytest` passes upstream tests, `uv run realdoc-bench --help` works, `import ocr_eval_ext` works, and `_env()` is a no-op unless `RDB_ALLOW_DOTENV=1`.

- [ ] **Step 1: Merge upstream at the pinned commit**

```bash
cd ~/repos/ocr-eval
git remote add upstream https://github.com/extend-hq/realdoc-bench.git
git fetch upstream
git merge --allow-unrelated-histories fb26a6876481de76dc293f722ab4efa71279904d \
  -m "Merge upstream extend-hq/realdoc-bench at pinned commit fb26a687"
git log --oneline -3   # expect: merge commit on top of our docs commits
```
No path conflicts are expected (our repo contains only `docs/superpowers/**`). If `README.md` conflicts, keep upstream's and move ours (none exists today).

- [ ] **Step 2: Record the pins**

Create `configs/pins.yaml`:
```yaml
# Both pins are asserted at load (preconditions) and stamped into run metadata.
harness_commit: fb26a6876481de76dc293f722ab4efa71279904d
dataset_revision: 906170ab201d7b8238a32a9115fc66b4b72e0710
dataset_repo: Extend-AI/RealDoc-Bench
```

- [ ] **Step 3: Create the package skeletons BEFORE touching pyproject** (hatchling errors on a declared-but-missing package dir)

```bash
mkdir -p ocr_eval_ext tests_ext
touch ocr_eval_ext/__init__.py tests_ext/__init__.py
```

- [ ] **Step 4: Extend pyproject for our package**

In `pyproject.toml`: add to `[project.scripts]` → `ocr-eval = "ocr_eval_ext.cli:app"`; change `[tool.hatch.build.targets.wheel]` → `packages = ["realdoc_bench", "ocr_eval_ext"]`; add `"tests_ext"` to `[tool.pytest.ini_options] testpaths`. No new runtime deps (openai, numpy, pymupdf, httpx already declared).

- [ ] **Step 4b: Environment + baseline test run**

```bash
uv sync --extra dev
uv run pytest; echo "rc=$?"       # UNPIPED exit status
```
Expected: **85 passed** across 8 test files, no keys, no network (verified at the pinned commit during plan review). This is the regression baseline.

- [ ] **Step 5: Write the failing test for dotenv disablement**

`tests_ext/test_dotenv_disabled.py`:
```python
import os
from pathlib import Path

from realdoc_bench.cli import _env


def test_env_is_noop_without_optin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RDB_ALLOW_DOTENV", raising=False)
    monkeypatch.delenv("SNEAKY_KEY", raising=False)
    Path(".env").write_text("SNEAKY_KEY=leaked\n")
    _env()
    assert "SNEAKY_KEY" not in os.environ


def test_env_loads_with_explicit_optin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RDB_ALLOW_DOTENV", "1")
    monkeypatch.delenv("OPTED_KEY", raising=False)
    Path(".env").write_text("OPTED_KEY=ok\n")
    _env()
    assert os.environ.get("OPTED_KEY") == "ok"
```

- [ ] **Step 6: Run it to make sure the first test fails**

Run: `uv run pytest tests_ext/test_dotenv_disabled.py -v`
Expected: `test_env_is_noop_without_optin` FAILS (upstream `_env()` loads `.env` unconditionally).

- [ ] **Step 7: Modify `_env()` in `realdoc_bench/cli.py`**

```python
def _env() -> None:
    # Fork policy: keys come from the environment only (see spec §Keys and secrets).
    # Upstream's .env loading is opt-in via RDB_ALLOW_DOTENV=1.
    if os.environ.get("RDB_ALLOW_DOTENV") != "1":
        return
    load_dotenv(Path.cwd() / ".env.local")
    load_dotenv(Path.cwd() / ".env")
```
`import os` is **not** currently imported in upstream cli.py — add it to the import block.

- [ ] **Step 8: Run tests — both new tests and full suite pass**

Run: `uv run pytest tests_ext/test_dotenv_disabled.py -v && uv run pytest`
Expected: PASS everywhere; upstream count unchanged.

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "chore: fork bring-up — pins, ocr_eval_ext package, dotenv opt-in"
```

---

### Task 2: Registry config

**Files:**
- Create: `ocr_eval_ext/config.py`, `configs/registry.yaml`, `tests_ext/test_config.py`

**Interfaces:**
- Produces:
```python
class RegistryEntry(BaseModel):
    id: str                      # e.g. "qwen3-vl-8b@openrouter"
    shape: Literal["vlm-chat", "transcriber"]
    transport: Literal["openai-compat", "upstream-parser"]
    base_url: str | None = None          # openai-compat only
    model: str | None = None             # served model id (openai-compat only)
    upstream_parser: str | None = None   # upstream registry name (upstream-parser only)
    api_key_env: str | None = None       # name of env var holding the key; value never stored
    precision: Literal["bf16", "fp8-vllm", "q8-gguf", "provider-default"]
    weights_licence: str
    provider_tos_commercial: Literal["ok", "blocked", "conditional"]
    tos_note: str = ""
    provenance: str
    release_date: str                     # YYYY-MM-DD; contamination flag if > 2026-05-24 (HF createdAt)
    provider_pin: dict | None = None      # OpenRouter: {"order": [...], "allow_fallbacks": False}
    local: bool = False                   # True → serialize cells, preflight required, cost renders "n/a"
    promptable: bool = True               # False for endpoints that accept no prompt (mistral_ocr_4)
    pricing: dict | None = None           # {"input_per_mtok": X, "output_per_mtok": Y} — used by
                                          # --dry-run estimates and as realized-cost fallback when the
                                          # provider returns no usage.cost (vLLM). None → cost "n/a"

def load_registry(path: Path) -> list[RegistryEntry]      # validates; raises on duplicate ids
def get_entry(entries: list[RegistryEntry], id: str) -> RegistryEntry
CONTAMINATION_CUTOFF = "2026-05-24"  # HF createdAt of Extend-AI/RealDoc-Bench (rev-2.1 note: spec cited lastModified 06-03)
```

- [ ] **Step 1: Write failing tests**

`tests_ext/test_config.py`:
```python
from pathlib import Path

import pytest

from ocr_eval_ext.config import CONTAMINATION_CUTOFF, RegistryEntry, get_entry, load_registry

MINIMAL = """
- id: m1@host
  shape: vlm-chat
  transport: openai-compat
  base_url: https://example.com/v1
  model: org/m1
  api_key_env: EXAMPLE_KEY
  precision: provider-default
  weights_licence: apache-2.0
  provider_tos_commercial: ok
  provenance: Example Org
  release_date: "2026-01-01"
"""


def test_load_minimal(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(MINIMAL)
    entries = load_registry(p)
    assert entries[0].id == "m1@host"
    assert entries[0].local is False


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(MINIMAL + MINIMAL)
    with pytest.raises(ValueError, match="duplicate"):
        load_registry(p)


def test_transport_field_consistency(tmp_path):
    bad = MINIMAL.replace("transport: openai-compat", "transport: upstream-parser")
    p = tmp_path / "r.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError, match="upstream_parser"):
        load_registry(p)


def test_contamination_flag_logic():
    e = RegistryEntry(
        id="x@y", shape="vlm-chat", transport="openai-compat",
        base_url="u", model="m", precision="bf16", weights_licence="mit",
        provider_tos_commercial="ok", provenance="p", release_date="2026-07-01",
    )
    assert e.release_date > CONTAMINATION_CUTOFF   # ISO strings compare correctly


def test_get_entry_unknown_raises():
    with pytest.raises(KeyError):
        get_entry([], "nope")
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests_ext/test_config.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `ocr_eval_ext/config.py`**

```python
"""Registry of (model, serving) pairs. One entry per pair — never merged."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

CONTAMINATION_CUTOFF = "2026-05-24"  # HF createdAt of Extend-AI/RealDoc-Bench (rev-2.1 note: spec cited lastModified 06-03)  # RealDoc-Bench HF publication date


class RegistryEntry(BaseModel):
    id: str
    shape: Literal["vlm-chat", "transcriber"]
    transport: Literal["openai-compat", "upstream-parser"]
    base_url: str | None = None
    model: str | None = None
    upstream_parser: str | None = None
    api_key_env: str | None = None
    precision: Literal["bf16", "fp8-vllm", "q8-gguf", "provider-default"]
    weights_licence: str
    provider_tos_commercial: Literal["ok", "blocked", "conditional"]
    tos_note: str = ""
    provenance: str
    release_date: str
    provider_pin: dict | None = None
    local: bool = False

    @model_validator(mode="after")
    def _check_transport_fields(self) -> "RegistryEntry":
        if self.transport == "openai-compat" and not (self.base_url and self.model):
            raise ValueError(f"{self.id}: openai-compat requires base_url and model")
        if self.transport == "upstream-parser" and not self.upstream_parser:
            raise ValueError(f"{self.id}: upstream-parser transport requires upstream_parser name")
        return self

    @property
    def contaminated(self) -> bool:
        return self.release_date > CONTAMINATION_CUTOFF


def load_registry(path: Path) -> list[RegistryEntry]:
    entries = [RegistryEntry(**raw) for raw in yaml.safe_load(path.read_text())]
    ids = [e.id for e in entries]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate registry ids: {sorted(dupes)}")
    return entries


def get_entry(entries: list[RegistryEntry], id: str) -> RegistryEntry:
    for e in entries:
        if e.id == id:
            return e
    raise KeyError(f"registry id not found: {id}")
```

- [ ] **Step 4: Run tests** — `uv run pytest tests_ext/test_config.py -v` → PASS.

- [ ] **Step 5: Seed `configs/registry.yaml`** with the Stage 1 set (values from the survey; ToS notes from `gate3-conclusion-ranking.md`):

```yaml
# One entry per (model, serving) pair. api_key_env names the env var; values never live here.
- id: qwen3-vl-8b@openrouter
  shape: vlm-chat
  transport: openai-compat
  base_url: https://openrouter.ai/api/v1
  model: qwen/qwen3-vl-8b-instruct
  api_key_env: OPENROUTER_API_KEY
  precision: provider-default          # resolved provider recorded per row
  provider_pin: {order: ["Alibaba"], allow_fallbacks: false}   # verify provider slug at first run
  weights_licence: apache-2.0
  provider_tos_commercial: ok
  provenance: Alibaba
  release_date: "2025-10-01"

- id: qwen3.5-9b@openrouter
  shape: vlm-chat
  transport: openai-compat
  base_url: https://openrouter.ai/api/v1
  model: qwen/qwen3.5-9b
  api_key_env: OPENROUTER_API_KEY
  precision: provider-default
  provider_pin: {order: ["Alibaba"], allow_fallbacks: false}
  weights_licence: apache-2.0
  provider_tos_commercial: ok
  provenance: Alibaba
  release_date: "2026-02-01"           # post-dataset → contamination flag expected

- id: gemini-3.5-flash@google
  shape: transcriber
  transport: upstream-parser
  upstream_parser: gemini_3_5_flash
  api_key_env: GEMINI_API_KEY
  precision: provider-default
  weights_licence: closed              # ceiling anchor — non-candidate, flagged in report
  provider_tos_commercial: ok
  provenance: Google
  release_date: "2026-03-01"

- id: qwen3-vl-32b@openrouter            # 3rd hosted VLM (input-cheaper than the 8B — survey anomaly)
  shape: vlm-chat
  transport: openai-compat
  base_url: https://openrouter.ai/api/v1
  model: qwen/qwen3-vl-32b-instruct
  api_key_env: OPENROUTER_API_KEY
  precision: provider-default
  provider_pin: {order: ["Alibaba"], allow_fallbacks: false}
  pricing: {input_per_mtok: 0.104, output_per_mtok: 0.416}
  weights_licence: apache-2.0
  provider_tos_commercial: ok
  provenance: Alibaba
  release_date: "2025-10-01"

- id: qwen3-vl-8b@openrouter-transcriber  # the direct-vs-two-stage CALIBRATION PAIR (DoD #3)
  shape: transcriber
  transport: openai-compat
  base_url: https://openrouter.ai/api/v1
  model: qwen/qwen3-vl-8b-instruct
  api_key_env: OPENROUTER_API_KEY
  precision: provider-default
  provider_pin: {order: ["Alibaba"], allow_fallbacks: false}
  pricing: {input_per_mtok: 0.117, output_per_mtok: 0.455}
  weights_licence: apache-2.0
  provider_tos_commercial: ok
  provenance: Alibaba
  release_date: "2025-10-01"

- id: gemini-3.5-flash@google-vlmchat     # frontier CEILING ANCHOR for Section A (vlm-chat)
  shape: vlm-chat
  transport: openai-compat
  base_url: https://generativelanguage.googleapis.com/v1beta/openai
  model: gemini-3.5-flash
  api_key_env: GEMINI_API_KEY
  precision: provider-default
  weights_licence: closed                 # non-candidate, flagged in report
  provider_tos_commercial: ok
  provenance: Google
  release_date: "2026-03-01"

- id: mistral-ocr@mistral
  shape: transcriber
  transport: upstream-parser
  upstream_parser: mistral_ocr_4          # upstream registers mistral_ocr_4, NOT mistral_ocr
  promptable: false                       # PDF-upload endpoint, no prompt — pdf-direct label in report
  api_key_env: MISTRAL_API_KEY
  precision: provider-default
  weights_licence: closed
  provider_tos_commercial: ok
  provenance: Mistral
  release_date: "2026-05-01"

- id: glm-ocr@local-vllm
  shape: transcriber
  transport: openai-compat
  base_url: http://localhost:8000/v1
  model: zai-org/GLM-OCR               # verify exact HF id in Task 8
  api_key_env: null
  precision: bf16
  weights_licence: mit
  provider_tos_commercial: ok          # self-hosted — no provider ToS
  provenance: Zhipu
  release_date: "2026-04-01"
  local: true

- id: dots-ocr@local-vllm
  shape: transcriber
  transport: openai-compat
  base_url: http://localhost:8000/v1
  model: rednote-hilab/dots.ocr        # verify exact HF id in Task 8
  api_key_env: null
  precision: bf16
  weights_licence: mit
  provider_tos_commercial: ok
  provenance: RedNote
  release_date: "2025-08-01"           # pre-dataset → reproduction target
  local: true
```
Release dates above are placeholders to the month — **Step 6 verifies each against the HF model card** and corrects; that verification is part of this task, not deferred.

- [ ] **Step 6: Verify registry facts** — for each entry: exact HF model id, release date (HF `createdAt`), licence tag; for every `upstream_parser` value, confirm it appears in `uv run realdoc-bench evaluate list` (name mismatches exit with BadParameter at parse time). Correct the YAML. Add a test asserting `load_registry(Path("configs/registry.yaml"))` parses, ids are unique, and DoD categories are satisfiable: ≥3 hosted `vlm-chat`, ≥1 `local: true` transcriber, ≥1 `upstream-parser` transcriber, ≥1 closed-weights anchor per shape, and at least one (model) present under both shapes (calibration pair).

- [ ] **Step 7: Run full suite** — `uv run pytest` → PASS.

- [ ] **Step 8: Commit** — `git add -A && git commit -m "feat: registry config with ToS/licence/contamination fields"`

---

### Task 3: Fail-closed preconditions

**Files:**
- Create: `ocr_eval_ext/preconditions.py`, `tests_ext/test_preconditions.py`

**Interfaces:**
- Consumes: bank items (list of dicts with `question_id`, `source_file`, `capabilities`, `gold_dict`).
- Produces:
```python
CHECKBOX_TAGS = ("checkbox_state", "handdrawn_check", "form_checkbox_grid")  # upstream bucket
EXPECTED = {...}   # the constants below
class PreconditionError(RuntimeError): ...
def check_bank(items: list[dict]) -> dict          # returns measured counts; raises PreconditionError
def items_with_tags(items, tags) -> list[dict]
def boolean_fields(items) -> list[tuple[str, str, bool, str]]        # (qid, key, gold, source_file)
def null_fields(items) -> list[tuple[str, str, None, str]]           # (qid, key, None, source_file)
    # SAME 4-tuple shape as boolean_fields — both feed metrics.field_outcomes directly
def assert_single_page(pdf_path: Path) -> int       # raises unless page_count == 1
def ink_coverage(png_bytes: bytes) -> float         # fraction of non-white pixels
```

- [ ] **Step 1: Write failing tests** (synthetic mini-bank; the real-bank check is a CLI command, Task 7)

`tests_ext/test_preconditions.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests_ext/test_preconditions.py -v` → FAIL.

- [ ] **Step 3: Implement `ocr_eval_ext/preconditions.py`**

```python
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
```

- [ ] **Step 4: Run tests** — PASS. **Step 5: Full suite** — PASS. **Step 6: Commit** — `git commit -m "feat: fail-closed bank cardinality preconditions + page/ink guards"`

---

### Task 4: Boolean/null-restricted metrics

**Files:**
- Create: `ocr_eval_ext/metrics.py`, `tests_ext/test_metrics.py`

**Interfaces:**
- Consumes: score-cache records (upstream shape: `field_matches: dict[str,bool]`, `match: bool`, `answer` present or `error`), `boolean_fields`/`null_fields` from Task 3.
- Produces:
```python
@dataclass
class FieldOutcome:
    qid: str; key: str; doc: str; gold: object
    status: Literal["correct", "incorrect", "error"]   # error = no scorable answer for the question

def field_outcomes(records: dict[str, dict], fields: list[tuple]) -> list[FieldOutcome]
    # records keyed by qid → cache record. Status rules (STRICTER than upstream scoring — D7):
    #   qid absent, record bears "error", or record's answer is None → "error"
    #   null gold: "correct" ONLY if the key is PRESENT in answer with explicit None —
    #     key-absent is "incorrect" (upstream's deep_equal(None, None) would call it correct;
    #     that rewards extractor collapse with a perfect no-hallucination score)
    #   non-null gold: field_matches[key] truthy → "correct", else "incorrect"

@dataclass
class MetricBlock:
    n: int; n_answered: int
    acc_over_all: float          # ranking key: errors count incorrect
    acc_over_answered: float
    error_rate: float

def checkbox_metrics(outcomes) -> dict   # MetricBlock + polarity: {"checked": MetricBlock, "unchecked": ...}
                                          # + confusion {"tt": int, "tf": int, "ft": int, "ff": int, "err": int}
def null_metrics(outcomes) -> dict       # MetricBlock + hallucination_rate (answered-non-null / n)
def baseline_rows(fields) -> dict        # {"always_true": acc, "always_false": acc, "majority": acc,
                                          #  "class_balance": {"true": int, "false": int}}
```

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run to verify failure.** — FAIL (module missing).

- [ ] **Step 3: Implement `ocr_eval_ext/metrics.py`**

```python
"""Boolean- and null-restricted metrics over upstream score-cache records.
Rule: accuracy-over-all (errors incorrect) is the ranking key. Every block carries n."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class FieldOutcome:
    qid: str
    key: str
    doc: str
    gold: object
    status: Literal["correct", "incorrect", "error"]


@dataclass
class MetricBlock:
    n: int
    n_answered: int
    acc_over_all: float
    acc_over_answered: float
    error_rate: float


def _block(outcomes: list[FieldOutcome]) -> MetricBlock:
    n = len(outcomes)
    answered = [o for o in outcomes if o.status != "error"]
    correct = sum(1 for o in outcomes if o.status == "correct")
    return MetricBlock(
        n=n,
        n_answered=len(answered),
        acc_over_all=correct / n if n else 0.0,
        acc_over_answered=correct / len(answered) if answered else 0.0,
        error_rate=(n - len(answered)) / n if n else 0.0,
    )


def field_outcomes(records: dict[str, dict], fields: list[tuple]) -> list[FieldOutcome]:
    out = []
    for qid, key, gold, doc in fields:
        rec = records.get(qid)
        if not rec or "answer" not in rec or rec.get("answer") is None:
            status = "error"                       # no scorable answer at all (D7)
        elif gold is None:
            # Null gold: correct ONLY on key-present explicit None. Upstream's
            # deep_equal(None, None) would score key-absent/None-answer as correct,
            # which rewards extractor collapse on the hallucination metric (D7).
            ans = rec["answer"]
            status = "correct" if (isinstance(ans, dict) and key in ans and ans[key] is None) \
                else "incorrect"
        else:
            status = "correct" if rec.get("field_matches", {}).get(key) else "incorrect"
        out.append(FieldOutcome(qid, key, doc, gold, status))
    return out


def checkbox_metrics(outcomes: list[FieldOutcome]) -> dict:
    checked = [o for o in outcomes if o.gold is True]
    unchecked = [o for o in outcomes if o.gold is False]
    conf = {"tt": 0, "tf": 0, "ft": 0, "ff": 0, "err": 0}
    for o in outcomes:
        if o.status == "error":
            conf["err"] += 1
        elif o.gold is True:
            conf["tt" if o.status == "correct" else "tf"] += 1
        else:
            conf["ff" if o.status == "correct" else "ft"] += 1
    return {"overall": _block(outcomes), "confusion": conf,
            "polarity": {"checked": _block(checked), "unchecked": _block(unchecked)}}


def null_metrics(outcomes: list[FieldOutcome]) -> dict:
    hall = sum(1 for o in outcomes if o.status == "incorrect")   # scored wrong on a null gold
    n = len(outcomes)
    return {"overall": _block(outcomes), "hallucination_rate": hall / n if n else 0.0}


def baseline_rows(fields: list[tuple]) -> dict:
    golds = [g for _, _, g, _ in fields]
    t = sum(1 for g in golds if g is True)
    f = len(golds) - t
    n = len(golds)
    always_true = t / n if n else 0.0
    always_false = f / n if n else 0.0        # computed from counts, not 1-x (float exactness)
    return {"always_true": always_true, "always_false": always_false,
            "majority": max(always_true, always_false),
            "class_balance": {"true": t, "false": f}}
```

- [ ] **Step 4: Run tests** — PASS. **Step 5: Full suite** — PASS. **Step 6: Commit** — `git commit -m "feat: boolean/null-restricted metrics with polarity split and trivial baselines"`

---

### Task 5: Document-clustered bootstrap CIs

**Files:**
- Create: `ocr_eval_ext/stats.py`, `tests_ext/test_stats.py`

**Interfaces:**
- Consumes: `list[FieldOutcome]` (Task 4).
- Produces:
```python
def cluster_bootstrap_ci(outcomes, *, iters=2000, seed=0, alpha=0.05) -> tuple[float, float]
    # resamples DOCS with replacement; statistic = acc_over_all over resampled fields
def paired_delta_ci(a, b, *, iters=2000, seed=0, alpha=0.05) -> tuple[float, float]
    # a, b: outcomes for two models over the SAME fields; resamples docs; statistic = acc_a - acc_b
def separable(delta_ci: tuple[float, float]) -> bool     # True iff 0 outside the interval
```

- [ ] **Step 1: Write failing tests**

```python
from ocr_eval_ext.metrics import FieldOutcome
from ocr_eval_ext.stats import cluster_bootstrap_ci, paired_delta_ci, separable


def mk(doc, status, qid="q", key="k", gold=True):
    return FieldOutcome(qid, key, doc, gold, status)


def test_ci_is_deterministic_and_ordered():
    outs = [mk(f"d{i}", "correct" if i % 2 else "incorrect") for i in range(40)]
    lo1, hi1 = cluster_bootstrap_ci(outs, seed=7)
    lo2, hi2 = cluster_bootstrap_ci(outs, seed=7)
    assert (lo1, hi1) == (lo2, hi2) and lo1 < 0.5 < hi1


def test_clustered_wider_than_degenerate_clusters():
    # 10 docs × 10 perfectly correlated fields vs 100 independent docs — same 50% accuracy
    correlated = [mk(f"d{i//10}", "correct" if (i // 10) % 2 else "incorrect") for i in range(100)]
    independent = [mk(f"d{i}", "correct" if i % 2 else "incorrect") for i in range(100)]
    wc = cluster_bootstrap_ci(correlated, seed=1)
    wi = cluster_bootstrap_ci(independent, seed=1)
    assert (wc[1] - wc[0]) > (wi[1] - wi[0])   # correlation must widen the interval


def test_paired_delta_and_separability():
    a = [mk(f"d{i}", "correct") for i in range(30)]
    b = [mk(f"d{i}", "correct" if i < 15 else "incorrect") for i in range(30)]
    ci = paired_delta_ci(a, b, seed=3)
    assert separable(ci) and ci[0] > 0
    ci_same = paired_delta_ci(a, a, seed=3)
    assert not separable(ci_same)
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement `ocr_eval_ext/stats.py`**

```python
"""Document-clustered bootstrap. Questions cluster on documents (429 q / 263 docs);
naive binomial intervals are too tight. Percentile method, docs resampled with replacement."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ocr_eval_ext.metrics import FieldOutcome


def _by_doc(outcomes: list[FieldOutcome]) -> dict[str, list[FieldOutcome]]:
    d: dict[str, list[FieldOutcome]] = defaultdict(list)
    for o in outcomes:
        d[o.doc].append(o)
    return d


def _acc(outs: list[FieldOutcome]) -> float:
    return sum(1 for o in outs if o.status == "correct") / len(outs) if outs else 0.0


def cluster_bootstrap_ci(outcomes, *, iters=2000, seed=0, alpha=0.05):
    docs = list(_by_doc(outcomes).items())
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(iters):
        idx = rng.integers(0, len(docs), len(docs))
        sample = [o for i in idx for o in docs[i][1]]
        stats.append(_acc(sample))
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def paired_delta_ci(a, b, *, iters=2000, seed=0, alpha=0.05):
    da, db = _by_doc(a), _by_doc(b)
    docs = sorted(set(da) | set(db))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(iters):
        idx = rng.integers(0, len(docs), len(docs))
        sa = [o for i in idx for o in da.get(docs[i], [])]
        sb = [o for i in idx for o in db.get(docs[i], [])]
        stats.append(_acc(sa) - _acc(sb))
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def separable(delta_ci: tuple[float, float]) -> bool:
    lo, hi = delta_ci
    return lo > 0 or hi < 0
```

- [ ] **Step 4: Run tests** — PASS. **Step 5: Full suite.** **Step 6: Commit** — `git commit -m "feat: document-clustered bootstrap CIs and paired separability"`

---

### Task 6: Direct (vlm-chat) runner

**Files:**
- Create: `ocr_eval_ext/direct.py`, `tests_ext/test_direct.py`, `tests_ext/mock_openai.py`

**Interfaces:**
- Consumes: `RegistryEntry` (Task 2), upstream `build_template`/`string_keys`/`score_typed`/`_render_pdf_pages`, `RunLayout`, `ink_coverage`/`assert_single_page` (Task 3).
- Produces:
```python
STAGE1_CONDITION = {"preprocess": "raw", "output_contract": "schema_prompted",
                    "render": {"engine": "pymupdf", "dpi": 150},
                    "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024},
                    "sample_index": 0}
def condition_hash(condition: dict) -> str                     # sha256 of canonical JSON, first 12 hex
def parser_key(entry_id: str, condition: dict) -> str          # f"vlm__{entry_id}__{condition_hash(...)}"
                                                               # (this is the 'parser' name in cache paths)
def direct_prompt(question: str, template: str) -> str
def run_direct(layout, entries, *, bank_path=None, condition=STAGE1_CONDITION,
               workers=8, force=False, dry_run=False, max_spend_usd=None,
               limit=None, no_image=False) -> dict              # summary counts
```
- Cache record = upstream score-record shape **plus** `{"raw_response", "usage", "resolved_provider", "retrieved_at", "condition", "image_sha", "error_class"}` — extras are additive so upstream `_worker` rescoring and `aggregate_results` still work on these files.

- [ ] **Step 1: Write the mock OpenAI server fixture** (`tests_ext/mock_openai.py`)

```python
"""Minimal OpenAI-compatible /chat/completions stub for tests. Configurable reply text."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockOpenAI:
    def __init__(self, reply_text='{"a": true}'):
        self.reply_text = reply_text
        self.requests: list[dict] = []
        handler_self = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                handler_self.requests.append(body)
                resp = {
                    "id": "cmpl-1", "model": body.get("model", "m"),
                    "provider": "MockProvider",
                    "choices": [{"message": {"role": "assistant",
                                             "content": handler_self.reply_text}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                }
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):  # silence
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
```

- [ ] **Step 2: Write failing tests** (`tests_ext/test_direct.py`)

```python
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
    with MockOpenAI(reply_text="I cannot help with that") as mock:
        run_direct(layout, [entry(mock.base_url)])
    pk = parser_key("m1@mock", STAGE1_CONDITION)
    rec = json.loads(layout.cache_path("q1", pk).read_text())
    assert rec["error_class"] == "parse_error" and "answer" not in rec


def test_dry_run_prices_without_calls(tmp_path):
    layout = make_run_dir(tmp_path)
    with MockOpenAI() as mock:
        s = run_direct(layout, [entry(mock.base_url)], dry_run=True)
        assert mock.requests == [] and s["cells"] == 1
```

- [ ] **Step 3: Run to verify failure.**

- [ ] **Step 4: Implement `ocr_eval_ext/direct.py`**

```python
"""Direct QA (vlm-chat): page image + question + typed template → JSON answer.
Writes upstream-score-cache-compatible records under parser key vlm__<id>__<cond>,
so upstream rescoring/aggregation work unchanged on our rows."""
from __future__ import annotations

import base64
import concurrent.futures
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

from openai import OpenAI

from ocr_eval_ext.config import RegistryEntry
from ocr_eval_ext.preconditions import assert_single_page, ink_coverage
from realdoc_bench.evaluate.parsers._vision_base import _render_pdf_pages
from realdoc_bench.evaluate.runs import RunLayout
from realdoc_bench.evaluate.score import _ensure_template, score_typed

STAGE1_CONDITION = {
    "preprocess": "raw",
    "output_contract": "schema_prompted",
    "render": {"engine": "pymupdf", "dpi": 150},
    "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024, "seed": None},
    "sample_index": 0,
    "no_image": False,     # in the dict from commit one — flipping a VALUE, never adding a key
}

SYSTEM = ("You answer questions about a scanned document page. "
          "Return JSON with exactly the keys and value types of the template. "
          "Booleans must be true/false. If a field is empty or not filled in, return null for it. "
          "Do not guess values that are not visible on the page.")

REFUSAL_MARKERS = ("i cannot", "i can't", "i'm unable", "i am unable", "cannot assist",
                   "can't help", "against my", "i won't")

MAX_RETRIES = 4
BACKOFF_BASE_SEC = 2.0


def condition_hash(condition: dict) -> str:
    return hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest()[:12]


def parser_key(entry_id: str, condition: dict) -> str:
    return f"vlm__{entry_id}__{condition_hash(condition)}"


def direct_prompt(question: str, template: str) -> str:
    return (f"Question:\n{question}\n\n"
            f"Answer template (return JSON with exactly these keys and value types):\n{template}")


def _extract_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if 0 <= s < e:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                return None
    return None


def _render_page(layout: RunLayout, stem: str, condition: dict, png_cache: Path) -> bytes:
    """Atomic write (os.replace) — write_bytes truncates first, and a concurrent reader seeing a
    half-written file kills the run (reproduced under 8-thread stress during plan review)."""
    import os
    import uuid

    dpi = condition["render"]["dpi"]
    pre = condition["preprocess"]
    png_cache.mkdir(parents=True, exist_ok=True)
    p = png_cache / f"{stem}@{dpi}@{pre}.png"     # preprocess in the name — Stage 2 deskew must not
    if p.exists():                                 # silently reuse raw renders
        data = p.read_bytes()
        if data:
            return data
    pdf = layout.docs_dir / f"{stem}.pdf"
    assert_single_page(pdf)
    pages = _render_pdf_pages(pdf, dpi)
    tmp = p.with_name(p.name + f".tmp.{uuid.uuid4().hex}")
    tmp.write_bytes(pages[0])
    os.replace(tmp, p)
    return pages[0]


def _one(client: OpenAI, entry: RegistryEntry, item: dict, png: bytes,
         condition: dict, png_dims: tuple[int, int]) -> dict:
    prompt = direct_prompt(item["question"], item["template"])
    base = {"qid": item["question_id"], "parser": parser_key(entry.id, condition),
            "source_file": item["source_file"], "domain": item.get("domain", ""),
            "condition": condition, "image_sha": hashlib.sha256(png).hexdigest(),
            "image_px": list(png_dims), "image_bytes": len(png),
            "prompt_sha": hashlib.sha256((SYSTEM + "\x00" + prompt).encode()).hexdigest()[:12],
            "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    content: list[dict] = [{"type": "text", "text": prompt}]
    if not condition.get("no_image"):
        b64 = base64.b64encode(png).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    extra_body = {}
    if entry.provider_pin:
        extra_body["provider"] = entry.provider_pin
    t0 = time.perf_counter()
    resp = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=entry.model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": content}],
                temperature=condition["sampling"]["temperature"],
                top_p=condition["sampling"]["top_p"],
                max_tokens=condition["sampling"]["max_tokens"],
                extra_body=extra_body or None,
            )
            break
        except Exception as e:  # noqa: BLE001 — bounded retry, then per-cell isolation
            retry_after = getattr(getattr(e, "response", None), "headers", {}) or {}
            wait = float(retry_after.get("retry-after") or BACKOFF_BASE_SEC * (2 ** attempt))
            if attempt == MAX_RETRIES - 1:
                return {**base, "error": str(e)[:300], "error_class": "api_error"}
            time.sleep(wait)
    raw = resp.model_dump()
    text = (resp.choices[0].message.content or "").strip()
    usage = raw.get("usage") or {}    # OpenRouter now always includes usage details (incl. cost)
    common = {**base, "raw_response": text, "usage": usage,
              "resolved_provider": raw.get("provider") or "",
              "latency_sec": time.perf_counter() - t0}
    if not text:
        return {**common, "error": "empty response", "error_class": "empty"}
    ans = _extract_json(text)
    if ans is None:
        cls = "refusal" if any(m in text.lower() for m in REFUSAL_MARKERS) else "parse_error"
        return {**common, "error": f"unparseable response ({cls})", "error_class": cls}
    fm, allc = score_typed(ans, item["gold_dict"], item["str_keys"])
    return {**common, "answer": ans, "field_matches": fm, "match": allc, "error_class": "none"}


# Deliberate divergence from spec (documented): the rendered-image hash lives in the ROW
# (image_sha), not the cache key — the render is deterministic given (pinned dataset revision,
# pymupdf version, dpi), so drift is detectable rather than auto-invalidating: `rescore` and
# `report` warn when a row's image_sha no longer matches the current render.


def run_direct(layout: RunLayout, entries: list[RegistryEntry], *, bank_path: Path | None = None,
               condition: dict = STAGE1_CONDITION, workers: int = 8, force: bool = False,
               dry_run: bool = False, max_spend_usd: float | None = None,
               limit: int | None = None, no_image: bool = False) -> dict:
    import os

    bank = json.loads((bank_path or layout.bank_path).read_text())
    items = bank["items"][:limit] if limit else bank["items"]
    for it in items:
        _ensure_template(it)
    cond = dict(condition)
    if no_image:
        cond = {**cond, "no_image": True}

    cells = []
    for e in entries:
        pk = parser_key(e.id, cond)
        for it in items:
            cpath = layout.cache_path(it["question_id"], pk)
            if force or not cpath.exists():
                cells.append((e, it, cpath))
    if dry_run:
        # ESTIMATE (labelled so in output): ~1,600 image tokens/page + prompt ~400 in, ~120 out.
        est = 0.0
        for e, _it, _c in cells:
            if e.pricing:
                est += (2000 / 1e6) * e.pricing["input_per_mtok"] \
                     + (120 / 1e6) * e.pricing["output_per_mtok"]
        return {"cells": len(cells), "entries": len(entries), "items": len(items),
                "estimated_usd": round(est, 2), "estimate_note": "±2x — token counts are guesses"}

    png_cache = layout.root / "docs_png"
    summary = {"ok": 0, "error": 0, "cached": len(entries) * len(items) - len(cells)}
    spend = 0.0
    clients = {}
    for e in entries:
        key = os.environ.get(e.api_key_env) if e.api_key_env else "local"
        if e.api_key_env and not key:
            raise RuntimeError(f"{e.id}: env var {e.api_key_env} not set")
        clients[e.id] = OpenAI(base_url=e.base_url, api_key=key or "none")

    local_cells = [c for c in cells if c[0].local]
    hosted_cells = [c for c in cells if not c[0].local]

    def do(cell):
        e, it, cpath = cell
        try:
            png = _render_page(layout, it["source_file"], cond, png_cache)
            if ink_coverage(png) < 0.001:
                rec = {"qid": it["question_id"], "parser": parser_key(e.id, cond),
                       "source_file": it["source_file"], "domain": it.get("domain", ""),
                       "error": "blank render", "error_class": "render_error"}
            else:
                import io as _io

                from PIL import Image as _Img
                dims = _Img.open(_io.BytesIO(png)).size
                rec = _one(clients[e.id], e, it, png, cond, dims)
        except Exception as exc:  # noqa: BLE001 — one bad doc costs one cell, never the run
            rec = {"qid": it["question_id"], "parser": parser_key(e.id, cond),
                   "source_file": it["source_file"], "domain": it.get("domain", ""),
                   "error": str(exc)[:300], "error_class": "render_error"}
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(rec, ensure_ascii=False))
        return rec

    def track(rec):
        # Runs single-threaded in the consumer loop — no lock needed. Raising here cancels
        # pending futures via the pool's generator cleanup; overshoot is bounded by ≤ workers
        # in-flight calls (measured 12/50 cells at workers=8 during plan review).
        nonlocal spend
        summary["ok" if rec.get("error_class") == "none" else "error"] += 1
        u = rec.get("usage") or {}
        cost = u.get("cost")
        if cost is None:                       # vLLM/local: no cost field — token×rate fallback
            e = next((x for x in entries if rec["parser"].startswith(f"vlm__{x.id}__")), None)
            if e and e.pricing:
                cost = (u.get("prompt_tokens", 0) / 1e6) * e.pricing["input_per_mtok"] \
                     + (u.get("completion_tokens", 0) / 1e6) * e.pricing["output_per_mtok"]
        spend += cost or 0.0
        if max_spend_usd is not None and spend > max_spend_usd:
            raise RuntimeError(f"--max-spend {max_spend_usd} exceeded (realized {spend:.2f})")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for rec in pool.map(do, hosted_cells):
            track(rec)
    for cell in local_cells:                   # one local model resident at a time — serialize
        track(do(cell))
    return summary
```

- [ ] **Step 5: Run tests** — `uv run pytest tests_ext/test_direct.py -v` → PASS. Note: `max_tokens` (not `max_completion_tokens`) is correct for OpenRouter/vLLM compatibility; the mock asserts it arrives.

- [ ] **Step 6: Full suite.** **Step 7: Commit** — `git commit -m "feat: direct vlm-chat runner with cache-compatible records, provider pin, max-spend"`

---

### Task 7: `ocr-eval` CLI — verify, direct, selftest

**Files:**
- Create: `ocr_eval_ext/cli.py`, `ocr_eval_ext/selftest.py`, `tests_ext/test_cli.py`, `tests_ext/test_selftest.py`

**Interfaces:**
- Produces (typer app `app`, installed as `ocr-eval` via Task 1 pyproject):
  - `ocr-eval verify --run-dir D` → preconditions vs the real bank (+ pins check vs `configs/pins.yaml` + git); exit 1 on any mismatch.
  - `ocr-eval direct --run-dir D --registry configs/registry.yaml -m ID [-m ID...] [--dry-run] [--max-spend X] [--limit N] [--no-image] [--workers N] [--force]` → runs selftest first (fail-closed), then `run_direct`; exits nonzero if any cell errored.
  - `ocr-eval selftest [--extractor]` → offline scorer fixtures; `--extractor` also validates `gemini_extract` against 5 known-answer transcripts (requires GEMINI_API_KEY).
  - `ocr-eval rescore --run-dir D` → re-runs `score_typed` with current templates over EVERY cache file that has an `"answer"` (both shapes); rewrites `field_matches`/`match` in place, prints a changed-row count. Implemented by iterating `layout.cache_dir.glob("*.json")` directly + `_ensure_template` per bank item — NOT by delegating to upstream `run_score`. (Corrected rationale: upstream `evaluate score` does **not** validate parser names — passing `-p vlm__…` actually works on the cache-hit path. We still need our own command because upstream's `run_score` demands `GEMINI_API_KEY` up front and `parsed_parsers()` never discovers `vlm__*` rows; and **upstream `evaluate score --force` pointed at a `vlm__*` key DESTROYS the row** — it skips the cache, finds no `parses/vlm__…/` markdown, and overwrites the record with `error: markdown missing`, losing the paid-for answer. Guard: `ocr-eval` prints a warning about this in `rescore --help`, and the runbook forbids upstream `--force` on `vlm__*` keys.) Test: corrupt one cached `field_matches` by hand, run rescore, assert it is healed and the count is 1.
- `selftest.py` produces: `run_offline() -> list[str]` (empty = pass; strings = failures), `run_extractor() -> list[str]`, `FIXTURES: list[dict]`.

- [ ] **Step 1: Write failing selftest tests**

`tests_ext/test_selftest.py`:
```python
from ocr_eval_ext.selftest import FIXTURES, run_offline


def test_offline_selftest_passes():
    assert run_offline() == []


def test_fixture_coverage():
    kinds = {f["kind"] for f in FIXTURES}
    assert {"correct", "polarity_inverted", "missing_field",
            "null_correct", "null_hallucinated", "refusal_is_error"} <= kinds
```

- [ ] **Step 2: Implement `ocr_eval_ext/selftest.py`** — hand-written fixtures exercising upstream `build_template` + `score_typed` end to end (never drawn from the bank):

```python
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
```

- [ ] **Step 3: Run selftest tests** — PASS (this simultaneously validates our understanding of upstream scoring; if `null_correct` fails, upstream template semantics differ from the spec's assumption — stop and re-read `jsonify.py:_placeholder_to_type_hint` before proceeding).

- [ ] **Step 4: Write failing CLI tests**

`tests_ext/test_cli.py`:
```python
import json

from typer.testing import CliRunner

from ocr_eval_ext.cli import app

runner = CliRunner()


def test_verify_fails_on_tiny_bank(tmp_path):
    run = tmp_path / "run"
    (run / "docs").mkdir(parents=True)
    (run / "qa_bank.json").write_text(json.dumps({"items": [
        {"question_id": "q1", "source_file": "d", "capabilities": [], "gold_dict": {"a": 1}}]}))
    result = runner.invoke(app, ["verify", "--run-dir", str(run)])
    assert result.exit_code == 1
    assert "cardinality mismatch" in result.output


def test_selftest_command_passes():
    result = runner.invoke(app, ["selftest"])
    assert result.exit_code == 0
    assert "offline scorer self-test: PASS" in result.output
```

- [ ] **Step 5: Implement `ocr_eval_ext/cli.py`**

```python
"""ocr-eval — Stage 1 commands. Wraps upstream for download/parse/score; adds
verify (fail-closed preconditions), direct (vlm-chat), selftest, report."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer
import yaml
from rich.console import Console

from ocr_eval_ext import selftest as st
from ocr_eval_ext.config import load_registry
from ocr_eval_ext.preconditions import PreconditionError, check_bank
from realdoc_bench.evaluate.runs import RunLayout

app = typer.Typer(no_args_is_help=True)
console = Console(width=200)          # fixed width — CI COLUMNS must not split assertion tokens

REPO_ROOT = Path(__file__).resolve().parents[1]
PINS_PATH = REPO_ROOT / "configs" / "pins.yaml"


def _run_meta_path(layout: RunLayout) -> Path:
    return layout.root / "run_meta.json"


def _preflight(layout: RunLayout) -> None:
    """The gate every spending/reporting command calls first. Fail-closed."""
    if st.run_offline():
        console.print("[red]scorer self-test failed — refusing to proceed[/red]")
        raise typer.Exit(1)
    pins = yaml.safe_load(PINS_PATH.read_text())
    head = subprocess.run(["git", "merge-base", "HEAD", pins["harness_commit"]],
                          capture_output=True, text=True, cwd=REPO_ROOT)
    if head.returncode != 0 or pins["harness_commit"] not in head.stdout:
        console.print(f"[red]harness pin {pins['harness_commit'][:10]} is not an ancestor of HEAD[/red]")
        raise typer.Exit(1)
    meta_p = _run_meta_path(layout)
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("dataset_revision") != pins["dataset_revision"]:
            console.print(f"[red]run dir was downloaded at revision "
                          f"{meta.get('dataset_revision')!r}, pins say "
                          f"{pins['dataset_revision']!r}[/red]")
            raise typer.Exit(1)
    try:
        items = json.loads(layout.bank_path.read_text())["items"]
        check_bank(items)
    except (FileNotFoundError, KeyError, PreconditionError) as e:
        console.print(f"[red]preflight FAILED: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def verify(run_dir: Path = typer.Option(..., "--run-dir")) -> None:
    """Full fail-closed sweep: pins, cardinalities, every PDF single-page + non-blank render.
    Warms the PNG cache as a side effect. Run once after download, before any spend."""
    from ocr_eval_ext.direct import STAGE1_CONDITION, _render_page
    from ocr_eval_ext.preconditions import assert_single_page, ink_coverage

    layout = RunLayout.at(run_dir)
    _preflight(layout)
    png_cache = layout.root / "docs_png"
    blank, multi = [], []
    for pdf in sorted(layout.docs_dir.glob("*.pdf")):
        try:
            png = _render_page(layout, pdf.stem, STAGE1_CONDITION, png_cache)
        except PreconditionError:
            multi.append(pdf.stem)
            continue
        if ink_coverage(png) < 0.001:
            blank.append(pdf.stem)
    if blank or multi:
        console.print(f"[red]verify FAILED — multi-page: {multi} blank-render: {blank}[/red]")
        raise typer.Exit(1)
    meta_p = _run_meta_path(layout)
    if not meta_p.exists():                # stamp pins + renderer version on first successful verify
        from importlib.metadata import version as _v
        pins = yaml.safe_load(PINS_PATH.read_text())
        meta_p.write_text(json.dumps({
            "dataset_revision": pins["dataset_revision"],
            "harness_commit": pins["harness_commit"],
            "pymupdf_version": _v("pymupdf"),
            # vLLM runs in its own serving env, not this one — its version is recorded per model
            # in docs/local-serving.md, which is the operative record for local rows.
        }, indent=2))
    console.print("[green]verify PASS[/green] — pins, cardinalities, pages, renders all green")


@app.command()
def selftest(extractor: bool = typer.Option(False, "--extractor")) -> None:
    fails = st.run_offline()
    if fails:
        console.print("[red]offline scorer self-test: FAIL[/red]")
        for f in fails:
            console.print(f"  - {f}")
        raise typer.Exit(1)
    console.print("offline scorer self-test: PASS")
    if extractor:
        efails = st.run_extractor()
        if efails:
            console.print("[red]extractor validation: FAIL[/red]")
            for f in efails:
                console.print(f"  - {f}")
            raise typer.Exit(1)
        console.print("extractor validation: PASS (5/5)")


@app.command()
def direct(
    run_dir: Path = typer.Option(..., "--run-dir"),
    registry: Path = typer.Option(Path("configs/registry.yaml"), "--registry"),
    model: list[str] = typer.Option(..., "--model", "-m"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    max_spend: float | None = typer.Option(None, "--max-spend"),
    limit: int | None = typer.Option(None, "--limit"),
    no_image: bool = typer.Option(False, "--no-image"),
    workers: int = typer.Option(8, "--workers"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    from ocr_eval_ext.config import get_entry
    from ocr_eval_ext.direct import run_direct

    layout = RunLayout.at(run_dir)
    _preflight(layout)                     # cardinalities + pins + scorer self-test, fail-closed
    entries = load_registry(registry)
    chosen = [get_entry(entries, m) for m in model]
    bad_shape = [e.id for e in chosen if e.shape != "vlm-chat" or e.transport != "openai-compat"]
    if bad_shape:
        console.print(f"[red]not vlm-chat/openai-compat: {bad_shape} — use ocr-eval parse for those[/red]")
        raise typer.Exit(1)
    summary = run_direct(layout, chosen, dry_run=dry_run, max_spend_usd=max_spend,
                         limit=limit, no_image=no_image, workers=workers, force=force)
    console.print(summary)
    if not dry_run and summary.get("error"):
        raise typer.Exit(2)   # fail-visible: errors occurred, report will mark them


# Additional commands added in Tasks 8-9 (parse/score/preflight/report) all call _preflight(layout)
# first. `ocr-eval score` additionally runs the BLOCKING extractor-validation gate:
#   fixture_hash = sha256(json.dumps(st.EXTRACTOR_FIXTURES, sort_keys=True))
#   stamp = layout.root / f".extractor_ok_{DEFAULT_MODEL}_{fixture_hash[:8]}"
#   if not stamp.exists(): run st.run_extractor(); [] required; then stamp.touch()
# — 5 Gemini calls per (extractor, fixture-set), cached per run dir, never skippable.
# `ocr-eval score` also records DEFAULT_MODEL into run_meta.json as extractor id; if a later
# invocation sees a different extractor id there, it refuses to mix generations unless
# --new-extractor-generation is passed, which archives eval/cache/ to eval/cache@<old-id>/ first.


@app.command()
def rescore(run_dir: Path = typer.Option(..., "--run-dir")) -> None:
    """Recompute field_matches/match from stored answers with CURRENT templates — both shapes,
    zero API calls. (Never point upstream `evaluate score --force` at vlm__* keys: it would
    overwrite them with 'markdown missing' and destroy the paid-for answers.)"""
    from realdoc_bench.evaluate.score import _ensure_template, score_typed

    layout = RunLayout.at(run_dir)
    _preflight(layout)
    items = {i["question_id"]: i for i in json.loads(layout.bank_path.read_text())["items"]}
    changed = 0
    for f in sorted(layout.cache_dir.glob("*.json")):
        rec = json.loads(f.read_text())
        item = items.get(rec.get("qid"))
        if item is None or "answer" not in rec or rec["answer"] is None:
            continue
        _ensure_template(item)
        fm, allc = score_typed(rec["answer"], item["gold_dict"], item["str_keys"])
        if fm != rec.get("field_matches") or allc != rec.get("match"):
            rec["field_matches"], rec["match"] = fm, allc
            f.write_text(json.dumps(rec, ensure_ascii=False))
            changed += 1
    console.print(f"rescore: {changed} row(s) changed")
```

- [ ] **Step 6: Run CLI tests** — PASS. **Step 7: Full suite.** **Step 8: Commit** — `git commit -m "feat: ocr-eval CLI with verify/selftest/direct, fail-closed gating"`

---

### Task 8: OpenAI-compatible transcriber parser + local serving docs

**Files:**
- Create: `ocr_eval_ext/parsers_openai.py`, `tests_ext/test_parsers_openai.py`, `docs/local-serving.md`
- Modify: `ocr_eval_ext/cli.py` (add `preflight` command; register parsers at import)

**Interfaces:**
- Consumes: `VisionParserBase` subclass contract; `_MARKDOWN_PROMPT` from `cloud_vlm.py`; `RegistryEntry`.
- Produces:
```python
class OpenAICompatVisionParser(VisionParserBase):
    def __init__(self, *, base_url: str, model: str, api_key_env: str | None = None): ...
    # prompt = _MARKDOWN_PROMPT; _call_page posts one chat completion per page image

def register_openai_parsers(registry_path: Path) -> list[str]
    # for each transcriber+openai-compat entry: creates a named subclass via type()
    # and register_parser(safe_name(entry.id), version=entry.model); returns names.
    # safe_name: entry id with '@'/'.' → '_' (parser names become dir names in parses/).

def preflight(entry: RegistryEntry) -> str   # GET {base_url}/models; returns served id; raises on mismatch
```
- Upstream `evaluate parse -p <safe_name>` and `evaluate score` then work unchanged for these transcribers; transcripts land in `parses/<safe_name>/`.

- [ ] **Step 1: Write failing tests** — parser calls mock server per page with the upstream markdown prompt and an image; registration creates buildable parsers; preflight matches/rejects.

```python
import fitz

from ocr_eval_ext.parsers_openai import OpenAICompatVisionParser, preflight, register_openai_parsers, safe_name
from tests_ext.mock_openai import MockOpenAI


def test_safe_name():
    assert safe_name("glm-ocr@local-vllm") == "glm-ocr_local-vllm"


def test_call_page_uses_markdown_prompt_and_image(tmp_path):
    pdf = tmp_path / "d.pdf"
    doc = fitz.open(); doc.new_page(width=612, height=792); doc.save(pdf)
    with MockOpenAI(reply_text="## Page\n**Box:** ☒ Yes") as mock:
        p = OpenAICompatVisionParser(base_url=mock.base_url, model="org/m")
        result = p.parse(pdf)
        assert "☒" in result.markdown
        req = mock.requests[0]
        text_parts = [c["text"] for c in req["messages"][-1]["content"] if c.get("type") == "text"]
        assert any("☒ (checked)" in t for t in text_parts)     # upstream prompt in use
        assert any(c.get("type") == "image_url" for c in req["messages"][-1]["content"])


def test_register_from_registry(tmp_path):
    (tmp_path / "r.yaml").write_text("""
- id: t1@local
  shape: transcriber
  transport: openai-compat
  base_url: http://localhost:9/v1
  model: org/t1
  precision: bf16
  weights_licence: mit
  provider_tos_commercial: ok
  provenance: X
  release_date: "2025-01-01"
  local: true
""")
    names = register_openai_parsers(tmp_path / "r.yaml")
    assert len(names) == 1
    assert names[0].startswith("t1_local__")            # safe_name + "__" + condition hash
    from realdoc_bench.evaluate.parsers.base import build
    assert build(names[0]).version == "org/t1"
```

- [ ] **Step 2: Verify failure, then implement.** Key implementation points (complete file expected ~90 lines):

```python
from pathlib import Path

from ocr_eval_ext.config import RegistryEntry
from realdoc_bench.evaluate.parsers._vision_base import VisionParserBase
from realdoc_bench.evaluate.parsers.base import register_parser
from realdoc_bench.evaluate.parsers.base import registry as parser_registry
from realdoc_bench.evaluate.parsers.cloud_vlm import _MARKDOWN_PROMPT


class OpenAICompatVisionParser(VisionParserBase):
    prompt = _MARKDOWN_PROMPT
    page_concurrency = 1        # local GPU: one in-flight request; hosted entries may override

    def __init__(self, *, base_url: str, model: str, api_key_env: str | None = None):
        import os
        from openai import OpenAI
        key = os.environ.get(api_key_env) if api_key_env else "none"
        if api_key_env and not key:
            raise RuntimeError(f"env var {api_key_env} not set")
        self._client = OpenAI(base_url=base_url, api_key=key or "none")
        self._model = model

    def _call_page(self, png_bytes: bytes) -> tuple[str, int, int]:
        import base64
        b64 = base64.b64encode(png_bytes).decode()
        resp = self._client.chat.completions.create(
            model=self._model, temperature=0.0, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": self.prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}])
        u = (resp.model_dump().get("usage") or {})
        return ((resp.choices[0].message.content or ""),
                u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
```
```python
def safe_name(entry_id: str) -> str:
    return entry_id.replace("@", "_").replace(".", "-").replace("/", "_")


def register_openai_parsers(registry_path: Path) -> list[str]:
    from ocr_eval_ext.config import load_registry
    names = []
    for e in load_registry(registry_path):
        if e.shape != "transcriber" or e.transport != "openai-compat":
            continue
        name = safe_name(e.id)
        if name in parser_registry:      # idempotent re-import
            names.append(name)
            continue

        def _init(self, *, _e=e):
            OpenAICompatVisionParser.__init__(self, base_url=_e.base_url, model=_e.model,
                                              api_key_env=_e.api_key_env)

        cls = type(name, (OpenAICompatVisionParser,), {"__init__": _init})
        register_parser(name, version=e.model)(cls)
        names.append(name)
    return names


def preflight(entry: RegistryEntry) -> str:
    import httpx
    r = httpx.get(f"{entry.base_url}/models", timeout=10)
    r.raise_for_status()
    served = [m.get("id", "") for m in r.json().get("data", [])]
    if entry.model not in served:
        raise RuntimeError(f"{entry.id}: served models {served} do not include {entry.model}")
    return entry.model
```
Note `safe_name` maps `.` → `-` (dots.ocr → `dots-ocr…`): parser names become directory names under `parses/` and cache-key suffixes; the test in Step 1 pins the exact mapping.

**Transcriber condition seam (Stage 2 protection — decided now because it is unwindable later without discarding cache):** the registered parser name is `safe_name(entry.id) + "__" + condition_hash(TRANSCRIBER_CONDITION)`, where `TRANSCRIBER_CONDITION = {"preprocess": "raw", "render": {"engine": "pymupdf", "dpi": 150}, "sampling": {"temperature": 0.0}, "sample_index": 0}` (import `condition_hash` from `direct.py`). Stage 2's deskew therefore creates `…__<newhash>` parser dirs and cache keys instead of silently overwriting raw transcripts. The `ocr-eval parse` wrapper also writes `parses/<parser>/condition.json` (the dict, verbatim) so transcriber rows are self-describing. Upstream-parser entries (gemini_3_5_flash, mistral_ocr_4) keep their upstream names — they are the published-comparability rows and never vary conditions.

**Also add to `ocr_eval_ext/cli.py` in this task** — upstream's `realdoc-bench` CLI never imports `ocr_eval_ext`, so our dynamically registered parsers do not exist there. Wrapper commands make them reachable:
- `ocr-eval parse --run-dir D -p NAME [--workers N] [--force] [--limit N]` → `register_openai_parsers(...)` then delegate to upstream `run_parse`.
- `ocr-eval score --run-dir D -p NAME [--force]` → register, then upstream `run_score` + `aggregate_results` (requires GEMINI_API_KEY; call `require_api_key()` first, fail fast).
Test: CliRunner invoking `ocr-eval parse -p t1_local` against a run dir + mock server produces `parses/t1_local/doc_1.md`.

- [ ] **Step 3: Write `docs/local-serving.md`** — per-model vLLM launch lines (BF16, one model resident):

```markdown
# Local serving (RTX 4070 Ti SUPER, 16 GB, CUDA)
Pin and record: `uv run python -c "import vllm; print(vllm.__version__)"` per run.
One model resident at a time. Start, preflight, run, stop.

## GLM-OCR (0.9B, BF16)
vllm serve zai-org/GLM-OCR --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --max-model-len 8192
# verify exact HF id + any --trust-remote-code / processor flags against the model card at first use

## dots.ocr (3B, BF16)
vllm serve rednote-hilab/dots.ocr --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --max-model-len 8192 --trust-remote-code

## Preflight (always, before any cells)
uv run ocr-eval preflight glm-ocr@local-vllm
```
The exact flags per model are verified against each model card / vLLM docs at first bring-up and the working command line is committed back into this file — the file is the record of what actually ran.

- [ ] **Step 4: Run tests, full suite, commit** — `git commit -m "feat: OpenAI-compatible transcriber parser, preflight, local-serving runbook"`

---

### Task 9: Shape-segregated markdown report

**Files:**
- Create: `ocr_eval_ext/report_md.py`, `tests_ext/test_report_md.py`
- Modify: `ocr_eval_ext/cli.py` (add `report` command)

**Interfaces:**
- Consumes: run dir cache records (both shapes: upstream parser keys + `vlm__*` keys), bank, registry, Tasks 3–5 modules.
- Produces: `build_markdown_report(layout, registry_entries, *, bank_path=None) -> str` and writes `<run-dir>/report.md`. Structure (assert in tests):
  1. Header: pins, dataset attribution (CC-BY-4.0 line), run date, caveat block (`origin=None` mixed provenance; ~18% born-digital estimate; provider downscaling uncontrolled).
  2. **Section A — Direct QA (vlm-chat):** one row per `vlm__<id>__<cond>` key. Columns: checkbox acc-over-all [CI] · polarity (checked/unchecked) · null hallucination rate [CI] (n=188) · general per-field · strict per-question · error-class rates · n · precision · licence · ToS stamp · contamination flag · resolved providers seen · median latency · realized cost.
  3. Baseline rows (always-true / always-false / majority + class balance line + no-image row when present).
  4. **Section B — Transcribe-then-extract:** same columns; rows labelled `<transcriber> + extractor gemini-3-flash-preview`; transcript-recall diagnostic column (fraction of checkbox-question transcripts containing any of `☒ ☐ ☑ [x] [ ]` — computed from `parses/<parser>/<stem>.md`); an `input` column (`raster-png` for our OpenAI-compat parsers, `pdf-direct` for upstream adapters like mistral_ocr that upload the PDF) with a footnote that pdf-direct rows can free-ride on embedded text layers (measured: 2 of 16 sampled docs carry one).
  4b. **Completeness marking:** any row whose cell count over the expected item set is short renders as `INCOMPLETE (k/N)` in bold at the row head — a partial matrix must be visibly partial.
  5. Cross-shape warning paragraph (auto-printed) + the direct-vs-two-stage calibration pair called out when both rows exist for one model.
  6. Separability appendix: pairwise paired-bootstrap deltas within each section; "not separable" where CI spans 0.
  7. Staleness warning when a row's `retrieved_at` values span > 7 days.

- [ ] **Step 1: Write failing tests** — build a synthetic run dir (two fake vlm parsers' cache records + one transcriber's records + parses/*.md), then assert on rendered markdown: sections in order, baseline row values, "not separable" appears for two identical models, ToS stamp text appears for a `blocked` entry, CC-BY line present, checkbox n == expected.

```python
# core assertions (abridged for the plan; write ~8 focused tests):
md = build_markdown_report(layout, entries)
assert md.index("## Section A — Direct QA") < md.index("## Section B — Transcribe-then-extract")
assert "majority-class" in md and "165" not in md.split("Section B")[1]  # balance line in A only
assert "not separable" in md
assert "CC-BY-4.0" in md and "origin=None" in md
assert "⚠ ToS: blocked" in md            # for the entry with provider_tos_commercial: blocked
assert "transcript-recall" in md.lower()
```

- [ ] **Step 2: Verify failure, implement.** Implementation notes: load all cache files once (`layout.cache_dir.glob("*.json")`), group by parser key; map parser keys → registry entries — `vlm__<id>__…` by prefix; openai-compat transcribers by `safe_name(e.id) + "__"` prefix; **upstream-parser entries by `e.upstream_parser` exactly** (`gemini_3_5_flash`, `mistral_ocr_4` — `safe_name(e.id)` produces the WRONG string for these and would strip the anchor's licence/ToS stamps); unknown parser keys render in Section B with a `(unregistered)` marker rather than crashing. Metrics/CI calls per row; keep pure-function core (`records → dict rows → markdown`) so tests don't need API. Additional mandatory behaviors:
  - **beats-majority column:** `paired_delta_ci(model_outcomes, majority_baseline_outcomes)` (the majority predictor is deterministic per field, so outcomes are well-defined) → `yes / no / not separable`. A model is never labelled above-baseline when not separable from majority.
  - **Bucket overlap line:** "40 questions appear in both the checkbox and blank-field buckets" printed with the metric definitions.
  - **Transcript-recall = glyph AND label:** a checkbox-question transcript counts as recalling the field only if it contains a checkbox glyph and a token of the gold field key/label — glyph-only counts "emitted checkboxes somewhere," not "emitted this field."
  - **STALE-RENDER check (D3):** recompute the current render hash per referenced doc once; any row whose `image_sha` differs is marked and the report exits nonzero without `--allow-stale-render`.
  - **Serving-identity guard (D4):** if one parser key's rows span >1 `resolved_provider`, the report hard-fails (not warns).
  - **Precision discipline:** `provider-default` renders as `unknown (not asserted)`; if models being compared in one section ran at differing known precisions, the section header carries the mixed-precision caveat (spec §Precision policy).
  - **Cost/latency columns:** `n/a` for `local: true` rows (never a fake $0.00) and, in Section B, scoped in the header to the transcription leg only (upstream extraction records carry no token/latency fields).
  - **Same-family flag:** any transcriber row whose `provenance` matches the extractor's family (Google + gemini extractor) is marked `same-family scoring`.
  - **Extractor id** read from `run_meta.json`, never hardcoded in the template.

- [ ] **Step 3: Add `report` CLI command** (runs preconditions on the bank first — a report over a wrong bank must not render).

- [ ] **Step 4: Run tests, full suite, commit** — `git commit -m "feat: shape-segregated markdown report with CIs, baselines, stamps"`

---

### Task 10: E2E smoke test + Stage 1 runbook + Table-3 snapshot

**Files:**
- Create: `tests_ext/test_e2e_smoke.py`, `docs/runbook-stage1.md`, `docs/superpowers/specs/table3-snapshot.md`

**Interfaces:** none new — this task proves the seams hold and writes the operational procedure.

- [ ] **Step 1: Write the E2E smoke test** — synthetic 2-item bank + 2 tiny PDFs + MockOpenAI: `run_direct` → `build_markdown_report` → assert the report table contains the mock model row with 100% checkbox accuracy and the baseline rows. No network, no keys.

- [ ] **Step 2: Snapshot the reproduction targets** — `docs/superpowers/specs/table3-snapshot.md`: copy the paper's Table 3 open-weight rows (dots.ocr 70.6±3.6 / 61.4±3.5 · olmOCR-2 79.5±2.6 / 67.9±3.0 · PaddleOCR-VL 59.6±4.0 / 48.5±3.6) and the README leaderboard as of `fb26a687` (Gemini 3.5 Flash 89.3 / 82.2), each with source (paper table vs README) and retrieval date. These are the numbers the reproduction gate compares against; the README will drift, this file must not.

- [ ] **Step 3: Write `docs/runbook-stage1.md`** — the exact operational sequence:

> **The sketch below is the plan's original draft, not the operative procedure.**
> [`docs/runbook-stage1.md`](../../runbook-stage1.md) is what actually ran and is authoritative
> where the two differ. Two known divergences: the `bws` wrapper lines are superseded
> (shell-profile exports — see the Global Constraints note above), and the rotation sign-off moved
> from `bws secret list` to the issuing providers' own consoles, since keys live only in the
> environment on the current host.

```markdown
# Stage 1 run procedure
Prereqs: OPENROUTER_API_KEY and GEMINI_API_KEY **rotated** (prior exposure) and present in bws.
All commands via: bws run --project-id 18f14ed9-8ba5-4cc6-bbd4-b45b01534270 -- <cmd>

1.  uv run ocr-eval selftest --extractor            # fail-closed gates (5 Gemini calls)
2.  uv run realdoc-bench evaluate download --run-dir runs/stage1 \
      --dataset Extend-AI/RealDoc-Bench --revision 906170ab201d7b8238a32a9115fc66b4b72e0710
    # NB: the CLI flag is --dataset (the function kwarg repo_id is not the flag name).
    # ocr-eval verify writes run_meta.json with dataset_revision on first pass.
3.  uv run ocr-eval verify --run-dir runs/stage1     # pins + cardinalities + pages + renders;
                                                     # warms the PNG cache. Abort on any red.
4.  # Environment reproduction anchor (validates render/keys/scoring end-to-end).
    # Use ocr-eval wrappers, NOT `evaluate run` — upstream run's report phase writes
    # dashboard.html, which globs ALL cache records and cross-ranks both shapes:
    uv run ocr-eval parse --run-dir runs/stage1 -p gemini_3_5_flash
    uv run ocr-eval score --run-dir runs/stage1 -p gemini_3_5_flash
    #   → compare to table3-snapshot.md (89.3/82.2); investigate before proceeding if outside ±2.5
5.  # Hosted OCR endpoint (upstream parser name is mistral_ocr_4 — NOT mistral_ocr):
    uv run ocr-eval parse --run-dir runs/stage1 -p mistral_ocr_4
    uv run ocr-eval score --run-dir runs/stage1 -p mistral_ocr_4
6.  # Local specialist (see docs/local-serving.md; one model at a time). NB: our registered
    # parsers exist only under the ocr-eval CLI (upstream never imports ocr_eval_ext), and
    # their names carry the transcriber-condition hash — `ocr-eval parse` prints them:
    vllm serve ... ; uv run ocr-eval preflight glm-ocr@local-vllm
    uv run ocr-eval parse --run-dir runs/stage1 -p glm-ocr_local-vllm__<cond>
    uv run ocr-eval score --run-dir runs/stage1 -p glm-ocr_local-vllm__<cond>
    # Scoring-leg cost note: each scored question = one gemini-3-flash call per transcriber
    # (~1,356 calls ≈ low single-digit $ per transcriber). No automated dry-run for this leg
    # (upstream has no hook — divergence D6). Budget it here, smoke with --limit 20 first.
    # NEVER run upstream `evaluate score --force` against vlm__* keys — it overwrites the
    # cached answers with 'markdown missing' (verified during plan review). dashboard.html,
    # if ever generated, is upstream's shape-mixed view: report.md is the only authoritative
    # output; `ocr-eval report` renames any dashboard.html to dashboard-upstream-UNSEGREGATED.html.
    # dots.ocr likewise → open-weight reproduction check vs 70.6±3.6 (setup caveats: our
    # serving stack, not the paper's; document any residual gap in the report)
7.  # Direct QA candidates + Section A ceiling anchor:
    uv run ocr-eval direct --run-dir runs/stage1 -m qwen3-vl-8b@openrouter -m qwen3.5-9b@openrouter \
      -m qwen3-vl-32b@openrouter -m gemini-3.5-flash@google-vlmchat \
      --dry-run                                    # inspect cells + estimate, then re-run with --max-spend 40
    # Calibration pair (same weights, transcriber shape — registry id qwen3-vl-8b@openrouter-transcriber):
    uv run ocr-eval parse --run-dir runs/stage1 -p qwen3-vl-8b_openrouter-transcriber__<cond>
    uv run ocr-eval score --run-dir runs/stage1 -p qwen3-vl-8b_openrouter-transcriber__<cond>
8.  # No-image control (one model):
    uv run ocr-eval direct --run-dir runs/stage1 -m qwen3-vl-8b@openrouter --no-image --max-spend 10
9.  uv run ocr-eval report --run-dir runs/stage1    # → runs/stage1/report.md
10. Re-run step 7 verbatim → expect "cached: <all>", zero API calls (DoD #5).
```

- [ ] **Step 4: Run full suite** — everything green. **Step 5: Commit** — `git commit -m "feat: e2e smoke, stage-1 runbook, reproduction-target snapshot"`

---

### Task 11: Definition-of-done checklist run

Not code — the Stage 1 acceptance pass, executed by a human (or an agent with keys) following `docs/runbook-stage1.md`. Every DoD item from the spec, verified and recorded in `report.md`:

- [ ] Preconditions, scorer self-test, extractor validation all green (fail-closed observed at least once by deliberately corrupting a fixture, then restoring).
- [ ] Reproduction: gemini_3_5_flash within tolerance of snapshot; dots.ocr local vs paper CI with caveats documented.
- [ ] ≥3 hosted VLM rows, ≥1 local specialist (BF16), ≥1 hosted OCR endpoint, frontier anchor, calibration pair.
- [ ] Baselines + CIs + polarity split + shape segregation present in `report.md`.
- [ ] Cache-hit rerun: 100% cached, zero API calls.
- [ ] Keys were rotated before first hosted call.
