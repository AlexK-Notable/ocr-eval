# Reproduction-target snapshot

Frozen numbers the Stage 1 reproduction gate (DoD #2) and the Gemini frontier anchor compare
against. Two independent sources, **never merged into one table**: the README drifts (it is
regenerated as this repo's own leaderboard changes over time); this file is pinned and must not
be silently updated to track it. If a future run's report disagrees with a number here, that is a
signal to investigate — not a license to edit this file to match.

**Gate rule:** the reproduction gate (DoD #2) targets come from the **paper Table 3** rows below —
they are the only open-weight rows with a published CI to reproduce *within*. The Gemini 3.5 Flash
frontier anchor (a closed model, not a reproduction-gate candidate) compares instead against the
**README row**, since Gemini 3.5 Flash is not one of the paper's open-weight Table 3 candidates —
it is upstream's own `cloud_vlm` entry, tracked here via this repo's leaderboard, not the paper.
Never compare a run's `gemini_3_5_flash` row against a paper-table number, and never compare a
run's `dots-ocr@local-vllm` row against the README (the README carries no dots.ocr QA-leaderboard
row at all — only a `dots.mocr` row in the unrelated *Layout* leaderboard, a different benchmark
mode entirely; see the caveat under that table below).

---

## 1. Paper Table 3 — open-weight rows (the reproduction-gate targets)

**Source:** paper Table 3 (transcribed by prior survey; re-verify against arXiv at first
reproduction run). These three rows were carried into `docs/superpowers/plans/
2026-08-01-stage1-eval-pipeline.md` from an earlier survey pass over the RealDocBench paper, not
independently re-derived from the arXiv PDF by this task. Format is `per-field accuracy ± 95% CI
half-width / per-question accuracy ± 95% CI half-width` — the same two-number shape as this
report's `general/field` and `strict/question` columns (per-field / per-question), not the
Layout benchmark's F1 family.

| Model | Per-field accuracy | Per-question accuracy | Reproduction candidate for |
| --- | ---: | ---: | --- |
| dots.ocr (2-stage) | 70.6% ± 3.6pp | 61.4% ± 3.5pp | `dots-ocr@local-vllm` (registry) |
| olmOCR-2 | 79.5% ± 2.6pp | 67.9% ± 3.0pp | not in `configs/registry.yaml` — no Stage 1 candidate |
| PaddleOCR-VL | 59.6% ± 4.0pp | 48.5% ± 3.6pp | not in `configs/registry.yaml` — no Stage 1 candidate |

**Verification status:** UNVERIFIED against the primary source. `docs/runbook-stage1.md` step 6
directs the operator to re-verify the `dots.ocr` row against the arXiv paper directly (not this
survey transcription) before treating a reproduction attempt as confirmed or refuted — a
transcription error in either the survey or this file would otherwise silently pass or fail the
gate on a wrong number. Only `dots.ocr` has a registered Stage 1 candidate (`dots-ocr@local-vllm`,
`configs/registry.yaml`); olmOCR-2 and PaddleOCR-VL have no local-serving entry in this repo and
are listed here for completeness/future reference only.

**CI overlap is the pass condition**, not exact-value match: DoD #2 asks for the local reproduction
run's own bootstrap CI (from `report.md`'s Section B row, `general/field` and `strict/question`
columns — note: `report.md` does not currently render a CI on those two columns the way it does on
`checkbox acc-over-all`/`blank-null acc-over-all`; compare point estimates against this table's
range and treat overlap-by-eye as the practical criterion until/unless a CI is added to those
columns) to overlap this table's stated interval. A residual gap is expected and must be
*documented*, not treated as an automatic fail — `configs/registry.yaml`'s own local-serving setup
(this repo's vLLM version/precision/prompt) is not the paper's serving stack; see
`docs/local-serving.md` for what is/isn't controlled.

---

## 2. README leaderboard — as of pinned commit `fb26a687`

**Source:** README @ fb26a687 (`harness_commit` in `configs/pins.yaml`:
`fb26a6876481de76dc293f722ab4efa71279904d`). **Retrieved:** 2026-08-02, copied verbatim from this
repo's own `README.md` at that commit — this is this repo's *own* leaderboard (parsers this
harness has actually run and scored), not the paper's. It will drift as the harness re-runs
parsers or adds new ones; this snapshot will not follow it.

### Document QA Leaderboard (1,356 questions / 3,742 fields)

*This is the table the Gemini 3.5 Flash frontier anchor compares against — the row is bolded
below for that reason, not because it ranks first.*

| Parser | Per-field Accuracy | Per-question Accuracy |
| --- | ---: | ---: |
| Extend Parse 2.0 | 96.0% | 90.9% |
| LlamaParse (Agentic) | 92.2% | 84.5% |
| Reducto (Agentic) | 91.4% | 83.8% |
| Extend Parse Light 1.0 | 90.5% | 82.7% |
| **Gemini 3.5 Flash** | **89.3%** | **82.2%** |
| LlamaParse | 89.2% | 80.8% |
| Azure DI | 89.1% | 79.6% |
| Mistral OCR 4 | 88.8% | 81.3% |
| Reducto | 88.7% | 80.5% |
| AWS Textract | 70.7% | 54.0% |

**Tolerance:** `docs/runbook-stage1.md` step 4 treats a Stage 1 `gemini_3_5_flash` run as within
tolerance of this row when both columns land within ±2.5pp of 89.3% / 82.2%. Outside that band,
investigate (harness/render/prompt/template divergence, dataset-revision drift, or an extractor
generation change) before proceeding to spend on the remaining candidates — this is the
pipeline-level positive control the paper-table reproduction gate (§1) can't stand in for on its
own, since it exercises a *closed* model this harness can call directly, live, on demand.

Mistral OCR 4 (88.8% / 81.3%) is listed here for completeness — it is `mistral-ocr@mistral` in
`configs/registry.yaml`, the hosted-OCR-endpoint candidate in the Stage 1 DoD categories (§3 of
the plan), and its own `mistral_ocr_4` run (`docs/runbook-stage1.md` step 5) can be sanity-checked
against this row the same way, though no formal tolerance is asserted for it — only the frontier
(Gemini) anchor has a stated ±2.5pp gate.

### Layout Leaderboard (unrelated benchmark mode — different metric family)

Included for completeness only. This table scores bounding-box/block-type layout prediction (F1 /
adjusted F1 / mAP), a **different RealDocBench mode** from the QA-extraction numbers everywhere
else in this file and in `report.md` — Stage 1 (this pipeline) only exercises the QA-extraction
mode. The `dots.mocr` row here is **not** the same measurement as the paper Table 3 `dots.ocr` row
in §1 above (different metric, and note the README's own `mocr` vs the registry/paper's `ocr`
spelling) — never treat this row as a README cross-check for the `dots-ocr@local-vllm` reproduction
attempt.

| Model | N | Strict F1 | Adjusted F1 | Macro F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Extend Parse 2.0 | 1500 | 0.781 | 0.847 | 0.702 | 0.818 | 0.748 |
| AWS Textract | 1500 | 0.626 | 0.709 | 0.526 | 0.598 | 0.656 |
| Paddle OCR VL 1.5 | 1500 | 0.584 | 0.684 | 0.450 | 0.661 | 0.524 |
| Azure DI | 1500 | 0.558 | 0.687 | 0.521 | 0.492 | 0.644 |
| dots.mocr | 1500 | 0.238 | 0.320 | 0.188 | 0.225 | 0.253 |
