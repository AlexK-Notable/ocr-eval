# HANDOFF — Gemini candidate runs, paused mid-flight (2026-08-05)

**Temporary working document.** Delete once the Gemini runs are finished and their results are folded
into `docs/results-stage1-2026-08-04.md`. Nothing here is a durable spec; the durable decisions live
in `docs/scoring.md`, `docs/api.md` and the registry headers.

Written at the user's request before a context compaction, because the session had run long enough
that recall was becoming less reliable than re-reading. Every number below was re-verified against
the repo at the time of writing, not recalled.

---

## 1. STOP STATE — what is running

**Nothing.** All `ocr-eval` and `run-gemini-tiers.sh` processes confirmed dead. The user's
instruction was: *"stop before initiating any runs"*. Do not start a run without fresh approval.

---

## 2. Git state

Two commits ahead of `origin/main`, unpushed:

| commit | what |
|---|---|
| `cd143ddb` | `feat: gemini-native transport — control mediaResolution` (+ `configs/registry-gemini.yaml`) |
| `d3849534` | `fix: an omitted token count killed a full-bank run 300 cells into paid spend` |

Uncommitted:

* **`ocr_eval_ext/direct.py`** — +38/−6. Adds `GEMINI_SAMPLING` and folds it into
  `_condition_for`. **This is the vendor-default sampling change and it is NOT committed.**
* **394 deleted `runs/stage1/eval/cache/*google-native*f7a267f691d0*.json`** — tracked deletions,
  staged as deletions in the working tree.

`main` was last pushed at `12bc228e`.

---

## 3. THE DECISION THAT PAUSED THE WORK

The user asked: *"temp 0? are we doing that under advisement of any kind of documentation?"* That
question found a real problem.

**Google's `ai.google.dev/gemini-api/docs/prompting-strategies` says, verbatim:**

> "we strongly recommend keeping them at their default values for Gemini 3.x models. Changing these
> parameters (for example, setting the temperature below 1.0) can cause unexpected behavior, such as
> **looping or degraded performance**"

So `STAGE1_CONDITION`'s `temperature: 0.0` is against Google's explicit guidance for 3.x. The named
failure mode (looping) is the same one `docs/api.md` already records for Qwen thinking models under
greedy decoding, where it was observed live.

**Scope matters and is easy to get wrong:** that warning is for **3.x only**. No equivalent warning
was found for the 2.5 family, and `docs/gemini-api/docs/thinking` says nothing about sampling at all.

**User's decision (via AskUserQuestion, recorded):**
1. Sampling → **"Re-run everything at vendor default"** (all six, not just 3.x).
2. Pro tier → **"Smoke 20 and check for looping first"** before the full 1,356.

**Provenance note, so nobody re-litigates this:** temperature 0 was NOT invented for these runs. It
is ratified in `specs/2026-08-01-ocr-eval-pipeline-design.md:90` and already documented in
`docs/api.md:105-111` as a deliberate divergence from *Qwen's* vendor guidance, accepted for
reproducibility. The error was carrying it onto six Gemini models without checking *Google's*
guidance first.

---

## 4. Condition hashes — verified, do not guess these

```
0373123f3b15   STAGE1_CONDITION           every non-gemini transport (unchanged, untouched)
872144e4aecb   gemini + vendor sampling   the NEW target condition (uncommitted code)
f7a267f691d0   gemini + temperature 0     ABANDONED, rows deleted
4caf39e9ac1f   max_tokens=1024            the 1,356 live Haiku rows (pre-12288 change)
```

`GEMINI_SAMPLING = {"temperature": 1.0, "top_p": 0.95}`, merged over
`STAGE1_CONDITION["sampling"]`, so `max_tokens` stays 12288 and `seed` stays None.

**Two traps here:**

* `_condition_for()` in `direct.py` is the ONLY place that may build a gemini condition. Row
  `condition` and the hash inside its `parser` key must come from the same dict or `report` reads one
  row as two conditions. `do()` already routes through it via the local `ecos`.
* The live Haiku rows are at `4caf39e9ac1f`, i.e. `max_tokens=1024`, **not** today's
  `STAGE1_CONDITION`. The other agent's 12288 change moved the baseline after those rows were
  scored. Any new non-Gemini row is therefore already a different condition from Haiku's.

---

## 5. Deleted rows — READ BEFORE RESTORING ANYTHING

3,624 temperature-0 Gemini rows were deleted. **Only 394 of them were ever committed.**

* 394 → recoverable with `git checkout -- runs/stage1/eval/cache/`
* the other ~3,230 → exist **only** in `~/ocr-eval-gemini-temp0-abandoned-2026-08-05/` (3,624 files)

They are a superseded condition and were removed so they would not render as extra Section A rows.
Roughly $2 of spend written off — a real cost of the temp-0 misstep, worth stating plainly rather
than hiding.

Other backups (do not delete without asking):

| path | contents |
|---|---|
| `~/ocr-eval-preD11rev2-backup-2026-08-05/` | 9,722 files — all cache rows before the extractor re-scores |
| `~/ocr-eval-gemini-temp0-abandoned-2026-08-05/` | 3,624 — the deleted temp-0 Gemini rows |
| `~/ocr-eval-haiku-backup-2026-08-04/` | 3 — early Haiku report/meta snapshots |

---

## 6. KNOWN GAP — fix before the next run

**`GEMINI_SAMPLING` is not pinned by any test.** `grep -rn 'GEMINI_SAMPLING' tests_ext/` returns
nothing. `GEMINI_MEDIA_RESOLUTION` *is* pinned (`test_gemini_native.py`), so the asymmetry is an
oversight, not a decision. A test should assert:

* `_condition_for` merges vendor sampling for gemini-native and leaves other transports
  byte-identical (the existing media_resolution test does exactly this — extend it);
* the resulting condition hash differs from both `STAGE1_CONDITION` and the abandoned
  `f7a267f691d0`.

Current state: **448 tests pass**, lint at the 14-error `B008` baseline (all pre-existing).

---

## 7. What is already DONE and must not be redone

* **Extractor** is `gemini-3.6-flash` (D11 rev 3), best of three graders measured head-to-head over
  4,068 cells (p=4.7e-09 vs 3.1-flash-lite, p=3.4e-16 vs 3.5-flash-lite). Contamination tested via
  `scripts/extractor_memorization_probe.py`: 0.0% on both blind arms, identical to a pre-cutoff
  control. **Committed and pushed** (`12bc228e`).
* **Section B** is fully re-scored under 3.6-flash: docstrange 93.8% checkbox / 84.4% strict,
  qwen3-vl-32b 90.3% / 79.4%, qwen3-vl-8b 81.8% / 72.7%. 4,068 cells, 0 errors.
* **All six Gemini models preflight PASS** at `MEDIA_RESOLUTION_HIGH`.
* **`media_resolution` confound is closed** — this was the substantive finding of the Gemini work:
  3.x defaults to HIGH (~1161 image tokens), 2.5 defaults to MEDIUM (~317) on the same page, and the
  budget is fixed rather than resolution-derived (identical counts across a 4x downscale). Not
  reachable through the OpenAI-compat shim, hence the new transport.

---

## 8. Resuming — the order to do things in

1. **Add the missing `GEMINI_SAMPLING` test** (§6), then commit the `direct.py` change together with
   it. Do not commit the sampling change untested.
2. **Get approval before any spend.** The user stopped the work; treat §3's decision as approving the
   *approach*, not as standing authorization to start burning money.
3. **Dry-run first** to re-establish the estimate under the new condition
   (`--registry configs/registry-gemini.yaml`, all six, `--dry-run`). Last estimate at temp 0 was
   **$20.46 for 8,136 cells (±2x)**; vendor-default sampling may change output length, so re-estimate
   rather than reusing that figure.
4. **Pro tier gets a 20-cell smoke** per the user's decision. Inspect for the loop signature:
   `completion_tokens` distribution, `finish_reason == "MAX_TOKENS"` count, repeated text in
   `raw_response`. Only then commit to the full bank.
5. Tier order the user specified: **flash-lite → flash → pro**.
6. Use `scripts/run-gemini-tiers.sh` (`TIERS=lite|flash|pro`, `WORKERS` overridable, default 4). It
   deliberately does **not** use `set -e` — see §9.

---

## 9. Traps that have already cost time today

* **`ocr-eval direct`/`score` exit 2 when any cell errors.** That is the fail-visible contract, not a
  crash. A `set -e` driver script treats it as fatal and silently abandons the remaining legs — this
  happened, killing two of three re-score legs while *appearing* to still run for 25 minutes, because
  rows are written incrementally. Exit **1** means a real crash; exit **2** means cell errors.
* **Upstream `_worker` treats ANY cached error as terminal.** A transient 429 freezes permanently;
  `--force` is the only built-in remedy and re-bills all 1,356 cells. Deleting just the affected rows
  re-attempts only those.
* **A provider may OMIT a token field rather than send 0.** Gemini omits `candidatesTokenCount` when
  the completion is empty; `.get(k, 0)` does NOT default a present-but-None key. Fixed in both the
  transport and `track()` (`d3849534`), but the pattern will recur with any new transport.
* **Preflight a reasoning model with a generous `max_tokens`.** `gemini-3.1-pro-preview` returned
  empty content with `finish_reason=MAX_TOKENS` at 64 tokens — a starved model looks exactly like a
  broken one. Preflight now uses 4096.
* **Google returns an INVALID key as HTTP 400, a MISSING key as 403.** A 400 does not mean "the key
  worked". This produced one wrong diagnosis earlier today.
* **`${VAR:+SET}` expands to the VALUE.** Use `[ -n "$(printenv VAR)" ] && echo set`. The unsafe idiom
  leaked a live key twice.
* **Keys load on demand:** run `gemkey` first (sources a 0600 `~/.config/ocr-eval/secrets.env`).
  A fresh shell has no key.

---

## 10. Comparability caveat that MUST reach the report

Once Gemini rows land at vendor-default sampling, **Section A will contain two sampling regimes**:

* temperature 0.0 — Haiku on Bedrock (88.8% checkbox), the qwen family on OpenRouter (90.7%), every
  Ollama validation row;
* temperature 1.0 / top_p 0.95 — all six Gemini rows.

The report renders a condition hash per row, so this is *visible* rather than hidden. But a
"Gemini beats Haiku" reading would be unsound: that delta spans two sampling regimes and is not a
pure capability comparison. This needs stating in `report.md`'s caveats and in the results doc, in
the same spirit as the existing cross-shape and precision caveats.

Second-order point worth recording honestly: applying a **3.x-scoped** warning to the three 2.5
models is a uniformity choice, not a vendor-backed one. It is the mirror image of the
per-generation-default confound the `media_resolution` work just closed, and a reader deserves to
know which way it cuts.
