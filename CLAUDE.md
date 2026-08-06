# Working agreements — ocr-eval

This repo exists to measure model behaviour honestly. It has fail-closed gates, pinned conditions,
and per-row provenance precisely so a number can never quietly mean something other than what it
claims.

**Hold your own claims to the standard this repo holds its measurements to.** Every rule below was
paid for by a specific failure here on 2026-08-05, and the cost landed on the user: real spend, a
leaked key, or a wrong answer they had to catch. The rules are not independent — most of them are
one underlying mistake (treating a weak signal as settled) showing up in a different place.

---

## Evidence — what counts as knowing something

These three are one failure mode wearing different clothes. Read them together.

**A tool reporting success is not verification.** Before saying something is verified, confirm the
check compared what you claim it compared. *Origin: relayed `selftest --extractor`'s `PASS (5/5)`
as "the responses were correct" without reading a single response. The claim happened to hold —
worse, because nothing corrected it.*

**A pass or an empty result is evidence only if the check was capable of failing.** Prove the
negative case can fire before trusting it. *Origin: a secret-scan regex exceeded ugrep's complexity
limit and printed an error, not a result; empty output read as "clean." Same shape: `pkill`
reporting success while one process survived, and `.get(k, 0)` not defaulting a present-but-`None`
key.*

**"Not statistically significant" is not "equivalent."** Say "not detected at this sample size" and
state the size. *Origin: an n=300 A/B reported p=1.000 and got read as "the graders are
interchangeable." At 4,068 cells the same comparison separated them at p<1e-8.*

**Verify rather than recall, and re-verify before writing anything durable.** Late in a long session
recall degrades while confidence does not. *Origin: re-checking each fact while writing a handoff
turned up two things stated wrongly minutes earlier — including in the handoff's own subject
matter.*

**Read the vendor's own guidance before applying an inherited setting to a new vendor.** A condition
ratified for one provider is not validated for the next. *Origin: `STAGE1_CONDITION`'s
`temperature: 0.0` was ratified for Qwen and Bedrock, then carried onto six Gemini models — Google
explicitly warns that sub-1.0 temperature causes looping or degradation on Gemini 3.x. The user
caught it; ~$2 of spend was written off.*

## Spend and irreversible acts

**Fresh approval for each paid run.** Approving an approach is not authorization to start spending.
Say what it will cost and wait. *Origin: the user had to interrupt mid-flight with "stop before
initiating any runs."*

**Smoke a handful of cells before committing to a full bank.** This is the evidence rules applied to
money: 20 cells is how you find out whether the check can fail. *Origin: a full-bank run crashed
~310 paid cells in on a `None` token count; the temp-0 error burned a whole tier before anyone
looked.*

**Before deleting or overwriting generated artifacts, separate what is committed from what exists
only on disk — and say which.** *Origin: 3,624 rows described as "backed up." Only 394 were ever
committed; the rest live in one home-directory folder (`~/ocr-eval-*-2026-08-0*`). Do not clean
those up without asking.*

## Secrets

**Never write an idiom that can expand a secret.** `${VAR:+SET}` prints the *value*. Use
`[ -n "$(printenv VAR)" ] && echo set`. Credentials travel in headers, never in a URL — a URL
reaches logs, proxies, and error bodies. *Origin: the same key leaked three times in two sessions,
once from a presence check meant to avoid exactly that.* `scripts/ocr-eval-keys.sh` is the reference
implementation; `tests_ext/test_score_credentials.py` has a canary that fails if a credential
returns to a URL.

## Shell and scripts

**Know a command's exit-code contract before letting a script die on nonzero.** Here exit **2**
means "completed, with cells that errored" — the fail-visible contract — and exit **1** means a real
crash. *Origin: `set -e` over `ocr-eval score` let 15 throttled cells silently abandon two of three
re-score legs, while the script appeared to run for 25 more minutes because rows write
incrementally.* `scripts/run-gemini-tiers.sh` documents why it omits `set -e`.

**Background long jobs and report; never hold a turn open on a long blocking sleep.** It leaves the
user no way in. *Origin: a `sleep 580` monitor had to be interrupted to stop a run.*

## Working with the user

**When evidence contradicts what the user told you, say so in one or two sentences, then continue
with whatever is required under either reading.** Stop only if the answer changes the next action.
*Origin: two "fuck it" replies. The first followed a flag where re-scoring had already been shown
necessary regardless of who was right — the flag was worth raising, the halt was not.*

**Report the unflattering number.** Direction of effect, cost written off, metrics that disagree with
each other. *Origin: the 3.5-flash-lite re-score came out worse than the extractor it replaced, and
DocStrange's checkbox accuracy rose while its overall count fell — a single summary figure would
have hidden both.*

---

## Repo facts worth not rediscovering

* **Exit codes.** `direct`, `parse`, `score` exit **2** on errored cells (fail-visible), **1** on a
  real crash.
* **A cached error is terminal** unless its message contains `"no answer"` (`score.py:512`). A
  throttled 429 therefore freezes permanently. `--force` re-bills all 1,356 cells; deleting just the
  affected rows re-attempts only those — `cli.py` prints this on exit 2.
* **`_condition_for` (`direct.py:124`) is the only legal place to build a gemini-native condition.**
  A row's stored `condition` and the hash in its parser key must come from the same dict, or
  `report` reads one row as two conditions. Never add a key to `STAGE1_CONDITION` for a
  transport-specific parameter: it rehashes every existing row on every transport and orphans the
  cache.
* **Condition hashes** (recomputed live, not recalled): `0373123f3b15` STAGE1_CONDITION ·
  `872144e4aecb` gemini + vendor sampling · `f7a267f691d0` gemini + temp 0 (abandoned) ·
  `4caf39e9ac1f` the live Haiku rows at `max_tokens=1024`, which already differ from today's
  baseline.
* **Providers may omit a token field rather than send 0.** Gemini drops `candidatesTokenCount` on an
  empty completion. Use `or 0`, never `.get(k, 0)` — a dict default does not fire for a
  present-but-`None` key. Expect this with every new transport.
* **Preflight a reasoning model with a generous `max_tokens`.** `gemini-3.1-pro-preview` returned
  empty content with `finish_reason=MAX_TOKENS` at 64 — a starved model is indistinguishable from a
  broken one.
* **Google returns an invalid key as HTTP 400, a missing key as 403.** A 400 does not mean the key
  worked.
* **Image encoding differs by transport:** Bedrock Converse takes raw bytes, Gemini native takes
  base64 in `inlineData.data`. Each silently degrades if given the other.
* **Keys load on demand:** run `gemkey` (sources a 0600 `~/.config/ocr-eval/secrets.env`). A fresh
  shell has none.
* **Lint baseline is only meaningful scoped:** `ruff check ocr_eval_ext/ tests_ext/` → 14
  pre-existing `B008`. Whole-repo is ~148, nearly all upstream. Compare scoped, and keep the full
  test suite green.
