# Which OCR service should we use for checkboxes? — cost/benefit, 2026-08-10

**Written for a reader with no prior context on this benchmark.** Sections 1–3 are the answer.
Sections 4–6 are the caveats you need before quoting any of it.

Full numbers: `results-checkbox-final-2026-08-10.md`. Generated report: `runs/stage1/report.md`.

---

## 0. What was measured, in plain terms

We gave 14 different AI services the same **581 scanned business documents** (insurance forms,
mortgage paperwork, medical and food-safety inspection reports) and asked each the same **1,356
questions** about them.

This analysis scores one specific skill: **reading whether a checkbox is ticked or empty.** There are
**258 checkbox questions** with a known right answer. That is the only thing scored below.

Two things make that skill worth isolating:

* It is **hard for AI**. A tick is a few dark pixels inside a small box, easily lost.
* It is **high-consequence**. "Coverage declined ☑" versus "☐" inverts the meaning of a document.
  A wrong checkbox is not a typo, it is a wrong fact.

**The number to beat is 64%.** Of the 258 boxes, 165 are ticked. So a service that ignored the
documents entirely and always answered "ticked" would score **64% for free**. Any option scoring
near that is worthless regardless of price.

---

## 1. The answer

**Use `gemini-3.5-flash-lite`.** It is the most accurate option to within measurement error, and the
cheapest of the accurate ones by a very wide margin: **1/12th the cost of the model it ties with**,
and **1/10th the cost of DocStrange**, which is materially less accurate.

If you already run a document-OCR pipeline and want the best *OCR-shaped* product rather than a
chat model, **DocStrange** is the pick, at roughly 10× the cost and ~2.3× the error rate.

---

## 2. The table

Sorted by accuracy. **Cost is all-in for the full 581-document set** — see §3 on why that matters.

| Service | Correct (of 258) | Accuracy | Errors per 1,000 checkboxes | All-in cost | Cost per 10,000 pages | Calls per answer |
|---|---|---|---|---|---|---|
| **gemini-3.6-flash** | 252 | **97.7%** | 23 | $8.06 | $139 | 1 |
| **gemini-3.5-flash-lite** ⭐ | 251 | **97.3%** | 27 | **$0.67** | **$11** | 1 |
| DocStrange (Nanonets) | 242 | 93.8% | 62 | $6.73 | $116 | 2 |
| Mistral OCR 4-0 | 234 | 90.7% | 93 | $3.25 | $56 | 2 |
| qwen3-vl-8b (open-weights) | 234 | 90.7% | 93 | $0.39 | $7 | 1 |
| Mistral OCR 4-0 + annotations | 232 | 89.9% | 101 | $3.83 | $66 | 3 |
| gemini-3.1-pro-preview | 229 | 88.8% | 112 | $9.51 | $164 | 1 |
| claude-haiku-4.5 | 229 | 88.8% | 112 | *unpriced* | *unpriced* | 1 |
| qwen3-vl-32b (self-run option) | 233 | 90.3% | 97 | *unmetered* | *unmetered* | 2 |
| Mistral OCR 4-1 ⚠ | 221 | 85.7% | 143 | $3.25 | $56 | 2 |
| qwen3-vl-8b on own hardware | 225 | 87.2% | 128 | $0 + hardware | $0 + hardware | 1 |
| qwen3-vl-8b (two-call mode) | 211 | 81.8% | 182 | *unmetered* | *unmetered* | 2 |
| Mistral Document QnA | 208 | 80.6% | 194 | $0.31 | $5 | 1 |
| *(do nothing — always answer "ticked")* | *165* | *64.0%* | *360* | *$0* | *$0* | *0* |

⚠ **Mistral OCR 4-1 is what you get by default.** Asking for `mistral-ocr-latest` gives you 4-1, the
weaker of the two live Mistral versions. You must name `mistral-ocr-4-0` explicitly. (Caveat: the
4-0/4-1 difference is suggestive, not statistically proven — see §5.)

### Is the extra accuracy worth the money?

Measured against the free 64% baseline — how much you pay for each checkbox error you avoid:

| Service | Errors avoided vs. free baseline | Cost per error avoided |
|---|---|---|
| qwen3-vl-8b (one-call) | +69 | **$0.006** |
| Mistral Document QnA | +43 | $0.007 |
| **gemini-3.5-flash-lite** ⭐ | **+86** | **$0.008** |
| Mistral OCR 4-0 | +69 | $0.047 |
| Mistral OCR 4-0 + annotations | +67 | $0.057 |
| Mistral OCR 4-1 | +56 | $0.058 |
| DocStrange | +77 | $0.087 |
| gemini-3.6-flash | +87 | $0.093 |
| gemini-3.1-pro-preview | +64 | $0.149 |

Read this column carefully: **cheap-and-inaccurate options look efficient here.** Doc QnA has a fine
cost-per-error-avoided and is still the second-worst option available, because it avoids only 43 of
the 93 possible errors. Efficiency per unit is not the same as being good enough.

**`gemini-3.5-flash-lite` is the only option that is simultaneously near-best on accuracy and
near-best on efficiency.** That is what makes it the answer rather than a compromise.

---

## 3. Why "cost per page" quotes are misleading

Services here work in one of three shapes, and a headline price usually covers only the first step.

| Shape | What happens | Who does this |
|---|---|---|
| **One call** | Model reads the page and answers the question directly | Gemini, Claude, qwen, Doc QnA |
| **Two calls** | Service 1 converts the page to text; service 2 reads that text and answers | DocStrange, Mistral OCR, qwen transcriber |
| **Three calls** | OCR → a second AI extracts structured fields → a third answers | Mistral OCR + annotations |

**Two-call options bill twice.** Mistral OCR's advertised price is $4 per 1,000 pages, which works
out to $2.32 here — but you also pay ~$0.92 for the second step, so the real figure is **$3.25**.
The table above already includes this. Vendor pricing pages do not.

**"Thinking" models bill for text you never see.** Some models reason internally before answering,
and that hidden reasoning is billed at the output rate. `gemini-3.6-flash` produced **672,517
thinking tokens** across this run — **11× more than its visible answers** — and
`gemini-3.1-pro-preview` produced 447,654. Counting only the visible answer understates those two
bills by **167% and 130%**. The table above counts both.

This is the single biggest reason the recommendation is as lopsided as it is:
`gemini-3.5-flash-lite` does no hidden reasoning at all, so it is **12× cheaper than the model it
statistically ties with** — not the ~4.5× a visible-tokens-only comparison suggests.

**A subtle consequence:** more stages means more accuracy loss, not less. Each hop can drop
information. The three-call Mistral option scores *worse* than the two-call one it was built to
improve.

---

## 4. What the accuracy number hides — read before deciding

**Two services with identical accuracy can fail in opposite ways.** For checkboxes this usually
matters more than the headline figure:

| Service | Gets ticked boxes right | Gets empty boxes right | Practical meaning |
|---|---|---|---|
| gemini-3.6-flash | 100.0% | 93.5% | Never misses a tick; occasionally invents one |
| gemini-3.5-flash-lite | 98.8% | 94.6% | Most balanced of the accurate options |
| claude-haiku-4.5 | 96.4% | **75.3%** | **Strongly biased toward "ticked"** — 1 in 4 empty boxes read as ticked |
| Mistral OCR 4-0 | 91.5% | 89.2% | Balanced but less accurate overall |
| Mistral OCR + annotations | 87.9% | **93.5%** | Best in class on *empty* boxes, worse on ticked |
| Mistral Document QnA | 88.5% | **66.7%** | Worst on empty boxes of any option |

Two services in the table tie at exactly 90.7% with completely different behaviour. If your documents
are mostly *unticked* — a compliance form where a tick is the exception — an option with 75% accuracy
on empty boxes is far worse for you than its headline suggests.

**Decide which error you can least afford before choosing on accuracy alone.**

---

## 5. Honest limits on these numbers

**The sample is small.** 258 checkbox questions. Differences of 1–2 percentage points are not real
differences:

* The two Gemini models (97.7% vs 97.3%) are **statistically tied** — a formal test finds no
  difference at all (p=1.0). Treat them as equally accurate and choose on price.
* Mistral 4-0 beating 4-1 (90.7% vs 85.7%) is **suggestive but not proven** (p=0.060, and p=0.108
  once three documents with a known text-generation glitch are excluded). Real difference, too small
  a sample to confirm.

The correct phrasing throughout is *"no difference detected at this sample size"* — never
*"equivalent."*

**Four services may score better here than they would for you.** `gemini-3.6-flash`,
`gemini-3.5-flash-lite` and all three Mistral OCR options were released *after* this test set became
public, so their training data may include it. They are best-case ceilings, not neutral baselines.
`gemini-3.1-pro-preview` and Doc QnA predate it and are clean.

**Three services have no usable cost figure.** Two qwen options had their costs recorded as literal
zero by a logging bug (real cost ~$0.70 each); claude-haiku's usage was recorded but no verified
price rate was available. These are shown as *unpriced* / *unmetered*, never as $0 — a printed $0.00
would read as "free."

**The comparison is not perfectly apples-to-apples.** Different services were tested under slightly
different settings (see §Caveats in the full results doc), and the one-call/two-call/three-call
shapes are genuinely different products ranked in one table for convenience.

**These documents are PDFs.** Several options are handed the PDF directly and can read its embedded
text layer where one exists, which is much easier than reading pixels. If your inputs are photos or
scans with no text layer, expect all options to do worse.

---

## 6. If you only remember three things

1. **`gemini-3.5-flash-lite`: 97.3%, $11 per 10,000 pages.** Statistically as accurate as anything
   measured, **~12× cheaper** than the model it ties with, ~10× cheaper than DocStrange, and one call
   instead of two.
2. **Advertised OCR prices understate the real cost**, because two- and three-stage pipelines bill at
   every stage. Mistral's "$4 per 1,000 pages" is $56 per 10,000 in practice.
3. **Check the ticked/empty split, not just the headline.** One popular option reads a quarter of all
   *empty* boxes as ticked, which a 88.8% accuracy figure conceals entirely.

**Not worth pursuing further:** every configuration option Mistral exposes was tested and none
improved its checkbox accuracy. Its ceiling on this material is about 90–91% whatever you do — the
details are in `results-checkbox-final-2026-08-10.md`.
