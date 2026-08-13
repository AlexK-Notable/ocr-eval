# Which OCR service should we use for checkboxes? — cost/benefit, 2026-08-10

**Written for a reader with no prior context on this benchmark.** Sections 1–3 are the answer.
Sections 4–6 are the caveats you need before quoting any of it.

Full numbers: `results-checkbox-final-2026-08-10.md`. Generated report: `runs/stage1/report.md`.

---

## 0. What was measured, in plain terms

We gave 15 different AI services the same **581 scanned business documents** (insurance forms,
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
cheapest of the accurate ones by a very wide margin: **1/13th the cost of the two models it ties
with**, and **1/10th the cost of DocStrange**, which is materially less accurate.

Three services now sit within measurement error of each other at the top — `claude-sonnet-4.6`
(98.1%), `gemini-3.6-flash` (97.7%) and `gemini-3.5-flash-lite` (97.3%) — and the formal tests find
no difference between any pair. They cost **$8.69, $8.06 and $0.67** for the same work. When
accuracy ties, price decides, and the spread is 13×.

If you already run a document-OCR pipeline and want the best *OCR-shaped* product rather than a
chat model, **DocStrange** is the pick, at roughly 10× the cost and ~2.3× the error rate.

---

## 2. The table

Sorted by accuracy. **Cost is all-in for the full 581-document set** — see §3 on why that matters.

| Service | Correct (of 258) | Accuracy | Errors per 1,000 checkboxes | All-in cost | Cost per 10,000 pages | Calls per answer |
|---|---|---|---|---|---|---|
| **claude-sonnet-4.6** | 253 | **98.1%** | 19 | $8.69 ⁺ | $150 | 1 |
| **gemini-3.6-flash** | 252 | **97.7%** | 23 | $8.06 | $139 | 1 |
| **gemini-3.5-flash-lite** ⭐ | 251 | **97.3%** | 27 | **$0.67** | **$11** | 1 |
| DocStrange (Nanonets) | 242 | 93.8% | 62 | $6.73 | $116 | 2 |
| Mistral OCR 4-0 | 234 | 90.7% | 93 | $3.25 | $56 | 2 |
| qwen3-vl-8b (open-weights) | 234 | 90.7% | 93 | $0.39 | $7 | 1 |
| Mistral OCR 4-0 + annotations | 232 | 89.9% | 101 | $3.83 | $66 | 3 |
| gemini-3.1-pro-preview | 229 | 88.8% | 112 | $9.51 | $164 | 1 |
| claude-haiku-4.5 | 229 | 88.8% | 112 | $2.87 ⁺ | $49 | 1 |
| qwen3-vl-32b (self-run option) | 233 | 90.3% | 97 | *unmetered* | *unmetered* | 2 |
| Mistral OCR 4-1 ⚠ | 221 | 85.7% | 143 | $3.25 | $56 | 2 |
| qwen3-vl-8b on own hardware | 225 | 87.2% | 128 | $0 + hardware | $0 + hardware | 1 |
| qwen3-vl-8b (two-call mode) | 211 | 81.8% | 182 | *unmetered* | *unmetered* | 2 |
| Mistral Document QnA | 208 | 80.6% | 194 | $0.31 | $5 | 1 |
| *(do nothing — always answer "ticked")* | *165* | *64.0%* | *360* | *$0* | *$0* | *0* |

⁺ **Both Bedrock figures come from a single source.** AWS publishes no cost figure through its API
for these models, so the rate was read off the Bedrock pricing page by hand on 2026-08-11 and could
not be confirmed against a second source. Both are also the cost of the 1,336 of 1,356 questions
that recorded usage — the other 20 fail on image size (see §5). Treat them as ±a few percent, not as
billed figures. `claude-sonnet-4.6` was measured on 2026-08-11; `claude-haiku-4.5` read *unpriced*
until then.

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
| claude-haiku-4.5 | +64 | $0.045 |
| Mistral OCR 4-0 | +69 | $0.047 |
| Mistral OCR 4-0 + annotations | +67 | $0.057 |
| Mistral OCR 4-1 | +56 | $0.058 |
| DocStrange | +77 | $0.087 |
| gemini-3.6-flash | +87 | $0.093 |
| claude-sonnet-4.6 | **+88** | $0.099 |
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

### The most important thing this table does NOT price

**The cheap options never produce a page transcript, and the OCR options always do.** That single
asymmetry explains most of the price gap, and it is not a quality difference — it is a difference in
what you are buying.

* A one-call option is handed the page image and returns *only the answer* — around **46 words'
  worth of output**. The page is read and discarded.
* A per-page OCR option transcribes **the entire page** — thousands of characters — whether your
  question needs one field or fifty. You are billed for a complete, reusable document you may not
  want.

Two consequences worth understanding before treating $0.67 vs $3.25 as a like-for-like.

**1. If the cheap option had to produce the transcript, it would cost about the same as the OCR
vendors.** Priced from this run's own numbers, `gemini-3.5-flash-lite` transcribing all 581 pages
would run **$2.0–$3.3** — that is **$0.0034–$0.0057 per page**, against Mistral OCR's advertised
**$0.0040**. Mistral's page rate is not a markup; it is competitive with what a token-billed model
charges to do the same job. (DocStrange's $0.010/page is 2–3× above that range.) The reason is that
writing a transcript costs **26–46× more output** than writing an answer, and output bills at
**8.3× the input rate**. See the full results doc for the derivation and its uncertainty — this
figure is an estimate, not a measured run.

**2. If you want a transcript anyway, the OCR cost stops being attributable to this task.** Building
a search index, an archive, or extracting fifty other fields from the same page? Then the page rate
is shared across all of that work and charging it entirely to "read one checkbox" overstates it.

**So read this table as pricing one specific job: *answer this question about this page*. It does not
price *obtain a reusable transcript*.** For that second job the ranking would look very different,
and nothing here measures transcript quality as a deliverable — only whether the checkbox answer
derived from it was right.

---

## 4. What the accuracy number hides — read before deciding

**Two services with identical accuracy can fail in opposite ways.** For checkboxes this usually
matters more than the headline figure:

| Service | Gets ticked boxes right | Gets empty boxes right | Practical meaning |
|---|---|---|---|
| claude-sonnet-4.6 | 100.0% | 94.6% | Best on both halves of any option measured |
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

* The top three (98.1%, 97.7%, 97.3%) are **statistically tied** — every pairwise test finds no
  difference: sonnet-4.6 vs 3.6-flash p=1.0, sonnet-4.6 vs flash-lite p=0.73, the two Gemini models
  p=1.0. Treat all three as equally accurate and choose on price, where they differ 13×.
* Mistral 4-0 beating 4-1 (90.7% vs 85.7%) is **suggestive but not proven** (p=0.060, and p=0.108
  once three documents with a known text-generation glitch are excluded). Real difference, too small
  a sample to confirm.

The correct phrasing throughout is *"no difference detected at this sample size"* — never
*"equivalent."*

**Five services may score better here than they would for you.** `gemini-3.6-flash`,
`gemini-3.5-flash-lite` and all three Mistral OCR options were released *after* this test set became
public, so their training data may include it. They are best-case ceilings, not neutral baselines.
`gemini-3.1-pro-preview`, Doc QnA, `claude-sonnet-4.6` and `claude-haiku-4.5` predate it and are
clean — which is worth knowing, because the *most accurate* option measured (`claude-sonnet-4.6`,
98.1%) is one of the clean ones. That is the best evidence available here that the ~97–98% scores
are real skill rather than memorised answers.

**Two services fail outright on 20 of the 1,356 questions.** Both Bedrock options
(`claude-sonnet-4.6`, `claude-haiku-4.5`) reject 10 of the 581 documents because the page image
exceeds a hard 5 MB request limit — the same 10 documents for both, deterministically. No checkbox
question happens to fall on those pages, so the accuracy figures above are unaffected, but a
production pipeline would need to downscale those pages first.

**Two services have no usable cost figure.** Two qwen options had their costs recorded as literal
zero by a logging bug (real cost ~$0.70 each). They are shown as *unmetered*, never as $0 — a
printed $0.00 would read as "free." claude-haiku-4.5 was the third until 2026-08-11; it is now
priced, from one source and over 98.5% of the questions (see the ⁺ note in §2).

**The comparison is not perfectly apples-to-apples.** Different services were tested under slightly
different settings (see §Caveats in the full results doc), and the one-call/two-call/three-call
shapes are genuinely different products ranked in one table for convenience.

**These documents are PDFs.** Several options are handed the PDF directly and can read its embedded
text layer where one exists, which is much easier than reading pixels. If your inputs are photos or
scans with no text layer, expect all options to do worse.

---

## 6. If you only remember three things

1. **`gemini-3.5-flash-lite`: 97.3%, $11 per 10,000 pages.** Statistically as accurate as anything
   measured, **~13× cheaper** than either model it ties with, ~10× cheaper than DocStrange, and one
   call instead of two. Adding a third top-tier option (`claude-sonnet-4.6`, 98.1%, $150 per 10,000)
   did not change this: it is nominally the most accurate thing here and the tests cannot separate it
   from an option costing a thirteenth as much.
2. **Advertised OCR prices understate the real cost**, because two- and three-stage pipelines bill at
   every stage. Mistral's "$4 per 1,000 pages" is $56 per 10,000 in practice.
3. **Check the ticked/empty split, not just the headline.** One popular option reads a quarter of all
   *empty* boxes as ticked, which a 88.8% accuracy figure conceals entirely.

**Not worth pursuing further:** every configuration option Mistral exposes was tested and none
improved its checkbox accuracy. Its ceiling on this material is about 90–91% whatever you do — the
details are in `results-checkbox-final-2026-08-10.md`.
