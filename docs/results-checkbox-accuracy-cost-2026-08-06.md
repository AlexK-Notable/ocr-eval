# Checkbox accuracy and cost per leg — 2026-08-06

Twelve complete legs, **1,356 cells each**, all measured against the same pinned bank
(dataset revision `906170ab201d7b8238a32a9115fc66b4b72e0710`).

**Checkbox denominator:** 258 boolean checkbox fields — 165 `true`, 93 `false`.
**Majority-class baseline: 64.0%** (always-answer-`true`). Any leg below that is worse than a
constant.

`cb-acc` is accuracy-over-**all** — errored cells count as incorrect, which is this repo's ranking
key. `strict` is the share of cells where *every* field in the answer was right, so it is a
harder bar than `cb-acc` and moves independently of it.

---

## The table

| Parser | cb-acc | Err | ✓chk | ✓unchk | Strict | Total cost | Per call | Cost basis |
|---|---|---|---|---|---|---|---|---|
| gemini-3.6-flash (native) | **97.7%** | 0.0% | 100.0% | 93.5% | 92.0% | $8.060 ⁺ | $0.00594 | token rates |
| gemini-3.5-flash-lite (native) | 97.3% | 0.0% | 98.8% | 94.6% | 86.2% | $0.666 | $0.00049 | token rates |
| docstrange (transcriber) | 93.8% | 0.0% | 95.8% | 90.3% | 84.4% | $5.810 | $0.00428 | per-page, exact |
| **mistral-ocr-4-0** (transcriber) | **90.7%** | 0.0% | 91.5% | 89.2% | 82.2% | $2.324 | $0.00171 | per-page, exact |
| qwen3-vl-8b (direct, OpenRouter) | 90.7% | 0.0% | 95.8% | 81.7% | 75.1% | $0.390 | $0.00029 | provider cost field |
| qwen3-vl-32b (transcriber) | 90.3% | 0.0% | 93.3% | 84.9% | 79.4% | unknown | unknown | see §Cost gaps |
| gemini-3.1-pro-preview (native) | 88.8% | **10.1%** | 89.7% | 87.1% | 89.0% | $9.509 ⁺ | $0.00701 | token rates |
| claude-haiku-4.5 (Bedrock) | 88.8% | 0.0% | 96.4% | 75.3% | 71.7% | unknown | unknown | see §Cost gaps |
| qwen3-vl-8b-instruct (ollama, local) | 87.2% | 0.0% | 93.3% | 76.3% | 70.3% | $0 (local) | $0 | no API billing |
| **mistral-ocr-4-1** (transcriber) | 85.7% | 0.8% | 86.1% | 84.9% | 80.7% | $2.324 | $0.00171 | per-page, exact |
| qwen3-vl-8b (transcriber) | 81.8% | 0.0% | 77.6% | 89.2% | 72.7% | unknown | unknown | see §Cost gaps |
| **mistral-small-4 (Doc QnA)** | **79.5%** | 0.0% | 87.3% | 65.6% | 67.3% | $0.309 | $0.00023 | token rates |

⁺ **CORRECTED 2026-08-10** — originally published as $3.016 and $4.137. Both understated their real
bill because thinking tokens (`thoughtsTokenCount`) are billed as output and were not counted. See
`results-checkbox-final-2026-08-10.md` §Correction. `gemini-3.5-flash-lite` emits none, so its
$0.666 was always correct.

**Session total: $25.04** (originally reported $14.63, before the thinking-token correction) against
a $25 authorization.

### Costs are per-leg totals for the legs run today

$4.65 Mistral OCR (both pins, exact per-page) · $9.51 gemini-3.1-pro · $8.06 gemini-3.6-flash ·
$1.85 extractor · $0.67 gemini-3.5-flash-lite · $0.31 Doc QnA.

---

## Cost gaps — read before quoting any "per call" figure

**Three legs report `unknown`, not `$0`.** The distinction matters: a printed `$0.0000` would read
as "this leg was free."

* **qwen3-vl-8b / qwen3-vl-32b transcribers** carry per-token rates in the registry, but their parse
  records persisted only `cost_usd: 0.0` — upstream writes `result.cost_estimate_usd or 0.0`, which
  converts an unknown into a literal zero, and the token counts were never stored. Real spend was
  roughly $0.70 per leg. **Not recoverable from disk**; it would take a re-run to measure.
* **claude-haiku-4.5** is unpriced by design — Bedrock Converse returns no cost field and
  `pricing:GetProducts` is IAM-denied in this account. Its token counts *are* recorded
  (2,252,308 in / 71,464 out) so it can be priced later from published rates.
* **qwen3-vl-8b-instruct (ollama)** is a genuine `$0` — a `local: true` entry, run on local
  hardware, no API billing. This is a real zero, unlike the two above.

**Transcriber "per call" understates the true cost.** A transcriber leg is two paid steps: OCR
(shown here) plus one `gemini-3.6-flash` extractor call per cell. **Extractor cost is not recorded
per row** — the score rows carry only an `extractor` id stamp, no usage or cost field — so it cannot
be attributed per leg. The ~$1.85 measured for 2,712 cells works out to roughly **$0.00068/cell**,
which should be added to every transcriber row above for a like-for-like comparison against the
one-call vlm-chat legs.

---

## What the numbers say

### The two Gemini flash models are statistically tied on checkboxes

McNemar over the 258 shared fields: 252 vs 251 correct, **p=1.0**, 96.5% agreement. The 97.7% vs
97.3% gap is noise. But **3.5-flash-lite costs 12× less** ($0.666 vs $8.060 per full bank), so on
checkbox state specifically it is the better buy. They *do* separate on `strict` whole-answer
accuracy — 92.0% vs 86.2% — which is a different claim about a harder task.

### mistral-ocr-4-0 beats 4-1, and `mistral-ocr-latest` points at the loser

234 vs 221 correct, discordant 27/14, **p=0.060**. That is **not significant at α=0.05** — the
honest statement is "not detected at n=258", never "equivalent". The direction held from partial
data through to full n, and it has a practical consequence: `mistral-ocr-latest` resolves to
**4-1**, the apparently weaker pin, so anyone trusting the alias silently gets it. Both are
undeprecated and they are demonstrably not interchangeable (4-0 keeps `ACORD®` where 4-1 emits
`ACORD`).

### Document QnA was the hypothesis test, and it failed

Doc QnA was built to test whether Mistral's *markdown flattening* — not its OCR engine — costs the
transcriber legs their accuracy. Letting Mistral read the document and answer directly, with no
markdown bottleneck and no external extractor, made things **worse**: 79.5%, last of twelve.

* vs mistral-ocr-4-1 (the leg it was meant to beat): 221 vs 205, **p=0.052**
* vs mistral-ocr-4-0: 234 vs 205, **p=0.00026**

So flattening is not the explanation. Its unchecked-box accuracy of **65.6%** is the worst of any
complete leg, meaning it over-reads boxes as checked. Cheapest leg measured ($0.00023/call, 15×
cheaper than the per-page OCR) and the least accurate.

> **CORRECTION 2026-08-07 — this section named the wrong mechanism.** See
> "Correction: why Mistral OCR 4 loses checkboxes" below. Flattening is real but is *not* the
> dominant failure, which is why removing it could not help. The dominant failure is wholesale
> glyph omission, and the premise that Doc QnA sees page pixels is unverified.

### Polarity bias separates legs that share a headline number

Two legs tie at 90.7% cb-acc with completely different profiles: **mistral-ocr-4-0** is balanced
(91.5% checked / 89.2% unchecked) while **qwen3-vl-8b direct** is skewed (95.8% / 81.7%).
claude-haiku-4.5 is the most skewed (96.4% / 75.3%) — it reads boxes as checked. A single accuracy
figure hides this, and for a checkbox benchmark the asymmetry is often the more useful number.

### The pro leg's error rate is the reason it ranks low

gemini-3.1-pro-preview has a **10.1% error rate** — 43 parse errors, 3 API errors, 1 empty response
out of 1,356. Errors count as incorrect under accuracy-over-all, which is why its 88.8% `cb-acc`
sits *below* its 89.0% `strict` figure. Its accuracy-over-answered is far higher. Those rows are
recoverable by deleting just the errored ones and re-running.

---

## Correction: why Mistral OCR 4 loses checkboxes (2026-08-07)

**The mechanism recorded on 2026-08-06 was wrong.** It was written up as *glyph position* — Mistral
flattening a block into one pipe-delimited table cell and dropping the leading glyph, producing
**polarity inversion**. That claim is in `ocr_eval_ext/mistral_docqna.py`'s module docstring and in
`configs/registry-mistral-docqna.yaml`'s header, both of which state it as settled. Flattening does
happen, but it is not what costs the legs their accuracy.

**The dominant failure is glyph OMISSION.** Mistral transcribes the labels correctly and does not
emit the checkbox character. `finance_74`, same page, same question, both feeding the same
`gemini-3.6-flash` extractor:

```
mistral_ocr_4_0                       docstrange_sync
  Select all that apply                 Select all that apply
  Methane (CH4)                         ☑ Methane (CH4)
  Nitrous oxide (N2O)                   ☑ Nitrous oxide (N2O)
  Carbon dioxide (CO2)                  ☑ Carbon dioxide (CO2)
```

Two details make this diagnostic rather than anecdotal. First, **four lines earlier on the same
page Mistral does emit `☑ Organization-wide`** — so it is not blind to checkboxes, it dropped them
on this construct. Second, `finance_74` is a plain list: no pipes, no table, nothing flattened. The
mechanism cannot be flattening.

`finance_5` shows the same loss inside a table cell — Mistral emits
`SECONDARY PHONE # (469) 555-0635` where DocStrange emits
`SECONDARY PHONE # ☐ HOME ☐ BUS ☑ CELL`. The option glyphs vanish; the phone number survives.

### Why it produces false negatives, not inversions

Unmarked labels read as `false`. Of `mistral_ocr_4_0`'s 24 wrong checkbox fields: **10 missed a
true mark**, 4 invented one, 10 other/null. DocStrange is balanced at 7 vs 5. Omission has a
direction; inversion would not.

### Aggregate statistics hide this almost perfectly

Every summary figure says the transcripts are comparable — which is why the first pass missed it:

| Measure | mistral_ocr_4_0 | docstrange_sync |
|---|---|---|
| docs containing ≥1 checkbox glyph | 59.0% (343/581) | 62.0% (360/581) |
| total glyphs emitted | 7,570 | 9,110 |
| glyph immediately before a label | **68.9%** | 61.7% |

Mistral scores *better* on glyph-label adjacency, the very statistic the flattening hypothesis
predicted it would lose.

### Omission is mostly PARTIAL, and that is what the first pass got wrong

The first version of this section measured omission as "docs where Mistral emitted **zero** glyphs
while DocStrange emitted ≥1" — 29 docs, holding only 19 of the 258 booleans, where the two legs
tie. It concluded omission's contribution to the gap was "not established". **That conclusion was
an artefact of the wrong measure.** Total absence is the rare case; the common one is a document
losing *some* of its glyphs.

Measuring Mistral's glyph count as a FRACTION of DocStrange's on the same document (n=360 docs
where DocStrange found ≥1 glyph):

| glyph retention | docs |
|---|---|
| 0% (total loss) | 29 |
| 1–50% (major partial) | 26 |
| 50–90% | 36 |
| 90–110% (parity) | 243 |
| >110% (Mistral emits more) | 26 |

**62 docs lose glyphs partially against 29 losing them entirely**, so the zero-glyph test missed
two thirds of the affected corpus.

### Glyph retention predicts checkbox accuracy, and explains the whole gap

Splitting the 258 booleans by their document's retention ratio:

| subset | n | mistral_ocr_4_0 | docstrange_sync | gap |
|---|---|---|---|---|
| glyph loss ≥10% (retention <90%) | 70 | **75.7%** | 88.6% | **+12.9 pt** |
| parity or better (retention ≥90%) | 181 | **96.1%** | 96.1% | **0.0 pt** |

On documents where Mistral keeps its glyphs the two engines are **exactly tied at 96.1%** —
Mistral has no checkbox deficit at all. The entire 90.7% vs 93.8% gap is carried by the 70 fields
on glyph-losing documents. McNemar on that subset: DocStrange-only-right 14 vs
Mistral-only-right 5, **p=0.064** — marginal at n=70, not significant at α=0.05, so read this as
"consistent with, and localised by, glyph loss", not as a proven effect size.

**Ceiling.** 17 of `mistral_ocr_4_0`'s 24 wrong booleans (**71% of its errors**) sit on glyph-loss
docs. Repairing all of them would put it at **97.3%**, above DocStrange's 93.8%. That is an upper
bound on the defect's cost, not a prediction — it assumes every such error is glyph-caused.

Method, so it can be checked: retention ratio = (count of ☐☑☒□▢■✓✔ in
`runs/stage1/parses/mistral_ocr_4_0/<stem>.md`) ÷ (same count in
`runs/stage1/parses/docstrange_sync/<stem>.md`), per stem, docs with a zero denominator excluded.
DocStrange is used as the reference transcript, not as ground truth — the ratio measures
disagreement between two engines on the same page, so a doc where DocStrange itself over-emits
would appear here as Mistral "losing" glyphs. Confirming against the PDFs is the outstanding
follow-up.

### A second premise that turns out to be unverified

`mistral_docqna.py` states that Document QnA "passes the extracted text PLUS the page image to a
vision model", which was the whole rationale for the transport: a model seeing pixels could
recover glyph-label association that markdown lost. **Mistral's docs do not say this.** The
Document QnA page says only *"The extracted document content is analyzed by a large language
model"* — no mention of pixels reaching the model.

Our own data is consistent with text-only: on the zero-glyph docs, where a pixel-reading model
should have a decisive advantage over a transcript with no glyphs in it, Doc QnA scores **17/19 —
exactly tying OCR 4-0** (n=19, nothing detectable). If Doc QnA is text-only, it is not an
independent test of flattening at all; it is the same OCR output read by a different model, which
would explain why it could not beat the leg it was built to beat.

**Do not quote the "sees the page image" claim until it is verified.** The registry header and the
transport docstring both need this correction.

### Not to be confused with the repetition defect

Separately, `mistral_ocr_4` (4-1) emits a degenerate repetition loop on `finance_1` — the phrase
`the following services: (N)` repeated to N=1000, a 31,634-char transcript from one page whose
source PDF has **no text layer at all**. 4-0 emits that phrase zero times on the same page.
Phrase-repetition ≥20× affects 3 docs per pin. Excluding loop-affected docs moves the 4-0 vs 4-1
comparison from **p=0.060 to p=0.108** (234 vs 221 → 231 vs 220, n=255), so three documents carry
a noticeable share of that result. The direction holds; the evidence is weaker than the headline
p-value suggests. Temperature is not exposed on `/v1/ocr`, so this is not tunable from our side.

### Vendor guidance: there is none

Checked 2026-08-07 — Mistral's OCR basic-usage page, annotations page, Document QnA page, and the
OCR launch announcement. **None mentions checkboxes, checkmarks, tick marks, radio buttons,
selection marks, or form-field state.** No documented glyph convention, and no stated limitation
covering them. The documented limitations are about feature availability by version
(`table_format`/headers need OCR 2512+, `include_blocks` needs OCR 4+) and about images/tables
being replaced by placeholders. So the omission is undocumented behaviour rather than a
configuration we chose wrongly — there is no setting the docs point to that would change it.

---

## Caveats that travel with this table

1. **Two sampling regimes.** All four Gemini/Doc-QnA rows and the Mistral legs are at vendor-default
   sampling; Haiku, the qwen family and the ollama rows were measured at `temperature: 0.0`. A
   cross-provider delta here is not a pure capability comparison.
2. **Contamination.** `gemini-3.6-flash` and `gemini-3.5-flash-lite` (both released 2026-07-21) and
   both Mistral OCR 4 pins post-date `CONTAMINATION_CUTOFF` (2026-05-24). They are **ceiling
   anchors, not clean baselines**. `gemini-3.1-pro-preview` (2026-02-19) and `mistral-small-4`
   (2026-03) are pre-cutoff.
3. **Shapes are not interchangeable.** `vlm-chat` legs answer in one call; `transcriber` legs
   OCR-then-extract in two. Doc QnA is `vlm-chat` with provider-side OCR — a third shape again.
   Ranking them in one table is convenient, not rigorous.
4. **pdf-direct free ride.** Doc QnA and the OCR legs upload the PDF, so a born-digital file may be
   read from its embedded text layer rather than from pixels — a materially easier task than OCR-ing
   a scan.
5. **`gemini-3.1-pro-preview` is a preview endpoint** and can be withdrawn without notice, so its
   row may not be reproducible later.
