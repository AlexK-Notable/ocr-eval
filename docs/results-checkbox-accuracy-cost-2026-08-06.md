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
| gemini-3.6-flash (native) | **97.7%** | 0.0% | 100.0% | 93.5% | 92.0% | $3.016 | $0.00222 | token rates |
| gemini-3.5-flash-lite (native) | 97.3% | 0.0% | 98.8% | 94.6% | 86.2% | $0.666 | $0.00049 | token rates |
| docstrange (transcriber) | 93.8% | 0.0% | 95.8% | 90.3% | 84.4% | $5.810 | $0.00428 | per-page, exact |
| **mistral-ocr-4-0** (transcriber) | **90.7%** | 0.0% | 91.5% | 89.2% | 82.2% | $2.324 | $0.00171 | per-page, exact |
| qwen3-vl-8b (direct, OpenRouter) | 90.7% | 0.0% | 95.8% | 81.7% | 75.1% | $0.390 | $0.00029 | provider cost field |
| qwen3-vl-32b (transcriber) | 90.3% | 0.0% | 93.3% | 84.9% | 79.4% | unknown | unknown | see §Cost gaps |
| gemini-3.1-pro-preview (native) | 88.8% | **10.1%** | 89.7% | 87.1% | 89.0% | $4.137 | $0.00305 | token rates |
| claude-haiku-4.5 (Bedrock) | 88.8% | 0.0% | 96.4% | 75.3% | 71.7% | unknown | unknown | see §Cost gaps |
| qwen3-vl-8b-instruct (ollama, local) | 87.2% | 0.0% | 93.3% | 76.3% | 70.3% | $0 (local) | $0 | no API billing |
| **mistral-ocr-4-1** (transcriber) | 85.7% | 0.8% | 86.1% | 84.9% | 80.7% | $2.324 | $0.00171 | per-page, exact |
| qwen3-vl-8b (transcriber) | 81.8% | 0.0% | 77.6% | 89.2% | 72.7% | unknown | unknown | see §Cost gaps |
| **mistral-small-4 (Doc QnA)** | **79.5%** | 0.0% | 87.3% | 65.6% | 67.3% | $0.309 | $0.00023 | token rates |

**Session total: $14.63** against a $25 authorization. No spend cap tripped.

### Costs are per-leg totals for the legs run today

$4.65 Mistral OCR (both pins, exact per-page) · $4.14 gemini-3.1-pro · $3.02 gemini-3.6-flash ·
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
97.3% gap is noise. But **3.5-flash-lite costs 4.5× less** ($0.666 vs $3.016 per full bank), so on
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
