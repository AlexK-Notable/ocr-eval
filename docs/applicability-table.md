# Condition-axis applicability by measurement shape

Referenced by the design spec (rev 2). Each condition axis binds to exactly one pipeline step per
shape; cells the axis cannot meaningfully vary are **skipped by the runner**, never silently run.

| Axis | `vlm-chat` binds to | `transcriber` binds to | Skipped cells |
|---|---|---|---|
| `preprocess` (raw / deskew) | the page image sent to the model | the page image sent to the transcriber (extractor never sees images) | none — applies to both shapes |
| `output_contract` (schema_prompted / schema_native / lenient) | the answering model's output | **the extractor's output** (the transcript contract is fixed: upstream `_MARKDOWN_PROMPT`, never varied by this axis) | `schema_native` × any provider without native structured output support (recorded per-registry, cell skipped not silently downgraded) |
| `render.dpi` | page render | page render | none (fixed at 150 in Stages 1–2; a DPI sweep would be a new axis value, not a new mechanism) |
| `sampling` + `sample_index` (K-sample) | the answering model | **the transcriber** (extractor stays temperature 0 always — extractor variance must never contaminate transcriber agreement) | K>1 × deterministic engines (Tesseract, PP-OCR classical): agreement is degenerately 1.0 and would poison the agreement-vs-accuracy table |
| `no_image` (control) | omit the image | — | all transcriber cells (a transcriber without an image has no input) |

Invariants:
- The transcript contract (`_MARKDOWN_PROMPT`) is constant across all conditions and stages; changing it is a new `extractor_generation`-class event, not a condition.
- The extractor is always `temperature 0`, always the pinned model. K-sampling never resamples the extractor.
- Every cell's full condition dict (including the axis values above) is in its cache key; a skipped cell appears in the run plan output as `skipped (inapplicable)`, so matrices are visibly, never silently, partial.
