"""Mistral Document QnA transport (`transport: mistral-docqna`).

⚠️ CORRECTION 2026-08-07 — READ THIS BEFORE THE RATIONALE BELOW. Two of this docstring's load-
bearing claims did not survive checking, and the section under "WHY THIS EXISTS" is preserved as
written only because it records what was believed when the transport was built.

  1. FLATTENING IS NOT THE DOMINANT FAILURE. The real mechanism is glyph OMISSION: Mistral
     transcribes checkbox labels and does not emit the checkbox character. On `finance_74` — a
     plain list, no table, nothing flattened — it emits bare `Methane (CH4)` where DocStrange emits
     `☑ Methane (CH4)`, while four lines earlier on the SAME page it does emit
     `☑ Organization-wide`. Omission yields FALSE NEGATIVES (10 of 4-0's 24 wrong fields missed a
     true mark; only 4 invented one), not the polarity inversions described below. Mistral also
     scores BETTER than DocStrange on glyph-immediately-before-a-label (68.9% vs 61.7%) — the
     opposite of what the flattening story predicts.

     Omission is mostly PARTIAL, which is easy to mis-measure: only 29 docs lose ALL their glyphs,
     but 62 lose SOME (retention <90% of DocStrange's count on the same page). Splitting the 258
     booleans on that ratio localises the entire accuracy gap: where retention is >=90% the two
     engines TIE at 96.1% (n=181); where glyphs are lost Mistral is 75.7% vs 88.6% (n=70,
     McNemar 14/5, p=0.064 — marginal, not significant). So Mistral has no intrinsic checkbox
     deficit; it has a transcription defect on specific pages.

  2. "PASSES THE TEXT PLUS THE PAGE IMAGE" IS UNVERIFIED. Mistral's Document QnA docs say only
     "The extracted document content is analyzed by a large language model" — nothing about pixels
     reaching the model. Our own data is consistent with text-only: on the 29 docs where the OCR
     leg emitted ZERO glyphs, Doc QnA scores 17/19 checkbox booleans, exactly tying OCR 4-0 (n=19,
     nothing detectable) — where a pixel-reading model should have had a decisive edge. If Doc QnA
     is text-only then this transport is not an independent test of flattening at all, which would
     explain why it lost to the leg it was built to beat (80.6% vs 90.7%).

The transport is still worth having — it is a real second measurement shape on the same vendor, and
it is the cheapest leg in the bench — but do not cite either claim above as established. Full
write-up: docs/results-checkbox-accuracy-cost-2026-08-06.md, "Correction: why Mistral OCR 4 loses
checkboxes".

WHY THIS EXISTS — it isolates the FLATTENING step, which nothing else in this bench can.
(As believed 2026-08-06. Superseded in part by the correction above.)

The transcriber legs (`mistral_ocr_4`, `mistral_ocr_4_0`) score 85.7% / 90.7% on checkbox state
while DocStrange scores 93.8% on the same corpus. Investigating that gap 2026-08-06 ruled out the
obvious explanations:

  * NOT a glyph-alphabet difference: DocStrange emits the SAME Unicode (☐ in 78/150 sampled docs,
    ☑ in 75) that Mistral does (☐ 73, ☑ 60). Both feed the same gemini-3.6-flash extractor.
  * IS a glyph-POSITION difference. On finance_4's STATUS OF TRANSACTION block, DocStrange emits
    `☐ QUOTE ☑ ISSUE POLICY ☐ RENEW` — glyph BEFORE its label, one line per row. Mistral's OCR
    flattens the same block into one pipe-delimited table cell and drops the leading glyph:
    `| QUOTE ☑ ISSUE POLICY ☐ RENEW BOUND ... CANCEL 32 ☐ AM ☐ PM |`. Read with the ordinary
    glyph-before-label convention that says quote=true, issue_policy=false. Gold is the exact
    inverse, and the extractor duly returned the inverse. Polarity inversion is the specific
    failure this benchmark exists to catch.

So the open question is whether Mistral's OCR *engine* is weak on checkboxes, or whether its
markdown flattening destroys glyph-label association that the engine actually resolved. A
markdown-level fixup cannot answer that — normalizing ☑ to [x] preserves the inversion exactly, and
post-processing one provider's output before scoring measures the post-processor, not the provider.

Document QnA answers it directly: Mistral runs its OWN OCR internally and passes the extracted text
PLUS the page image to a vision model, which can therefore resolve glyph-label association from
layout instead of from a flattened table. Same vendor, same OCR engine, no markdown bottleneck, no
external extractor. If checkbox accuracy jumps, the flattening was the problem; if it does not, the
engine's checkbox reading is the ceiling.

MEASUREMENT SHAPE. This is `vlm-chat`, not `transcriber`: one call per cell, answering the question
directly, with no separate extractor. It is therefore NOT comparable with the transcriber rows above
— it changes both the model doing the reading and the number of hops. It is closest to the Gemini
native rows, with the crucial difference that the provider does its own rasterization
(`input_mode: pdf-direct`), so the pinned dpi/render conditions do not apply and the
embedded-text-layer caveat does.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.mistral.ai/v1"

# Same taxonomy as gemini_native/score: 408/429/5xx and connection-level failures are transient.
# Mistral's /v1/ocr returned two isolated 500s across 1,162 pages on 2026-08-06, so 5xx here is a
# real, observed, retryable condition rather than a defensive guess.
RETRYABLE_STATUSES = frozenset({408, 429})


class MistralCredentialError(RuntimeError):
    """The API rejected the credential itself — a configuration fault, never a transport blip."""


def is_retryable_status(status: int) -> bool:
    return status in RETRYABLE_STATUSES or 500 <= status < 600


def pdf_data_url(pdf_bytes: bytes) -> str:
    """Base64 data URL.

    Mistral's Document QnA doc says the document URL must be "public and accessible by our API",
    which would rule out this corpus entirely. Probed live 2026-08-06: a base64 data URL is in fact
    accepted (HTTP 200, `mistral-small-2603`, 2,778 prompt tokens on finance_4), matching the
    documented behaviour of `/v1/ocr`. Pinned by test so a docs-driven "fix" to public URLs cannot
    land silently.
    """
    return f"data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode('ascii')}"


class MistralDocQnAClient:
    """`POST /v1/chat/completions` with a `document_url` content part.

    `max_retries` is 0 by design: `direct.py` owns the retry loop for every transport, and a client
    retrying internally would nest inside it. Same discipline as `OpenAI(max_retries=0)`,
    botocore's `max_attempts: 1`, and `GeminiNativeClient`.
    """

    def __init__(self, api_key: str, *, model: str, base_url: str = DEFAULT_BASE_URL,
                 timeout: float = 300.0, client: httpx.Client | None = None) -> None:
        if not api_key:
            raise MistralCredentialError("mistral-docqna: no API key provided")
        self._key = api_key
        self._model = model
        self._base = base_url.rstrip("/")
        self._own = client is None
        self._c = client or httpx.Client(timeout=timeout)

    @property
    def resolved_provider(self) -> str:
        """Stamped on every row; `report_md`'s D4 gate hard-fails a parser key whose rows span more
        than one provider, so this describes the serving stack, not just the vendor."""
        return "mistral:api"

    def close(self) -> None:
        if self._own:
            self._c.close()

    def generate(self, *, system: str, prompt: str, pdf: bytes,
                 temperature: float | None, top_p: float | None,
                 max_tokens: int | None) -> tuple[str, dict, str]:
        """Returns the `(text, usage, provider)` triple every other transport returns.

        `pdf` is required: unlike the raster transports there is no `no_image` control here, because
        removing the document would remove the OCR step this transport exists to exercise.

        A `None` sampling parameter is OMITTED from the body rather than sent as JSON `null`, and
        the two are not interchangeable:

          * `max_tokens: null` is documented as accepted (`integer|null`) but its behaviour is not
            documented, whereas an absent key unambiguously takes the provider's own behaviour.
            Omitting is the only way to say "no cap from us" without asserting a number — and the
            number is exactly what we do not want to assert (a 12288 cap on a `reasoning: true`
            model silently truncates thinking before the answer is written).
          * Mistral does not publish a default temperature at all; the API reference says to call
            `/models` for it, which returns `default_model_temperature` (0.3 for
            `mistral-small-2603`). Omitting the key gets that value from the server, which stays
            correct if the vendor retunes it. Hardcoding 0.3 here would freeze today's reading of
            a value the vendor treats as theirs to change.
          * Mistral's guidance is to alter `temperature` OR `top_p`, never both. Omitting both
            respects that; the previous run set both (temperature 0.0 AND top_p 1.0).
        """
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "document_url", "document_url": pdf_data_url(pdf)},
                ]},
            ],
            "response_format": {"type": "json_object"},
        }
        for key, value in (("temperature", temperature), ("top_p", top_p),
                           ("max_tokens", max_tokens)):
            if value is not None:
                body[key] = value
        # Credential in a HEADER, never the URL: httpx embeds the full URL in HTTPStatusError, so a
        # query-param key prints the live secret into any non-2xx traceback.
        resp = self._c.post(f"{self._base}/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {self._key}"})
        if resp.status_code != 200:
            msg = _error_message(resp)
            if resp.status_code in (401, 403):
                raise MistralCredentialError(
                    f"mistral-docqna: API key rejected (HTTP {resp.status_code}): {msg}")
            raise httpx.HTTPStatusError(
                f"mistral-docqna {self._model}: HTTP {resp.status_code} {msg}".strip(),
                request=resp.request, response=resp)

        raw = resp.json()
        choices = raw.get("choices") or []
        first = choices[0] if choices else {}
        text = ((first.get("message") or {}).get("content") or "").strip()

        u = raw.get("usage") or {}
        # `or 0`, NOT `.get(k, 0)`: a provider that OMITS a token field yields a present-but-None
        # key and a dict default never fires for that. Gemini's omitted `candidatesTokenCount`
        # killed a full-bank run ~300 paid cells in; assume every new transport can do the same.
        usage = {
            "prompt_tokens": u.get("prompt_tokens") or 0,
            "completion_tokens": u.get("completion_tokens") or 0,
            "total_tokens": u.get("total_tokens") or 0,
            "mistral_usage": u,
            "finish_reason": first.get("finish_reason"),
            "served_model": raw.get("model"),   # server-reported id, catches an alias re-pointing
        }
        return text, usage, self.resolved_provider


def _error_message(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except (ValueError, TypeError):
        return resp.text[:300]
    err = body.get("error") or body.get("message") or body
    if isinstance(err, dict):
        return str(err.get("message") or err)[:300]
    return str(err)[:300]


def preflight_mistral_docqna(entry, api_key: str, pdf_path: Path,
                             max_tokens: int | None = None) -> str:
    """One tiny real call — being able to LIST a model is not being able to invoke it (learned on
    `gemini-3.1-pro-preview`, which listed fine and returned empty content at a starved budget).

    Sampling defaults to fully unset (provider defaults), matching what the run itself sends. A
    preflight that passed temperature 0.0 while the run sends nothing would be exercising a
    different condition than the one it certifies — and the failure it exists to catch (empty
    content from a starved reasoning budget) is precisely a sampling-dependent one.
    """
    client = MistralDocQnAClient(api_key, model=entry.model)
    try:
        text, usage, provider = client.generate(
            system="Reply with a JSON object.",
            prompt='Return exactly {"ok": true}',
            pdf=pdf_path.read_bytes(), temperature=None, top_p=None, max_tokens=max_tokens)
    finally:
        client.close()
    if not text:
        raise RuntimeError(
            f"empty response from {entry.model} at max_tokens={max_tokens} "
            f"(finish_reason={usage.get('finish_reason')})")
    return f"{entry.model} (served {usage.get('served_model')}, provider {provider})"
