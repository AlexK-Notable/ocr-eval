"""Mistral Document QnA transport (`transport: mistral-docqna`).

WHY THIS EXISTS — it isolates the FLATTENING step, which nothing else in this bench can.

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
                 temperature: float, top_p: float, max_tokens: int) -> tuple[str, dict, str]:
        """Returns the `(text, usage, provider)` triple every other transport returns.

        `pdf` is required: unlike the raster transports there is no `no_image` control here, because
        removing the document would remove the OCR step this transport exists to exercise.
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
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
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


def preflight_mistral_docqna(entry, api_key: str, pdf_path: Path, max_tokens: int = 1024) -> str:
    """One tiny real call — being able to LIST a model is not being able to invoke it (learned on
    `gemini-3.1-pro-preview`, which listed fine and returned empty content at a starved budget).
    """
    client = MistralDocQnAClient(api_key, model=entry.model)
    try:
        text, usage, provider = client.generate(
            system="Reply with a JSON object.",
            prompt='Return exactly {"ok": true}',
            pdf=pdf_path.read_bytes(), temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    finally:
        client.close()
    if not text:
        raise RuntimeError(
            f"empty response from {entry.model} at max_tokens={max_tokens} "
            f"(finish_reason={usage.get('finish_reason')})")
    return f"{entry.model} (served {usage.get('served_model')}, provider {provider})"
