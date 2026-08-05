"""Gemini native `generateContent` transport (`transport: gemini-native`).

WHY THIS EXISTS — `media_resolution`, and only that.

The vlm-chat leg already reaches Gemini perfectly well through Google's OpenAI-compat shim
(`gemini-3.5-flash@google-vlmchat` in `configs/registry.yaml` is exactly that). This module is not a
second way to do the same thing; it exists because ONE parameter that materially changes what the
model sees is unreachable through that shim, and leaving it unset silently confounds any comparison
that spans Gemini generations.

Measured live 2026-08-05, same 1275x1650 render, image tokens only (total minus a 92-token text
baseline):

    generation   default            at MEDIA_RESOLUTION_HIGH
    3.x          1161  (= HIGH)     1161
    2.5           317  (= MEDIUM)   1865

So on defaults a 2.5 model is handed roughly a QUARTER of the image budget a 3.x model gets. For a
checkbox benchmark — where the entire signal can be a few dark pixels inside a small box — that is
not a footnote: a 2.5-vs-3.x gap measured on defaults conflates model capability with how much image
detail each generation was given. Setting `mediaResolution` explicitly on every row makes image
detail a PINNED CONDITION rather than a per-generation accident.

The budget is fixed, not resolution-derived: token count was identical across a 4x downscale of the
same page (1275x1650 -> 318x412 all reported the same count), so this cannot be equalized by
re-rendering at a different dpi. It has to be sent on the wire.

It is NOT reachable via the compat shim. Probed five spellings/nestings; every one returned
`400 Unknown name "media_resolution" at 'extra_body.google': Cannot find field`. The probe method is
sound rather than merely unlucky: `thinking_config` at that same path IS accepted (it fails on the
VALUE, not the field name), which proves `extra_body.google` resolves and `media_resolution` simply
is not a member of it.

Deliberately a separate transport rather than a flag on the openai-compat path, for the same reason
`bedrock.py` is separate: this speaks a different wire protocol (`contents`/`parts`/`inlineData`,
`generationConfig`, `usageMetadata`), and making two protocols share one code path would let a
request-shape bug in one silently produce plausible-looking rows for the other. Everything AFTER the
call is identical, so rows are byte-compatible with both other transports.
"""
from __future__ import annotations

import base64
from typing import Any

import httpx

# The four documented enum values. `UNSPECIFIED` is accepted by the API but means "use the model
# default", which is precisely the per-generation accident this module exists to remove — it is
# listed for completeness and rejected by `validate_media_resolution`.
MEDIA_RESOLUTIONS = (
    "MEDIA_RESOLUTION_UNSPECIFIED",
    "MEDIA_RESOLUTION_LOW",
    "MEDIA_RESOLUTION_MEDIUM",
    "MEDIA_RESOLUTION_HIGH",
)
EXPLICIT_MEDIA_RESOLUTIONS = tuple(r for r in MEDIA_RESOLUTIONS if not r.endswith("UNSPECIFIED"))

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Retryable HTTP statuses, matching `direct.py`'s `_is_retryable` and `score.py`'s
# `_post_with_retry`: 408/429/5xx plus connection-level failures. Everything else is permanent.
RETRYABLE_STATUSES = frozenset({408, 429})

# Google returns an INVALID key as 400 INVALID_ARGUMENT (not 401/403 — a MISSING key gives 403),
# so a bare status check cannot tell a bad credential from a malformed request. Same taxonomy as
# `score.py`'s `_CREDENTIAL_REASONS`; see that module for the live probe that established it.
CREDENTIAL_REASONS = frozenset({
    "API_KEY_INVALID", "API_KEY_SERVICE_BLOCKED", "API_KEY_HTTP_REFERRER_BLOCKED",
    "API_KEY_IP_ADDRESS_BLOCKED", "API_KEY_ANDROID_APP_BLOCKED", "API_KEY_IOS_APP_BLOCKED",
    "ACCESS_TOKEN_EXPIRED", "CREDENTIALS_MISSING",
})


class GeminiCredentialError(RuntimeError):
    """The API rejected the credential itself. `RuntimeError` so it surfaces as a configuration
    fault rather than a retryable transport blip."""


def validate_media_resolution(value: Any) -> str:
    """Fail closed on anything that is not an explicit resolution.

    Rejects `None` and `MEDIA_RESOLUTION_UNSPECIFIED` as hard as it rejects a typo: both mean "let
    the model decide", which reintroduces the 4x cross-generation gap this transport exists to
    remove — and would do so INVISIBLY, since the condition hash would still claim a pinned
    condition. An unset value must never be the quiet default here."""
    if value not in EXPLICIT_MEDIA_RESOLUTIONS:
        raise ValueError(
            f"media_resolution must be one of {list(EXPLICIT_MEDIA_RESOLUTIONS)} (got {value!r}). "
            f"'MEDIA_RESOLUTION_UNSPECIFIED'/None are refused on purpose: they defer to a "
            f"per-generation default (3.x=HIGH ~1161 image tokens, 2.5=MEDIUM ~317), which makes a "
            f"cross-generation comparison measure image budget instead of model capability.")
    return str(value)


def error_reason(resp: httpx.Response) -> str | None:
    """Google's machine-readable `error.details[].reason`, or None."""
    try:
        err = resp.json().get("error") or {}
    except (ValueError, TypeError):
        return None
    for d in err.get("details") or []:
        if isinstance(d, dict) and d.get("reason"):
            return str(d["reason"])
    return None


def error_message(resp: httpx.Response) -> str:
    try:
        return str((resp.json().get("error") or {}).get("message") or "")[:300]
    except (ValueError, TypeError):
        return resp.text[:300]


def is_retryable_status(status: int) -> bool:
    return status in RETRYABLE_STATUSES or 500 <= status < 600


class GeminiNativeClient:
    """Minimal `generateContent` client. Deliberately not an `openai.OpenAI` look-alike — see the
    module docstring on why the two protocols stay visibly distinct.

    `max_retries` is 0 by design: `direct.py` owns the retry loop for every transport, and a client
    that retried internally would nest under it (up to MAX_RETRIES x this client's own attempts for
    what should be at most MAX_RETRIES real requests). Same discipline as `OpenAI(max_retries=0)`
    and botocore's `retries={"max_attempts": 1}`.
    """

    def __init__(self, api_key: str, *, model: str, base_url: str = DEFAULT_BASE_URL,
                 timeout: float = 300.0, client: httpx.Client | None = None) -> None:
        if not api_key:
            raise GeminiCredentialError("gemini-native: no API key provided")
        self._key = api_key
        self._model = model
        self._base = base_url.rstrip("/")
        self._own = client is None
        self._c = client or httpx.Client(timeout=timeout)

    @property
    def resolved_provider(self) -> str:
        """Serving identity stamped on every row. `report_md.py`'s D4 gate hard-fails a parser key
        whose rows span more than one provider, so this must describe the serving stack, not just
        the vendor."""
        return "google:generativelanguage"

    def close(self) -> None:
        if self._own:
            self._c.close()

    def generate(self, *, system: str, prompt: str, png: bytes | None,
                 temperature: float, top_p: float, max_tokens: int,
                 media_resolution: str) -> tuple[str, dict, str]:
        """Returns the same `(text, usage, provider)` triple the other transports do.

        `png=None` sends no image part at all (the `no_image` language-prior control). NB the native
        API takes BASE64 in `inlineData.data` — unlike Bedrock Converse, which takes raw bytes and
        silently degrades a base64 string that is handed to it as if it were bytes.
        """
        media_resolution = validate_media_resolution(media_resolution)
        parts: list[dict] = [{"text": prompt}]
        if png is not None:
            parts.append({"inlineData": {"mimeType": "image/png",
                                         "data": base64.b64encode(png).decode()}})
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
                "mediaResolution": media_resolution,
            },
        }
        url = f"{self._base}/models/{self._model}:generateContent"
        resp = self._c.post(url, json=body, headers={"x-goog-api-key": self._key})
        if resp.status_code != 200:
            reason = error_reason(resp)
            msg = error_message(resp)
            if reason in CREDENTIAL_REASONS:
                raise GeminiCredentialError(
                    f"gemini-native: API key rejected ({reason}, HTTP {resp.status_code}): {msg}")
            # Status is preserved on the exception so `direct.py`'s `_is_retryable` can classify it
            # without re-parsing a string.
            raise httpx.HTTPStatusError(
                f"gemini-native {self._model}: HTTP {resp.status_code} {reason or ''} {msg}".strip(),
                request=resp.request, response=resp)

        raw = resp.json()
        cands = raw.get("candidates") or []
        first = cands[0] if cands else {}
        parts_out = (first.get("content") or {}).get("parts") or []
        text = "\n".join(p.get("text", "") for p in parts_out if p.get("text")).strip()

        um = raw.get("usageMetadata") or {}
        usage = {
            # Remapped to the openai-compat field names so metrics/report code needs no branch.
            "prompt_tokens": um.get("promptTokenCount"),
            "completion_tokens": um.get("candidatesTokenCount"),
            "total_tokens": um.get("totalTokenCount"),
            "gemini_usage": um,                      # native shape kept verbatim alongside
            "finish_reason": first.get("finishReason"),
            "media_resolution": media_resolution,    # what the row was ACTUALLY served at
        }
        return text, usage, self.resolved_provider


def preflight_gemini_native(entry, api_key: str, media_resolution: str,
                            max_tokens: int = 4096) -> str:
    """One tiny real call — proves the model is invokable by THIS key at THIS resolution before any
    bulk spend. Listing a model is not the same as being able to invoke it.

    `max_tokens` defaults to 4096 rather than something token-frugal like 64, because a REASONING
    model spends this budget on thinking before it emits any answer: `gemini-3.1-pro-preview`
    returned `finish_reason=MAX_TOKENS` with completely empty content at 64, which looks exactly
    like a broken model and is really a starved one. The real runs use STAGE1_CONDITION's 12288 for
    the same reason (see `direct.py`'s note on the 1024 -> 12288 change), so preflighting at a
    budget the run itself would never use would test the wrong thing."""
    client = GeminiNativeClient(api_key, model=entry.model)
    try:
        text, usage, provider = client.generate(
            system="Reply with a JSON object.", prompt='Return exactly {"ok": true}',
            png=None, temperature=0.0, top_p=1.0, max_tokens=max_tokens,
            media_resolution=media_resolution)
    finally:
        client.close()
    if not text:
        raise RuntimeError(
            f"empty response from {entry.model} at max_tokens={max_tokens} "
            f"(finish_reason={usage.get('finish_reason')}); MAX_TOKENS here means the model spent "
            f"the whole budget thinking — raise it rather than treating the model as broken")
    return f"{entry.model} @ {media_resolution} (provider {provider})"
