"""AWS Bedrock transport (`transport: bedrock-converse`).

Why this exists as its own transport rather than another `base_url` on the openai-compat path:
**Bedrock exposes no OpenAI-compatible `/chat/completions` endpoint reachable from this account.**
Probed 2026-08-04 — `https://bedrock-runtime.<region>.amazonaws.com/openai/v1/models` returns
`404 UnknownOperationException`, `bedrock-runtime` advertises no chat-completions operation
(`Converse`, `ConverseStream`, `InvokeModel`, ... only), `bedrock-runtime get-api-key` is not
permitted for this role, and an unauthenticated POST is `403` — every call must be SigV4-signed.
So the `openai` client cannot reach Bedrock at all, and `run_direct`'s openai-compat precondition
is genuinely inapplicable rather than merely inconvenient.

What this module provides is deliberately narrow: a `_one`-compatible call surface returning the
same `(text, usage, provider)` triple the openai-compat path yields, so **every downstream
contract is unchanged** — cache-row shape, `condition_hash`, `parser_key`, `score_typed`,
`track()`'s spend arithmetic, and `report_md.py`'s gates all see exactly what they saw before.

Two Bedrock-specific facts that drive the code below:

1. **Converse returns token counts but no cost field.** `usage` is mapped onto the
   `prompt_tokens`/`completion_tokens` names `track()` already reads, so the registry-`pricing`
   fallback works unmodified. There is no `usage.cost` to opportunistically read (unlike
   OpenRouter), so an entry with no `pricing` and an active `--max-spend` hits `run_direct`'s
   existing fail-closed rule and raises — correct behaviour, documented in `docs/api.md`.
2. **Retry classification is exception-type-based, not status-code-based.** botocore raises
   `ClientError` with a string error code rather than an `APIStatusError` carrying `.status_code`,
   so `direct.py`'s `_is_retryable` cannot classify these. `is_retryable_bedrock` below is the
   Bedrock-side equivalent and is consulted by `_is_retryable` for botocore exceptions.
"""
from __future__ import annotations

# Transient Bedrock failures — worth the retry budget. `ThrottlingException` is the one that
# actually matters at full-bank scale (1,356 cells x N models against on-demand quotas).
RETRYABLE_ERROR_CODES = frozenset({
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelTimeoutException",
    "ModelNotReadyException",
})

# Permanent: retrying cannot help, and burning 4 attempts per cell on 1,356 cells turns a
# misconfiguration into a very slow one. `AccessDeniedException` is the expected symptom of a
# model that exists in `list-foundation-models` but is not enabled for this role — the exact
# failure mode probed on this account for the Anthropic and cross-region-profile model ids.
PERMANENT_ERROR_CODES = frozenset({
    "AccessDeniedException",
    "ValidationException",
    "ResourceNotFoundException",
    "ModelErrorException",
    "SerializationException",
    "ExpiredTokenException",
    "UnrecognizedClientException",
})

# Anthropic models served through Bedrock reject `temperature` and `top_p` in the SAME request:
#   ValidationException: `temperature` and `top_p` cannot both be specified for this model.
# Verified live 2026-08-04 across all six models invokable on the probed account: ONLY the
# Anthropic one rejects the pair — Nova Lite/Pro, Gemma 3 27B, Mistral Large 3 and Kimi K2.5 all
# accept both. So this is a per-model-family wire constraint, not a Bedrock-wide one, and is
# handled by narrowing the request for exactly those models rather than by weakening
# STAGE1_CONDITION for everyone.
#
# STAGE1_CONDITION pins `top_p: 1.0`, which is the IDENTITY value (consider the full distribution)
# — so for that specific value, omitting it is provably a no-op and the measurement is unchanged.
# `narrow_sampling_for_model` therefore drops top_p ONLY when it is exactly 1.0, and RAISES for any
# other value rather than silently altering what a condition claims to pin.
TEMPERATURE_TOPP_EXCLUSIVE_PREFIXES = ("anthropic.", "us.anthropic.", "eu.anthropic.",
                                        "apac.anthropic.")


def excludes_temperature_with_top_p(model_id: str) -> bool:
    return model_id.startswith(TEMPERATURE_TOPP_EXCLUSIVE_PREFIXES)


def narrow_sampling_for_model(model_id: str, *, temperature: float,
                              top_p: float) -> tuple[dict, str]:
    """Return `(inferenceConfig fragment, note)` for this model's sampling constraints.

    `note` is "" when the full condition went on the wire verbatim, or a short description of the
    narrowing when it did not — recorded per row as `sampling_note` so a Bedrock row is
    self-describing about what was actually sent, rather than silently implying the condition dict
    was honored in full.

    Raises `ValueError` when a genuine (non-identity) top_p cannot be sent, instead of dropping a
    parameter the caller explicitly pinned."""
    if not excludes_temperature_with_top_p(model_id):
        return {"temperature": temperature, "topP": top_p}, ""
    if top_p != 1.0:
        raise ValueError(
            f"{model_id}: Anthropic-on-Bedrock rejects temperature and top_p in the same request, "
            f"and top_p={top_p} is not the identity value (1.0), so it cannot be dropped without "
            f"changing the measurement. Use a condition with top_p=1.0 for this model, or serve it "
            f"through a transport that accepts both.")
    return ({"temperature": temperature},
            "top_p=1.0 omitted (identity value; Anthropic-on-Bedrock forbids sending it "
            "alongside temperature)")


def is_botocore_error(exc: BaseException) -> bool:
    """True for botocore's `ClientError`/`BotoCoreError` without importing botocore at module
    import time (this module is imported by `direct.py`, which must stay usable in environments
    that never touch Bedrock — boto3 is a declared dependency, but a broken/partial install must
    not take the openai-compat path down with it)."""
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        return False
    return isinstance(exc, ClientError | BotoCoreError)


def is_retryable_bedrock(exc: BaseException) -> bool:
    """Bedrock-side counterpart to `direct.py`'s `_is_retryable`.

    `BotoCoreError` (connection-level: DNS, refused, read timeout, endpoint resolution) is
    retryable for the same reason `APIConnectionError` is on the openai path — no HTTP response
    was obtained, so nothing has been established about permanence.

    An unrecognised error code is treated as **permanent**. That is the deliberate direction: a
    genuinely transient code we failed to enumerate costs one lost cell (recorded honestly as an
    `api_error` row with the code in it, so it is visible and fixable), whereas treating an
    unknown permanent code as retryable silently quadruples the wall-clock and spend of a
    misconfigured full-bank run."""
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        return False
    if isinstance(exc, BotoCoreError):
        return True
    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code", "") in RETRYABLE_ERROR_CODES
    return False


def retry_after_bedrock(exc: BaseException) -> float | None:
    """Server-suggested wait from a throttling response's HTTP headers, or None to let
    `direct.py`'s `_retry_wait` fall back to exponential backoff. botocore surfaces headers under
    `response["ResponseMetadata"]["HTTPHeaders"]` — a dict, not the `.headers` attribute
    `_retry_wait` probes on openai exceptions, which is why this is extracted here instead."""
    try:
        from botocore.exceptions import ClientError
    except ImportError:
        return None
    if not isinstance(exc, ClientError):
        return None
    headers = (exc.response.get("ResponseMetadata") or {}).get("HTTPHeaders") or {}
    raw = headers.get("retry-after") or headers.get("x-amzn-retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


class BedrockConverseClient:
    """Thin `bedrock-runtime` wrapper presenting the one call `_one` needs.

    Deliberately NOT an `openai.OpenAI` look-alike: faking `chat.completions.create` and a
    `model_dump()` shape would make two genuinely different wire protocols look interchangeable
    at the call site, which is exactly the kind of silent-divergence this codebase's registry
    design exists to prevent. `_one` branches on `entry.transport` explicitly instead.

    Retries are owned by `direct.py` (`MAX_RETRIES` / `_is_retryable` / `_retry_wait`), so the
    botocore client is configured with `retries={"max_attempts": 1, "mode": "standard"}` — the
    same no-nested-retries discipline `OpenAI(..., max_retries=0)` establishes on the other path.
    botocore's default is 3 adaptive attempts, which would nest under ours for up to 12 real API
    calls per cell.
    """

    def __init__(self, *, region: str, model_id: str):
        import boto3
        from botocore.config import Config

        self._model_id = model_id
        self._region = region
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"max_attempts": 1, "mode": "standard"},
                          read_timeout=300, connect_timeout=15),
        )

    @property
    def resolved_provider(self) -> str:
        """Serving identity recorded on every row. Region is part of it deliberately: the same
        `modelId` served from two regions is two serving stacks, and `report_md.py`'s D4 gate
        hard-fails a parser key whose rows span more than one distinct provider — so a run
        accidentally split across regions is caught rather than silently averaged."""
        return f"bedrock:{self._region}"

    def converse(self, *, system: str, prompt: str, png: bytes | None,
                 temperature: float, top_p: float, max_tokens: int) -> tuple[str, dict, str]:
        """Returns `(text, usage, provider)` — the same triple the openai-compat path produces.

        `usage` is remapped onto `prompt_tokens`/`completion_tokens` so `track()`'s registry-
        pricing arithmetic needs no Bedrock special-casing; the native Converse field names are
        preserved alongside them so a cache row stays self-describing about which wire protocol
        produced it. When a model's constraints forced the request to be narrowed, `usage` also
        carries a `sampling_note` saying so (see `narrow_sampling_for_model`).
        """
        content: list[dict] = []
        if png is not None:
            # Converse takes raw bytes, NOT base64 — boto3 signs and encodes the blob itself.
            # Passing a base64 string here is accepted as bytes and silently degrades the image.
            content.append({"image": {"format": "png", "source": {"bytes": png}}})
        content.append({"text": prompt})

        sampling, note = narrow_sampling_for_model(
            self._model_id, temperature=temperature, top_p=top_p)
        resp = self._client.converse(
            modelId=self._model_id,
            messages=[{"role": "user", "content": content}],
            system=[{"text": system}],
            inferenceConfig={**sampling, "maxTokens": max_tokens},
        )
        text = "".join(
            block.get("text", "")
            for block in (resp.get("output", {}).get("message", {}).get("content") or [])
        )
        u = resp.get("usage") or {}
        usage = {
            "prompt_tokens": u.get("inputTokens", 0),
            "completion_tokens": u.get("outputTokens", 0),
            "total_tokens": u.get("totalTokens", 0),
            "bedrock_usage": dict(u),          # native field names, verbatim
            "stop_reason": resp.get("stopReason", ""),
        }
        if note:
            usage["sampling_note"] = note      # row stays honest about what actually went on the wire
        return text, usage, self.resolved_provider


def preflight_bedrock(entry) -> str:
    """Confirm the entry's `modelId` is actually invokable by THIS role before spending a full
    bank against it.

    A control-plane `ListFoundationModels` check would be misleading here and is deliberately not
    used: on the probed account 119 models are listed and only 7 are invokable — `AccessDeniedException`
    on invoke is the normal state for a listed-but-not-enabled model. So this sends one real
    minimal Converse call (a few tokens, effectively free) and reports the concrete error code on
    failure, mirroring what `preflight`'s `GET /models` does for a local vLLM server: catch a
    serving-identity problem before a single page is paid for.
    """
    from botocore.exceptions import ClientError

    if not (entry.region and entry.model):      # guaranteed by RegistryEntry's validator
        raise RuntimeError(f"{entry.id}: bedrock-converse entry missing region/model")
    client = BedrockConverseClient(region=entry.region, model_id=entry.model)
    try:
        client.converse(system="Reply with one word.", prompt="Reply with exactly: OK",
                        png=None, temperature=0.0, top_p=1.0, max_tokens=8)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "?")
        msg = e.response.get("Error", {}).get("Message", "")[:200]
        raise RuntimeError(
            f"{entry.id}: Bedrock preflight FAILED for modelId {entry.model!r} in "
            f"{entry.region} — {code}: {msg}"
        ) from e
    return f"{entry.model} @ bedrock:{entry.region}"


def probe_invokable(region: str, model_ids: list[str]) -> dict[str, str]:
    """Diagnostic helper (not used by the run path): map model ids to "ok" or their error code.

    Exists because "is this model available?" has no single answer on Bedrock — a model can be
    ACTIVE in the control plane, carry an `inferenceTypesSupported` of `INFERENCE_PROFILE`
    (meaning the bare id is not invokable; the `us.`-prefixed profile id is), and still be
    IAM-denied for the calling role. Cheapest reliable answer is one tiny call each.
    """
    from botocore.exceptions import ClientError

    out: dict[str, str] = {}
    for mid in model_ids:
        try:
            BedrockConverseClient(region=region, model_id=mid).converse(
                system="Reply with one word.", prompt="Reply with exactly: OK",
                png=None, temperature=0.0, top_p=1.0, max_tokens=8)
            out[mid] = "ok"
        except ClientError as e:
            out[mid] = e.response.get("Error", {}).get("Code", "?")
        except Exception as e:      # diagnostic breadth is the point — report, never propagate
            out[mid] = type(e).__name__
    return out
