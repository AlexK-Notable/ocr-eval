"""Regression tests for the two `gemini_extract` fork changes (2026-08-04).

Both exist because of things that actually happened, not hypotheticals:

  * A transient 503 during an extractor-gate run printed a LIVE Google API key into a session
    transcript, because upstream sent the key as a URL query parameter and httpx embeds the full
    URL in `HTTPStatusError`. The canary test below fails if the key ever returns to the URL.
  * That same 503 aborted the whole gate, because upstream called `raise_for_status()` once with no
    retry — so a blip in a free Google endpoint failed a run mid-flight.

No network: `httpx.MockTransport` serves every response.
"""
from __future__ import annotations

import httpx
import pytest

from realdoc_bench.evaluate import score

CANARY = "SECRET-CANARY-do-not-leak"


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", CANARY)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Retry waits are real `time.sleep` calls — 2s + 4s + 8s would make this file slow."""
    monkeypatch.setattr(score.time, "sleep", lambda _s: None)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def ok_body(payload: str = '{"a": true}') -> dict:
    return {"candidates": [{"content": {"parts": [{"text": payload}]}}]}


# ── credential handling ──────────────────────────────────────────────────────────────────────

def test_key_is_sent_as_header_not_url_param():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        seen["header"] = request.headers.get("x-goog-api-key")
        seen["url"] = str(request.url)
        return httpx.Response(200, json=ok_body())

    assert score.gemini_extract("q?", '{"a": <boolean>}', "md", client=_client(handler)) == {"a": True}
    assert seen["header"] == CANARY, "key must travel in the x-goog-api-key header"
    assert "key" not in seen["params"], "key must NOT be a URL query parameter"
    assert CANARY not in seen["url"], "key must not appear anywhere in the URL"


def test_key_absent_from_http_error_message():
    """The exact leak that occurred: httpx puts the request URL in HTTPStatusError, so a key in the
    URL ends up in tracebacks, logs, and session transcripts."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    with pytest.raises(httpx.HTTPStatusError) as ei:
        score.gemini_extract("q?", '{"a": <boolean>}', "md", client=_client(handler))
    assert CANARY not in str(ei.value)
    assert CANARY not in str(ei.value.request.url)


def test_missing_key_raises_before_any_request(monkeypatch):
    """A configuration problem must surface as RuntimeError (what callers expect), and must not
    reach the wire at all.

    Uses monkeypatch.delenv rather than os.environ.pop so the deletion is reverted at teardown —
    a raw pop would leak into every later test in the session (the autouse `_key` fixture sets the
    var, but an unmanaged pop still fights it across ordering changes)."""
    called = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        called["n"] += 1
        return httpx.Response(200, json=ok_body())

    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY / GOOGLE_API_KEY not set"):
        score.gemini_extract("q?", '{"a": <boolean>}', "md", client=_client(handler))
    assert called["n"] == 0


# ── retry behaviour ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_transient_statuses_are_retried_then_succeed(status):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(status, json={"error": {"message": "transient"}})
        return httpx.Response(200, json=ok_body())

    assert score.gemini_extract("q?", '{"a": <boolean>}', "md", client=_client(handler)) == {"a": True}
    assert calls["n"] == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_statuses_fail_on_first_attempt(status):
    """Retrying a bad key or malformed request is pure waste — and at 1,356 questions per
    transcriber it is very slow waste."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(status, json={"error": {"message": "permanent"}})

    with pytest.raises(httpx.HTTPStatusError):
        score.gemini_extract("q?", '{"a": <boolean>}', "md", client=_client(handler))
    assert calls["n"] == 1


def test_retry_budget_is_bounded_and_then_raises():
    """A genuinely unavailable extractor must still fail the run — the gate is fail-closed by
    design; retries only absorb blips."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": {"message": "down"}})

    with pytest.raises(httpx.HTTPStatusError):
        score.gemini_extract("q?", '{"a": <boolean>}', "md", client=_client(handler))
    assert calls["n"] == score._EXTRACT_MAX_ATTEMPTS == 4


def test_connection_level_failure_is_retried():
    """No HTTP response obtained — nothing established about permanence."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=ok_body())

    assert score.gemini_extract("q?", '{"a": <boolean>}', "md", client=_client(handler)) == {"a": True}
    assert calls["n"] == 2


def test_retry_after_header_is_honored_and_clamped(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(score.time, "sleep", lambda s: waits.append(s))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "7"}, json={"error": {}})
        if calls["n"] == 2:
            # hostile value — must be clamped, never honored literally
            return httpx.Response(429, headers={"retry-after": "9000"}, json={"error": {}})
        return httpx.Response(200, json=ok_body())

    score.gemini_extract("q?", '{"a": <boolean>}', "md", client=_client(handler))
    assert waits[0] == pytest.approx(7.0)
    assert waits[1] <= score._EXTRACT_MAX_WAIT_SEC == 60.0


def test_unparseable_retry_after_falls_back_to_backoff(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(score.time, "sleep", lambda s: waits.append(s))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, headers={"retry-after": "not-a-number"}, json={"error": {}})
        return httpx.Response(200, json=ok_body())

    score.gemini_extract("q?", '{"a": <boolean>}', "md", client=_client(handler))
    assert waits == [pytest.approx(score._EXTRACT_BACKOFF_BASE_SEC)]


def test_client_lifecycle_unchanged_on_caller_supplied_client():
    """A caller-supplied client must NOT be closed by gemini_extract (upstream `run_score` shares
    one across every question) — the retry rewrite must not have changed that."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body())

    c = _client(handler)
    score.gemini_extract("q?", '{"a": <boolean>}', "md", client=c)
    assert not c.is_closed
    score.gemini_extract("q?", '{"a": <boolean>}', "md", client=c)   # still usable
