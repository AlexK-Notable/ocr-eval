"""Minimal OpenAI-compatible /chat/completions stub for tests. Configurable reply text, plus a
scripted-response mode (status codes + headers) for retry/robustness tests."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockOpenAI:
    def __init__(self, reply_text='{"a": true}', responses=None):
        """
        responses: optional list of per-request response specs, consumed in order across
        successive POSTs — the LAST entry repeats once the list is exhausted, so a test only
        needs to script the interesting prefix (e.g. `[{"status": 429, ...}]` then every
        subsequent call gets a 200). Each spec is a dict:
          - "status": int, default 200.
          - "headers": dict[str, str] of extra response headers (e.g. Retry-After), default {}.
          - "body": full response JSON body, if given, used verbatim (lets a test control
            `usage.cost`, `provider`, etc. exactly).
          - "content": for status == 200 with no explicit "body" — the assistant message content
            for just this one scripted response (overrides `reply_text`).
          - "message": for status != 200 with no explicit "body" — the mock error message text.
        `responses=None` (the default) preserves the original single-reply behavior: every
        request gets a 200 with `reply_text` as the assistant content.
        """
        self.reply_text = reply_text
        self.responses = responses
        self.requests: list[dict] = []
        handler_self = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                handler_self.requests.append(body)
                spec = handler_self._spec_for(len(handler_self.requests) - 1)
                status = spec.get("status", 200)
                headers = spec.get("headers") or {}
                if "body" in spec:
                    resp_body = spec["body"]
                elif status == 200:
                    resp_body = {
                        "id": "cmpl-1", "model": body.get("model", "m"),
                        "provider": "MockProvider",
                        "choices": [{"message": {"role": "assistant",
                                                 "content": spec.get("content",
                                                                     handler_self.reply_text)}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                    }
                else:
                    resp_body = {"error": {"message": spec.get("message", "mock error"),
                                           "type": "mock_error"}}
                data = json.dumps(resp_body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):  # silence
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def _spec_for(self, index: int) -> dict:
        if not self.responses:
            return {"status": 200}
        i = min(index, len(self.responses) - 1)
        return self.responses[i]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
