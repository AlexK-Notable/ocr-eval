"""Minimal OpenAI-compatible /chat/completions stub for tests. Configurable reply text."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockOpenAI:
    def __init__(self, reply_text='{"a": true}'):
        self.reply_text = reply_text
        self.requests: list[dict] = []
        handler_self = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                handler_self.requests.append(body)
                resp = {
                    "id": "cmpl-1", "model": body.get("model", "m"),
                    "provider": "MockProvider",
                    "choices": [{"message": {"role": "assistant",
                                             "content": handler_self.reply_text}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                }
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):  # silence
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
