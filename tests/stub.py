"""A tiny stand-in for the DeepSeek API used by the test suite."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

VALID_KEY = "test-key"

CONTENT_CHUNKS = ["Hello", " from", " the", " stub"]
REASONING_CHUNKS = ["weighing ", "options"]


class StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    counters: Dict[str, int] = {}
    requests: List[Dict[str, Any]] = []

    def log_message(self, *args: Any) -> None:  # keep test output clean
        pass

    # ------------------------------------------------------------------ helpers
    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, events: List[str]) -> None:
        body = "".join(f"data: {event}\n\n" for event in events).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path in ("/models", "/user/balance"):
            if self.headers.get("Authorization") != f"Bearer {VALID_KEY}":
                self._json(401, {"error": {"message": "invalid api key"}})
                return
        if self.path == "/models":
            self._json(200, {"data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]})
        elif self.path == "/user/balance":
            self._json(
                200,
                {
                    "is_available": True,
                    "balance_infos": [
                        {
                            "currency": "USD",
                            "total_balance": "12.34",
                            "granted_balance": "0.00",
                            "topped_up_balance": "12.34",
                        }
                    ],
                },
            )
        else:
            self._json(404, {"error": {"message": "unknown path"}})

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw or b"{}")
        StubHandler.requests.append(
            {"path": self.path, "payload": payload, "auth": self.headers.get("Authorization")}
        )

        if self.headers.get("Authorization") != f"Bearer {VALID_KEY}":
            self._json(401, {"error": {"message": "invalid api key"}})
            return

        text = json.dumps(payload)
        if "TRIGGER_429" in text:
            StubHandler.counters["429"] = StubHandler.counters.get("429", 0) + 1
            if StubHandler.counters["429"] <= 2:
                self.send_response(429)
                body = json.dumps({"error": {"message": "slow down"}}).encode()
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(body)
                return
        if "TRIGGER_400" in text:
            self._json(400, {"error": {"message": "bad request from test"}})
            return

        wants_reasoning = "WITH_REASONING" in text
        usage = {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "prompt_cache_hit_tokens": 3,
            "prompt_cache_miss_tokens": 8,
        }
        model = payload.get("model", "deepseek-chat")

        if not payload.get("stream"):
            message: Dict[str, Any] = {"role": "assistant", "content": "".join(CONTENT_CHUNKS)}
            if wants_reasoning:
                message["reasoning_content"] = "".join(REASONING_CHUNKS)
            self._json(
                200,
                {
                    "id": "stub",
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": usage,
                },
            )
            return

        events: List[str] = []
        for chunk in REASONING_CHUNKS if wants_reasoning else []:
            events.append(json.dumps({"choices": [{"delta": {"reasoning_content": chunk}}]}))
        for chunk in CONTENT_CHUNKS:
            events.append(json.dumps({"choices": [{"delta": {"content": chunk}}]}))
        events.append(json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
        events.append(json.dumps({"choices": [], "usage": usage}))
        events.append("[DONE]")
        self._sse(events)


class StubServer:
    """Context manager that runs :class:`StubHandler` on a background thread."""

    def __init__(self) -> None:
        StubHandler.counters = {}
        StubHandler.requests = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "StubServer":
        self.thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
