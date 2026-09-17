"""A small, dependency-free client for the DeepSeek chat completions API.

The API is OpenAI compatible, so anything that speaks
``POST /chat/completions`` with server-sent events works here. Only the
standard library is used (``urllib``, ``json``, ``time``).
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, Iterator, List, Optional

DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class DeepSeekError(Exception):
    """Base class for every error raised by this module."""


class AuthError(DeepSeekError):
    """The API rejected the credentials (HTTP 401/403)."""


class APIError(DeepSeekError):
    """The API returned an error response."""

    def __init__(self, message: str, status: Optional[int] = None, code: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.code = code


class RateLimitError(APIError):
    """HTTP 429 - too many requests."""


class NetworkError(DeepSeekError):
    """The request never produced a usable response."""


def estimate_cost(usage: Dict[str, Any], pricing: Dict[str, Any], model: str) -> Optional[float]:
    """Estimate the USD cost of a turn, or ``None`` when rates are unknown.

    ``pricing`` maps a model name to per-million-token rates, for example::

        {"deepseek-chat": {"input_cache_hit": 0.028,
                           "input_cache_miss": 0.28,
                           "output": 0.42}}

    ``"*"`` may be used as a catch-all model key. Rates are user supplied so
    that this tool never has to guess at current provider prices.
    """
    if not pricing or not usage:
        return None
    rates = pricing.get(model) or pricing.get("*")
    if not isinstance(rates, dict):
        return None

    def rate(name: str) -> Optional[float]:
        value = rates.get(name)
        return float(value) if isinstance(value, (int, float)) else None

    hit = usage.get("prompt_cache_hit_tokens", 0) or 0
    miss = usage.get("prompt_cache_miss_tokens", 0) or 0
    if not hit and not miss:
        # No cache breakdown available: treat the whole prompt as a miss.
        miss = usage.get("prompt_tokens", 0) or 0
    output = usage.get("completion_tokens", 0) or 0

    cost = 0.0
    counted = False
    for name, tokens in (("input_cache_hit", hit), ("input_cache_miss", miss), ("output", output)):
        value = rate(name)
        if value is None:
            continue
        counted = True
        cost += value * tokens / 1_000_000.0
    return cost if counted else None


class DeepSeekClient:
    """Minimal synchronous client for ``/chat/completions``."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if not api_key:
            raise AuthError("an API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.extra_headers = dict(extra_headers or {})

    # ------------------------------------------------------------------ utils
    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "deepseek-cli/0.1.0",
        }
        headers.update(self.extra_headers)
        return headers

    def _build_request(self, path: str, payload: Optional[Dict[str, Any]], method: str):
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        return urllib.request.Request(url, data=data, headers=self._headers(), method=method)

    @staticmethod
    def _retry_delay(attempt: int, response_headers: Any = None) -> float:
        if response_headers is not None:
            retry_after = response_headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.0, min(30.0, float(retry_after)))
                except (TypeError, ValueError):
                    pass
        return min(20.0, (2 ** attempt) * 0.75) + random.uniform(0, 0.4)

    @staticmethod
    def _describe_http_error(status: int, body: bytes) -> str:
        detail = ""
        try:
            parsed = json.loads(body.decode("utf-8", "replace"))
            error = parsed.get("error") if isinstance(parsed, dict) else None
            if isinstance(error, dict):
                detail = str(error.get("message") or error.get("type") or "")
            elif isinstance(error, str):
                detail = error
            elif isinstance(parsed, dict) and parsed.get("message"):
                detail = str(parsed["message"])
        except (ValueError, AttributeError):
            detail = body.decode("utf-8", "replace").strip()[:400]

        hints = {
            400: "the request was malformed",
            401: "the API key is missing or invalid",
            402: "the account has insufficient balance",
            403: "the API key is not allowed to perform this request",
            404: "the endpoint or model does not exist",
            422: "the request parameters were rejected",
            429: "rate limit exceeded",
            500: "the server hit an internal error",
            503: "the service is temporarily overloaded",
        }
        hint = hints.get(status, "unexpected response")
        message = f"HTTP {status}: {hint}"
        if detail:
            message += f" - {detail}"
        return message

    # ------------------------------------------------------------- transport
    def _open(self, request, *, timeout: Optional[float] = None):
        return urllib.request.urlopen(request, timeout=timeout or self.timeout)

    def _request_with_retries(self, request) -> Any:
        """Open ``request``, retrying transient failures.

        Only used for non-streaming calls and the very first streaming call:
        once a stream has started emitting bytes we can no longer retry safely.
        """
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._open(request)
            except urllib.error.HTTPError as exc:
                body = b""
                try:
                    body = exc.read()
                except Exception:  # pragma: no cover - body already consumed
                    pass
                message = self._describe_http_error(exc.code, body)
                if exc.code in (401, 403):
                    raise AuthError(message) from exc
                if exc.code == 429:
                    last_error = RateLimitError(message, status=429)
                elif exc.code in RETRY_STATUSES:
                    last_error = APIError(message, status=exc.code)
                else:
                    raise APIError(message, status=exc.code)
                if attempt >= self.max_retries:
                    raise last_error
                time.sleep(self._retry_delay(attempt, exc.headers))
            except urllib.error.URLError as exc:
                last_error = NetworkError(f"could not reach {request.full_url}: {exc.reason}")
                if attempt >= self.max_retries:
                    raise last_error
                time.sleep(self._retry_delay(attempt))
            except TimeoutError as exc:  # pragma: no cover - socket timeouts
                last_error = NetworkError(f"request timed out after {self.timeout}s: {exc}")
                if attempt >= self.max_retries:
                    raise last_error
                time.sleep(self._retry_delay(attempt))
        raise last_error if last_error else NetworkError("request failed")

    @staticmethod
    def _iter_sse(response) -> Iterator[str]:
        """Yield ``data:`` payload strings from a server-sent event stream."""
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                yield line[5:].strip()

    # ------------------------------------------------------------------ calls
    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: str = "deepseek-chat",
        stream: bool = True,
        **options: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Run a chat completion.

        Yields small event dicts, which keeps streaming and non-streaming
        callers on the same code path:

        * ``{"type": "reasoning", "text": "..."}``
        * ``{"type": "content", "text": "..."}``
        * ``{"type": "usage", "usage": {...}}``
        * ``{"type": "done", "finish_reason": "stop"}``
        """
        payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": bool(stream)}
        payload.update({k: v for k, v in options.items() if v is not None})

        request = self._build_request("/chat/completions", payload, "POST")
        response = self._request_with_retries(request)
        try:
            if not stream:
                yield from self._parse_full_response(response)
            else:
                yield from self._parse_stream(response)
        finally:
            response.close()

    def _parse_full_response(self, response) -> Iterator[Dict[str, Any]]:
        raw = response.read()
        try:
            data = json.loads(raw.decode("utf-8", "replace"))
        except ValueError as exc:
            raise APIError(f"the API returned invalid JSON: {exc}") from exc

        if isinstance(data, dict) and data.get("error"):
            error = data["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise APIError(str(message))

        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            reasoning = message.get("reasoning_content")
            if reasoning:
                yield {"type": "reasoning", "text": reasoning}
            content = message.get("content")
            if content:
                yield {"type": "content", "text": content}
            yield {"type": "done", "finish_reason": choices[0].get("finish_reason")}
        if data.get("usage"):
            yield {"type": "usage", "usage": _normalise_usage(data["usage"])}

    def _parse_stream(self, response) -> Iterator[Dict[str, Any]]:
        usage: Optional[Dict[str, Any]] = None
        finish_reason: Optional[str] = None
        for payload in self._iter_sse(response):
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue

            if isinstance(chunk, dict) and chunk.get("error"):
                error = chunk["error"]
                message = error.get("message") if isinstance(error, dict) else str(error)
                raise APIError(str(message))

            if chunk.get("usage"):
                usage = _normalise_usage(chunk["usage"])

            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    yield {"type": "reasoning", "text": reasoning}
                content = delta.get("content")
                if content:
                    yield {"type": "content", "text": content}
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

        yield {"type": "done", "finish_reason": finish_reason}
        if usage:
            yield {"type": "usage", "usage": usage}

    def list_models(self) -> List[str]:
        """Return the model ids advertised by ``GET /models``."""
        request = self._build_request("/models", None, "GET")
        response = self._request_with_retries(request)
        try:
            data = json.loads(response.read().decode("utf-8", "replace"))
        except ValueError as exc:
            raise APIError(f"the API returned invalid JSON: {exc}") from exc
        entries = data.get("data") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            return []
        return [str(entry.get("id")) for entry in entries if isinstance(entry, dict) and entry.get("id")]

    def balance(self) -> Dict[str, Any]:
        """Return the account balance from ``GET /user/balance``."""
        request = self._build_request("/user/balance", None, "GET")
        response = self._request_with_retries(request)
        try:
            return json.loads(response.read().decode("utf-8", "replace"))
        except ValueError as exc:
            raise APIError(f"the API returned invalid JSON: {exc}") from exc


def _normalise_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise token accounting across API shapes."""
    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if hit is None and miss is None:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            hit = details.get("cached_tokens")
            if hit is not None:
                miss = max(0, prompt - int(hit))
    normalised = {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(usage.get("total_tokens") or (prompt + completion)),
    }
    if hit is not None:
        normalised["prompt_cache_hit_tokens"] = int(hit)
    if miss is not None:
        normalised["prompt_cache_miss_tokens"] = int(miss)
    return normalised


def collect(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Fold a stream of events into ``{content, reasoning_content, usage, ...}``."""
    content: List[str] = []
    reasoning: List[str] = []
    usage: Dict[str, Any] = {}
    finish_reason: Optional[str] = None
    for event in events:
        kind = event.get("type")
        if kind == "content":
            content.append(event.get("text", ""))
        elif kind == "reasoning":
            reasoning.append(event.get("text", ""))
        elif kind == "usage":
            usage = event.get("usage") or {}
        elif kind == "done":
            finish_reason = event.get("finish_reason")
    return {
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "usage": usage,
        "finish_reason": finish_reason,
    }
