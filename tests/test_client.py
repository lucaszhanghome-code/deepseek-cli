"""Tests for the DeepSeek API client."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseek_cli.client import (  # noqa: E402
    APIError,
    AuthError,
    DeepSeekClient,
    collect,
    estimate_cost,
)
from tests.stub import VALID_KEY, StubServer  # noqa: E402


class ClientTests(unittest.TestCase):
    def test_streaming_collects_content_reasoning_and_usage(self) -> None:
        with StubServer() as server:
            client = DeepSeekClient(VALID_KEY, server.base_url)
            result = collect(
                client.chat([{"role": "user", "content": "hi WITH_REASONING"}], model="deepseek-chat")
            )
        self.assertEqual(result["content"], "Hello from the stub")
        self.assertEqual(result["reasoning_content"], "weighing options")
        self.assertEqual(result["usage"]["prompt_tokens"], 11)
        self.assertEqual(result["usage"]["completion_tokens"], 7)
        self.assertEqual(result["usage"]["prompt_cache_hit_tokens"], 3)

    def test_non_streaming_returns_full_message(self) -> None:
        with StubServer() as server:
            client = DeepSeekClient(VALID_KEY, server.base_url)
            result = collect(
                client.chat(
                    [{"role": "user", "content": "hi"}],
                    model="deepseek-reasoner",
                    stream=False,
                )
            )
        self.assertEqual(result["content"], "Hello from the stub")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["usage"]["total_tokens"], 18)

    def test_payload_carries_model_messages_and_options(self) -> None:
        with StubServer() as server:
            client = DeepSeekClient(VALID_KEY, server.base_url)
            collect(
                client.chat(
                    [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
                    model="deepseek-chat",
                    temperature=0.3,
                    max_tokens=64,
                )
            )
        from tests.stub import StubHandler

        payload = StubHandler.requests[-1]["payload"]
        self.assertEqual(payload["model"], "deepseek-chat")
        self.assertEqual(payload["temperature"], 0.3)
        self.assertEqual(payload["max_tokens"], 64)
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "system")

    def test_retries_transient_errors_then_succeeds(self) -> None:
        with StubServer() as server:
            client = DeepSeekClient(VALID_KEY, server.base_url, max_retries=3)
            result = collect(client.chat([{"role": "user", "content": "TRIGGER_429"}]))
        self.assertEqual(result["content"], "Hello from the stub")
        from tests.stub import StubHandler

        self.assertEqual(StubHandler.counters["429"], 3)

    def test_bad_key_raises_auth_error(self) -> None:
        with StubServer() as server:
            client = DeepSeekClient("wrong-key", server.base_url, max_retries=0)
            with self.assertRaises(AuthError) as ctx:
                collect(client.chat([{"role": "user", "content": "hi"}]))
        self.assertIn("401", str(ctx.exception))
        self.assertIn("invalid api key", str(ctx.exception))

    def test_bad_request_surfaces_api_message(self) -> None:
        with StubServer() as server:
            client = DeepSeekClient(VALID_KEY, server.base_url, max_retries=0)
            with self.assertRaises(APIError) as ctx:
                collect(client.chat([{"role": "user", "content": "TRIGGER_400"}]))
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("bad request from test", str(ctx.exception))

    def test_missing_key_rejected_early(self) -> None:
        with self.assertRaises(AuthError):
            DeepSeekClient("")

    def test_list_models_and_balance(self) -> None:
        with StubServer() as server:
            client = DeepSeekClient(VALID_KEY, server.base_url)
            self.assertEqual(client.list_models(), ["deepseek-chat", "deepseek-reasoner"])
            balance = client.balance()
        self.assertTrue(balance["is_available"])
        self.assertEqual(balance["balance_infos"][0]["total_balance"], "12.34")

    def test_unreachable_host_raises_network_error(self) -> None:
        from deepseek_cli.client import NetworkError

        client = DeepSeekClient(VALID_KEY, "http://127.0.0.1:1", timeout=2, max_retries=0)
        with self.assertRaises(NetworkError):
            collect(client.chat([{"role": "user", "content": "hi"}]))

    def test_api_error_body_is_parsed_for_non_json(self) -> None:
        message = DeepSeekClient._describe_http_error(503, b"<html>down</html>")
        self.assertIn("503", message)
        self.assertIn("down", message)


class CostTests(unittest.TestCase):
    PRICING = {"deepseek-chat": {"input_cache_hit": 0.028, "input_cache_miss": 0.28, "output": 0.42}}

    def test_cost_uses_cache_breakdown(self) -> None:
        usage = {
            "prompt_cache_hit_tokens": 1_000_000,
            "prompt_cache_miss_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
        }
        cost = estimate_cost(usage, self.PRICING, "deepseek-chat")
        self.assertAlmostEqual(cost, 0.028 + 0.28 + 0.42, places=6)

    def test_cost_without_cache_breakdown_treats_prompt_as_miss(self) -> None:
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
        cost = estimate_cost(usage, self.PRICING, "deepseek-chat")
        self.assertAlmostEqual(cost, 0.28 + 0.42, places=6)

    def test_no_pricing_means_no_estimate(self) -> None:
        self.assertIsNone(estimate_cost({"prompt_tokens": 10}, {}, "deepseek-chat"))
        self.assertIsNone(estimate_cost({"prompt_tokens": 10}, self.PRICING, "other-model"))

    def test_wildcard_pricing_key(self) -> None:
        pricing = {"*": {"input_cache_miss": 1.0, "output": 1.0}}
        cost = estimate_cost({"prompt_tokens": 500_000, "completion_tokens": 500_000}, pricing, "x")
        self.assertAlmostEqual(cost, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
