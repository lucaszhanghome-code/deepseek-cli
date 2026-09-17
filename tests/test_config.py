"""Tests for configuration handling."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseek_cli import config  # noqa: E402


class ConfigDirTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._previous = os.environ.get("DEEPSEEK_CLI_HOME")
        os.environ["DEEPSEEK_CLI_HOME"] = str(self.tmp)

    def tearDown(self) -> None:
        if self._previous is None:
            os.environ.pop("DEEPSEEK_CLI_HOME", None)
        else:
            os.environ["DEEPSEEK_CLI_HOME"] = self._previous
        self._tmp.cleanup()

    def test_config_dir_honours_override(self) -> None:
        self.assertEqual(config.config_dir(), self.tmp)
        self.assertEqual(config.config_path(), self.tmp / "config.json")
        self.assertEqual(config.sessions_dir(), self.tmp / "sessions")

    def test_defaults_when_file_is_missing(self) -> None:
        loaded = config.load_config()
        self.assertEqual(loaded["model"], "deepseek-chat")
        self.assertEqual(loaded["base_url"], config.DEFAULT_BASE_URL)
        self.assertTrue(loaded["stream"])

    def test_values_override_defaults(self) -> None:
        path = self.tmp / "config.json"
        path.write_text(json.dumps({"model": "deepseek-reasoner", "temperature": 0.2}), encoding="utf-8")
        loaded = config.load_config()
        self.assertEqual(loaded["model"], "deepseek-reasoner")
        self.assertEqual(loaded["temperature"], 0.2)
        self.assertEqual(loaded["max_tokens"], config.DEFAULT_CONFIG["max_tokens"])

    def test_unknown_keys_are_rejected(self) -> None:
        path = self.tmp / "config.json"
        path.write_text(json.dumps({"modle": "typo"}), encoding="utf-8")
        with self.assertRaises(config.ConfigError):
            config.load_config()

    def test_malformed_json_is_rejected(self) -> None:
        (self.tmp / "config.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(config.ConfigError):
            config.load_config()

    def test_save_and_reload_roundtrip(self) -> None:
        written = config.save_config({"model": "deepseek-reasoner", "pricing": {"*": {"output": 1.0}}})
        self.assertTrue(written.exists())
        reloaded = config.load_config()
        self.assertEqual(reloaded["model"], "deepseek-reasoner")
        self.assertEqual(reloaded["pricing"], {"*": {"output": 1.0}})

    def test_save_creates_missing_directories(self) -> None:
        nested = self.tmp / "a" / "b" / "config.json"
        config.save_config({"model": "deepseek-chat"}, nested)
        self.assertTrue(nested.exists())


class ApiKeyTests(unittest.TestCase):
    def test_cli_value_wins(self) -> None:
        key = config.resolve_api_key("from-flag", {"api_key": "from-file"}, {"DEEPSEEK_API_KEY": "from-env"})
        self.assertEqual(key, "from-flag")

    def test_environment_beats_config_file(self) -> None:
        key = config.resolve_api_key(None, {"api_key": "from-file"}, {"DEEPSEEK_API_KEY": "from-env"})
        self.assertEqual(key, "from-env")

    def test_config_file_is_used_as_fallback(self) -> None:
        self.assertEqual(config.resolve_api_key(None, {"api_key": "from-file"}, {}), "from-file")

    def test_blank_values_are_ignored(self) -> None:
        self.assertIsNone(config.resolve_api_key("  ", {"api_key": ""}, {"DEEPSEEK_API_KEY": " "}))

    def test_alternate_env_name(self) -> None:
        self.assertEqual(config.resolve_api_key(None, None, {"DEEPSEEK_KEY": "alt"}), "alt")

    def test_require_api_key_raises_with_guidance(self) -> None:
        saved = dict(os.environ)
        for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_KEY"):
            os.environ.pop(name, None)
        try:
            with self.assertRaises(config.ConfigError) as ctx:
                config.require_api_key(None, None)
        finally:
            os.environ.clear()
            os.environ.update(saved)
        message = str(ctx.exception)
        self.assertIn("DEEPSEEK_API_KEY", message)
        self.assertIn("--api-key", message)


class BaseUrlTests(unittest.TestCase):
    def test_trailing_slash_is_stripped(self) -> None:
        self.assertEqual(config.resolve_base_url("https://example.com/", None), "https://example.com")

    def test_config_value_is_used(self) -> None:
        self.assertEqual(config.resolve_base_url(None, {"base_url": "https://proxy.local"}), "https://proxy.local")

    def test_default_is_used_last(self) -> None:
        previous = os.environ.pop("DEEPSEEK_BASE_URL", None)
        try:
            self.assertEqual(config.resolve_base_url(None, {}), config.DEFAULT_BASE_URL)
        finally:
            if previous is not None:
                os.environ["DEEPSEEK_BASE_URL"] = previous


if __name__ == "__main__":
    unittest.main()
