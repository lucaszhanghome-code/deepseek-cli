"""Tests for first-run API key onboarding."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseek_cli import config as config_mod  # noqa: E402
from deepseek_cli import onboard  # noqa: E402
from tests.stub import VALID_KEY, StubServer  # noqa: E402

DEAD_URL = "http://127.0.0.1:9"


class _FakeStdin:
    """A stdin stand-in with a controllable ``isatty``."""

    def __init__(self, tty: bool) -> None:
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty

    def read(self) -> str:
        return ""


class KeyPromptTests(unittest.TestCase):
    """Unit level: prompting, verification and storage."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self._env = dict(os.environ)
        os.environ["DEEPSEEK_CLI_HOME"] = str(self.home)
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ.pop("DEEPSEEK_KEY", None)
        os.environ.pop(onboard.NO_PROMPT_ENV, None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env)
        self._tmp.cleanup()

    # ------------------------------------------------------------ prompt gates
    def test_prompting_requires_a_terminal(self) -> None:
        self.assertTrue(onboard.can_prompt(stdin=_FakeStdin(True)))
        self.assertFalse(onboard.can_prompt(stdin=_FakeStdin(False)))
        self.assertFalse(onboard.can_prompt(interactive=False, stdin=_FakeStdin(True)))

    def test_prompting_can_be_switched_off_by_the_environment(self) -> None:
        os.environ[onboard.NO_PROMPT_ENV] = "1"
        self.assertFalse(onboard.can_prompt(stdin=_FakeStdin(True)))
        self.assertFalse(onboard.prompting_allowed())
        self.assertTrue(onboard.prompting_allowed({"DEEPSEEK_CLI_NO_PROMPT": "no"}))

    # ------------------------------------------------------------- verification
    def test_verification_accepts_a_good_key(self) -> None:
        with StubServer() as server:
            self.assertIsNone(onboard.verify_key(VALID_KEY, base_url=server.base_url))

    def test_verification_reports_a_rejected_key(self) -> None:
        with StubServer() as server:
            problem = onboard.verify_key("not-the-key", base_url=server.base_url)
        self.assertIsNotNone(problem)
        self.assertIn("rejected", problem or "")

    def test_verification_tolerates_an_unreachable_api(self) -> None:
        self.assertIsNone(onboard.verify_key(VALID_KEY, base_url=DEAD_URL, timeout=1))

    # ------------------------------------------------------------------ prompting
    def test_pasted_key_is_verified_saved_and_returned(self) -> None:
        err = io.StringIO()
        with StubServer() as server:
            settings = {"base_url": server.base_url, "timeout": 10}
            with mock.patch.object(onboard, "read_key", return_value=VALID_KEY):
                key = onboard.ask_for_api_key(settings, err=err)
        self.assertEqual(key, VALID_KEY)
        self.assertIn("accepted", err.getvalue())
        self.assertEqual(config_mod.load_config()["api_key"], VALID_KEY)
        self.assertEqual(config_mod.resolve_api_key(config=config_mod.load_config()), VALID_KEY)

    def test_saved_key_wins_on_the_next_run(self) -> None:
        with StubServer() as server:
            with mock.patch.object(onboard, "read_key", return_value=VALID_KEY):
                onboard.ask_for_api_key({"base_url": server.base_url}, err=io.StringIO())
        self.assertEqual(config_mod.require_api_key(None, config_mod.load_config()), VALID_KEY)

    def test_a_rejected_key_is_asked_for_again(self) -> None:
        err = io.StringIO()
        with StubServer() as server:
            with mock.patch.object(
                onboard, "read_key", side_effect=["wrong-key", VALID_KEY]
            ) as reader:
                key = onboard.ask_for_api_key({"base_url": server.base_url}, err=err)
        self.assertEqual(reader.call_count, 2)
        self.assertEqual(key, VALID_KEY)
        self.assertIn("rejected", err.getvalue())

    def test_pasting_nothing_asks_again(self) -> None:
        err = io.StringIO()
        with StubServer() as server:
            with mock.patch.object(onboard, "read_key", side_effect=["   ", VALID_KEY]):
                key = onboard.ask_for_api_key({"base_url": server.base_url}, err=err)
        self.assertEqual(key, VALID_KEY)
        self.assertIn("empty", err.getvalue())

    def test_cancelling_returns_none_and_saves_nothing(self) -> None:
        err = io.StringIO()
        with mock.patch.object(onboard, "read_key", return_value=None):
            key = onboard.ask_for_api_key({}, err=err)
        self.assertIsNone(key)
        self.assertIn("cancelled", err.getvalue())
        self.assertFalse(config_mod.config_path().exists())

    def test_running_out_of_attempts_returns_none(self) -> None:
        err = io.StringIO()
        with StubServer() as server:
            with mock.patch.object(
                onboard, "read_key", return_value="not-the-key"
            ) as reader:
                key = onboard.ask_for_api_key(
                    {"base_url": server.base_url}, err=err, attempts=2
                )
        self.assertIsNone(key)
        self.assertEqual(reader.call_count, 2)
        self.assertIn("too many attempts", err.getvalue())

    def test_an_unverifiable_key_is_still_accepted_when_offline(self) -> None:
        with mock.patch.object(onboard, "read_key", return_value="sk-offline"):
            key = onboard.ask_for_api_key(
                {"base_url": DEAD_URL, "timeout": 1}, err=io.StringIO()
            )
        self.assertEqual(key, "sk-offline")
        self.assertEqual(config_mod.load_config()["api_key"], "sk-offline")

    def test_a_key_can_be_used_without_saving_it(self) -> None:
        with StubServer() as server:
            with mock.patch.object(onboard, "read_key", return_value=VALID_KEY):
                key = onboard.ask_for_api_key(
                    {"base_url": server.base_url}, err=io.StringIO(), save=False
                )
        self.assertEqual(key, VALID_KEY)
        self.assertFalse(config_mod.config_path().exists())

    # -------------------------------------------------------------------- storage
    def test_save_api_key_round_trips(self) -> None:
        path = config_mod.save_api_key("sk-pasted-by-the-user")
        self.assertTrue(path.exists())
        self.assertIn("sk-pasted-by-the-user", path.read_text(encoding="utf-8"))
        self.assertEqual(config_mod.load_config()["api_key"], "sk-pasted-by-the-user")

    def test_save_api_key_rejects_an_empty_value(self) -> None:
        with self.assertRaises(config_mod.ConfigError):
            config_mod.save_api_key("   ")

    def test_mask_key_never_reveals_the_middle(self) -> None:
        self.assertEqual(config_mod.mask_key("sk-1234567890abcdef"), "sk-123...cdef")
        self.assertEqual(config_mod.mask_key("short"), "*****")
        self.assertEqual(config_mod.mask_key(None), "(none)")

    def test_key_source_is_reported(self) -> None:
        self.assertEqual(config_mod.api_key_source("flag", {"api_key": "x"}), "flag")
        self.assertEqual(
            config_mod.api_key_source(None, {"api_key": "x"}, {"DEEPSEEK_API_KEY": "x"}),
            "$DEEPSEEK_API_KEY",
        )
        self.assertEqual(config_mod.api_key_source(None, {"api_key": "x"}, {}), "config file")
        self.assertEqual(config_mod.api_key_source(None, {}, {}), "none")


if __name__ == "__main__":
    unittest.main()
