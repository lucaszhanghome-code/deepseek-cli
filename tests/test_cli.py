"""End-to-end tests for the command line interface."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseek_cli import cli  # noqa: E402
from deepseek_cli import onboard  # noqa: E402
from deepseek_cli.args import build_parser, resolve_settings, apply_overrides  # noqa: E402
from deepseek_cli import config as config_mod  # noqa: E402
from deepseek_cli.chat import Chat  # noqa: E402
from deepseek_cli.client import DeepSeekClient  # noqa: E402
from deepseek_cli.session import SessionStore  # noqa: E402
from tests.stub import VALID_KEY, StubServer  # noqa: E402


def feed(lines: Iterable[str]):
    """Return a ``read_input`` stand-in that replays ``lines`` then EOFs."""
    iterator = iter(lines)

    def reader() -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError

    return reader


class BaseStubTest(unittest.TestCase):
    """Shared fixture: stub API server plus an isolated config directory."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self._env = dict(os.environ)
        os.environ["DEEPSEEK_CLI_HOME"] = str(self.home)
        os.environ["DEEPSEEK_API_KEY"] = VALID_KEY
        os.environ.pop("DEEPSEEK_BASE_URL", None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env)
        self._tmp.cleanup()

    def make_chat(self, server: StubServer, **overrides) -> Chat:
        settings: Dict[str, object] = {
            "model": "deepseek-chat",
            "system_prompt": "",
            "temperature": 1.0,
            "max_tokens": 256,
            "stream": True,
            "reasoning": True,
            "history_limit": 40,
            "timeout": 10,
            "max_retries": 0,
            "pricing": {},
            "no_save": False,
            "quiet": True,
        }
        settings.update(overrides)
        client = DeepSeekClient(VALID_KEY, server.base_url, timeout=10, max_retries=0)
        chat = Chat(client, settings, store=SessionStore(self.home / "sessions"), interactive=False)
        chat.out = io.StringIO()
        chat.err = io.StringIO()
        return chat


class ParserTests(BaseStubTest):
    def test_prompt_is_joined(self) -> None:
        ns = build_parser().parse_args(["hello", "there"])
        self.assertEqual(ns.prompt, ["hello", "there"])

    def test_defaults_are_unset_so_config_can_fill_them(self) -> None:
        ns = build_parser().parse_args([])
        self.assertIsNone(ns.model)
        self.assertIsNone(ns.temperature)
        self.assertIsNone(ns.stream)

    def test_settings_fall_back_to_config(self) -> None:
        cfg = dict(config_mod.DEFAULT_CONFIG)
        cfg["model"] = "deepseek-reasoner"
        ns = build_parser().parse_args([])
        settings = resolve_settings(ns, cfg)
        self.assertEqual(settings["model"], "deepseek-reasoner")
        self.assertTrue(settings["stream"])

    def test_flags_beat_config(self) -> None:
        cfg = dict(config_mod.DEFAULT_CONFIG)
        cfg["model"] = "deepseek-reasoner"
        ns = build_parser().parse_args(["-m", "deepseek-chat", "--no-stream", "-t", "0.5"])
        settings = resolve_settings(ns, cfg)
        self.assertEqual(settings["model"], "deepseek-chat")
        self.assertFalse(settings["stream"])
        self.assertEqual(settings["temperature"], 0.5)

    def test_overrides_are_coerced_and_persisted(self) -> None:
        cfg = apply_overrides(["model=deepseek-reasoner", "temperature=0.2", "stream=false"])
        self.assertEqual(cfg["model"], "deepseek-reasoner")
        self.assertEqual(cfg["temperature"], 0.2)
        self.assertFalse(cfg["stream"])
        self.assertEqual(config_mod.load_config()["temperature"], 0.2)

    def test_unknown_override_key_is_rejected(self) -> None:
        with self.assertRaises(config_mod.ConfigError):
            apply_overrides(["nope=1"])

    def test_bad_override_value_is_rejected(self) -> None:
        with self.assertRaises(config_mod.ConfigError):
            apply_overrides(["temperature=hot"])


class OneShotTests(BaseStubTest):
    def run_main(self, argv: List[str]) -> "tuple[int, str, str]":
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), _stdin(io.StringIO("")):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_prompt_is_answered_and_streamed(self) -> None:
        with StubServer() as server:
            code, out, _ = self.run_main(["--base-url", server.base_url, "say hello"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Hello from the stub", out)

    def test_json_mode_emits_a_single_object(self) -> None:
        with StubServer() as server:
            code, out, _ = self.run_main(["--base-url", server.base_url, "--json", "hi WITH_REASONING"])
        self.assertEqual(code, cli.EXIT_OK)
        payload = json.loads(out)
        self.assertEqual(payload["content"], "Hello from the stub")
        self.assertEqual(payload["reasoning_content"], "weighing options")
        self.assertEqual(payload["usage"]["total_tokens"], 18)
        self.assertEqual(payload["model"], "deepseek-chat")

    def test_stdin_prompt_is_used_when_no_argument(self) -> None:
        with StubServer() as server:
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err), _stdin(io.StringIO("piped question\n")):
                code = cli.main(["--base-url", server.base_url])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Hello from the stub", out.getvalue())

    def test_output_file_is_appended(self) -> None:
        target = self.home / "reply.txt"
        with StubServer() as server:
            code, _, _ = self.run_main(["--base-url", server.base_url, "-o", str(target), "hi"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Hello from the stub", target.read_text(encoding="utf-8"))

    def test_auth_failure_exits_one(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "wrong-key"
        with StubServer() as server:
            code, _, err = self.run_main(["--base-url", server.base_url, "hi"])
        self.assertEqual(code, cli.EXIT_ERROR)
        self.assertIn("401", err)

    def test_missing_api_key_exits_two(self) -> None:
        os.environ.pop("DEEPSEEK_API_KEY", None)
        code, _, err = self.run_main(["hi"])
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertIn("DEEPSEEK_API_KEY", err)

    def test_model_flag_is_sent_to_the_api(self) -> None:
        from tests.stub import StubHandler

        with StubServer() as server:
            self.run_main(["--base-url", server.base_url, "-m", "deepseek-reasoner", "--no-stream", "hi"])
        self.assertEqual(StubHandler.requests[-1]["payload"]["model"], "deepseek-reasoner")
        self.assertFalse(StubHandler.requests[-1]["payload"]["stream"])

    def test_print_config_does_not_need_a_key(self) -> None:
        os.environ.pop("DEEPSEEK_API_KEY", None)
        code, out, _ = self.run_main(["--print-config"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("base_url", out)

    def test_no_prompt_and_no_stdin_is_a_usage_error(self) -> None:
        code, _, err = self.run_main([])
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertIn("no prompt", err)

    def test_list_models_hits_the_api(self) -> None:
        with StubServer() as server:
            code, out, _ = self.run_main(["--base-url", server.base_url, "--list-models"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("deepseek-reasoner", out)

    def test_sessions_listing_works_when_empty(self) -> None:
        code, out, _ = self.run_main(["--sessions"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("no saved sessions", out)

    def test_saved_session_continues_with_history(self) -> None:
        store = SessionStore(self.home / "sessions")
        from deepseek_cli.session import Session

        session = Session()
        session.add("user", "earlier question")
        session.add("assistant", "earlier answer")
        store.save(session)

        with StubServer() as server:
            code, out, _ = self.run_main(["--base-url", server.base_url, "-c", "and now?"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Hello from the stub", out)

        from tests.stub import StubHandler

        sent = StubHandler.requests[-1]["payload"]["messages"]
        self.assertEqual([m["role"] for m in sent], ["user", "assistant", "user"])
        self.assertEqual(sent[0]["content"], "earlier question")


class CommandTests(BaseStubTest):
    def test_run_answers_then_quits(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            chat.read_input = feed(["hello", "/quit"])
            code = chat.run()
        self.assertEqual(code, 0)
        roles = [m["role"] for m in chat.session.messages]
        self.assertEqual(roles, ["user", "assistant"])
        self.assertEqual(chat.session.messages[1]["content"], "Hello from the stub")

    def test_eof_exits_cleanly(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            chat.read_input = feed([])
            self.assertEqual(chat.run(), 0)

    def test_unknown_command_is_reported(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            chat.read_input = feed(["/nope", "/quit"])
            out, err = io.StringIO(), io.StringIO()
            chat.out, chat.err = out, err
            chat.run()
        self.assertIn("unknown command", err.getvalue())

    def test_model_temperature_and_system_commands(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            chat.handle_command("/model deepseek-reasoner")
            chat.handle_command("/temp 0.3")
            chat.handle_command("/system be brief")
            chat.handle_command("/maxtokens 512")
            chat.handle_command("/stream off")
            chat.handle_command("/reasoning off")
        self.assertEqual(chat.model, "deepseek-reasoner")
        self.assertEqual(chat.settings["temperature"], 0.3)
        self.assertEqual(chat.session.system_prompt, "be brief")
        self.assertEqual(chat.settings["max_tokens"], 512)
        self.assertFalse(chat.settings["stream"])
        self.assertFalse(chat.show_reasoning)

    def test_invalid_command_arguments_are_rejected(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            err = io.StringIO()
            chat.err = err
            chat.handle_command("/temp hot")
            chat.handle_command("/temp 5")
            chat.handle_command("/maxtokens zero")
            chat.handle_command("/stream maybe")
        self.assertIn("expected a number", err.getvalue())
        self.assertIn("between 0 and 2", err.getvalue())
        self.assertEqual(chat.settings["temperature"], 1.0)
        self.assertTrue(chat.settings["stream"])

    def test_clear_undo_retry_and_history(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            chat.send("first", display=False)
            chat.send("second", display=False)
            self.assertEqual(len(chat.session.messages), 4)
            chat.handle_command("/undo")
            self.assertEqual(len(chat.session.messages), 2)
            chat.handle_command("/clear")
            self.assertEqual(chat.session.messages, [])
            self.assertEqual(chat.session.usage["turns"], 0)
            chat.handle_command("/retry")  # last_prompt is still "second"
            self.assertEqual(chat.session.messages[0]["content"], "second")

    def test_new_and_load_sessions(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            chat.send("keep me", display=False)
            first_id = chat.session.id
            chat.handle_command("/new")
            self.assertNotEqual(chat.session.id, first_id)
            chat.handle_command(f"/load {first_id}")
        self.assertEqual(chat.session.id, first_id)
        self.assertEqual(chat.session.messages[0]["content"], "keep me")

    def test_sessions_save_list_and_delete(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            out = io.StringIO()
            chat.out = out
            chat.send("hello", display=False)
            chat.handle_command("/sessions")
            self.assertIn(chat.session.id, out.getvalue())
            chat.handle_command("/sessions")  # ids stay stable
            chat.handle_command(f"/delete {chat.session.id}")
        self.assertTrue(out.getvalue().count(chat.session.id) >= 2)
        self.assertEqual(chat.store.list_sessions(), [])

    def test_usage_and_balance_and_models_render(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            out = io.StringIO()
            chat.out = out
            chat.send("hi", display=False)
            chat.handle_command("/usage")
            chat.handle_command("/balance")
            chat.handle_command("/models")
        text = out.getvalue()
        self.assertIn("prompt_tokens", text)
        self.assertIn("12.34", text)
        self.assertIn("deepseek-chat", text)

    def test_footer_reports_tokens_and_cost(self) -> None:
        pricing = {"deepseek-chat": {"input_cache_hit": 1.0, "input_cache_miss": 1.0, "output": 1.0}}
        with StubServer() as server:
            chat = self.make_chat(server, pricing=pricing, quiet=False)
            chat.interactive = True
            out = io.StringIO()
            chat.out = out
            chat.send("hi", display=False)
        self.assertIn("11 in / 7 out", out.getvalue())
        self.assertIn("cache 3 hit / 8 miss", out.getvalue())
        self.assertIn("$", out.getvalue())

    def test_failed_request_rolls_back_the_prompt(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            chat.client.api_key = "bad"
            err = io.StringIO()
            chat.err = err
            result = chat.send("doomed", display=False)
        self.assertIn("error", result)
        self.assertEqual(chat.session.messages, [])
        self.assertIn("401", err.getvalue())

    def test_retry_after_failure_resends(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            chat.client.api_key = "bad"
            chat.err = io.StringIO()
            chat.send("doomed", display=False)
            chat.client.api_key = VALID_KEY
            chat.settings["quiet"] = True
            chat.handle_command("/retry")
        self.assertEqual(chat.session.messages[0]["content"], "doomed")
        self.assertEqual(chat.session.messages[1]["role"], "assistant")

    def test_help_lists_commands(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            out = io.StringIO()
            chat.out = out
            chat.handle_command("/help")
        self.assertIn("/quit", out.getvalue())
        self.assertIn("/system", out.getvalue())

    def test_multiline_input_uses_backslash(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            with _input_lines(["line one \\", "line two"]):
                self.assertEqual(chat.read_input(), "line one \nline two")

    def test_multiline_prompt_is_sent_as_one_message(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            with _input_lines(["line one \\", "line two", "/quit"]):
                chat.run()
        self.assertEqual(chat.session.messages[0]["content"], "line one \nline two")

    def test_reply_prefix_is_printed_once(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            out = io.StringIO()
            chat.out = out
            chat.send("hi", display=True)
        self.assertEqual(out.getvalue().count("deepseek>"), 1)

    def test_no_save_skips_persistence(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server, no_save=True)
            chat.send("transient", display=False)
            chat.handle_command("/save")
        self.assertEqual(chat.store.list_sessions(), [])


class _tty:
    """A ``sys.stdin`` stand-in that reports itself as a terminal."""

    def isatty(self) -> bool:
        return True

    def read(self) -> str:
        return ""


class FirstRunKeyTests(BaseStubTest):
    """The paste-your-own-key onboarding flow."""

    def setUp(self) -> None:
        super().setUp()
        for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_KEY", "DEEPSEEK_CLI_NO_PROMPT"):
            os.environ.pop(name, None)

    def run_main(
        self, argv: List[str], stdin: Optional[io.StringIO] = None
    ) -> "tuple[int, str, str]":
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), _stdin(stdin or io.StringIO("")):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_pasting_a_key_on_first_run_starts_the_chat(self) -> None:
        with StubServer() as server:
            argv = ["--base-url", server.base_url, "say hello"]
            with mock.patch.object(onboard, "read_key", return_value=VALID_KEY):
                code, out, err = self.run_main(argv, stdin=_tty())
        self.assertEqual(code, cli.EXIT_OK, err)
        self.assertIn("Hello from the stub", out)
        self.assertIn("accepted", err)
        self.assertEqual(config_mod.load_config()["api_key"], VALID_KEY)

    def test_the_saved_key_is_reused_without_asking_again(self) -> None:
        with StubServer() as server:
            argv = ["--base-url", server.base_url, "say hello"]
            with mock.patch.object(onboard, "read_key", return_value=VALID_KEY):
                self.run_main(argv, stdin=_tty())
            with mock.patch.object(onboard, "read_key", side_effect=AssertionError):
                code, out, _ = self.run_main(argv, stdin=_tty())
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Hello from the stub", out)

    def test_cancelling_the_prompt_leaves_the_run_keyless(self) -> None:
        with StubServer() as server:
            argv = ["--base-url", server.base_url, "say hello"]
            with mock.patch.object(onboard, "read_key", return_value=None):
                code, out, err = self.run_main(argv, stdin=_tty())
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertEqual(out, "")
        self.assertIn("cancelled", err)

    def test_no_prompt_flag_never_reads_a_key(self) -> None:
        with StubServer() as server:
            with mock.patch.object(onboard, "read_key", side_effect=AssertionError) as reader:
                code, _, err = self.run_main(
                    ["--no-prompt", "--base-url", server.base_url, "hi"], stdin=_tty()
                )
        self.assertEqual(reader.call_count, 0)
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertIn("DEEPSEEK_API_KEY", err)

    def test_no_prompt_environment_variable_never_reads_a_key(self) -> None:
        os.environ["DEEPSEEK_CLI_NO_PROMPT"] = "1"
        with StubServer() as server:
            with mock.patch.object(onboard, "read_key", side_effect=AssertionError) as reader:
                code, _, _ = self.run_main(["--base-url", server.base_url, "hi"], stdin=_tty())
        self.assertEqual(reader.call_count, 0)
        self.assertEqual(code, cli.EXIT_USAGE)

    def test_json_mode_never_reads_a_key(self) -> None:
        with StubServer() as server:
            with mock.patch.object(onboard, "read_key", side_effect=AssertionError) as reader:
                code, out, err = self.run_main(
                    ["--json", "--base-url", server.base_url, "hi"], stdin=_tty()
                )
        self.assertEqual(reader.call_count, 0)
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertEqual(out, "")
        self.assertIn("DEEPSEEK_API_KEY", err)

    def test_piped_stdin_without_a_key_fails_instead_of_hanging(self) -> None:
        with StubServer() as server:
            with mock.patch.object(onboard, "read_key", side_effect=AssertionError):
                code, _, err = self.run_main(
                    ["--base-url", server.base_url], stdin=io.StringIO("piped\n")
                )
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertIn("DEEPSEEK_API_KEY", err)

    def test_config_never_prints_the_raw_key(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            chat.client.api_key = "sk-abcdef123456789"
            chat.handle_command("/config")
        printed = chat.out.getvalue()
        self.assertNotIn("sk-abcdef123456789", printed)
        self.assertIn(config_mod.mask_key("sk-abcdef123456789"), printed)

    def test_print_config_reports_a_missing_key(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), _stdin(io.StringIO("")):
            code = cli.main(["--print-config"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("MISSING", out.getvalue())

    def test_print_config_reports_the_saved_key_as_masked(self) -> None:
        config_mod.save_api_key("sk-abcdef123456789")
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), _stdin(io.StringIO("")):
            code = cli.main(["--print-config"])
        self.assertEqual(code, cli.EXIT_OK)
        printed = out.getvalue()
        self.assertNotIn("sk-abcdef123456789", printed)
        self.assertIn("config file", printed)


class KeyCommandTests(BaseStubTest):
    """/key pastes, verifies and stores a key mid-session."""

    def test_inline_key_is_verified_and_saved(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            chat.handle_command("/key " + VALID_KEY)
        self.assertEqual(chat.client.api_key, VALID_KEY)
        self.assertIn("accepted", chat.out.getvalue())
        self.assertEqual(config_mod.load_config()["api_key"], VALID_KEY)

    def test_a_rejected_key_is_not_stored(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            chat.handle_command("/key sk-wrong")
        self.assertEqual(chat.client.api_key, VALID_KEY)
        self.assertIn("rejected", chat.err.getvalue())
        self.assertEqual(config_mod.load_config()["api_key"], "")

    def test_the_prompt_is_used_when_no_key_is_given(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            with mock.patch.object(onboard, "can_prompt", return_value=True):
                with mock.patch.object(onboard, "read_key", return_value=VALID_KEY) as reader:
                    chat.handle_command("/key")
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(chat.client.api_key, VALID_KEY)

    def test_usage_is_shown_when_there_is_no_terminal(self) -> None:
        with StubServer() as server:
            chat = self.make_chat(server)
            with mock.patch.object(onboard, "can_prompt", return_value=False):
                with mock.patch.object(onboard, "read_key", side_effect=AssertionError):
                    chat.handle_command("/key")
        self.assertIn("usage", chat.err.getvalue())
        self.assertIn("/key", chat.err.getvalue())


class _input_lines:
    """Feed canned answers to ``input()``."""

    def __init__(self, lines: Iterable[str]) -> None:
        self._iterator = iter(lines)
        self._original = None

    def __enter__(self):
        import builtins

        self._original = builtins.input

        def fake_input(prompt: str = "") -> str:
            try:
                return next(self._iterator)
            except StopIteration:
                raise EOFError

        builtins.input = fake_input
        return self

    def __exit__(self, *exc_info) -> None:
        import builtins

        builtins.input = self._original


class _stdin:
    """Temporarily replace ``sys.stdin``."""

    def __init__(self, stream: Optional[io.StringIO]) -> None:
        self.stream = stream
        self.previous = None

    def __enter__(self):
        self.previous = sys.stdin
        sys.stdin = self.stream
        return self.stream

    def __exit__(self, *exc_info) -> None:
        sys.stdin = self.previous


if __name__ == "__main__":
    unittest.main()
