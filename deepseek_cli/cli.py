"""Entry point for the ``deepseek`` command."""

from __future__ import annotations

import io
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import args as args_mod, config as config_mod, onboard, render
from .chat import Chat
from .client import DeepSeekClient, DeepSeekError
from .session import Session, SessionStore

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPT = 130


def _read_stdin() -> str:
    try:
        return sys.stdin.read()
    except (OSError, UnicodeDecodeError):
        return ""


def _print_config(settings: Dict[str, Any], stream) -> None:
    cfg = config_mod.load_config()
    source = config_mod.api_key_source(config=cfg)
    key = config_mod.resolve_api_key(config=cfg)
    rows: List[List[str]] = [
        ["model", str(settings["model"])],
        ["base_url", str(settings["base_url"])],
        ["temperature", str(settings["temperature"])],
        ["max_tokens", str(settings["max_tokens"])],
        ["stream", str(settings["stream"])],
        ["history_limit", str(settings["history_limit"])],
        ["timeout", str(settings["timeout"])],
        ["max_retries", str(settings["max_retries"])],
        ["system_prompt", str(settings["system_prompt"]) or "(none)"],
        ["pricing", "configured" if settings["pricing"] else "(none)"],
        ["api key", f"{config_mod.mask_key(key)} (from {source})" if key else "MISSING"],
    ]
    render.print_table(rows, ["setting", "value"], stream)
    stream.write(f"\nconfig file : {config_mod.config_path()}\n")
    stream.write(f"sessions dir: {config_mod.sessions_dir()}\n")


def _print_sessions(store: SessionStore, stream) -> None:
    sessions = store.list_sessions()
    if not sessions:
        stream.write(f"no saved sessions in {store.directory}\n")
        return
    rows = [
        [
            session.id,
            str(len([m for m in session.messages if m.get("role") == "user"])),
            session.model,
            (session.title or "(untitled)")[:48],
        ]
        for session in sessions
    ]
    render.print_table(rows, ["id", "turns", "model", "title"], stream)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = args_mod.build_parser()
    ns = parser.parse_args(argv)

    try:
        cfg = args_mod.apply_overrides(ns.overrides)
        settings = args_mod.resolve_settings(ns, cfg)
    except config_mod.ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    style = render.Style(enabled=not ns.no_color and render.supports_ansi(sys.stdout))
    store = SessionStore()

    if ns.print_config:
        _print_config(settings, sys.stdout)
        return EXIT_OK
    if ns.sessions:
        _print_sessions(store, sys.stdout)
        return EXIT_OK

    try:
        api_key: Optional[str] = config_mod.require_api_key(ns.api_key, cfg)
    except config_mod.ConfigError as exc:
        message = str(exc)
        api_key = None
        if not ns.no_prompt and not ns.as_json and onboard.can_prompt():
            api_key = onboard.ask_for_api_key(settings, style=style, err=sys.stderr)
        if api_key is None:
            print(f"error: {message}", file=sys.stderr)
            return EXIT_USAGE

    client = DeepSeekClient(
        api_key,
        base_url=settings["base_url"],
        timeout=int(settings["timeout"]),
        max_retries=int(settings["max_retries"]),
    )

    if ns.list_models or ns.balance:
        probe = Chat(client, settings, style=style, store=store, interactive=False)
        probe.out, probe.err = sys.stdout, sys.stderr
        if ns.list_models:
            probe.cmd_models("")
        if ns.balance:
            probe.cmd_balance("")
        return EXIT_OK

    prompt = " ".join(ns.prompt).strip() or None
    if prompt is None:
        if sys.stdin.isatty():
            one_shot = False
        else:
            prompt = _read_stdin().strip() or None
            one_shot = prompt is not None
            if prompt is None:
                print(
                    "error: no prompt given (pass one as an argument or pipe it on stdin)",
                    file=sys.stderr,
                )
                return EXIT_USAGE
    else:
        one_shot = True

    # ------------------------------------------------------------------ session
    session: Optional[Session] = None
    try:
        if ns.session:
            session = store.load(ns.session)
        elif ns.continue_last:
            session = store.latest()
            if session is None:
                print("no saved sessions to continue", file=sys.stderr)
                return EXIT_USAGE
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    settings["quiet"] = bool(ns.quiet)

    capture = io.StringIO() if (one_shot and ns.as_json) else sys.stdout
    chat = Chat(
        client,
        settings,
        session=session,
        store=store,
        style=style,
        out=capture,
        interactive=not one_shot,
    )

    if one_shot:
        result = chat.once(prompt or "")
        if ns.output:
            try:
                with open(ns.output, "a", encoding="utf-8") as handle:
                    handle.write(result.get("content", "") + "\n")
            except OSError as exc:
                print(f"error: could not write {ns.output}: {exc}", file=sys.stderr)
                return EXIT_ERROR
        if ns.as_json:
            payload = {
                "model": chat.model,
                "content": result.get("content", ""),
                "reasoning_content": result.get("reasoning_content", ""),
                "usage": result.get("usage", {}),
                "cost_usd": result.get("cost_usd"),
                "finish_reason": result.get("finish_reason"),
                "elapsed_seconds": round(float(result.get("elapsed", 0.0)), 3),
                "session_id": chat.session.id,
            }
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        if result.get("error"):
            return EXIT_ERROR
        if result.get("interrupted"):
            return EXIT_INTERRUPT
        return EXIT_OK

    try:
        return chat.run()
    except KeyboardInterrupt:
        sys.stderr.write("\n")
        chat.save()
        return EXIT_INTERRUPT


__all__ = ["main", "EXIT_OK", "EXIT_ERROR", "EXIT_USAGE", "EXIT_INTERRUPT"]
