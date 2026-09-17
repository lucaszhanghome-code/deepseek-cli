"""Command line parsing and settings resolution."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

from . import __version__, config as config_mod

BOOL_KEYS = {"stream"}
INT_KEYS = {"max_tokens", "timeout", "max_retries", "history_limit"}
FLOAT_KEYS = {"temperature"}
JSON_KEYS = {"pricing"}


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the ``deepseek`` command."""
    parser = argparse.ArgumentParser(
        prog="deepseek",
        description="Chat with the DeepSeek API from your terminal.",
        epilog=(
            "Run with no prompt to open an interactive chat; type /help inside it for "
            "commands, or /exit to quit."
        ),
    )
    parser.add_argument("prompt", nargs="*", help="prompt to send; omit for interactive mode")

    model = parser.add_argument_group("model")
    model.add_argument("-m", "--model", help="model id, e.g. deepseek-chat or deepseek-reasoner")
    model.add_argument("-s", "--system", help="system prompt for this conversation")
    model.add_argument("-t", "--temperature", type=float, help="sampling temperature (0-2)")
    model.add_argument("--max-tokens", type=int, help="maximum tokens to generate")
    model.add_argument("--stream", dest="stream", action="store_true", default=None, help="stream the reply")
    model.add_argument("--no-stream", dest="stream", action="store_false", help="wait for the whole reply")
    model.add_argument("--reasoning", dest="reasoning", action="store_true", default=None,
                       help="show reasoning_content when the model provides it")
    model.add_argument("--no-reasoning", dest="reasoning", action="store_false",
                       help="hide reasoning_content")

    conn = parser.add_argument_group("connection")
    conn.add_argument("--api-key", help="DeepSeek API key (defaults to $DEEPSEEK_API_KEY)")
    conn.add_argument("--no-prompt", action="store_true",
                      help="never ask for an API key on a terminal")
    conn.add_argument("--base-url", help="API base URL (defaults to $DEEPSEEK_BASE_URL)")
    conn.add_argument("--timeout", type=int, help="per-request timeout in seconds")
    conn.add_argument("--max-retries", type=int, help="retries for transient failures")

    convo = parser.add_argument_group("conversation")
    convo.add_argument("-c", "--continue", dest="continue_last", action="store_true",
                       help="resume the most recently updated saved session")
    convo.add_argument("--session", metavar="ID", help="resume a specific saved session id")
    convo.add_argument("--history-limit", type=int,
                       help="maximum number of past messages to resend as context")
    convo.add_argument("--no-save", action="store_true", help="do not persist the session to disk")

    out = parser.add_argument_group("output")
    out.add_argument("-o", "--output", metavar="FILE", help="append the final reply to FILE")
    out.add_argument("--json", dest="as_json", action="store_true",
                     help="print a JSON object instead of plain text (one-shot mode)")
    out.add_argument("-q", "--quiet", action="store_true",
                     help="suppress the banner and token/usage footers")
    out.add_argument("--no-color", action="store_true", help="disable ANSI colours")

    misc = parser.add_argument_group("utility")
    misc.add_argument("--list-models", action="store_true", help="list models available on the account")
    misc.add_argument("--balance", action="store_true", help="show the account balance")
    misc.add_argument("--sessions", action="store_true", help="list saved sessions and exit")
    misc.add_argument("--print-config", action="store_true", help="show effective settings and exit")
    misc.add_argument("--set", dest="overrides", metavar="KEY=VALUE", action="append",
                      help="persist a config value, e.g. --set model=deepseek-reasoner")
    misc.add_argument("--version", action="version", version=f"deepseek-cli {__version__}")
    return parser


def _coerce(key: str, value: str) -> Any:
    if key in BOOL_KEYS:
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{key} expects a boolean, got {value!r}")
    if key in INT_KEYS:
        return int(value)
    if key in FLOAT_KEYS:
        return float(value)
    if key in JSON_KEYS:
        return json.loads(value)
    return value


def apply_overrides(pairs, path: Optional[str] = None) -> Dict[str, Any]:
    """Persist ``KEY=VALUE`` overrides and return the new config."""
    cfg = config_mod.load_config(path)
    if not pairs:
        return cfg
    for pair in pairs:
        if "=" not in pair:
            raise config_mod.ConfigError(f"--set expects KEY=VALUE, got {pair!r}")
        key, _, value = pair.partition("=")
        key = key.strip()
        if key not in config_mod.DEFAULT_CONFIG:
            raise config_mod.ConfigError(
                f"unknown config key {key!r}. Supported: {', '.join(sorted(config_mod.DEFAULT_CONFIG))}"
            )
        try:
            cfg[key] = _coerce(key, value)
        except (ValueError, json.JSONDecodeError) as exc:
            raise config_mod.ConfigError(f"invalid value for {key}: {exc}") from exc
    config_mod.save_config(cfg, path)
    return cfg


def resolve_settings(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge flags, environment variables and config into one settings dict."""
    env = os.environ
    stream = args.stream
    if stream is None:
        stream = bool(cfg.get("stream", True))
    reasoning = args.reasoning
    if reasoning is None:
        reasoning = True

    return {
        "model": args.model or env.get("DEEPSEEK_MODEL") or cfg.get("model") or config_mod.DEFAULT_MODEL,
        "base_url": config_mod.resolve_base_url(args.base_url, cfg),
        "system_prompt": args.system if args.system is not None else cfg.get("system_prompt", ""),
        "temperature": args.temperature if args.temperature is not None else cfg.get("temperature", 1.0),
        "max_tokens": args.max_tokens if args.max_tokens is not None else cfg.get("max_tokens"),
        "stream": stream,
        "reasoning": reasoning,
        "timeout": args.timeout if args.timeout is not None else cfg.get("timeout", 120),
        "max_retries": args.max_retries if args.max_retries is not None else cfg.get("max_retries", 3),
        "history_limit": args.history_limit if args.history_limit is not None else cfg.get("history_limit", 40),
        "pricing": cfg.get("pricing") or {},
        "no_save": bool(args.no_save),
    }
