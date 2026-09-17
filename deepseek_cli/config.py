"""Configuration and credential resolution for deepseek-cli.

Values are resolved with the following precedence (highest first):

1. Command line flags
2. Environment variables (``DEEPSEEK_API_KEY``, ``DEEPSEEK_BASE_URL``, ...)
3. ``config.json`` in the config directory
4. Built-in defaults

The config directory is ``~/.deepseek-cli`` unless the ``DEEPSEEK_CLI_HOME``
environment variable points somewhere else.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

APP_NAME = "deepseek-cli"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
KEY_ENV_VARS = ("DEEPSEEK_API_KEY", "DEEPSEEK_KEY")

DEFAULT_CONFIG: Dict[str, Any] = {
    "api_key": "",
    "model": DEFAULT_MODEL,
    "base_url": DEFAULT_BASE_URL,
    "system_prompt": "",
    "temperature": 1.0,
    "max_tokens": 4096,
    "stream": True,
    "timeout": 120,
    "max_retries": 3,
    "history_limit": 40,
    "pricing": {},
}


class ConfigError(Exception):
    """Raised when the configuration cannot be loaded or is invalid."""


def config_dir() -> Path:
    """Return the directory that holds config and saved sessions."""
    override = os.environ.get("DEEPSEEK_CLI_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".deepseek-cli"


def config_path() -> Path:
    """Return the path of ``config.json``."""
    return config_dir() / "config.json"


def sessions_dir() -> Path:
    """Return the directory used to persist conversations."""
    return config_dir() / "sessions"


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load ``config.json`` merged over the built-in defaults.

    A missing file is not an error; a malformed file is.
    """
    target = Path(path) if path is not None else config_path()
    config = dict(DEFAULT_CONFIG)
    if not target.exists():
        return config

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"could not read {target}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{target} must contain a JSON object")

    unknown = sorted(set(raw) - set(DEFAULT_CONFIG))
    if unknown:
        raise ConfigError(
            f"{target} has unknown key(s): {', '.join(unknown)}. "
            f"Supported keys: {', '.join(sorted(DEFAULT_CONFIG))}"
        )

    config.update(raw)
    return config


def _restrict(path: Path) -> None:
    """Best-effort tightening of a file's permissions to owner-only."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def save_config(config: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Write ``config`` to disk, creating parents as needed."""
    target = Path(path) if path is not None else config_path()
    merged = dict(DEFAULT_CONFIG)
    merged.update(config)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not write {target}: {exc}") from exc
    if merged.get("api_key"):
        _restrict(target)
    return target


def save_api_key(key: str, path: Optional[Path] = None) -> Path:
    """Store ``key`` in ``config.json`` so later runs need no setup."""
    cleaned = (key or "").strip()
    if not cleaned:
        raise ConfigError("the API key cannot be empty")
    config = load_config(path)
    config["api_key"] = cleaned
    return save_config(config, path)


def mask_key(key: Optional[str]) -> str:
    """Return a display-safe hint for a key; never the whole secret."""
    if not key:
        return "(none)"
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"


def resolve_api_key(
    cli_value: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    environ: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Return the API key from the first source that provides one."""
    if cli_value and cli_value.strip():
        return cli_value.strip()

    env = os.environ if environ is None else environ
    for name in KEY_ENV_VARS:
        value = env.get(name)
        if value and value.strip():
            return value.strip()

    if config:
        value = config.get("api_key")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def api_key_source(
    cli_value: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    environ: Optional[Dict[str, str]] = None,
) -> str:
    """Return where the effective API key comes from (for display)."""
    if cli_value and cli_value.strip():
        return "flag"
    env = os.environ if environ is None else environ
    for name in KEY_ENV_VARS:
        value = env.get(name)
        if value and value.strip():
            return f"${name}"
    if config:
        value = config.get("api_key")
        if isinstance(value, str) and value.strip():
            return "config file"
    return "none"


def resolve_base_url(
    cli_value: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the API base URL, normalised without a trailing slash."""
    url = cli_value or os.environ.get("DEEPSEEK_BASE_URL") or None
    if not url and config:
        url = config.get("base_url")
    if not url:
        url = DEFAULT_BASE_URL
    return str(url).rstrip("/")


def require_api_key(
    cli_value: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Like :func:`resolve_api_key` but raises when nothing is found."""
    key = resolve_api_key(cli_value, config)
    if not key:
        raise ConfigError(missing_key_message())
    return key


def missing_key_message() -> str:
    """Return the guidance shown when no API key could be found."""
    return (
        "no API key found.\n"
        "Paste your own key when prompted, or provide one of:\n"
        "  * the DEEPSEEK_API_KEY environment variable\n"
        "  * --api-key on the command line\n"
        f"  * an \"api_key\" entry in {config_path()}\n"
        "Get a key at https://platform.deepseek.com/api_keys"
    )
