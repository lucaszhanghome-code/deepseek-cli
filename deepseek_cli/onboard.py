"""First-run onboarding: ask the user to paste their own API key.

Nothing in this package ever ships with a key baked in, so when no key can be
resolved the CLI offers to take one from the person using it, checks that the
API accepts it, and stores it in the user's own config file.
"""

from __future__ import annotations

import getpass
import os
import sys
from typing import Any, Dict, Optional, TextIO

from . import config as config_mod, render
from .client import AuthError, DeepSeekClient, DeepSeekError

NO_PROMPT_ENV = "DEEPSEEK_CLI_NO_PROMPT"
KEY_URL = "https://platform.deepseek.com/api_keys"
MAX_ATTEMPTS = 3

TRUE_VALUES = {"1", "true", "yes", "on"}


def prompting_allowed(environ: Optional[Dict[str, str]] = None) -> bool:
    """Return ``False`` when the environment forbids interactive prompts."""
    env = os.environ if environ is None else environ
    value = str(env.get(NO_PROMPT_ENV, "")).strip().lower()
    return value not in TRUE_VALUES


def _console_attached(stream: TextIO) -> bool:
    """Return ``False`` when a Windows "terminal" is really the NUL device.

    ``isatty()`` cannot tell the two apart on Windows: the NUL device is a
    character device just like the console, so ``prog < NUL`` would otherwise
    hang waiting for a key that will never arrive. Only a real console accepts
    ``GetConsoleMode``.
    """
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = ctypes.c_uint32()
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        return bool(kernel32.GetConsoleMode(ctypes.c_void_p(handle), ctypes.byref(mode)))
    except Exception:  # noqa: BLE001 - never block a legitimate prompt
        return True


def can_prompt(
    *,
    interactive: bool = True,
    environ: Optional[Dict[str, str]] = None,
    stdin: Optional[TextIO] = None,
) -> bool:
    """Return ``True`` when it is safe to ask a human to paste a key."""
    if not interactive or not prompting_allowed(environ):
        return False
    stream = sys.stdin if stdin is None else stdin
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    return _console_attached(stream)


def _writable_stream(stream: Optional[TextIO]) -> Optional[TextIO]:
    """Return ``stream`` only when it can back a real terminal."""
    if stream is None:
        return None
    try:
        stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None
    return stream


def read_key(stream: Optional[TextIO] = None, prompt: str = "API key: ") -> Optional[str]:
    """Read a key without echoing it. ``None`` when the user cancels."""
    try:
        return getpass.getpass(prompt, stream=_writable_stream(stream))
    except (EOFError, KeyboardInterrupt, OSError):
        return None


def verify_key(
    key: str,
    *,
    base_url: str,
    timeout: int = 30,
    max_retries: int = 0,
) -> Optional[str]:
    """Return ``None`` when the API accepts ``key``, else a message.

    ``GET /user/balance`` is used because it is cheap and never generates
    tokens. A network failure is not treated as a bad key: the user is likely
    offline, and the key may well be fine.
    """
    try:
        client = DeepSeekClient(key, base_url=base_url, timeout=timeout, max_retries=max_retries)
    except AuthError as exc:
        return str(exc)
    try:
        client.balance()
    except AuthError as exc:
        return f"the API rejected that key ({exc})"
    except DeepSeekError:
        return None
    return None


def ask_for_api_key(
    settings: Optional[Dict[str, Any]] = None,
    *,
    style: Optional[render.Style] = None,
    err: Optional[TextIO] = None,
    save: bool = True,
    verify: bool = True,
    attempts: int = MAX_ATTEMPTS,
) -> Optional[str]:
    """Prompt for a DeepSeek API key, check it and store it.

    Returns the accepted key, or ``None`` when the user cancels or runs out of
    attempts. Everything is written to ``err`` so that stdout stays machine
    readable.
    """
    settings = settings or {}
    style = style or render.Style(False)
    err = sys.stderr if err is None else err

    err.write(
        "No DeepSeek API key found.\n"
        f"Get one at {KEY_URL}, then paste it here (input is hidden).\n"
        + (
            f"It will be saved to {config_mod.config_path()} and sent only to the DeepSeek API.\n"
            if save
            else "It will be used for this run only.\n"
        )
    )
    err.flush()

    base_url = config_mod.resolve_base_url(settings.get("base_url"))
    timeout = int(settings.get("timeout") or 30)

    for _ in range(max(1, attempts)):
        pasted = read_key(err)
        if pasted is None:
            err.write("cancelled\n")
            return None
        key = pasted.strip()
        if not key:
            err.write("that was empty - try again\n")
            continue
        if verify:
            problem = verify_key(key, base_url=base_url, timeout=timeout)
            if problem:
                err.write(style(f"{problem}\n", "yellow"))
                continue
        where = "(not saved)"
        if save:
            try:
                where = str(config_mod.save_api_key(key))
            except config_mod.ConfigError as exc:
                err.write(style(f"could not save the key: {exc}\n", "yellow"))
                where = "(not saved)"
        err.write(style(f"API key accepted ({config_mod.mask_key(key)}) {where}\n", "green"))
        return key

    err.write("gave up after too many attempts\n")
    return None


__all__ = [
    "NO_PROMPT_ENV",
    "KEY_URL",
    "ask_for_api_key",
    "can_prompt",
    "prompting_allowed",
    "read_key",
    "verify_key",
]
