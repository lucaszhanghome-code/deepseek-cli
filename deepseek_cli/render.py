"""Terminal output helpers: colours, spinner, streaming writer, tables.

Everything degrades gracefully: on a non-TTY (or with ``NO_COLOR`` set) all
styling is dropped so output stays pipeable.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, TextIO

RESET = "\x1b[0m"
CODES = {
    "bold": "1",
    "dim": "2",
    "italic": "3",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "grey": "90",
}


def _enable_windows_ansi() -> bool:
    """Turn on virtual terminal processing so ANSI codes work on Windows."""
    if os.name != "nt":  # pragma: no cover - non-Windows
        return True
    try:  # pragma: no cover - Windows only
        import ctypes

        kernel32 = ctypes.windll.kernel32
        for handle_id in (-11, -12):  # stdout, stderr
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                continue
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        return True
    except Exception:  # pragma: no cover - best effort only
        return False


def supports_ansi(stream: TextIO) -> bool:
    """Best-effort check for colour support on ``stream``."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    if os.name == "nt":
        return _enable_windows_ansi()
    return True


class Style:
    """Applies ANSI codes when colour is enabled."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, text: str, *names: str) -> str:
        if not self.enabled or not names:
            return text
        codes = ";".join(CODES[name] for name in names if name in CODES)
        if not codes:
            return text
        return f"\x1b[{codes}m{text}{RESET}"


class Spinner:
    """A tiny status spinner that writes to stderr so stdout stays clean."""

    FRAMES = ("|", "/", "-", "\\")

    def __init__(self, message: str = "thinking", stream: Optional[TextIO] = None, enabled: bool = True):
        self.message = message
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "Spinner":
        if not self.enabled:
            return self
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self) -> None:  # pragma: no cover - timing dependent
        index = 0
        while not self._stop.is_set():
            frame = self.FRAMES[index % len(self.FRAMES)]
            try:
                self.stream.write(f"\r\x1b[2m{frame} {self.message}\x1b[0m")
                self.stream.flush()
            except (OSError, ValueError):
                return
            index += 1
            self._stop.wait(0.1)

    def stop(self) -> "Spinner":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        if self.enabled:
            try:
                self.stream.write("\r" + " " * (len(self.message) + 4) + "\r")
                self.stream.flush()
            except (OSError, ValueError):
                pass
        return self

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()


class StreamWriter:
    """Writes streamed content, keeping reasoning output visually distinct."""

    def __init__(self, stream: Optional[TextIO] = None, style: Optional[Style] = None, show_reasoning: bool = True):
        self.stream = stream if stream is not None else sys.stdout
        self.style = style or Style(False)
        self.show_reasoning = show_reasoning
        self._reasoning_open = False
        self.content_chars = 0
        self.reasoning_chars = 0
        self.first_token_at: Optional[float] = None
        self.started_at = time.time()

    def _mark_first_token(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = time.time()

    def write_reasoning(self, text: str) -> None:
        if not self.show_reasoning or not text:
            return
        self._mark_first_token()
        if not self._reasoning_open:
            self.stream.write(self.style("\n[reasoning] ", "dim", "italic"))
            self._reasoning_open = True
        self.stream.write(self.style(text, "dim"))
        self.stream.flush()
        self.reasoning_chars += len(text)

    def write_content(self, text: str) -> None:
        if not text:
            return
        self._mark_first_token()
        if self._reasoning_open:
            self.stream.write("\n\n")
            self._reasoning_open = False
        self.stream.write(text)
        self.stream.flush()
        self.content_chars += len(text)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    @property
    def time_to_first_token(self) -> Optional[float]:
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.started_at


def format_usage(
    usage: Optional[Dict[str, Any]],
    *,
    model: str = "",
    elapsed: Optional[float] = None,
    cost: Optional[float] = None,
    style: Optional[Style] = None,
) -> str:
    """Render a one-line token/timing summary."""
    paint = style or Style(False)
    usage = usage or {}
    parts: List[str] = []
    if model:
        parts.append(model)
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is not None and completion is not None:
        parts.append(f"{prompt} in / {completion} out")
        hit = usage.get("prompt_cache_hit_tokens")
        miss = usage.get("prompt_cache_miss_tokens")
        if hit is not None or miss is not None:
            parts.append(f"cache {hit or 0} hit / {miss or 0} miss")
    if cost is not None:
        parts.append(f"~${cost:.5f}")
    if elapsed is not None:
        parts.append(f"{elapsed:.1f}s")
    return paint("[" + " - ".join(parts) + "]", "dim") if parts else ""


def print_table(rows: Sequence[Sequence[str]], headers: Sequence[str], stream: Optional[TextIO] = None) -> None:
    """Print a simple left-aligned table."""
    out = stream if stream is not None else sys.stdout
    table = [list(headers)] + [list(row) for row in rows]
    widths = [max(len(str(row[i])) for row in table) for i in range(len(headers))]
    for index, row in enumerate(table):
        line = "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        out.write(line + "\n")
        if index == 0:
            out.write("  ".join("-" * width for width in widths) + "\n")


def terminal_width(default: int = 80) -> int:
    """Return the usable terminal width."""
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except (ValueError, OSError):  # pragma: no cover - exotic terminals
        return default


def print_markdown(text: str, stream: Optional[TextIO] = None, style: Optional[Style] = None) -> None:
    """Print text with light Markdown emphasis for headings, lists and code."""
    out = stream if stream is not None else sys.stdout
    paint = style or Style(False)
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.write(paint(line, "dim") + "\n" if not in_fence else paint(line, "dim") + "\n")
            continue
        if in_fence:
            out.write(paint("  " + line, "grey") + "\n")
        elif line.startswith("#"):
            out.write(paint(line.lstrip("# ").upper(), "bold", "cyan") + "\n")
        elif line.lstrip().startswith(("- ", "* ")):
            out.write(paint("  * ", "cyan") + line.lstrip()[2:] + "\n")
        else:
            out.write(line + "\n")


__all__ = [
    "Style",
    "Spinner",
    "StreamWriter",
    "format_usage",
    "print_table",
    "print_markdown",
    "supports_ansi",
    "terminal_width",
]
