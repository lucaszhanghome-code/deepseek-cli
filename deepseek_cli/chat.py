"""The interactive chat loop: input handling, streaming, persistence."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, TextIO

from . import __version__, render
from .client import (
    APIError,
    AuthError,
    DeepSeekClient,
    DeepSeekError,
    NetworkError,
    estimate_cost,
)
from .commands import CommandMixin
from .session import Session, SessionStore

BANNER = r"""
   __                 __             __     ______ __
  / /____  ___  ___  / /_______  ___ / /__  / ___// /   /  _/
 / __/ _ \/ _ \/ _ \/ __/ ___/ _ \/ _ `/ _ \/ /__ / /    / /
 \__/\___/ .__/\___/\__/\__/  .__/\_,_/\___/\___//_/   /_/
        /_/                /_/
"""


class Chat(CommandMixin):
    """Drives one conversation end to end."""

    def __init__(
        self,
        client: DeepSeekClient,
        settings: Dict[str, Any],
        session: Optional[Session] = None,
        store: Optional[SessionStore] = None,
        style: Optional[render.Style] = None,
        out: Optional[TextIO] = None,
        err: Optional[TextIO] = None,
        interactive: bool = True,
    ) -> None:
        self.client = client
        self.settings = settings
        self.style = style or render.Style(False)
        self.out = out if out is not None else sys.stdout
        self.err = err if err is not None else sys.stderr
        self.interactive = interactive
        self.store = store if store is not None else SessionStore()
        self.no_save = bool(settings.get("no_save"))

        self.model: str = settings.get("model", "deepseek-chat")
        if session is None:
            session = Session(model=self.model, system_prompt=str(settings.get("system_prompt") or ""))
        else:
            session.model = self.model
            if settings.get("system_prompt"):
                session.system_prompt = str(settings["system_prompt"])
        self.session: Session = session

        self.last_prompt: Optional[str] = None
        self.history: List[str] = []
        self.show_reasoning = bool(settings.get("reasoning", True))

    # ------------------------------------------------------------ persistence
    def save(self) -> None:
        if self.no_save:
            return
        try:
            self.store.save(self.session)
        except OSError as exc:
            self.err.write(f"warning: could not save session: {exc}\n")

    # --------------------------------------------------------------- display
    def print_banner(self) -> None:
        self.out.write(self.style(BANNER, "cyan"))
        resumed = f" (resumed, {len(self.session.messages)} messages)" if self.session.messages else ""
        self.out.write(
            f"  deepseek-cli {__version__} - model {self.style(self.model, 'bold')}{resumed}\n"
        )
        self.out.write(self.style("  /help for commands, /exit to quit\n\n", "dim"))
        if self.session.messages:
            summary = self.session.messages[-1]
            preview = str(summary.get("content", ""))[:120].replace("\n", " ")
            self.out.write(self.style(f"  last {summary.get('role')}: {preview}\n\n", "dim"))

    def footer(self, usage: Optional[Dict[str, Any]], elapsed: float, cost: Optional[float]) -> None:
        if self.settings.get("quiet") or not self.interactive:
            return
        line = render.format_usage(usage, model=self.model, elapsed=elapsed, cost=cost, style=self.style)
        if line:
            self.out.write("\n" + line + "\n")

    def error(self, message: str) -> None:
        self.err.write(self.style(f"error: {message}", "red") + "\n")

    # ------------------------------------------------------------------ turns
    def send(self, prompt: str, *, display: bool = True) -> Dict[str, Any]:
        """Send ``prompt``, stream the reply and record the exchange.

        Returns the assistant result dict (``content``, ``usage``, ``cost``...).
        """
        self.last_prompt = prompt
        if display:
            self.history.append(prompt)
        self.session.add("user", prompt)

        messages = self.session.context(self.settings.get("history_limit"))
        options: Dict[str, Any] = {"max_tokens": self.settings.get("max_tokens")}
        temperature = self.settings.get("temperature")
        if temperature is not None:
            options["temperature"] = temperature

        streaming = bool(self.settings.get("stream", True))
        writer = render.StreamWriter(self.out, self.style, show_reasoning=self.show_reasoning)
        spinner: Optional[render.Spinner] = None
        if self.interactive and render.supports_ansi(self.err):
            spinner = render.Spinner("thinking", self.err).start()

        usage: Dict[str, Any] = {}
        finish_reason: Optional[str] = None
        interrupted = False
        parts: List[str] = []
        reasoning_parts: List[str] = []
        prefix_written = False

        def begin_reply() -> None:
            """Stop the spinner and print the assistant prefix exactly once."""
            nonlocal spinner, prefix_written
            if spinner is not None:
                spinner.stop()
                spinner = None
            if display and not prefix_written:
                prefix_written = True
                self.out.write(self.style("deepseek> ", "green"))

        try:
            for event in self.client.chat(messages, model=self.model, stream=streaming, **options):
                kind = event.get("type")
                if kind == "content":
                    text = event.get("text", "")
                    if text:
                        begin_reply()
                        parts.append(text)
                        writer.write_content(text)
                elif kind == "reasoning":
                    text = event.get("text", "")
                    if text:
                        begin_reply()
                        reasoning_parts.append(text)
                        writer.write_reasoning(text)
                elif kind == "usage":
                    usage = event.get("usage") or {}
                elif kind == "done":
                    finish_reason = event.get("finish_reason")
        except KeyboardInterrupt:
            interrupted = True
            self.err.write("\n" + self.style("[interrupted]", "yellow") + "\n")
        except AuthError as exc:
            if spinner:
                spinner.stop()
            self.session.drop_last_exchange()
            self.error(f"{exc}")
            self.err.write(self.style("  check DEEPSEEK_API_KEY or --api-key\n", "dim"))
            return {"content": "", "usage": {}, "error": str(exc)}
        except DeepSeekError as exc:
            if spinner:
                spinner.stop()
            self.session.drop_last_exchange()
            self.error(str(exc))
            if isinstance(exc, (NetworkError, APIError)):
                self.err.write(self.style("  the prompt was discarded - use /retry to try again\n", "dim"))
            return {"content": "", "usage": {}, "error": str(exc)}
        finally:
            if spinner is not None:
                spinner.stop()

        content = "".join(parts)
        cost = estimate_cost(usage, self.settings.get("pricing") or {}, self.model)
        if usage:
            self.session.record_usage(usage, cost)

        if content:
            self.session.add("assistant", content, interrupted=True if interrupted else None)
        if display:
            if not parts:
                begin_reply()
            self.out.write("\n")
        if interrupted and not content:
            self.out.write(self.style("[no output]\n", "dim"))
        self.footer(usage, writer.elapsed, cost)
        self.save()
        return {
            "content": content,
            "reasoning_content": "".join(reasoning_parts),
            "usage": usage,
            "cost_usd": cost,
            "finish_reason": finish_reason,
            "elapsed": writer.elapsed,
            "interrupted": interrupted,
        }

    # -------------------------------------------------------------- one-shot
    def once(self, prompt: str) -> Dict[str, Any]:
        """Non-interactive variant: no banner, no spinner, no footer."""
        self.interactive = False
        result = self.send(prompt, display=False)
        content = result.get("content") or ""
        if content and not content.endswith("\n"):
            self.out.write("\n")
        return result

    # ------------------------------------------------------------- main loop
    def run(self) -> int:
        self.print_banner()
        if not self.session.messages and self.session.title:
            self.session.title = ""
        while True:
            try:
                line = self.read_input()
            except EOFError:
                self.out.write("\n")
                return 0
            except KeyboardInterrupt:
                self.err.write("\n" + self.style("(interrupted - /exit or Ctrl+D quits)", "dim") + "\n")
                continue
            if line is None:
                continue
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if not self.handle_command(line):
                    return 0
            else:
                self.send(line)

    def read_input(self) -> Optional[str]:
        """Read one line, honouring a trailing backslash for continuations."""
        prompt = self.style("you> ", "bold", "cyan")
        continuation = self.style("  ..> ", "dim")
        try:
            first = input(prompt)
        except EOFError:
            raise
        lines = [first]
        while lines[-1].rstrip().endswith("\\"):
            lines[-1] = lines[-1].rstrip()[:-1]
            try:
                lines.append(input(continuation))
            except EOFError:
                break
        return "\n".join(lines)


__all__ = ["Chat", "BANNER"]
