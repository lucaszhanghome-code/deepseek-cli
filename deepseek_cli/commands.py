"""Slash commands for the interactive loop.

``CommandMixin`` is mixed into :class:`deepseek_cli.chat.Chat`; it relies on the
plumbing that class provides (``self.client``, ``self.session``, ``self.out``,
``self.settings`` and friends) and only implements command behaviour.
"""

from __future__ import annotations

from typing import Dict, List

from . import config as config_mod, onboard, render
from .client import DeepSeekError
from .session import Session

COMMANDS: "Dict[str, Dict[str, str]]" = {
    "help": {"args": "", "help": "list these commands"},
    "quit": {"args": "", "help": "save and exit (also /exit)"},
    "new": {"args": "", "help": "start a fresh conversation"},
    "clear": {"args": "", "help": "forget the messages in this conversation"},
    "retry": {"args": "", "help": "resend the previous prompt"},
    "undo": {"args": "", "help": "remove the last prompt and reply"},
    "history": {"args": "[n]", "help": "show the last n prompts (default 10)"},
    "usage": {"args": "", "help": "token totals for this conversation"},
    "model": {"args": "[id]", "help": "show or switch the model"},
    "models": {"args": "", "help": "list models offered by the API"},
    "system": {"args": "[text|-]", "help": "show, set or clear the system prompt"},
    "temp": {"args": "[0-2]", "help": "show or set the temperature"},
    "maxtokens": {"args": "[n]", "help": "show or set max output tokens"},
    "stream": {"args": "on|off", "help": "toggle streaming"},
    "reasoning": {"args": "on|off", "help": "toggle reasoning output"},
    "config": {"args": "", "help": "show effective settings and config paths"},
    "key": {"args": "[key]", "help": "show, replace and save the API key"},
    "balance": {"args": "", "help": "show the account balance"},
    "save": {"args": "", "help": "save this conversation to disk"},
    "sessions": {"args": "", "help": "list saved sessions"},
    "load": {"args": "ID|last", "help": "resume a saved session"},
    "delete": {"args": "ID", "help": "delete a saved session"},
}

ALIASES = {
    "exit": "quit",
    "q": "quit",
    "h": "help",
    "?": "help",
    "reset": "clear",
    "last": "load",
    "tokens": "usage",
}

TRUE_VALUES = {"on", "true", "yes", "1", "enable", "enabled"}
FALSE_VALUES = {"off", "false", "no", "0", "disable", "disabled"}


class CommandMixin:
    """Implements every ``/command``."""

    def handle_command(self, line: str) -> bool:
        """Run a command line. Returns ``False`` when the loop should stop."""
        body = line[1:].strip()
        if not body:
            return True
        name, _, rest = body.partition(" ")
        name = ALIASES.get(name.lower(), name.lower())
        handler = getattr(self, f"cmd_{name}", None)
        if handler is None:
            self.error(f"unknown command /{name} - try /help")
            return True
        result = handler(rest.strip())
        return result is not False

    # ------------------------------------------------------------------ basics
    def cmd_help(self, rest: str) -> None:
        rows = [[f"/{name} {spec['args']}".strip(), spec["help"]] for name, spec in COMMANDS.items()]
        rows.append(["/exit", "alias for /quit"])
        render.print_table(rows, ["command", "description"], self.out)
        self.out.write(
            self.style(
                "\nTrailing \\ continues a line. Anything not starting with / is sent to the model.\n",
                "dim",
            )
        )

    def cmd_quit(self, rest: str) -> bool:
        self.save()
        if self.interactive:
            self.out.write(self.style("bye\n", "dim"))
        return False

    def cmd_new(self, rest: str) -> None:
        self.save()
        self.session = Session(model=self.model, system_prompt=self.session.system_prompt)
        self.out.write(f"started session {self.style(self.session.id, 'cyan')}\n")

    def cmd_clear(self, rest: str) -> None:
        self.session.messages = []
        self.session.title = ""
        self.session.usage = Session._blank_usage()
        self.save()
        self.out.write("conversation cleared\n")

    def cmd_retry(self, rest: str) -> None:
        if not self.last_prompt:
            self.out.write("nothing to retry yet\n")
            return
        self.out.write(self.style(f"resending: {self.last_prompt[:80]}\n", "dim"))
        self.send(self.last_prompt, display=False)

    def cmd_undo(self, rest: str) -> None:
        if self.session.drop_last_exchange():
            self.save()
            self.out.write("removed the last exchange\n")
        else:
            self.out.write("nothing to undo\n")

    def cmd_history(self, rest: str) -> None:
        limit = 10
        if rest:
            try:
                limit = max(1, int(rest))
            except ValueError:
                self.error(f"expected a number, got {rest!r}")
                return
        prompts = [m["content"] for m in self.session.messages if m.get("role") == "user"]
        if not prompts:
            self.out.write("no prompts yet\n")
            return
        for index, prompt in enumerate(prompts[-limit:], start=max(1, len(prompts) - limit + 1)):
            preview = prompt.replace("\n", " ")[:100]
            self.out.write(f"{self.style(str(index).rjust(3), 'dim')}  {preview}\n")

    # ----------------------------------------------------------------- numbers
    def cmd_usage(self, rest: str) -> None:
        usage = self.session.usage or {}
        rows = [[key, str(value)] for key, value in usage.items()]
        render.print_table(rows, ["metric", "value"], self.out)
        pricing = self.settings.get("pricing") or {}
        if not pricing:
            self.out.write(
                self.style(
                    "\ncost estimation is off - add per-model rates under \"pricing\" in "
                    f"{config_mod.config_path()}\n",
                    "dim",
                )
            )

    def cmd_model(self, rest: str) -> None:
        if not rest:
            self.out.write(f"model: {self.style(self.model, 'bold')}\n")
            return
        self.model = rest
        self.session.model = rest
        self.save()
        self.out.write(f"model set to {self.style(rest, 'bold')}\n")

    def cmd_models(self, rest: str) -> None:
        try:
            models = self.client.list_models()
        except DeepSeekError as exc:
            self.error(str(exc))
            return
        for model in models:
            marker = "*" if model == self.model else " "
            style = "bold" if marker == "*" else "dim"
            self.out.write(f" {marker} {self.style(model, style)}\n")

    def cmd_system(self, rest: str) -> None:
        if not rest:
            current = self.session.system_prompt
            self.out.write((current if current else "(none set)") + "\n")
            return
        if rest in ("-", "clear", "none"):
            self.session.system_prompt = ""
            self.out.write("system prompt cleared\n")
        else:
            self.session.system_prompt = rest
            self.out.write(f"system prompt set to {self.style(rest[:80], 'dim')}\n")
        self.save()

    def cmd_temp(self, rest: str) -> None:
        if not rest:
            self.out.write(f"temperature: {self.settings.get('temperature')}\n")
            return
        try:
            value = float(rest)
        except ValueError:
            self.error(f"expected a number, got {rest!r}")
            return
        if not 0.0 <= value <= 2.0:
            self.error("temperature must be between 0 and 2")
            return
        self.settings["temperature"] = value
        self.out.write(f"temperature set to {value}\n")

    def cmd_maxtokens(self, rest: str) -> None:
        if not rest:
            self.out.write(f"max tokens: {self.settings.get('max_tokens')}\n")
            return
        try:
            value = int(rest)
        except ValueError:
            self.error(f"expected an integer, got {rest!r}")
            return
        if value <= 0:
            self.error("max tokens must be positive")
            return
        self.settings["max_tokens"] = value
        self.out.write(f"max tokens set to {value}\n")

    # ----------------------------------------------------------------- toggles
    def cmd_stream(self, rest: str) -> None:
        if not rest:
            self.out.write(f"streaming: {'on' if self.settings.get('stream') else 'off'}\n")
            return
        if rest.lower() in TRUE_VALUES:
            self.settings["stream"] = True
        elif rest.lower() in FALSE_VALUES:
            self.settings["stream"] = False
        else:
            self.error(f"expected on or off, got {rest!r}")
            return
        self.out.write(f"streaming {'on' if self.settings['stream'] else 'off'}\n")

    def cmd_reasoning(self, rest: str) -> None:
        if not rest:
            self.out.write(f"reasoning output: {'on' if self.show_reasoning else 'off'}\n")
            return
        if rest.lower() in TRUE_VALUES:
            self.show_reasoning = True
        elif rest.lower() in FALSE_VALUES:
            self.show_reasoning = False
        else:
            self.error(f"expected on or off, got {rest!r}")
            return
        self.out.write(f"reasoning output {'on' if self.show_reasoning else 'off'}\n")

    # ------------------------------------------------------------------- state
    def cmd_config(self, rest: str) -> None:
        config = config_mod.load_config()
        resolved = config_mod.resolve_api_key(config=config)
        key = getattr(self.client, "api_key", "") or resolved
        source = config_mod.api_key_source(config=config) if resolved else "this session"
        rows: List[List[str]] = [
            ["model", str(self.model)],
            ["temperature", str(self.settings.get("temperature"))],
            ["max_tokens", str(self.settings.get("max_tokens"))],
            ["stream", str(self.settings.get("stream"))],
            ["history_limit", str(self.settings.get("history_limit"))],
            ["timeout", str(self.settings.get("timeout"))],
            ["max_retries", str(self.settings.get("max_retries"))],
            ["system_prompt", self.session.system_prompt or "(none)"],
            ["pricing", "configured" if self.settings.get("pricing") else "(none)"],
            ["api key", f"{config_mod.mask_key(key)} (from {source})" if key else "MISSING"],
        ]
        render.print_table(rows, ["setting", "value"], self.out)
        self.out.write(self.style(f"\nconfig file : {config_mod.config_path()}\n", "dim"))
        self.out.write(self.style(f"sessions dir: {config_mod.sessions_dir()}\n", "dim"))

    def cmd_key(self, rest: str) -> None:
        """Show, replace and store the API key."""
        current = getattr(self.client, "api_key", "") or ""
        supplied = rest.strip()
        if not supplied:
            self.out.write(f"current key: {config_mod.mask_key(current)}\n")
            if not onboard.can_prompt():
                self.err.write(
                    "usage: /key <api-key>  (paste one from "
                    f"{onboard.KEY_URL}, or run without a key to be asked)\n"
                )
                return
            supplied = (onboard.read_key(self.err, prompt="new API key: ") or "").strip()
            if not supplied:
                self.out.write("cancelled\n")
                return

        problem = onboard.verify_key(
            supplied,
            base_url=getattr(self.client, "base_url", config_mod.DEFAULT_BASE_URL),
            timeout=int(self.settings.get("timeout") or 30),
        )
        if problem:
            self.error(problem)
            return
        try:
            path = config_mod.save_api_key(supplied)
        except config_mod.ConfigError as exc:
            self.error(str(exc))
            return
        self.client.api_key = supplied
        self.out.write(
            f"API key accepted and saved ({config_mod.mask_key(supplied)}) to {path}\n"
        )

    def cmd_balance(self, rest: str) -> None:
        try:
            data = self.client.balance()
        except DeepSeekError as exc:
            self.error(str(exc))
            return
        infos = data.get("balance_infos") if isinstance(data, dict) else None
        if isinstance(infos, list) and infos:
            rows = [
                [str(info.get("currency", "")), str(info.get("total_balance", "")),
                 str(info.get("granted_balance", "")), str(info.get("topped_up_balance", ""))]
                for info in infos
                if isinstance(info, dict)
            ]
            render.print_table(rows, ["currency", "total", "granted", "topped up"], self.out)
        else:
            self.out.write(str(data) + "\n")

    # ---------------------------------------------------------------- sessions
    def cmd_save(self, rest: str) -> None:
        if self.no_save:
            self.out.write("session saving is disabled for this run (--no-save)\n")
            return
        try:
            path = self.store.save(self.session)
        except OSError as exc:
            self.error(str(exc))
            return
        self.out.write(f"saved to {self.style(str(path), 'dim')}\n")

    def cmd_sessions(self, rest: str) -> None:
        sessions = self.store.list_sessions()
        if not sessions:
            self.out.write("no saved sessions\n")
            return
        rows = []
        for session in sessions:
            marker = "*" if session.id == self.session.id else " "
            rows.append([
                marker,
                session.id,
                str(len([m for m in session.messages if m.get("role") == "user"])),
                session.model,
                (session.title or "(untitled)")[:48],
            ])
        render.print_table(rows, ["", "id", "turns", "model", "title"], self.out)

    def cmd_load(self, rest: str) -> None:
        target = rest or "last"
        if target in ("last", "latest"):
            session = self.store.latest()
            if session is None:
                self.out.write("no saved sessions\n")
                return
        else:
            try:
                session = self.store.load(target)
            except (OSError, ValueError) as exc:
                self.error(str(exc))
                return
        self.session = session
        if session.model:
            self.model = session.model
        self.out.write(
            f"resumed {self.style(session.id, 'cyan')} "
            f"({len(session.messages)} messages, model {session.model})\n"
        )

    def cmd_delete(self, rest: str) -> None:
        if not rest:
            self.error("usage: /delete ID")
            return
        try:
            removed = self.store.delete(rest)
        except OSError as exc:
            self.error(str(exc))
            return
        self.out.write("deleted\n" if removed else f"no session with id {rest!r}\n")


__all__ = ["CommandMixin", "COMMANDS", "ALIASES"]
