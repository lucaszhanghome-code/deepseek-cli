"""Conversation state and on-disk persistence.

Every conversation is a :class:`Session`. Sessions are JSON documents under
``<config dir>/sessions/<id>.json`` so they stay inspectable and greppable.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config

TITLE_LIMIT = 60


def _new_id() -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:4]}"


@dataclass
class Session:
    """A single conversation with the model."""

    id: str = field(default_factory=_new_id)
    model: str = config.DEFAULT_MODEL
    system_prompt: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    title: str = ""

    def __post_init__(self) -> None:
        self.usage = self._blank_usage() | dict(self.usage or {})

    @staticmethod
    def _blank_usage() -> Dict[str, Any]:
        return {
            "turns": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "cost_usd": 0.0,
        }

    # ----------------------------------------------------------- mutations
    def add(self, role: str, content: str, **extra: Any) -> Dict[str, Any]:
        """Append a message and refresh metadata."""
        message: Dict[str, Any] = {"role": role, "content": content}
        message.update({k: v for k, v in extra.items() if v is not None})
        self.messages.append(message)
        if role == "user" and not self.title:
            self.title = content.strip().replace("\n", " ")[:TITLE_LIMIT]
        self.updated_at = time.time()
        return message

    def record_usage(self, usage: Optional[Dict[str, Any]], cost: Optional[float] = None) -> None:
        """Accumulate token accounting for one completed turn."""
        self.usage["turns"] = int(self.usage.get("turns", 0)) + 1
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
        ):
            self.usage[key] = int(self.usage.get(key, 0)) + int((usage or {}).get(key, 0) or 0)
        if cost:
            self.usage["cost_usd"] = round(float(self.usage.get("cost_usd", 0.0)) + float(cost), 6)
        self.updated_at = time.time()

    def drop_last_exchange(self) -> bool:
        """Remove the most recent user/assistant pair. Returns ``True`` on success."""
        while self.messages and self.messages[-1]["role"] != "user":
            self.messages.pop()
        if self.messages:
            self.messages.pop()
            self.updated_at = time.time()
            return True
        return False

    def context(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Build the message list to send, newest turns first when trimming."""
        history = [m for m in self.messages if m.get("role") in ("user", "assistant")]
        if limit and limit > 0 and len(history) > limit:
            history = history[-limit:]
        payload: List[Dict[str, str]] = []
        if self.system_prompt:
            payload.append({"role": "system", "content": self.system_prompt})
        payload.extend({"role": m["role"], "content": m.get("content", "")} for m in history)
        return payload

    # ------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "usage": self.usage,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            id=str(data.get("id") or _new_id()),
            model=str(data.get("model") or config.DEFAULT_MODEL),
            system_prompt=str(data.get("system_prompt") or ""),
            messages=list(data.get("messages") or []),
            usage=dict(data.get("usage") or {}),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            title=str(data.get("title") or ""),
        )


class SessionStore:
    """Reads and writes :class:`Session` JSON files."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.directory = Path(directory) if directory is not None else config.sessions_dir()

    def path_for(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.json"

    def save(self, session: Session) -> Path:
        path = self.path_for(session.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(session.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    def load(self, session_id: str) -> Session:
        path = self.path_for(session_id)
        if not path.exists():
            raise FileNotFoundError(f"no saved session with id {session_id!r} in {self.directory}")
        return Session.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def delete(self, session_id: str) -> bool:
        path = self.path_for(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> List[Session]:
        """Return every saved session, most recently updated first."""
        if not self.directory.exists():
            return []
        sessions: List[Session] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                sessions.append(Session.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                continue
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def latest(self) -> Optional[Session]:
        sessions = self.list_sessions()
        return sessions[0] if sessions else None
