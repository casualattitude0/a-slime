"""JSON-backed registry for named conversation chats.

Each chat entry maps to a session_id (== chat_id) in the history store.
The registry only tracks metadata (title, timestamps). Message content
lives in Chroma via history_store.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatEntry:
    __slots__ = ("chat_id", "version_id", "title", "created_at", "updated_at", "metadata")

    def __init__(
        self,
        chat_id: str,
        version_id: str,
        title: str,
        created_at: str,
        updated_at: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.chat_id = chat_id
        self.version_id = version_id
        self.title = title
        self.created_at = created_at
        self.updated_at = updated_at
        self.metadata = dict(metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "version_id": self.version_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChatEntry":
        return cls(
            chat_id=d["chat_id"],
            version_id=d.get("version_id", ""),
            title=d.get("title", "New Chat"),
            created_at=d["created_at"],
            updated_at=d.get("updated_at", d["created_at"]),
            metadata=d.get("metadata") if isinstance(d.get("metadata"), dict) else {},
        )


class ChatRegistry:
    def __init__(self, persist_path: Path) -> None:
        self._path = persist_path
        self._lock = Lock()
        self._chats: dict[str, ChatEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for c in data.get("chats", []):
                entry = ChatEntry.from_dict(c)
                self._chats[entry.chat_id] = entry
        except Exception:
            pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"chats": [c.to_dict() for c in self._chats.values()]}
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def create(
        self,
        version_id: str,
        title: str = "New Chat",
        chat_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatEntry:
        with self._lock:
            cid = chat_id or str(uuid.uuid4())
            now = _now_iso()
            entry = ChatEntry(
                chat_id=cid,
                version_id=version_id,
                title=title,
                created_at=now,
                updated_at=now,
                metadata=metadata,
            )
            self._chats[cid] = entry
            self._save()
            return entry

    def get(self, chat_id: str) -> ChatEntry | None:
        with self._lock:
            return self._chats.get(chat_id)

    def list_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(
                [c.to_dict() for c in self._chats.values()],
                key=lambda c: c["updated_at"],
                reverse=True,
            )

    def rename(self, chat_id: str, title: str) -> bool:
        with self._lock:
            entry = self._chats.get(chat_id)
            if entry is None:
                return False
            entry.title = title.strip() or "New Chat"
            entry.updated_at = _now_iso()
            self._save()
            return True

    def delete(self, chat_id: str) -> bool:
        with self._lock:
            if chat_id not in self._chats:
                return False
            del self._chats[chat_id]
            self._save()
            return True

    def touch(self, chat_id: str) -> None:
        """Update updated_at without changing title."""
        with self._lock:
            entry = self._chats.get(chat_id)
            if entry:
                entry.updated_at = _now_iso()
                self._save()

    def set_title_if_default(self, chat_id: str, first_message: str) -> None:
        """On first user message, replace default 'New Chat' title."""
        with self._lock:
            entry = self._chats.get(chat_id)
            if entry and entry.title == "New Chat":
                entry.title = first_message[:60].strip() or "New Chat"
                entry.updated_at = _now_iso()
                self._save()

    def set_generated_title_if_default(self, chat_id: str, title: str) -> str | None:
        """Apply generated title only when current title is still default."""
        with self._lock:
            entry = self._chats.get(chat_id)
            if entry is None or entry.title != "New Chat":
                return None
            clean = title.strip() or "New Chat"
            entry.title = clean
            entry.updated_at = _now_iso()
            self._save()
            return entry.title

    def ensure(self, chat_id: str, version_id: str) -> ChatEntry:
        """Return existing entry or create one (used for legacy session adoption)."""
        with self._lock:
            entry = self._chats.get(chat_id)
            if entry is not None:
                return entry
        return self.create(version_id=version_id, chat_id=chat_id)
