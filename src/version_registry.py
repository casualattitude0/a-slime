from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

_DEFAULT_VERSION_ID = "default"


class VersionEntry:
    __slots__ = (
        "version_id",
        "name",
        "session_id",
        "memory_collection",
        "rag_collection",
        "model_profile",
        "created_at",
    )

    def __init__(
        self,
        version_id: str,
        name: str,
        session_id: str,
        memory_collection: str,
        rag_collection: str,
        model_profile: str,
        created_at: str,
    ) -> None:
        self.version_id = version_id
        self.name = name
        self.session_id = session_id
        self.memory_collection = memory_collection
        self.rag_collection = rag_collection
        self.model_profile = model_profile
        self.created_at = created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "name": self.name,
            "session_id": self.session_id,
            "memory_collection": self.memory_collection,
            "rag_collection": self.rag_collection,
            "model_profile": self.model_profile,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VersionEntry":
        return cls(
            version_id=d["version_id"],
            name=d["name"],
            session_id=d["session_id"],
            memory_collection=d["memory_collection"],
            rag_collection=d.get("rag_collection", "langchain"),
            model_profile=d.get("model_profile", "default"),
            created_at=d["created_at"],
        )


class VersionRegistry:
    DEFAULT_RAG_COLLECTION = "langchain"
    DEFAULT_MEMORY_COLLECTION = "agent_memory"

    def __init__(self, persist_path: Path) -> None:
        self._path = persist_path
        self._lock = Lock()
        self._versions: dict[str, VersionEntry] = {}
        self._active_version_id: str = _DEFAULT_VERSION_ID
        self._load()
        if not self._versions:
            self._create_default()

    def _create_default(self) -> None:
        entry = VersionEntry(
            version_id=_DEFAULT_VERSION_ID,
            name="Default",
            session_id=str(uuid.uuid4()),
            memory_collection=self.DEFAULT_MEMORY_COLLECTION,
            rag_collection=self.DEFAULT_RAG_COLLECTION,
            model_profile="default",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._versions[_DEFAULT_VERSION_ID] = entry
        self._active_version_id = _DEFAULT_VERSION_ID
        self._save()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._active_version_id = data.get("active_version_id", _DEFAULT_VERSION_ID)
            for v in data.get("versions", []):
                entry = VersionEntry.from_dict(v)
                self._versions[entry.version_id] = entry
            if self._active_version_id not in self._versions:
                self._active_version_id = next(iter(self._versions), _DEFAULT_VERSION_ID)
        except Exception:
            pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active_version_id": self._active_version_id,
            "versions": [v.to_dict() for v in self._versions.values()],
        }
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_active(self) -> VersionEntry:
        with self._lock:
            v = self._versions.get(self._active_version_id)
            if v is None:
                v = next(iter(self._versions.values()))
            return v

    def get_active_id(self) -> str:
        with self._lock:
            return self._active_version_id

    def list_versions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {**v.to_dict(), "is_active": v.version_id == self._active_version_id}
                for v in self._versions.values()
            ]

    def switch(self, version_id: str) -> bool:
        with self._lock:
            if version_id not in self._versions:
                return False
            self._active_version_id = version_id
            self._save()
            return True

    def create(self, name: str, model_profile: str = "default") -> VersionEntry:
        with self._lock:
            vid = str(uuid.uuid4())
            entry = VersionEntry(
                version_id=vid,
                name=name,
                session_id=str(uuid.uuid4()),
                memory_collection=f"agent_memory_{vid[:8]}",
                rag_collection=self.DEFAULT_RAG_COLLECTION,
                model_profile=model_profile,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._versions[vid] = entry
            self._save()
            return entry

    def delete(self, version_id: str) -> bool:
        with self._lock:
            if version_id not in self._versions:
                return False
            if version_id == _DEFAULT_VERSION_ID and len(self._versions) == 1:
                return False
            if self._active_version_id == version_id:
                remaining = [vid for vid in self._versions if vid != version_id]
                self._active_version_id = remaining[0] if remaining else _DEFAULT_VERSION_ID
            del self._versions[version_id]
            self._save()
            return True

    def get(self, version_id: str) -> VersionEntry | None:
        with self._lock:
            return self._versions.get(version_id)

    def update_session(self, version_id: str, session_id: str) -> None:
        with self._lock:
            if version_id in self._versions:
                self._versions[version_id].session_id = session_id
                self._save()
