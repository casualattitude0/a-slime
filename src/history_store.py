"""Chroma-backed chat history store.

Each message is persisted as a Chroma document with deterministic ordering
metadata so sessions can be restored in chronological order across restarts.

Collection name: ``chat_history`` (one collection, namespaced by session_id).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

_COLLECTION_NAME = "chat_history"
_MAX_MESSAGES = 40


def _make_store(chroma_dir: Path, embeddings: Any):
    from langchain_community.vectorstores import Chroma

    return Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=embeddings,
        collection_name=_COLLECTION_NAME,
    )


def _row_to_message(doc_id: str, content: str, metadata: dict) -> BaseMessage:
    role = metadata.get("role", "human")
    if role == "ai":
        return AIMessage(content=content)
    return HumanMessage(content=content)


def append_message(
    chroma_dir: Path,
    embeddings: Any,
    session_id: str,
    message: BaseMessage,
    *,
    version_id: str | None = None,
) -> None:
    """Persist a single message to the history collection."""
    store = _make_store(chroma_dir, embeddings)
    collection = store._collection

    existing = collection.get(
        where={"session_id": {"$eq": session_id}},
        include=[],
    )
    turn_index = len(existing.get("ids") or [])

    role = "ai" if isinstance(message, AIMessage) else "human"
    doc_id = str(uuid.uuid4())
    metadata: dict[str, Any] = {
        "session_id": session_id,
        "role": role,
        "turn_index": turn_index,
        "created_at": time.time(),
    }
    if version_id is not None:
        metadata["version_id"] = version_id

    content = message.content if isinstance(message.content, str) else str(message.content)
    collection.add(
        ids=[doc_id],
        documents=[content],
        metadatas=[metadata],
    )


def load_session_messages(
    chroma_dir: Path,
    embeddings: Any,
    session_id: str,
) -> list[BaseMessage]:
    """Load all messages for a session, ordered by turn_index."""
    store = _make_store(chroma_dir, embeddings)
    collection = store._collection

    result = collection.get(
        where={"session_id": {"$eq": session_id}},
        include=["documents", "metadatas"],
    )
    ids = result.get("ids") or []
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []

    if not ids:
        return []

    rows = sorted(
        zip(ids, docs, metas),
        key=lambda r: r[2].get("turn_index", 0),
    )
    return [_row_to_message(rid, doc, meta) for rid, doc, meta in rows]


def trim_session_messages(
    chroma_dir: Path,
    embeddings: Any,
    session_id: str,
    max_messages: int = _MAX_MESSAGES,
) -> None:
    """Delete oldest messages beyond max_messages for a session."""
    store = _make_store(chroma_dir, embeddings)
    collection = store._collection

    result = collection.get(
        where={"session_id": {"$eq": session_id}},
        include=["metadatas"],
    )
    ids = result.get("ids") or []
    metas = result.get("metadatas") or []

    if len(ids) <= max_messages:
        return

    rows = sorted(
        zip(ids, metas),
        key=lambda r: r[1].get("turn_index", 0),
    )
    to_delete = [rid for rid, _ in rows[: len(rows) - max_messages]]
    if to_delete:
        collection.delete(ids=to_delete)


def get_session_messages_as_dicts(
    chroma_dir: Path,
    embeddings: Any,
    session_id: str,
) -> list[dict]:
    """Return messages as [{role: 'user'|'bot', text: str}] for API responses."""
    msgs = load_session_messages(chroma_dir, embeddings, session_id)
    result = []
    for m in msgs:
        if isinstance(m, HumanMessage):
            result.append({"role": "user", "text": m.content if isinstance(m.content, str) else str(m.content)})
        elif isinstance(m, AIMessage):
            result.append({"role": "bot", "text": m.content if isinstance(m.content, str) else str(m.content)})
    return result


def clear_session_messages(
    chroma_dir: Path,
    embeddings: Any,
    session_id: str,
) -> None:
    """Delete all messages for a session."""
    store = _make_store(chroma_dir, embeddings)
    collection = store._collection

    result = collection.get(
        where={"session_id": {"$eq": session_id}},
        include=[],
    )
    ids = result.get("ids") or []
    if ids:
        collection.delete(ids=ids)
