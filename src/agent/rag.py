"""Vector store helpers, RAG prefetch policy, admin CRUD."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma

from src.agent.config import _DEFAULT_RAG_COLLECTION


def has_supported_data_files(data_dir: Path) -> bool:
    if not data_dir.exists():
        return False
    for pat in ("*.pdf", "*.txt", "*.md"):
        if any(data_dir.rglob(pat)):
            return True
    return False


def list_rag_items(
    chroma_dir: Path,
    embeddings: Any,
    rag_collection: str = _DEFAULT_RAG_COLLECTION,
) -> list[dict]:
    try:
        store = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=embeddings,
            collection_name=rag_collection,
        )
        result = store._collection.get(include=["documents", "metadatas"])
        items = []
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        for i, doc_id in enumerate(ids):
            items.append(
                {
                    "id": doc_id,
                    "content": (docs[i][:200] if i < len(docs) else ""),
                    "metadata": metas[i] if i < len(metas) else {},
                }
            )
        return items
    except Exception as exc:
        return [{"error": str(exc)}]


def delete_rag_item(
    chroma_dir: Path,
    embeddings: Any,
    item_id: str,
    rag_collection: str = _DEFAULT_RAG_COLLECTION,
) -> bool:
    try:
        store = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=embeddings,
            collection_name=rag_collection,
        )
        store._collection.delete(ids=[item_id])
        return True
    except Exception:
        return False


def delete_all_rag_items(
    chroma_dir: Path,
    embeddings: Any,
    rag_collection: str = _DEFAULT_RAG_COLLECTION,
) -> bool:
    try:
        store = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=embeddings,
            collection_name=rag_collection,
        )
        result = store._collection.get(include=[])
        ids = result.get("ids") or []
        if ids:
            store._collection.delete(ids=ids)
        return True
    except Exception:
        return False


__all__ = [
    "_DEFAULT_RAG_COLLECTION",
    "delete_all_rag_items",
    "delete_rag_item",
    "has_supported_data_files",
    "list_rag_items",
]
