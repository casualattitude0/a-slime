"""Formatting and lightweight heuristics."""

from __future__ import annotations

import re
from typing import Any


def maybe_trim_chat_messages(history: Any, limit: int | None) -> Any:
    if limit is None or not history:
        return history
    if isinstance(history, list) and len(history) > limit:
        return history[-limit:]
    return history


def extract_memory_save_content(message: str) -> str | None:
    s = (message or "").strip()
    if not s:
        return None
    cues = ("記錄", "紀錄", "記住", "幫我記", "存起來", "save to memory")
    low = s.lower()
    if not any((c in s) for c in cues[:-1]) and cues[-1] not in low:
        return None
    cleaned = re.sub(r"^(ollama|gemini)\s*", "", s, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^(請|幫我|麻煩|可以)?\s*(記錄|紀錄|記住|幫我記|存起來)\s*", "", cleaned).strip()
    cleaned = re.sub(r"(哦|喔|吧|一下)$", "", cleaned).strip()
    return cleaned or None


def format_chat_history_as_text(history: Any) -> str:
    if not history:
        return ""
    if isinstance(history, str):
        return history
    lines: list[str] = []
    for m in history:
        role = getattr(m, "type", "") or ""
        content = getattr(m, "content", "") or ""
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        role_lower = role.lower()
        if "system" in role_lower:
            label = "System"
        elif "ai" in role_lower:
            label = "Assistant"
        else:
            label = "User"
        if str(content).strip():
            lines.append(f"{label}: {content}")
    return "\n".join(lines)


def normalize_search_query(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        for key in ("query", "input", "q", "search_query", "text"):
            v = raw.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        desc = raw.get("description")
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
    return str(raw).strip() if raw is not None else ""


_RETRIEVAL_INTENT_KEYWORDS = (
    "@data",
    "document",
    "documents",
    "doc",
    "search",
    "retrieve",
    "rag",
    "檔案",
    "文件",
    "資料",
    "內文",
    "內容",
    "根據",
    "查",
    "搜尋",
    "查詢",
)

_NON_VAGUE_QUESTION_HINTS = (
    "怎麼",
    "如何",
    "為什麼",
    "什麼",
    "哪",
    "幾",
    "請",
    "幫我",
    "what",
    "why",
    "how",
    "which",
)

_VAGUE_SHORT_RE = re.compile(
    r"^(?:哇(?:塞)?|真的假的|嗯+|喔+|哦+|蛤+|哈+|是喔|原來如此|好吧|太多了|這麼多)+[!！?？~～。．…\s]*$",
    re.IGNORECASE,
)


def has_retrieval_intent(msg: str) -> bool:
    low = msg.lower()
    return any(k in low for k in _RETRIEVAL_INTENT_KEYWORDS)


def is_vague_short_utterance(msg: str) -> bool:
    s = (msg or "").strip()
    if not s:
        return True
    normalized = re.sub(r"\s+", "", s)
    if len(normalized) > 14:
        return False
    if has_retrieval_intent(s):
        return False
    if re.fullmatch(r"[!！?？~～,，。.、…\s]+", s):
        return True
    if _VAGUE_SHORT_RE.fullmatch(s):
        return True
    if any(h in s.lower() for h in _NON_VAGUE_QUESTION_HINTS):
        return False
    if re.search(r"[A-Za-z]{4,}", s):
        return False
    if any(ch.isdigit() for ch in s):
        return False
    if len(normalized) <= 3:
        return True
    return len(normalized) <= 8 and ("?" not in s and "？" not in s)


def should_prefetch_rag(user_input: str, chat_history: list[Any] | None = None) -> bool:
    from src.agent.config import prefetch_vague_gate_enabled, rag_prefetch_mode

    msg = (user_input or "").strip()
    if not msg:
        return False
    if has_retrieval_intent(msg):
        return True
    mode = rag_prefetch_mode()
    if mode == "broad":
        if not prefetch_vague_gate_enabled():
            return True
        _ = chat_history
        return not is_vague_short_utterance(msg)
    return False


__all__ = [
    "extract_memory_save_content",
    "format_chat_history_as_text",
    "has_retrieval_intent",
    "is_vague_short_utterance",
    "maybe_trim_chat_messages",
    "normalize_search_query",
    "should_prefetch_rag",
]
