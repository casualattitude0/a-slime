"""Constants and environment-driven toggles."""

from __future__ import annotations

import os

DEFAULT_NVIDIA_CATALOG_MODEL = "z-ai/glm-5.1"
_DEFAULT_RAG_COLLECTION = "langchain"


def prefetch_rag_into_input_enabled() -> bool:
    v = (os.environ.get("AGENT_RAG_PREFETCH") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def prefetch_vague_gate_enabled() -> bool:
    v = (os.environ.get("AGENT_RAG_PREFETCH_VAGUE_GATE") or "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def rag_prefetch_mode() -> str:
    v = (os.environ.get("AGENT_RAG_PREFETCH_MODE") or "intent").strip().lower()
    return "broad" if v == "broad" else "intent"


def nvidia_agent_max_iterations() -> int:
    raw = (os.environ.get("AGENT_NVIDIA_MAX_ITERATIONS") or "5").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 5
    return max(1, min(n, 15))


def standard_agent_max_iterations() -> int:
    """Gemini / Ollama AgentExecutor loop cap (AGENT_MAX_ITERATIONS, default 30, max 100)."""
    raw = (
        os.environ.get("AGENT_MAX_ITERATIONS")
        or os.environ.get("AGENT_GEMINI_MAX_ITERATIONS")
        or "30"
    ).strip()
    try:
        n = int(raw)
    except ValueError:
        n = 30
    return max(1, min(n, 100))


def nvidia_react_history_tail_limit() -> int | None:
    raw = (os.environ.get("AGENT_NVIDIA_REACT_HISTORY_MSGS") or "16").strip().lower()
    if raw in ("", "all", "full", "unlimited", "0"):
        return None
    try:
        n = int(raw)
    except ValueError:
        n = 16
    return max(2, min(n, 48))


__all__ = [
    "DEFAULT_NVIDIA_CATALOG_MODEL",
    "_DEFAULT_RAG_COLLECTION",
    "prefetch_rag_into_input_enabled",
    "prefetch_vague_gate_enabled",
    "rag_prefetch_mode",
    "nvidia_agent_max_iterations",
    "standard_agent_max_iterations",
    "nvidia_react_history_tail_limit",
]
