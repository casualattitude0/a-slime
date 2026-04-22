"""Construct chat models from environment / explicit selection."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.agent.config import DEFAULT_NVIDIA_CATALOG_MODEL


def make_ollama_llm() -> BaseChatModel:
    from langchain_ollama import ChatOllama

    model = (os.environ.get("OLLAMA_MODEL") or "").strip()
    if not model:
        raise ValueError("OLLAMA_MODEL is not set.")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    return ChatOllama(model=model, base_url=base_url, temperature=0)


def make_gemini_llm(*, model: str | None = None) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini.")
    m = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    return ChatGoogleGenerativeAI(model=m, temperature=0, google_api_key=api_key)


def make_nvidia_llm(*, model: str | None = None) -> BaseChatModel:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    api_key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("Set NVIDIA_API_KEY for NVIDIA chat models.")
    m = (
        model
        or (os.environ.get("NVIDIA_MODEL") or DEFAULT_NVIDIA_CATALOG_MODEL).strip()
    )
    base_url = (os.environ.get("NVIDIA_BASE_URL") or "").strip() or None
    kwargs: dict[str, Any] = {"model": m, "api_key": api_key, "temperature": 0}
    if base_url:
        kwargs["base_url"] = base_url

    return ChatNVIDIA(**kwargs)


def default_chat_model_from_env() -> BaseChatModel:
    """Prefer Ollama when OLLAMA_MODEL is set; otherwise Gemini."""
    if (os.environ.get("OLLAMA_MODEL") or "").strip():
        return make_ollama_llm()
    return make_gemini_llm()


__all__ = [
    "DEFAULT_NVIDIA_CATALOG_MODEL",
    "default_chat_model_from_env",
    "make_gemini_llm",
    "make_nvidia_llm",
    "make_ollama_llm",
]
