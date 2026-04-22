"""Runtime profile for selecting ReAct vs tool-calling agent behavior."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel


def _type_name(llm: BaseChatModel) -> str:
    return type(llm).__name__.lower()


def is_ollama_llm(llm: BaseChatModel) -> bool:
    return "ollama" in _type_name(llm)


def is_nvidia_llm(llm: BaseChatModel) -> bool:
    return "chatnvidia" in _type_name(llm)


def use_react_agent_llm(llm: BaseChatModel) -> bool:
    import os

    if is_ollama_llm(llm):
        return True
    if is_nvidia_llm(llm):
        raw = (os.environ.get("AGENT_NVIDIA_USE_REACT") or "0").strip().lower()
        return raw in ("1", "true", "yes", "on")
    return False


@dataclass(frozen=True)
class LLMRuntimeProfile:
    """Backend-specific agent loop parameters (see strategies + executor)."""

    llm: BaseChatModel
    use_react: bool
    is_nvidia: bool
    is_ollama: bool


def profile_for_chat_model(llm: BaseChatModel) -> LLMRuntimeProfile:
    return LLMRuntimeProfile(
        llm=llm,
        use_react=use_react_agent_llm(llm),
        is_nvidia=is_nvidia_llm(llm),
        is_ollama=is_ollama_llm(llm),
    )


__all__ = [
    "LLMRuntimeProfile",
    "is_nvidia_llm",
    "is_ollama_llm",
    "profile_for_chat_model",
    "use_react_agent_llm",
]
