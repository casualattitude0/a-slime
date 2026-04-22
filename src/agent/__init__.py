"""Agent package: compose LLM, tools, RAG, and LangChain executor."""

from __future__ import annotations

from src.agent.builder import build_executor
from src.agent.callbacks import set_invocation_status_sink
from src.agent.config import DEFAULT_NVIDIA_CATALOG_MODEL
from src.agent.errors import LLMErrorInfo, classify_llm_error
from src.agent.errors import rich_exception_message as _rich_exception_message
from src.agent.executor import astream_executor, invoke_executor
from src.agent.interfaces import LLMRuntimeProfile, profile_for_chat_model
from src.agent.llm_factory import (
    default_chat_model_from_env,
    make_gemini_llm,
    make_nvidia_llm,
    make_ollama_llm,
)
from src.agent.output import _PYTHON_TOOL_CALL_RE
from src.agent.output import normalize_agent_output
from src.agent.prefetch_executor import PrefetchExecutor as _PrefetchExecutor
from src.agent.rag import delete_all_rag_items, delete_rag_item, has_supported_data_files, list_rag_items
from src.agent.router import agent_mode_reply, local_quick_reply, should_escalate_to_gemini
from src.agent.text_utils import is_vague_short_utterance as _is_vague_short_utterance

_make_llm = default_chat_model_from_env
_has_supported_data_files = has_supported_data_files

__all__ = [
    "DEFAULT_NVIDIA_CATALOG_MODEL",
    "LLMErrorInfo",
    "LLMRuntimeProfile",
    "_PrefetchExecutor",
    "_PYTHON_TOOL_CALL_RE",
    "_has_supported_data_files",
    "_is_vague_short_utterance",
    "_make_llm",
    "_rich_exception_message",
    "agent_mode_reply",
    "astream_executor",
    "build_executor",
    "classify_llm_error",
    "default_chat_model_from_env",
    "delete_all_rag_items",
    "delete_rag_item",
    "invoke_executor",
    "list_rag_items",
    "local_quick_reply",
    "make_gemini_llm",
    "make_nvidia_llm",
    "make_ollama_llm",
    "normalize_agent_output",
    "profile_for_chat_model",
    "set_invocation_status_sink",
    "should_escalate_to_gemini",
]
