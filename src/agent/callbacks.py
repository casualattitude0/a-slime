"""Status sink and LangChain callback → UI labels (zh-TW)."""

from __future__ import annotations

import threading
from typing import Any, Callable

from langchain_core.callbacks.base import BaseCallbackHandler

_status_tls = threading.local()


def set_invocation_status_sink(sink: Callable[[dict[str, Any]], None] | None) -> None:
    _status_tls.sink = sink


def emit_status(phase: str, label: str, **extra: Any) -> None:
    sink: Callable[[dict[str, Any]], None] | None = getattr(_status_tls, "sink", None)
    if sink:
        ev: dict[str, Any] = {"phase": phase, "label": label, **extra}
        sink(ev)


def llm_vendor_label(serialized: dict[str, Any] | None) -> str:
    s = serialized or {}
    text_parts = [
        str(s.get("name") or ""),
        str(s.get("id") or ""),
        str((s.get("kwargs") or {}).get("model") or ""),
        str((s.get("kwargs") or {}).get("model_name") or ""),
        str((s.get("config") or {}).get("model") or ""),
    ]
    joined = " ".join(text_parts).lower()
    if "gemini" in joined or "google" in joined:
        return "Gemini"
    if "ollama" in joined:
        return "Ollama"
    if "openai" in joined or "gpt" in joined:
        return "OpenAI"
    if "anthropic" in joined or "claude" in joined:
        return "Claude"
    if "nvidia" in joined or "chatnvidia" in joined:
        return "NVIDIA"
    return "LLM"


class AgentStatusCallbackHandler(BaseCallbackHandler):
    """Maps LangChain events to zh-TW status labels for the UI."""

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        name = str((serialized or {}).get("name") or "").lower()
        id_path = str((serialized or {}).get("id") or "").lower()
        if "agent" in name or "agent" in id_path or "executor" in name:
            emit_status("thinking", "分析問題中")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        vendor = llm_vendor_label(serialized)
        emit_status("llm_requesting", f"正在與 {vendor} 溝通")

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        name = str((serialized or {}).get("name") or "")
        phase, label = tool_name_to_status(name)
        emit_status(phase, label)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        emit_status("tool_result_processing", "整合工具結果")


def tool_name_to_status(name: str) -> tuple[str, str]:
    _map = {
        "document_search": ("tool_running", "查詢本機文件"),
        "web_search": ("tool_running", "搜尋網路"),
        "web_fetch": ("tool_running", "擷取網頁內容"),
        "search_memory": ("tool_running", "查詢長期記憶"),
        "save_to_memory": ("tool_running", "儲存至長期記憶"),
        "execute_shell_command": ("tool_running", "執行 Shell 指令"),
        "get_local_datetime": ("tool_running", "取得本機日期時間"),
        "ask_reasoning_model": ("tool_running", "委派推理模型"),
        "calendar_create_event": ("tool_running", "新增行事曆事件"),
        "calendar_update_event": ("tool_running", "修改行事曆事件"),
        "calendar_delete_event": ("tool_running", "刪除行事曆事件"),
    }
    if name in _map:
        return _map[name]
    return ("tool_running", f"執行工具：{name}" if name else "執行工具")


def status_callbacks() -> list[AgentStatusCallbackHandler]:
    return [AgentStatusCallbackHandler()]


__all__ = [
    "AgentStatusCallbackHandler",
    "emit_status",
    "llm_vendor_label",
    "set_invocation_status_sink",
    "status_callbacks",
    "tool_name_to_status",
]
