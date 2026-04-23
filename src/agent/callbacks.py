"""Status sink and LangChain callback → UI labels (zh-TW)."""

from __future__ import annotations

import threading
import re
from typing import Any, Callable

from langchain_core.callbacks.base import BaseCallbackHandler

_status_tls = threading.local()
_status_global_lock = threading.Lock()
_status_global_sink: Callable[[dict[str, Any]], None] | None = None


_REDACT_PATTERNS = [
    re.compile(r"(api[_-]?key\s*[=:]\s*)([^\s,;]+)", re.IGNORECASE),
    re.compile(r"(token\s*[=:]\s*)([^\s,;]+)", re.IGNORECASE),
    re.compile(r"(password\s*[=:]\s*)([^\s,;]+)", re.IGNORECASE),
    re.compile(r"(bearer\s+)([^\s,;]+)", re.IGNORECASE),
]


def _redact_text(text: str) -> str:
    out = text
    for pat in _REDACT_PATTERNS:
        out = pat.sub(r"\1***", out)
    return out


def _single_line(text: str, limit: int = 140) -> str:
    s = " ".join((text or "").split())
    if len(s) > limit:
        return s[: limit - 1] + "…"
    return s


def _tool_input_snippet(name: str, input_str: str) -> str:
    raw = (input_str or "").strip()
    if not raw:
        return ""
    safe = _single_line(_redact_text(raw), 180)
    if name == "execute_shell_command":
        return f"命令：{safe}"
    if name == "web_fetch":
        return f"URL：{safe}"
    if name == "web_search":
        return f"查詢：{safe}"
    if name in ("delegate_to_subagent", "delegate_to_subagents_parallel"):
        return f"任務：{safe}"
    return safe


def _tool_output_snippet(output: Any) -> str:
    text = _single_line(_redact_text(str(output or "")), 180)
    return text


def set_invocation_status_sink(sink: Callable[[dict[str, Any]], None] | None) -> None:
    global _status_global_sink
    _status_tls.sink = sink
    with _status_global_lock:
        _status_global_sink = sink


def get_invocation_status_sink() -> Callable[[dict[str, Any]], None] | None:
    sink = getattr(_status_tls, "sink", None)
    if sink is not None:
        return sink
    with _status_global_lock:
        return _status_global_sink


def emit_status(phase: str, label: str, **extra: Any) -> None:
    sink = get_invocation_status_sink()
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
        detail = _tool_input_snippet(name, input_str)
        emit_status(phase, f"{label}｜{detail}" if detail else label)

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
        detail = _tool_output_snippet(output)
        emit_status(
            "tool_result_processing",
            f"整合工具結果｜{detail}" if detail else "整合工具結果",
        )


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
        "delegate_to_subagent": ("tool_running", "委派 Sub-agent"),
        "delegate_to_subagents_parallel": ("tool_running", "平行委派 Sub-agent"),
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
    "get_invocation_status_sink",
    "llm_vendor_label",
    "set_invocation_status_sink",
    "status_callbacks",
    "tool_name_to_status",
]
