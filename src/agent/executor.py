"""Invoke / stream agent runs; orchestrate LLM ↔ tool iterations."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator
from typing import Any, Callable

from langchain_classic.agents import AgentExecutor

from src.agent.callbacks import (
    format_tool_start_status,
    get_invocation_status_sink,
    llm_vendor_label,
    set_invocation_status_sink,
    status_callbacks,
    tool_name_to_status,
)
from src.agent.config import prefetch_rag_into_input_enabled
from src.agent.prefetch_executor import PrefetchExecutor
from src.agent.text_utils import extract_memory_save_content, should_prefetch_rag


def run_agent_loop(
    agent_executor: AgentExecutor,
    payload: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """LangChain performs iterative LLM steps and tool execution until a final answer."""
    if config is not None:
        return agent_executor.invoke(payload, config=config)
    return agent_executor.invoke(payload)


def invoke_executor(
    executor: Any,
    payload: dict[str, Any],
    *,
    status_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    set_invocation_status_sink(status_sink)
    try:
        cb = status_callbacks()
        return executor.invoke(payload, config={"callbacks": cb})
    finally:
        set_invocation_status_sink(None)


def chunk_to_text(chunk: Any) -> str:
    if chunk is None:
        return ""
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                t = p.get("text", "")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""


async def astream_executor(
    executor: Any,
    payload: dict[str, Any],
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    if isinstance(executor, PrefetchExecutor):
        inner_exec = object.__getattribute__(executor, "_executor")
        retriever = object.__getattribute__(executor, "_retriever")
        is_ollama = bool(object.__getattribute__(executor, "_is_ollama"))
        save_memory_tool = object.__getattribute__(executor, "_save_memory_tool")
    else:
        inner_exec = executor
        retriever = None
        is_ollama = False
        save_memory_tool = None

    run_payload = dict(payload)
    if is_ollama and isinstance(run_payload.get("input"), str):
        mem_content = extract_memory_save_content(run_payload.get("input", ""))
        if mem_content and save_memory_tool is not None:
            save_result = await asyncio.to_thread(save_memory_tool.func, mem_content, "提醒,會議")
            final_text = f"已幫你記住：{mem_content}\n（{save_result}）"
            yield {"event": "delta", "text": final_text}
            yield {"event": "_done", "output": {"output": final_text}}
            return

    if (
        prefetch_rag_into_input_enabled()
        and retriever is not None
        and isinstance(run_payload.get("input"), str)
        and run_payload["input"].strip()
        and should_prefetch_rag(run_payload["input"], run_payload.get("chat_history"))
    ):
        q = run_payload["input"].strip()
        yield {"event": "status", "phase": "reading", "label": "調閱文件"}
        docs = await asyncio.to_thread(retriever.invoke, q)
        if docs:
            ctx = "\n\n".join(d.page_content for d in docs)
            run_payload["input"] = (
                f"{run_payload['input']}\n\n---\n"
                "[Embedded local documents — answer from this text when it applies; "
                "say if something is not covered here.]\n"
                f"{ctx}"
            )

    executor_run_id: str | None = None
    final_output: dict[str, Any] = {}
    seen_thinking = False
    seen_first_text_delta = False
    streamed_text_so_far = ""
    status_queue: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
    previous_sink = get_invocation_status_sink()

    def _stream_status_sink(ev: dict[str, Any]) -> None:
        try:
            status_queue.put_nowait(dict(ev))
        except Exception:
            return

    def _drain_status_events() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        extra_keys = ("detail", "content_hint")
        while True:
            try:
                queued = status_queue.get_nowait()
            except queue.Empty:
                break
            phase = str((queued or {}).get("phase") or "").strip()
            label = str((queued or {}).get("label") or "").strip()
            if not phase or not label:
                continue
            ev = {"event": "status", "phase": phase, "label": label}
            tool_name = str((queued or {}).get("tool") or "").strip()
            if tool_name:
                ev["tool"] = tool_name
            for key in extra_keys:
                value = (queued or {}).get(key)
                if value is None:
                    continue
                if isinstance(value, str):
                    value = value.strip()
                    if not value:
                        continue
                ev[key] = value
            out.append(ev)
        return out

    set_invocation_status_sink(_stream_status_sink)

    try:
        async for ev in inner_exec.astream_events(run_payload, version="v2"):
            for queued_ev in _drain_status_events():
                yield queued_ev

            if cancel_check is not None and cancel_check():
                yield {"event": "_cancelled"}
                return

            ev_name: str = ev.get("event", "")
            run_id: str = ev.get("run_id", "")

            if ev_name == "on_chain_start":
                if executor_run_id is None and not ev.get("parent_ids"):
                    executor_run_id = run_id
                if not seen_thinking:
                    seen_thinking = True
                    yield {"event": "status", "phase": "thinking", "label": "分析問題中"}

            elif ev_name == "on_chain_end":
                if run_id == executor_run_id:
                    output_data = (ev.get("data") or {}).get("output")
                    if isinstance(output_data, dict):
                        final_output = output_data

            elif ev_name == "on_chat_model_start":
                vendor = llm_vendor_label(ev.get("data", {}).get("serialized"))
                yield {
                    "event": "status",
                    "phase": "llm_requesting",
                    "label": f"正在與 {vendor} 溝通",
                }

            elif ev_name == "on_chat_model_end":
                output = (ev.get("data") or {}).get("output")
                tool_calls = getattr(output, "tool_calls", None) or []
                if tool_calls:
                    yield {"event": "status", "phase": "tool_planning", "label": "規劃工具呼叫"}

            elif ev_name == "on_tool_start":
                tool_name = ev.get("name") or ""
                phase, _label = tool_name_to_status(tool_name)
                raw_input = (ev.get("data") or {}).get("input")
                formatted_label = format_tool_start_status(tool_name, raw_input)
                yield {"event": "status", "phase": phase, "label": formatted_label, "tool": tool_name}

            elif ev_name == "on_tool_end":
                yield {"event": "status", "phase": "tool_result_processing", "label": "整合工具結果"}

            elif ev_name == "on_chat_model_stream":
                chunk = (ev.get("data") or {}).get("chunk")
                text = chunk_to_text(chunk)
                if text:
                    if streamed_text_so_far and text.startswith(streamed_text_so_far):
                        delta_text = text[len(streamed_text_so_far) :]
                    else:
                        delta_text = text
                    streamed_text_so_far = text
                    if not delta_text:
                        continue
                    if not seen_first_text_delta:
                        seen_first_text_delta = True
                        yield {"event": "status", "phase": "llm_streaming", "label": "模型回覆中"}
                    yield {"event": "delta", "text": delta_text}

        for queued_ev in _drain_status_events():
            yield queued_ev
        yield {"event": "_done", "output": final_output}

    except Exception as exc:
        for queued_ev in _drain_status_events():
            yield queued_ev
        yield {"event": "_error", "exc": exc}
    finally:
        set_invocation_status_sink(previous_sink)


__all__ = [
    "astream_executor",
    "chunk_to_text",
    "invoke_executor",
    "run_agent_loop",
]
