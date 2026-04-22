"""Wraps LangChain AgentExecutor with RAG prefetch + Ollama memory shortcut."""

from __future__ import annotations

from typing import Any

from langchain_classic.agents import AgentExecutor

from src.agent.callbacks import emit_status
from src.agent.config import prefetch_rag_into_input_enabled
from src.agent.text_utils import extract_memory_save_content, should_prefetch_rag


class PrefetchExecutor:
    """Delegates to AgentExecutor; optional prefetch merges retriever chunks into input."""

    __slots__ = ("_executor", "_retriever", "_is_ollama", "_save_memory_tool")

    def __init__(
        self,
        executor: AgentExecutor,
        retriever: Any,
        *,
        is_ollama: bool = False,
        save_memory_tool: Any = None,
    ) -> None:
        object.__setattr__(self, "_executor", executor)
        object.__setattr__(self, "_retriever", retriever)
        object.__setattr__(self, "_is_ollama", is_ollama)
        object.__setattr__(self, "_save_memory_tool", save_memory_tool)

    def invoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        from src.agent.executor import run_agent_loop

        config = kwargs.pop("config", None)
        if len(args) >= 2 and config is None:
            config = args[1]
        payload: dict[str, Any]
        if args and isinstance(args[0], dict):
            payload = {**args[0], **kwargs}
        elif kwargs:
            payload = dict(kwargs)
        else:
            payload = {}
        base_input = payload.get("input")
        if object.__getattribute__(self, "_is_ollama") and isinstance(base_input, str):
            mem_content = extract_memory_save_content(base_input)
            if mem_content:
                save_tool = object.__getattribute__(self, "_save_memory_tool")
                if save_tool is not None:
                    save_result = save_tool.func(mem_content, tags="提醒,會議")
                    return {"output": f"已幫你記住：{mem_content}\n（{save_result}）"}
        retriever = object.__getattribute__(self, "_retriever")
        if (
            prefetch_rag_into_input_enabled()
            and isinstance(base_input, str)
            and base_input.strip()
            and retriever is not None
            and should_prefetch_rag(base_input, payload.get("chat_history"))
        ):
            q = base_input.strip()
            emit_status("reading", "調閱文件")
            docs = retriever.invoke(q)
            if docs:
                ctx = "\n\n".join(d.page_content for d in docs)
                payload["input"] = (
                    f"{base_input}\n\n---\n"
                    "[Embedded local documents — answer from this text when it applies; "
                    "say if something is not covered here.]\n"
                    f"{ctx}"
                )
        inner = object.__getattribute__(self, "_executor")
        if config is not None:
            return run_agent_loop(inner, payload, config=config)
        return run_agent_loop(inner, payload)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_executor"), name)


__all__ = ["PrefetchExecutor"]
