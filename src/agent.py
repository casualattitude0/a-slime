import ast
import json
import os
import re
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field, field_validator

from src.tools import (
    make_memory_tools,
    make_reasoning_tool,
    make_web_fetch_tool,
    make_web_search_tool,
)

_DEFAULT_RAG_COLLECTION = "langchain"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _has_supported_data_files(data_dir: Path) -> bool:
    if not data_dir.exists():
        return False
    for pat in ("*.pdf", "*.txt", "*.md"):
        if any(data_dir.rglob(pat)):
            return True
    return False


_status_tls = threading.local()


def _prefetch_rag_into_input_enabled() -> bool:
    v = (os.environ.get("AGENT_RAG_PREFETCH") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def set_invocation_status_sink(sink: Callable[[dict[str, Any]], None] | None) -> None:
    _status_tls.sink = sink


def _emit_status(phase: str, label: str, **extra: Any) -> None:
    sink: Callable[[dict[str, Any]], None] | None = getattr(_status_tls, "sink", None)
    if sink:
        ev: dict[str, Any] = {"phase": phase, "label": label, **extra}
        sink(ev)


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
            _emit_status("thinking", "分析問題中")

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
        _emit_status("llm_requesting", "正在與 LLM 溝通")

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
        phase, label = _tool_name_to_status(name)
        _emit_status(phase, label)

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
        _emit_status("tool_result_processing", "整合工具結果")


def _normalize_search_query(raw: Any) -> str:
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


class DocumentSearchArgs(BaseModel):
    query: str = Field(description="Keywords or question to search in the documents")

    @field_validator("query", mode="before")
    @classmethod
    def coerce_query(cls, v: Any) -> str:
        return _normalize_search_query(v)


def make_ollama_llm() -> BaseChatModel:
    from langchain_ollama import ChatOllama

    model = (os.environ.get("OLLAMA_MODEL") or "").strip()
    if not model:
        raise ValueError("OLLAMA_MODEL is not set.")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    return ChatOllama(model=model, base_url=base_url, temperature=0)


def make_gemini_llm(*, model: str | None = None) -> BaseChatModel:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini.")
    m = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    return ChatGoogleGenerativeAI(model=m, temperature=0, google_api_key=api_key)


def _make_llm() -> BaseChatModel:
    if (os.environ.get("OLLAMA_MODEL") or "").strip():
        return make_ollama_llm()
    return make_gemini_llm()


def should_escalate_to_gemini(user_message: str, history_message_count: int) -> bool:
    """When Ollama is available and Gemini API key exists, ask Gemini whether to escalate."""
    root = _project_root()
    load_dotenv(root / ".env")
    enabled = (os.environ.get("AGENT_ROUTER_ENABLED") or "1").strip().lower()
    if enabled not in ("1", "true", "yes", "on"):
        return False
    if not (os.environ.get("OLLAMA_MODEL") or "").strip():
        return False
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return False
    try:
        skip_max = int((os.environ.get("AGENT_ROUTER_SKIP_MAX_CHARS") or "80").strip())
    except ValueError:
        skip_max = 80
    msg = user_message.strip()
    if len(msg) <= skip_max and history_message_count == 0:
        return False

    router_model = (os.environ.get("GEMINI_ROUTER_MODEL") or "").strip() or (
        os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash"
    )
    llm = ChatGoogleGenerativeAI(
        model=router_model,
        temperature=0,
        google_api_key=api_key,
    )
    router_prompt = (
        "You are a routing classifier. Decide if the user message needs deep reasoning, "
        "multi-step analysis, subtle judgment, or is likely too difficult for a small local model. "
        "Reply with JSON only, no markdown: {\"escalate\": true} or {\"escalate\": false}\n\n"
        f"User message:\n{msg[:8000]}"
    )
    try:
        resp = llm.invoke(router_prompt)
        text = str(getattr(resp, "content", None) or resp).strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
        m = re.search(r"\{[^{}]*\"escalate\"[^{}]*\}", text, re.DOTALL)
        chunk = m.group(0) if m else text
        data = json.loads(chunk)
        return bool(data.get("escalate"))
    except Exception:
        return False


def local_quick_reply(user_message: str, history: list) -> str | None:
    """Use local Ollama model to decide simple-vs-complex and optionally answer directly.

    ``history`` is the list of BaseMessage objects from the current session.
    The router receives the last few turns so it can answer context-dependent
    questions (e.g. "what did I just ask?") without escalating to the full agent.
    """
    from langchain_core.messages import AIMessage as _AI, HumanMessage as _HM

    root = _project_root()
    load_dotenv(root / ".env")
    enabled = (os.environ.get("AGENT_LOCAL_ROUTER_ENABLED") or "1").strip().lower()
    if enabled not in ("1", "true", "yes", "on"):
        return None
    if not (os.environ.get("OLLAMA_MODEL") or "").strip():
        return None
    msg = (user_message or "").strip()
    if not msg:
        return None

    history_message_count = len(history) if history else 0

    # Build recent-history context (last 6 messages = 3 turns).
    history_context = ""
    if history:
        recent = history[-6:]
        lines: list[str] = []
        for m in recent:
            role = "Assistant" if isinstance(m, _AI) else "User"
            content = m.content if isinstance(m.content, str) else str(m.content)
            lines.append(f"{role}: {content[:300]}")
        if lines:
            history_context = "Recent conversation (use this to answer context-dependent questions):\n"
            history_context += "\n".join(lines) + "\n\n"

    llm = make_ollama_llm()
    router_prompt = (
        "You are a lightweight local router.\n"
        "Task:\n"
        "1) Decide whether the user's message is SIMPLE.\n"
        "2) If SIMPLE, provide a direct short answer using the conversation history below if relevant.\n"
        "3) If COMPLEX (needs web search, document retrieval, or multi-step reasoning), do not answer.\n\n"
        "SIMPLE includes: greetings, trivial math, direct factual questions, AND questions about "
        "what was said earlier in this conversation (those can be answered from the history).\n"
        "NOT SIMPLE: anything requiring current information, file search, or deep analysis.\n\n"
        "Return JSON only, no markdown:\n"
        '{"simple": true, "reply": "..."}\n'
        "or\n"
        '{"simple": false}\n\n'
        + history_context
        + f"history_message_count={history_message_count}\n"
        f"User message:\n{msg[:4000]}"
    )
    try:
        resp = llm.invoke(router_prompt)
        text = str(getattr(resp, "content", None) or resp).strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
        m = re.search(r"\{[\s\S]*\}", text)
        chunk = m.group(0) if m else text
        data = json.loads(chunk)
        if bool(data.get("simple")):
            reply = str(data.get("reply") or "").strip()
            return reply or None
        return None
    except Exception:
        return None


class _PrefetchExecutor:
    """Delegates to AgentExecutor; optional prefetch merges retriever chunks into input."""

    __slots__ = ("_executor", "_retriever")

    def __init__(self, executor: AgentExecutor, retriever: Any) -> None:
        object.__setattr__(self, "_executor", executor)
        object.__setattr__(self, "_retriever", retriever)

    def invoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
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
        retriever = object.__getattribute__(self, "_retriever")
        if (
            _prefetch_rag_into_input_enabled()
            and isinstance(base_input, str)
            and base_input.strip()
            and retriever is not None
        ):
            q = base_input.strip()
            _emit_status("reading", "調閱文件")
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
            return inner.invoke(payload, config=config)
        return inner.invoke(payload)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_executor"), name)


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


def build_executor(
    chroma_dir: Path | None = None,
    *,
    llm: BaseChatModel | None = None,
    retriever_k: int = 4,
    memory_collection: str | None = None,
    rag_collection: str = _DEFAULT_RAG_COLLECTION,
) -> AgentExecutor | _PrefetchExecutor:
    from src.ingest import ingest as run_ingest

    root = _project_root()
    load_dotenv(root / ".env")
    chroma_path = (chroma_dir or (root / "chroma_db")).resolve()
    chroma_path.mkdir(parents=True, exist_ok=True)

    embed_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not embed_key:
        raise ValueError(
            "Embeddings use Gemini; set GOOGLE_API_KEY or GEMINI_API_KEY in .env."
        )
    embed_model = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embed_model,
        google_api_key=embed_key,
    )
    vectorstore = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
        collection_name=rag_collection,
    )
    try:
        rag_count = int(vectorstore._collection.count())
    except Exception:
        rag_count = 0
    if rag_count == 0:
        data_dir = root / "data"
        if _has_supported_data_files(data_dir):
            run_ingest(
                data_dir=data_dir.resolve(),
                chroma_dir=chroma_path.resolve(),
                chunk_size=1000,
                chunk_overlap=200,
            )
            vectorstore = Chroma(
                persist_directory=str(chroma_path),
                embedding_function=embeddings,
                collection_name=rag_collection,
            )
    retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})

    def _run_document_search(query: str) -> str:
        _emit_status("reading", "調閱文件")
        q = _normalize_search_query(query)
        if not q:
            return "Empty search query."
        docs = retriever.invoke(q)
        if not docs:
            return "No matching passages found."
        return "\n\n".join(d.page_content for d in docs)

    retriever_tool = StructuredTool.from_function(
        name="document_search",
        description=(
            "Search ingested local documents (data/ folder, embedded via "
            "src/ingest.py). Use when the user references local files, "
            "mentions @data, or asks about content known to be ingested."
        ),
        func=_run_document_search,
        args_schema=DocumentSearchArgs,
    )

    web_search_tool = make_web_search_tool()
    web_fetch_tool = make_web_fetch_tool()
    mem_col = memory_collection or "agent_memory"
    memory_tools = make_memory_tools(chroma_path, embeddings, collection_name=mem_col)
    reasoning_tool = make_reasoning_tool()

    chat_model = llm or _make_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一位研究助理，目標是協助使用者完成工作。\n"
                "語言規則：\n"
                "- 一律使用繁體中文回覆。\n"
                "- 嚴禁使用任何簡體中文字。\n"
                "- 即使使用者輸入英文或簡體中文，仍以繁體中文回覆。\n\n"
                "可用工具：\n"
                "- search_memory：持久化語意記憶，保存過往事實與筆記。若問題可能依賴既有脈絡，優先先查詢。\n"
                "- save_to_memory：儲存可長期重用的重要資訊（使用者偏好、決策、關鍵發現）。僅保存有意義且可重用的內容。\n"
                "- web_search：使用 DuckDuckGo 搜尋最新網路資訊。\n"
                "- web_fetch：擷取並清理指定網址文字內容，可搭配 web_search 讀取候選結果。\n"
                "- ask_reasoning_model：將複雜、多步驟的分析或綜整委派給更強的推理模型，並明確附上問題與已蒐集脈絡。\n"
                "- document_search：搜尋已匯入向量資料庫的本機文件；當使用者提到 @data 或詢問本機匯入內容時優先使用。\n\n"
                "工作流程：\n"
                "若任務非常簡單（例如打招呼、瑣碎事實查詢、直接澄清），可直接回覆而不呼叫工具。否則先拆解任務並判斷是否需要子代理。\n"
                "1. 區分任務：將使用者需求拆成有順序的子任務，並決定各子任務要使用的工具（search_memory、web_search、web_fetch、document_search、ask_reasoning_model）。\n"
                "2. 任務執行：依序執行子任務；需要複雜推理或綜整時，將已蒐集脈絡交給 ask_reasoning_model。\n"
                "3. 回覆：整合結果後精簡作答，僅引用實際使用到的來源；必要時用 save_to_memory 保存可持續利用的結論。\n"
                "禁止捏造引用。若輸入中出現嵌入的本機文件（[Embedded local documents — ...]），視為可選參考資料。",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    tools = [
        *memory_tools,
        web_search_tool,
        web_fetch_tool,
        reasoning_tool,
        retriever_tool,
    ]
    agent = create_tool_calling_agent(chat_model, tools, prompt)
    verbose = os.environ.get("AGENT_VERBOSE", "").lower() in ("1", "true", "yes")
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=10,
    )
    return _PrefetchExecutor(executor, retriever)


class LLMErrorInfo:
    """Structured info extracted from an LLM provider exception."""

    __slots__ = ("is_llm_error", "error_type", "message", "retry_after_seconds")

    def __init__(
        self,
        *,
        is_llm_error: bool,
        error_type: str,
        message: str,
        retry_after_seconds: float | None,
    ) -> None:
        self.is_llm_error = is_llm_error
        self.error_type = error_type
        self.message = message
        self.retry_after_seconds = retry_after_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_llm_error": self.is_llm_error,
            "error_type": self.error_type,
            "message": self.message,
            "retry_after_seconds": self.retry_after_seconds,
        }


def _parse_retry_delay(text: str) -> float | None:
    m = re.search(r"retry[_\s](?:in|after)[:\s]+([0-9]+(?:\.[0-9]+)?)\s*s", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"retryDelay[\"']?\s*:\s*[\"']([0-9]+(?:\.[0-9]+)?)s[\"']", text)
    if m:
        return float(m.group(1))
    return None


_LLM_STATUS_CODES = {429, 500, 502, 503, 504}
_LLM_ERROR_KEYWORDS = (
    "resource_exhausted",
    "rate_limit",
    "ratelimit",
    "quota",
    "too many requests",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "internal server error",
    "overloaded",
    "timeout",
)


def classify_llm_error(exc: BaseException) -> LLMErrorInfo:
    """Return LLMErrorInfo for any exception; ``is_llm_error`` is True only for provider errors."""
    text = str(exc).lower()
    raw = str(exc)

    is_llm = False
    error_type = "unknown"

    for kw in _LLM_ERROR_KEYWORDS:
        if kw in text:
            is_llm = True
            break

    for code in _LLM_STATUS_CODES:
        if str(code) in raw:
            is_llm = True
            break

    if is_llm:
        if "resource_exhausted" in text or "429" in raw or "quota" in text or "rate" in text:
            error_type = "quota_exceeded"
        elif "timeout" in text:
            error_type = "timeout"
        elif any(k in text for k in ("unavailable", "bad gateway", "gateway timeout", "overloaded")):
            error_type = "service_unavailable"
        else:
            error_type = "provider_error"

    retry_after = _parse_retry_delay(raw) if is_llm else None

    return LLMErrorInfo(
        is_llm_error=is_llm,
        error_type=error_type,
        message=raw,
        retry_after_seconds=retry_after,
    )


def status_callbacks() -> list[AgentStatusCallbackHandler]:
    return [AgentStatusCallbackHandler()]


def invoke_executor(
    executor: Any,
    payload: dict[str, Any],
    *,
    status_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Invoke with optional status sink (thread-local) and LangChain callbacks."""
    set_invocation_status_sink(status_sink)
    try:
        cb = status_callbacks()
        return executor.invoke(payload, config={"callbacks": cb})
    finally:
        set_invocation_status_sink(None)


def _tool_name_to_status(name: str) -> tuple[str, str]:
    """Map a LangChain tool name to a (phase, label) pair for UI status events."""
    _map = {
        "document_search": ("tool_running", "查詢本機文件"),
        "web_search": ("tool_running", "搜尋網路"),
        "web_fetch": ("tool_running", "擷取網頁內容"),
        "search_memory": ("tool_running", "查詢長期記憶"),
        "save_to_memory": ("tool_running", "儲存至長期記憶"),
        "ask_reasoning_model": ("tool_running", "委派推理模型"),
    }
    if name in _map:
        return _map[name]
    return ("tool_running", f"執行工具：{name}" if name else "執行工具")


def _chunk_to_text(chunk: Any) -> str:
    """Extract plain text from an AIMessageChunk (ignores tool-call chunks)."""
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
    """Async generator that streams events from the executor.

    Yields dicts with ``event`` key:
      - ``status``   – ``{phase, label}`` UI hint
      - ``delta``    – ``{text}`` incremental output token
      - ``_done``    – ``{output}`` raw executor output dict (terminal, internal)
      - ``_error``   – ``{exc}`` exception (terminal, internal)
      - ``_cancelled`` – cancelled by caller (terminal, internal)

    Terminal ``_*`` events are consumed by the web layer; clients only see
    ``status``, ``delta``, ``done``, and ``error`` frames.
    """
    import asyncio

    # Resolve the inner AgentExecutor and optional retriever from _PrefetchExecutor.
    if isinstance(executor, _PrefetchExecutor):
        inner_exec = object.__getattribute__(executor, "_executor")
        retriever = object.__getattribute__(executor, "_retriever")
    else:
        inner_exec = executor
        retriever = None

    run_payload = dict(payload)

    # RAG prefetch (same logic as _PrefetchExecutor.invoke).
    if (
        _prefetch_rag_into_input_enabled()
        and retriever is not None
        and isinstance(run_payload.get("input"), str)
        and run_payload["input"].strip()
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

    # executor_run_id is set on the first on_chain_start for the AgentExecutor,
    # so we can capture its on_chain_end output later.
    executor_run_id: str | None = None
    final_output: dict[str, Any] = {}
    seen_thinking = False
    seen_first_text_delta = False

    try:
        async for ev in inner_exec.astream_events(run_payload, version="v2"):
            if cancel_check is not None and cancel_check():
                yield {"event": "_cancelled"}
                return

            ev_name: str = ev.get("event", "")
            run_id: str = ev.get("run_id", "")

            if ev_name == "on_chain_start":
                # Identify the root AgentExecutor run.
                if executor_run_id is None and not ev.get("parent_ids"):
                    executor_run_id = run_id
                if not seen_thinking:
                    seen_thinking = True
                    yield {"event": "status", "phase": "thinking", "label": "分析問題中"}

            elif ev_name == "on_chain_end":
                # Capture the AgentExecutor's final output dict.
                if run_id == executor_run_id:
                    output_data = (ev.get("data") or {}).get("output")
                    if isinstance(output_data, dict):
                        final_output = output_data

            elif ev_name == "on_chat_model_start":
                yield {"event": "status", "phase": "llm_requesting", "label": "正在與 LLM 溝通"}

            elif ev_name == "on_chat_model_end":
                # If the model decided to call tools, signal that.
                output = (ev.get("data") or {}).get("output")
                tool_calls = getattr(output, "tool_calls", None) or []
                if tool_calls:
                    yield {"event": "status", "phase": "tool_planning", "label": "規劃工具呼叫"}

            elif ev_name == "on_tool_start":
                tool_name = ev.get("name") or ""
                phase, label = _tool_name_to_status(tool_name)
                yield {"event": "status", "phase": phase, "label": label, "tool": tool_name}

            elif ev_name == "on_tool_end":
                yield {"event": "status", "phase": "tool_result_processing", "label": "整合工具結果"}

            elif ev_name == "on_chat_model_stream":
                chunk = (ev.get("data") or {}).get("chunk")
                text = _chunk_to_text(chunk)
                if text:
                    if not seen_first_text_delta:
                        seen_first_text_delta = True
                        yield {"event": "status", "phase": "llm_streaming", "label": "模型回覆中"}
                    yield {"event": "delta", "text": text}

        yield {"event": "_done", "output": final_output}

    except Exception as exc:
        yield {"event": "_error", "exc": exc}


def _blocks_to_plain_text(parts: Any) -> str:
    """Gemini/LC content blocks → plain text (drops extras/signature/tool metadata)."""
    if parts is None:
        return ""
    if isinstance(parts, str):
        return parts
    if isinstance(parts, list):
        chunks: list[str] = []
        for p in parts:
            t = _blocks_to_plain_text(p)
            if t:
                chunks.append(t)
        return "\n\n".join(chunks).strip()
    if isinstance(parts, dict):
        if parts.get("type") == "text" and isinstance(parts.get("text"), str):
            return parts["text"]
        if isinstance(parts.get("text"), str):
            return parts["text"]
        return ""
    content = getattr(parts, "content", None)
    if content is not None and content is not parts:
        return _blocks_to_plain_text(content)
    return str(parts) if parts else ""


def _line_looks_like_tool_json(line: str) -> bool:
    s = line.strip()
    if not s.startswith("{"):
        return False
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return False
    return _is_tool_call_object(obj)


def _strip_tool_json_lines(text: str) -> str:
    lines = text.splitlines()
    kept = [ln for ln in lines if ln.strip() and not _line_looks_like_tool_json(ln)]
    return "\n".join(kept).strip()


def _is_tool_call_object(obj: Any) -> bool:
    if not isinstance(obj, dict) or not obj.get("name"):
        return False
    return "parameters" in obj or "arguments" in obj


def _strip_tool_json_objects_anywhere(text: str) -> str:
    """Remove tool-calling JSON objects (e.g. document_search) even if not on their own line."""
    decoder = json.JSONDecoder()
    i = 0
    out: list[str] = []
    n = len(text)
    while i < n:
        j = text.find("{", i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        tail = text[j:]
        ws = len(tail) - len(tail.lstrip())
        snippet = tail[ws : ws + 400]
        if '"name"' not in snippet and "'name'" not in snippet:
            out.append("{")
            i = j + 1
            continue
        try:
            obj, consumed = decoder.raw_decode(tail, ws)
        except json.JSONDecodeError:
            out.append("{")
            i = j + 1
            continue
        end = j + consumed
        if _is_tool_call_object(obj):
            i = end
            while i < n and text[i] in " \t\r\n":
                i += 1
        else:
            out.append(text[j:end])
            i = end
    return "".join(out).strip()


def _extract_embedded_block_list(text: str) -> tuple[str, str] | None:
    """If `text` embeds a Python-literal list of Gemini blocks, return (prefix, plain)."""
    idx = text.find("[{")
    if idx < 0:
        return None
    
    depth = 0
    in_string = False
    escape = False
    quote_char = ''
    
    for i in range(idx, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if in_string:
            if c == quote_char:
                in_string = False
        else:
            if c in ('"', "'"):
                in_string = True
                quote_char = c
            elif c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    chunk = text[idx:i+1]
                    try:
                        val = ast.literal_eval(chunk)
                        if isinstance(val, list) and (
                            not val or isinstance(val[0], dict) or isinstance(val[0], str)
                        ):
                            plain = _blocks_to_plain_text(val).strip()
                            prefix = text[:idx].strip()
                            return prefix, plain
                    except (ValueError, SyntaxError):
                        pass
                    break
    return None


def normalize_agent_output(raw: Any) -> str:
    """Executor `output` → plain user-visible string."""
    cur: Any = raw
    depth = 0
    while depth < 12:
        depth += 1
        if cur is None:
            return ""
        content = getattr(cur, "content", None)
        if content is not None:
            cur = content
            continue
        break

    base = _blocks_to_plain_text(cur).strip()
    extracted = _extract_embedded_block_list(base)
    if extracted is not None:
        prefix, plain = extracted
        base = "\n\n".join(x for x in (prefix, plain) if x).strip()

    base = _strip_tool_json_lines(base)
    base = _strip_tool_json_objects_anywhere(base)

    extracted2 = _extract_embedded_block_list(base)
    if extracted2 is not None:
        prefix, plain = extracted2
        base = "\n\n".join(x for x in (prefix, plain) if x).strip()

    base = _strip_tool_json_objects_anywhere(_strip_tool_json_lines(base))
    return base.strip()
