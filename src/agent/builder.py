"""Compose vector store, tools, and LangChain AgentExecutor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor
from langchain_community.vectorstores import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.agent.callbacks import emit_status
from src.agent.config import (
    _DEFAULT_RAG_COLLECTION,
    nvidia_agent_max_iterations,
    standard_agent_max_iterations,
)
from src.agent.interfaces import is_nvidia_llm, profile_for_chat_model
from src.agent.llm_factory import default_chat_model_from_env
from src.agent.paths import project_root
from src.agent.prefetch_executor import PrefetchExecutor
from src.agent.prompts import build_chat_system_message
from src.agent.rag import has_supported_data_files
from src.agent.strategies import build_agent_runnable
from src.agent.tools_registry import build_tool_list


def build_executor(
    chroma_dir: Path | None = None,
    *,
    llm: BaseChatModel | None = None,
    retriever_k: int = 4,
    memory_collection: str | None = None,
    rag_collection: str = _DEFAULT_RAG_COLLECTION,
    prefer_native_tool_agent: bool = False,
) -> AgentExecutor | PrefetchExecutor:
    from src.ingest import ingest as run_ingest

    root = project_root()
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
        if has_supported_data_files(data_dir):
            try:
                ingest_rc = run_ingest(
                    data_dir=data_dir.resolve(),
                    chroma_dir=chroma_path.resolve(),
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                if ingest_rc == 0:
                    vectorstore = Chroma(
                        persist_directory=str(chroma_path),
                        embedding_function=embeddings,
                        collection_name=rag_collection,
                    )
            except Exception:
                pass
    retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})

    chat_model = llm or default_chat_model_from_env()
    profile = profile_for_chat_model(chat_model)
    use_react = profile.use_react
    if prefer_native_tool_agent and profile.is_nvidia and use_react:
        use_react = False
    system_message = build_chat_system_message(root, chat_model)

    mem_col = memory_collection or "agent_memory"
    tools, save_memory_tool = build_tool_list(
        retriever=retriever,
        chroma_path=chroma_path,
        embeddings=embeddings,
        memory_collection=mem_col,
        emit_reading_status=lambda: emit_status("reading", "調閱文件"),
        include_local_datetime=is_nvidia_llm(chat_model),
    )

    agent = build_agent_runnable(
        chat_model,
        tools,
        system_message=system_message,
        use_react=use_react,
    )

    verbose = os.environ.get("AGENT_VERBOSE", "").lower() in ("1", "true", "yes")
    exec_max_iter = (
        nvidia_agent_max_iterations()
        if profile.is_nvidia
        else standard_agent_max_iterations()
    )
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=exec_max_iter,
    )
    return PrefetchExecutor(
        executor,
        retriever,
        is_ollama=profile.is_ollama,
        save_memory_tool=save_memory_tool,
    )


__all__ = ["build_executor"]
