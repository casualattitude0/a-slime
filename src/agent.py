import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalize_search_query(raw: Any) -> str:
    """Ollama tool calls sometimes pass nested JSON or wrong-shaped args."""
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


def _make_llm() -> BaseChatModel:
    ollama_model = (os.environ.get("OLLAMA_MODEL") or "").strip()
    if ollama_model:
        from langchain_ollama import ChatOllama

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        return ChatOllama(
            model=ollama_model,
            base_url=base_url,
            temperature=0,
        )
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "For Gemini, set GOOGLE_API_KEY or GEMINI_API_KEY. "
            "For a local agent LLM, set OLLAMA_MODEL (Ollama must be running)."
        )
    return ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)


def build_executor(
    chroma_dir: Path | None = None,
    *,
    retriever_k: int = 4,
) -> AgentExecutor:
    root = _project_root()
    load_dotenv(root / ".env")
    chroma_path = (chroma_dir or (root / "chroma_db")).resolve()
    if not chroma_path.is_dir():
        raise FileNotFoundError(
            f"Chroma store not found at {chroma_path}. Run: python src/ingest.py"
        )

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
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})

    def _run_document_search(query: str) -> str:
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
            "Search for information in the local documents. Pass a clear search query "
            "or question as the query argument."
        ),
        func=_run_document_search,
        args_schema=DocumentSearchArgs,
    )

    llm = _make_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the document_search tool when the user "
                "asks about information that may be in the local document store.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    tools = [retriever_tool]
    agent = create_tool_calling_agent(llm, tools, prompt)
    verbose = os.environ.get("AGENT_VERBOSE", "").lower() in ("1", "true", "yes")
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=10,
    )
