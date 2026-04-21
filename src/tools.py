from __future__ import annotations

import os
import shlex
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

_MEMORY_COLLECTION = "agent_memory"


def _memory_store_for(
    chroma_dir: Path,
    embeddings: GoogleGenerativeAIEmbeddings,
    collection_name: str,
) -> Chroma:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def list_memory_items(
    chroma_dir: Path,
    embeddings: GoogleGenerativeAIEmbeddings,
    collection_name: str = _MEMORY_COLLECTION,
) -> list[dict]:
    try:
        store = _memory_store_for(chroma_dir, embeddings, collection_name)
        result = store._collection.get(include=["documents", "metadatas"])
        items = []
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        for i, mem_id in enumerate(ids):
            items.append(
                {
                    "id": mem_id,
                    "content": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                }
            )
        return items
    except Exception as exc:
        return [{"error": str(exc)}]


def delete_memory_item(
    chroma_dir: Path,
    embeddings: GoogleGenerativeAIEmbeddings,
    item_id: str,
    collection_name: str = _MEMORY_COLLECTION,
) -> bool:
    try:
        store = _memory_store_for(chroma_dir, embeddings, collection_name)
        store._collection.delete(ids=[item_id])
        return True
    except Exception:
        return False


def delete_all_memory_items(
    chroma_dir: Path,
    embeddings: GoogleGenerativeAIEmbeddings,
    collection_name: str = _MEMORY_COLLECTION,
) -> bool:
    try:
        store = _memory_store_for(chroma_dir, embeddings, collection_name)
        result = store._collection.get(include=[])
        ids = result.get("ids") or []
        if ids:
            store._collection.delete(ids=ids)
        return True
    except Exception:
        return False


class WebSearchArgs(BaseModel):
    query: str = Field(description="Web search query")
    max_results: int = Field(default=5, description="Max results (1-10)")


def _run_web_search(query: str, max_results: int = 5) -> str:
    q = (query or "").strip()
    if not q:
        return "Empty search query."
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "Web search unavailable: install 'ddgs' (pip install ddgs)."
    n = max(1, min(int(max_results or 5), 10))
    rows: list[str] = []
    try:
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(q, max_results=n), 1):
                title = r.get("title") or ""
                url = r.get("href") or r.get("url") or ""
                body = r.get("body") or ""
                rows.append(f"{i}. {title}\n   {url}\n   {body}")
    except Exception as exc:
        return f"Web search failed: {exc}"
    return "\n\n".join(rows) if rows else "No results."


def make_web_search_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="web_search",
        description=(
            "Search the public web (DuckDuckGo) for fresh information. Returns "
            "titles, URLs, and snippets. Use for current events, facts not in "
            "memory, or to discover URLs to fetch with web_fetch."
        ),
        func=_run_web_search,
        args_schema=WebSearchArgs,
    )


class WebFetchArgs(BaseModel):
    url: str = Field(description="Absolute http(s) URL to fetch")


def _run_web_fetch(url: str) -> str:
    u = (url or "").strip()
    if not u.startswith(("http://", "https://")):
        return "URL must start with http:// or https://"
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=20.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"},
        ) as client:
            resp = client.get(u)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "").lower()
            body = resp.text
    except Exception as exc:
        return f"Fetch failed: {exc}"

    if "html" in ct or "<html" in body[:500].lower():
        soup = BeautifulSoup(body, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
    else:
        text = body

    if len(text) > 12000:
        text = text[:12000] + "\n... [truncated]"
    return text or "Empty response."


def make_web_fetch_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="web_fetch",
        description=(
            "Fetch a URL and return readable text (HTML stripped). Use after "
            "web_search to read the contents of a specific page."
        ),
        func=_run_web_fetch,
        args_schema=WebFetchArgs,
    )


def _memory_store(
    chroma_dir: Path,
    embeddings: GoogleGenerativeAIEmbeddings,
    collection_name: str = _MEMORY_COLLECTION,
) -> Chroma:
    return _memory_store_for(chroma_dir, embeddings, collection_name)


class SaveMemoryArgs(BaseModel):
    content: str = Field(description="Fact or note to store for future recall")
    tags: str = Field(default="", description="Optional comma-separated tags")


class SearchMemoryArgs(BaseModel):
    query: str = Field(description="Question or keywords to search remembered facts")
    k: int = Field(default=5, description="Max number of results (1-10)")


def make_memory_tools(
    chroma_dir: Path,
    embeddings: GoogleGenerativeAIEmbeddings,
    collection_name: str = _MEMORY_COLLECTION,
) -> list[StructuredTool]:
    store = _memory_store(chroma_dir, embeddings, collection_name)

    def _save(content: str, tags: str = "") -> str:
        c = (content or "").strip()
        if not c:
            return "Empty memory content."
        meta: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tags": (tags or "").strip(),
        }
        mem_id = f"mem-{uuid.uuid4()}"
        try:
            store.add_texts(texts=[c], metadatas=[meta], ids=[mem_id])
        except Exception as exc:
            return f"Memory save failed: {exc}"
        return f"Saved memory ({mem_id})."

    def _search(query: str, k: int = 5) -> str:
        q = (query or "").strip()
        if not q:
            return "Empty memory query."
        n = max(1, min(int(k or 5), 10))
        try:
            docs = store.similarity_search(q, k=n)
        except Exception as exc:
            return f"Memory search failed: {exc}"
        if not docs:
            return "No matching memories."
        rows: list[str] = []
        for i, d in enumerate(docs, 1):
            md = d.metadata or {}
            ts = md.get("ts", "")
            tags = md.get("tags", "")
            head_parts = [p for p in (ts, f"[{tags}]" if tags else "") if p]
            head = " ".join(head_parts)
            rows.append(f"{i}. {head}\n{d.page_content}".strip())
        return "\n\n".join(rows)

    save_tool = StructuredTool.from_function(
        name="save_to_memory",
        description=(
            "Persist a concrete fact, decision, or piece of context for future "
            "sessions. Use sparingly for durable, useful information the user "
            "or agent will benefit from recalling later."
        ),
        func=_save,
        args_schema=SaveMemoryArgs,
    )
    search_tool = StructuredTool.from_function(
        name="search_memory",
        description=(
            "Search the agent's persistent memory for previously saved facts "
            "or notes. Call BEFORE web_search when a question may rely on "
            "earlier context."
        ),
        func=_search,
        args_schema=SearchMemoryArgs,
    )
    return [save_tool, search_tool]


class AskReasoningArgs(BaseModel):
    question: str = Field(
        description="Complex question or task to delegate to a reasoning LLM"
    )
    context: str = Field(default="", description="Optional background information")


def make_reasoning_tool() -> StructuredTool:
    def _ask(question: str, context: str = "") -> str:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "Reasoning model unavailable: no Gemini API key."
        model = (
            (os.environ.get("GEMINI_REASONING_MODEL") or "").strip()
            or (os.environ.get("GEMINI_PRO_MODEL") or "").strip()
            or "gemini-2.5-pro"
        )
        try:
            llm = ChatGoogleGenerativeAI(
                model=model, temperature=0, google_api_key=api_key
            )
            prompt = (
                "You are a careful reasoning assistant. Think step by step and "
                "produce a precise, well-justified answer.\n\n"
                f"Context:\n{(context or '').strip() or '(none)'}\n\n"
                f"Question:\n{(question or '').strip()}"
            )
            resp = llm.invoke(prompt)
            text = getattr(resp, "content", None)
            if isinstance(text, list):
                parts = []
                for p in text:
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        parts.append(p["text"])
                    elif isinstance(p, str):
                        parts.append(p)
                return "\n\n".join(parts).strip()
            return str(text or resp).strip()
        except Exception as exc:
            return f"Reasoning model error: {exc}"

    return StructuredTool.from_function(
        name="ask_reasoning_model",
        description=(
            "Delegate a complex analytical question to a stronger reasoning "
            "LLM (e.g., Gemini 2.5 Pro). Use for deep analysis, multi-step "
            "logic, or synthesizing large context. Pass relevant context "
            "explicitly in the context argument."
        ),
        func=_ask,
        args_schema=AskReasoningArgs,
    )


class ShellCommandArgs(BaseModel):
    command: str = Field(description="Shell command to execute locally")


def _run_shell_command(command: str) -> str:
    cmd = (command or "").strip()
    if not cmd:
        return "Empty shell command."
    try:
        # Validate basic shell syntax early for clearer feedback.
        shlex.split(cmd)
    except ValueError as exc:
        return f"Invalid shell command: {exc}"
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return "Shell command timed out after 10 seconds."
    except Exception as exc:
        return f"Shell command failed: {exc}"

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    parts: list[str] = [f"exit_code: {result.returncode}"]
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    if len(parts) == 1:
        parts.append("No output.")
    text = "\n\n".join(parts)
    if len(text) > 10000:
        text = text[:10000] + "\n\n... [truncated]"
    return text


def make_shell_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="execute_shell_command",
        description=(
            "Execute local shell commands (e.g. date, grep, tail, ls) to retrieve "
            "system data, inspect logs, or extract specific text."
        ),
        func=_run_shell_command,
        args_schema=ShellCommandArgs,
    )
