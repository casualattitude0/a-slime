"""Register StructuredTools for the agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from src.agent.text_utils import normalize_search_query
from src.tools import (
    make_calendar_delete_tool,
    make_calendar_tool,
    make_calendar_update_tool,
    make_local_datetime_tool,
    make_mac_calendar_create_tool,
    make_mac_calendar_delete_tool,
    make_mac_calendar_update_tool,
    make_memory_tools,
    make_reasoning_tool,
    make_shell_tool,
    make_subagent_tool,
    make_web_fetch_tool,
    make_web_search_tool,
)


class DocumentSearchArgs(BaseModel):
    query: str = Field(description="Keywords or question to search in the documents")

    @field_validator("query", mode="before")
    @classmethod
    def coerce_query(cls, v: Any) -> str:
        return normalize_search_query(v)


def build_tool_list(
    *,
    retriever: Any,
    chroma_path: Any,
    embeddings: Any,
    memory_collection: str,
    emit_reading_status: Any,
    include_local_datetime: bool,
) -> tuple[list[Any], Any]:
    """Returns (tools, save_to_memory_tool_or_none)."""

    def run_document_search(query: str) -> str:
        emit_reading_status()
        q = normalize_search_query(query)
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
        func=run_document_search,
        args_schema=DocumentSearchArgs,
    )

    web_search_tool = make_web_search_tool()
    web_fetch_tool = make_web_fetch_tool()
    shell_tool = make_shell_tool()
    memory_tools = make_memory_tools(chroma_path, embeddings, collection_name=memory_collection)
    reasoning_tool = make_reasoning_tool()
    subagent_tool = make_subagent_tool()
    calendar_create_tool = make_calendar_tool(
        chroma_dir=Path(chroma_path),
        embeddings=embeddings,
        memory_collection=memory_collection,
    )
    calendar_update_tool = make_calendar_update_tool(
        chroma_dir=Path(chroma_path),
        embeddings=embeddings,
        memory_collection=memory_collection,
    )
    calendar_delete_tool = make_calendar_delete_tool(
        chroma_dir=Path(chroma_path),
        embeddings=embeddings,
        memory_collection=memory_collection,
    )
    mac_calendar_create_tool = make_mac_calendar_create_tool(
        chroma_dir=Path(chroma_path),
        embeddings=embeddings,
        memory_collection=memory_collection,
    )
    mac_calendar_update_tool = make_mac_calendar_update_tool()
    mac_calendar_delete_tool = make_mac_calendar_delete_tool(
        chroma_dir=Path(chroma_path),
        embeddings=embeddings,
        memory_collection=memory_collection,
    )

    tools: list[Any] = [
        *memory_tools,
        web_search_tool,
        web_fetch_tool,
        *([make_local_datetime_tool()] if include_local_datetime else []),
        shell_tool,
        reasoning_tool,
        subagent_tool,
        retriever_tool,
        calendar_create_tool,
        calendar_update_tool,
        calendar_delete_tool,
        mac_calendar_create_tool,
        mac_calendar_update_tool,
        mac_calendar_delete_tool,
    ]
    save_memory_tool = next(
        (t for t in memory_tools if getattr(t, "name", "") == "save_to_memory"),
        None,
    )
    return tools, save_memory_tool


__all__ = ["DocumentSearchArgs", "build_tool_list"]
