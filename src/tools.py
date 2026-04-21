from __future__ import annotations

import os
import json
import shlex
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

_MEMORY_COLLECTION = "agent_memory"
_ReminderSink = Callable[[dict[str, Any]], None]
_reminder_sink_lock = Lock()
_reminder_sink: _ReminderSink | None = None


def set_reminder_sink(sink: _ReminderSink | None) -> None:
    global _reminder_sink
    with _reminder_sink_lock:
        _reminder_sink = sink


def _emit_reminder(payload: dict[str, Any]) -> None:
    with _reminder_sink_lock:
        sink = _reminder_sink
    if sink is None:
        return
    try:
        sink(payload)
    except Exception:
        return


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


class CalendarCreateEventArgs(BaseModel):
    title: str = Field(description="Event title")
    start_at: str = Field(description="Event start datetime in ISO-8601")
    end_at: str = Field(description="Event end datetime in ISO-8601")
    timezone: str = Field(default="Asia/Taipei", description="IANA timezone")
    description: str = Field(default="", description="Event description")
    reminder_message: str = Field(default="", description="Reminder message for chat")
    source_session_id: str = Field(default="", description="Origin session id")


class CalendarUpdateEventArgs(BaseModel):
    event_id: str = Field(description="Google Calendar event id")
    title: str = Field(default="", description="Event title")
    start_at: str = Field(default="", description="Event start datetime in ISO-8601")
    end_at: str = Field(default="", description="Event end datetime in ISO-8601")
    timezone: str = Field(default="Asia/Taipei", description="IANA timezone")
    description: str = Field(default="", description="Event description")


class CalendarDeleteEventArgs(BaseModel):
    event_id: str = Field(description="Google Calendar event id")


def _normalize_iso_datetime(raw: str, timezone_name: str) -> datetime:
    s = (raw or "").strip()
    if not s:
        raise ValueError("datetime is required")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    parsed = datetime.fromisoformat(s)
    tz = ZoneInfo(timezone_name)
    if parsed.tzinfo is None:
        # Interpret naive datetime as local wall-clock time in requested timezone.
        parsed = parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)


def _create_google_calendar_event(
    *,
    title: str,
    start_at: datetime,
    end_at: datetime,
    timezone_name: str,
    description: str,
) -> dict[str, Any]:
    token = (os.environ.get("GOOGLE_CALENDAR_ACCESS_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("Missing GOOGLE_CALENDAR_ACCESS_TOKEN")
    calendar_id = (os.environ.get("GOOGLE_CALENDAR_ID") or "primary").strip() or "primary"
    payload = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start_at.isoformat(), "timeZone": timezone_name},
        "end": {"dateTime": end_at.isoformat(), "timeZone": timezone_name},
    }
    url = f"https://www.googleapis.com/calendar/v3/calendars/{quote(calendar_id, safe='')}/events"
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    return {"id": data.get("id", ""), "html_link": data.get("htmlLink", "")}


def _google_calendar_headers() -> tuple[str, dict[str, str]]:
    token = (os.environ.get("GOOGLE_CALENDAR_ACCESS_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("Missing GOOGLE_CALENDAR_ACCESS_TOKEN")
    calendar_id = (os.environ.get("GOOGLE_CALENDAR_ID") or "primary").strip() or "primary"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    return calendar_id, headers


def _update_google_calendar_event(
    *,
    event_id: str,
    title: str,
    start_at: datetime | None,
    end_at: datetime | None,
    timezone_name: str,
    description: str,
) -> dict[str, Any]:
    event_id_clean = (event_id or "").strip()
    if not event_id_clean:
        raise ValueError("event_id is required")
    calendar_id, headers = _google_calendar_headers()
    patch_payload: dict[str, Any] = {}
    if title.strip():
        patch_payload["summary"] = title.strip()
    if description.strip():
        patch_payload["description"] = description.strip()
    if start_at is not None:
        patch_payload["start"] = {"dateTime": start_at.isoformat(), "timeZone": timezone_name}
    if end_at is not None:
        patch_payload["end"] = {"dateTime": end_at.isoformat(), "timeZone": timezone_name}
    if not patch_payload:
        raise ValueError("no updatable fields provided")
    url = (
        "https://www.googleapis.com/calendar/v3/calendars/"
        f"{quote(calendar_id, safe='')}/events/{quote(event_id_clean, safe='')}"
    )
    with httpx.Client(timeout=20.0) as client:
        resp = client.patch(url, headers=headers, json=patch_payload)
        resp.raise_for_status()
        data = resp.json()
    return {"id": data.get("id", ""), "html_link": data.get("htmlLink", "")}


def _delete_google_calendar_event(*, event_id: str) -> None:
    event_id_clean = (event_id or "").strip()
    if not event_id_clean:
        raise ValueError("event_id is required")
    calendar_id, headers = _google_calendar_headers()
    url = (
        "https://www.googleapis.com/calendar/v3/calendars/"
        f"{quote(calendar_id, safe='')}/events/{quote(event_id_clean, safe='')}"
    )
    with httpx.Client(timeout=20.0) as client:
        resp = client.delete(url, headers=headers)
        resp.raise_for_status()


def _create_apple_calendar_event(
    *,
    title: str,
    start_at: datetime,
    end_at: datetime,
    timezone_name: str,
    description: str,
) -> dict[str, Any]:
    endpoint = (os.environ.get("APPLE_CALENDAR_API_URL") or "").strip()
    token = (os.environ.get("APPLE_CALENDAR_API_TOKEN") or "").strip()
    if not endpoint or not token:
        raise RuntimeError("Apple calendar API not configured")
    payload = {
        "title": title,
        "description": description,
        "start_at": start_at.isoformat(),
        "end_at": end_at.isoformat(),
        "timezone": timezone_name,
    }
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            endpoint,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
    return {"id": data.get("id", ""), "url": data.get("url", "")}


def make_calendar_tool() -> StructuredTool:
    def _create_event(
        title: str,
        start_at: str,
        end_at: str,
        timezone: str = "Asia/Taipei",
        description: str = "",
        reminder_message: str = "",
        source_session_id: str = "",
    ) -> str:
        t = (title or "").strip()
        if not t:
            return '{"ok": false, "error": "title is required"}'
        tz_name = (timezone or "Asia/Taipei").strip() or "Asia/Taipei"
        try:
            start_dt = _normalize_iso_datetime(start_at, tz_name)
            end_dt = _normalize_iso_datetime(end_at, tz_name)
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"invalid datetime: {str(exc)}"}, ensure_ascii=False)
        if end_dt <= start_dt:
            return json.dumps({"ok": False, "error": "end_at must be after start_at"}, ensure_ascii=False)
        desc = (description or "").strip()
        warning_messages: list[str] = []
        google_result: dict[str, Any] = {}
        apple_result: dict[str, Any] = {}

        try:
            google_result = _create_google_calendar_event(
                title=t,
                start_at=start_dt,
                end_at=end_dt,
                timezone_name=tz_name,
                description=desc,
            )
            google_success = True
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "google_success": False,
                    "apple_success": False,
                    "error": f"google calendar create failed: {str(exc)}",
                },
                ensure_ascii=False,
            )

        try:
            apple_result = _create_apple_calendar_event(
                title=t,
                start_at=start_dt,
                end_at=end_dt,
                timezone_name=tz_name,
                description=desc,
            )
            apple_success = True
        except Exception as exc:
            apple_success = False
            warning_messages.append(f"apple calendar create skipped/failed: {str(exc)}")

        _emit_reminder(
            {
                "title": t,
                "reminder_at": start_dt.isoformat(),
                "message": (reminder_message or "").strip() or f"提醒：{t}",
                "source_session_id": (source_session_id or "").strip(),
            }
        )
        result = {
            "ok": True,
            "google_success": google_success,
            "apple_success": apple_success,
            "event_ids": {
                "google": google_result.get("id", ""),
                "apple": apple_result.get("id", ""),
            },
            "event_links": {
                "google": google_result.get("html_link", ""),
                "apple": apple_result.get("url", ""),
            },
            "warnings": warning_messages,
        }
        return json.dumps(result, ensure_ascii=False)

    return StructuredTool.from_function(
        name="calendar_create_event",
        description=(
            "Create a calendar event in Google Calendar and attempt Apple Calendar sync. "
            "Google failure is fatal; Apple failure becomes warning. Also schedules an in-app reminder."
        ),
        func=_create_event,
        args_schema=CalendarCreateEventArgs,
    )


def make_calendar_update_tool() -> StructuredTool:
    def _update_event(
        event_id: str,
        title: str = "",
        start_at: str = "",
        end_at: str = "",
        timezone: str = "Asia/Taipei",
        description: str = "",
    ) -> str:
        tz_name = (timezone or "Asia/Taipei").strip() or "Asia/Taipei"
        start_dt: datetime | None = None
        end_dt: datetime | None = None
        if (start_at or "").strip():
            start_dt = _normalize_iso_datetime(start_at, tz_name)
        if (end_at or "").strip():
            end_dt = _normalize_iso_datetime(end_at, tz_name)
        if start_dt and end_dt and end_dt <= start_dt:
            return json.dumps({"ok": False, "error": "end_at must be after start_at"}, ensure_ascii=False)
        try:
            result = _update_google_calendar_event(
                event_id=event_id,
                title=title,
                start_at=start_dt,
                end_at=end_dt,
                timezone_name=tz_name,
                description=description,
            )
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"google calendar update failed: {str(exc)}"}, ensure_ascii=False)
        return json.dumps(
            {
                "ok": True,
                "event_id": result.get("id", ""),
                "event_link": result.get("html_link", ""),
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        name="calendar_update_event",
        description="Update an existing Google Calendar event by event_id.",
        func=_update_event,
        args_schema=CalendarUpdateEventArgs,
    )


def make_calendar_delete_tool() -> StructuredTool:
    def _delete_event(event_id: str) -> str:
        try:
            _delete_google_calendar_event(event_id=event_id)
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"google calendar delete failed: {str(exc)}"}, ensure_ascii=False)
        return json.dumps({"ok": True, "event_id": (event_id or "").strip()}, ensure_ascii=False)

    return StructuredTool.from_function(
        name="calendar_delete_event",
        description="Delete an existing Google Calendar event by event_id.",
        func=_delete_event,
        args_schema=CalendarDeleteEventArgs,
    )
