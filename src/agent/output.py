"""Normalize executor output for display."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

_PYTHON_TOOL_CALL_RE = re.compile(
    r"\{[A-Za-z_]\w*\([^{}]*\)\}",
    re.DOTALL,
)


def blocks_to_plain_text(parts: Any) -> str:
    """Gemini/LC content blocks → plain text (drops extras/signature/tool metadata)."""
    if parts is None:
        return ""
    if isinstance(parts, str):
        return parts
    if isinstance(parts, list):
        chunks: list[str] = []
        for p in parts:
            t = blocks_to_plain_text(p)
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
        return blocks_to_plain_text(content)
    return str(parts) if parts else ""


def line_looks_like_tool_json(line: str) -> bool:
    s = line.strip()
    if not s.startswith("{"):
        return False
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return False
    return is_tool_call_object(obj)


def strip_tool_json_lines(text: str) -> str:
    lines = text.splitlines()
    kept = [ln for ln in lines if ln.strip() and not line_looks_like_tool_json(ln)]
    return "\n".join(kept).strip()


def is_tool_call_object(obj: Any) -> bool:
    if not isinstance(obj, dict) or not obj.get("name"):
        return False
    return "parameters" in obj or "arguments" in obj


def strip_tool_json_objects_anywhere(text: str) -> str:
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
        if is_tool_call_object(obj):
            i = end
            while i < n and text[i] in " \t\r\n":
                i += 1
        else:
            out.append(text[j:end])
            i = end
    return "".join(out).strip()


def extract_embedded_block_list(text: str) -> tuple[str, str] | None:
    """If `text` embeds a Python-literal list of Gemini blocks, return (prefix, plain)."""
    idx = text.find("[{")
    if idx < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    quote_char = ""

    for i in range(idx, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if in_string:
            if c == quote_char:
                in_string = False
        else:
            if c in ('"', "'"):
                in_string = True
                quote_char = c
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    chunk = text[idx : i + 1]
                    try:
                        val = ast.literal_eval(chunk)
                        if isinstance(val, list) and (
                            not val or isinstance(val[0], dict) or isinstance(val[0], str)
                        ):
                            plain = blocks_to_plain_text(val).strip()
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

    base = blocks_to_plain_text(cur).strip()
    extracted = extract_embedded_block_list(base)
    if extracted is not None:
        prefix, plain = extracted
        base = "\n\n".join(x for x in (prefix, plain) if x).strip()

    base = strip_tool_json_lines(base)
    base = strip_tool_json_objects_anywhere(base)

    extracted2 = extract_embedded_block_list(base)
    if extracted2 is not None:
        prefix, plain = extracted2
        base = "\n\n".join(x for x in (prefix, plain) if x).strip()

    base = strip_tool_json_objects_anywhere(strip_tool_json_lines(base))
    base = _PYTHON_TOOL_CALL_RE.sub("", base).strip()
    return base.strip()


__all__ = ["_PYTHON_TOOL_CALL_RE", "normalize_agent_output"]
