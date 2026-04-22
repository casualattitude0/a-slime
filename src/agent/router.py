"""Gemini escalation router, Ollama local quick reply, agent-mode rules."""

from __future__ import annotations

import ast
import json
import os
import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agent.llm_factory import make_ollama_llm
from src.agent.paths import project_root


def should_escalate_to_gemini(user_message: str, history_message_count: int) -> bool:
    """When Ollama is available and Gemini API key exists, ask Gemini whether to escalate."""
    root = project_root()
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


def safe_eval_math(expr: str) -> str | None:
    s = (expr or "").strip()
    if not s or len(s) > 80:
        return None
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\%\(\)]+", s):
        return None
    try:
        node = ast.parse(s, mode="eval")
    except Exception:
        return None

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Pow,
        ast.FloorDiv,
    )
    if any(not isinstance(n, allowed_nodes) for n in ast.walk(node)):
        return None

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = _eval(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp):
            l = _eval(n.left)
            r = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return l + r
            if isinstance(n.op, ast.Sub):
                return l - r
            if isinstance(n.op, ast.Mult):
                return l * r
            if isinstance(n.op, ast.Div):
                return l / r
            if isinstance(n.op, ast.Mod):
                return l % r
            if isinstance(n.op, ast.FloorDiv):
                return l // r
            if isinstance(n.op, ast.Pow):
                return l**r
        raise ValueError("Unsupported expression")

    try:
        out = _eval(node)
    except Exception:
        return None
    if float(out).is_integer():
        return str(int(out))
    return str(out)


def last_user_message(history: list, current_message: str) -> str | None:
    cur = (current_message or "").strip()
    for m in reversed(history or []):
        content = getattr(m, "content", None)
        if not isinstance(content, str):
            continue
        text = content.strip()
        if not text:
            continue
        if text != cur:
            return text
    return None


def looks_like_identity_request(message: str) -> bool:
    s = (message or "").strip().lower()
    if not s:
        return False
    cues = (
        "自我介紹",
        "介紹一下你自己",
        "介紹你自己",
        "你是誰",
        "你的名字",
        "你叫什麼",
        "你是哪個模型",
        "who are you",
        "introduce yourself",
        "what is your name",
    )
    return any(c in s for c in cues)


def agent_mode_reply(user_message: str, history: list | None = None) -> str:
    """Handle requests in 'agent' mode without using an LLM."""
    msg = (user_message or "").strip()
    if not msg:
        return "請輸入內容。"

    low = msg.lower()
    if low in {"hi", "hello", "hey", "嗨", "你好", "哈囉"}:
        return "你好，我是 Agent 模式（不使用 LLM）。"
    if low in {"thanks", "thank you", "謝謝", "感謝"}:
        return "不客氣。"
    if low in {"bye", "掰掰", "再見"}:
        return "再見。"

    math_result = safe_eval_math(msg)
    if math_result is not None:
        return math_result

    if any(k in low for k in ("今天", "現在", "日期", "時間", "幾點", "today", "current time", "current date")):
        from datetime import datetime

        now = datetime.now()
        return f"目前時間：{now.strftime('%Y-%m-%d %H:%M:%S')}"

    if any(k in low for k in ("你認為", "你覺得", "怎麼看", "what do you think")):
        prev = last_user_message(history or [], msg)
        if prev:
            return f"如果你是指上一句「{prev}」，我可以幫你列出優缺點；請告訴我你要比較的選項。"
        return "請先提供主題或選項，我可以用規則幫你做優缺點比較。"

    if len(msg) <= 12 and ("?" in msg or "？" in msg):
        return "問題內容不夠完整，請補充主題、目標或限制條件。"

    return "Agent 模式不使用 LLM。請給我更具體的目標，我可提供規則化協助（算式、時間、簡單比較、格式整理）。"


def local_quick_reply(user_message: str, history: list) -> str | None:
    """Use local Ollama model to decide simple-vs-complex and optionally answer directly."""
    from langchain_core.messages import AIMessage as _AI, HumanMessage as _HM

    root = project_root()
    load_dotenv(root / ".env")
    enabled = (os.environ.get("AGENT_LOCAL_ROUTER_ENABLED") or "1").strip().lower()
    if enabled not in ("1", "true", "yes", "on"):
        return None
    if not (os.environ.get("OLLAMA_MODEL") or "").strip():
        return None
    msg = (user_message or "").strip()
    if not msg:
        return None
    if looks_like_identity_request(msg):
        return None

    history_message_count = len(history) if history else 0

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
        "Language rule:\n"
        '- If "simple" is true and you provide "reply", reply must be in zh-TW (Traditional Chinese).\n'
        "- Do not use Simplified Chinese.\n\n"
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


__all__ = [
    "agent_mode_reply",
    "local_quick_reply",
    "should_escalate_to_gemini",
]
