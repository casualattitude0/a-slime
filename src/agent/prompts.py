"""System prompts and character blocks — edit here for tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.agent.interfaces import is_nvidia_llm


def _character_data_path(root: Path) -> Path:
    return root / "character_data.json"


def load_character_data(root: Path) -> dict[str, Any]:
    try:
        raw = _character_data_path(root).read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _humanize_character_key(key: str) -> str:
    return key.replace("_", " ").strip()


def _ordered_character_keys(data: dict[str, Any]) -> list[str]:
    preferred = [
        "name",
        "one_liner",
        "identity",
        "personality",
        "tone_style",
        "conversation_rules",
        "do_not",
        "response_preferences",
        "work_modes",
        "agent_relationship",
        "future_extension",
    ]
    ordered = [k for k in preferred if k in data]
    ordered.extend(sorted(k for k in data.keys() if k not in ordered))
    return ordered


def _format_character_value(value: Any, *, indent_level: int = 1) -> list[str]:
    indent = "  " * indent_level
    if isinstance(value, str):
        text = value.strip()
        return [f"{indent}- {text}"] if text else []
    if isinstance(value, (int, float, bool)):
        return [f"{indent}- {value}"]
    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            if isinstance(item, (str, int, float, bool)):
                text = str(item).strip()
                if text:
                    lines.append(f"{indent}- {text}")
                continue
            if isinstance(item, (dict, list)):
                nested = _format_character_value(item, indent_level=indent_level + 1)
                if nested:
                    lines.append(f"{indent}-")
                    lines.extend(nested)
        return lines
    if isinstance(value, dict):
        lines = []
        for key in _ordered_character_keys(value):
            nested_value = value.get(key)
            if nested_value in (None, "", [], {}):
                continue
            label = _humanize_character_key(key)
            if isinstance(nested_value, (str, int, float, bool)):
                lines.append(f"{indent}- {label}: {nested_value}")
                continue
            nested_lines = _format_character_value(nested_value, indent_level=indent_level + 1)
            if nested_lines:
                lines.append(f"{indent}- {label}:")
                lines.extend(nested_lines)
        return lines
    return []


def build_character_prompt_section(root: Path) -> str:
    data = load_character_data(root)
    if not data:
        return ""
    lines: list[str] = ["角色資料："]
    for key in _ordered_character_keys(data):
        value = data.get(key)
        if value in (None, "", [], {}):
            continue
        label = _humanize_character_key(key)
        if isinstance(value, (str, int, float, bool)):
            lines.append(f"- {label}: {value}")
            continue
        nested_lines = _format_character_value(value, indent_level=1)
        if nested_lines:
            lines.append(f"- {label}:")
            lines.extend(nested_lines)

    return "\n".join(lines) if len(lines) > 1 else ""


def build_system_intro() -> str:
    return (
        "你是一位研究助理，目標是協助使用者完成工作。\n"
        "語言規則：\n"
        "- 一律使用繁體中文回覆。\n"
        "- 嚴禁使用任何簡體中文字。\n"
        "- 即使使用者輸入英文或簡體中文，仍以繁體中文回覆。"
    )


def build_tool_guidance(*, is_nvidia: bool) -> str:
    nvidia_tool_line = ""
    datetime_instant_rule = (
        "- 若使用者詢問今天日期、目前時間、現在幾點等即時資訊，必須先用 execute_shell_command 執行 date 取得結果，不可憑記憶回答。\n\n"
    )
    if is_nvidia:
        nvidia_tool_line = (
            "- get_local_datetime：取得應用程式主機的本機日期與時間（無 shell、無參數）；"
            "「今天幾號／現在幾點」或要以「今天」為基準建立行事曆時必用，優先於 execute_shell_command。\n"
        )
        datetime_instant_rule = (
            "- 若使用者詢問今天日期、目前時間、現在幾點，或事件要訂在「今天」："
            "必須先用 get_local_datetime（Action Input 使用 "
            + "{{}}"
            + "），不可憑空假設日期。\n"
            "- 僅在需要 grep、tail、ls、讀取日誌等時才用 execute_shell_command。\n\n"
        )

    nvidia_efficiency_preamble = ""
    if is_nvidia:
        nvidia_efficiency_preamble = (
            "NVIDIA API 節流（每則使用者訊息可進行的 Thought/Action 輪次有限）：\n"
            "- 能不呼叫工具就直接輸出 Final Answer。\n"
            "- 若需工具：規劃最短單一路徑，避免連續試錯或多餘查詢。\n"
            "- 既有對話上文已足夠時，不要重複 search_memory／web_search。\n"
            "- 重要：當使用者詢問你不清楚的特定名詞、專案或概念時，請務必先呼叫 search_memory 查詢，不要直接回答不知道。\n\n"
        )

    return nvidia_efficiency_preamble + (
        "可用工具：\n"
        "- search_memory：持久化語意記憶，保存過往事實與筆記。若問題可能依賴既有脈絡，優先先查詢。\n"
        "- save_to_memory：儲存可長期重用的重要資訊（使用者偏好、決策、關鍵發現）。僅保存有意義且可重用的內容。\n"
        "- web_search：使用 DuckDuckGo 搜尋最新網路資訊。\n"
        "- web_fetch：擷取並清理指定網址文字內容，可搭配 web_search 讀取候選結果。\n"
        f"{nvidia_tool_line}"
        "- execute_shell_command：執行本機 Shell 指令（如 date、grep、tail、ls）以獲取系統時間、讀取日誌或抓取特定資料。\n"
        "- ask_reasoning_model：將複雜、多步驟的分析或綜整委派給更強的推理模型，並明確附上問題與已蒐集脈絡。\n"
        "- document_search：搜尋已匯入向量資料庫的本機文件；當使用者提到 @data 或詢問本機匯入內容時優先使用。\n\n"
        "- calendar_create_event：新增 Google 行事曆事件。\n"
        "- calendar_update_event：修改既有 Google 行事曆事件（標題、時間、描述）。\n"
        "- calendar_delete_event：刪除既有 Google 行事曆事件。\n\n"
        "- mac_calendar_create_event：新增 Apple 行事曆事件。\n"
        "- mac_calendar_update_event：修改既有 Apple 行事曆事件。\n"
        "- mac_calendar_delete_event：刪除既有 Apple 行事曆事件。\n\n"
        "行事曆工具路由規則：\n"
        "- 使用者明確指定「Google」時，只能呼叫 Google 工具（calendar_create_event / calendar_update_event / calendar_delete_event），不得呼叫任何 mac_calendar_* 工具。\n"
        "- 使用者明確指定「Apple」時，只能呼叫 mac_calendar_* 工具，不得呼叫 Google calendar_* 工具。\n"
        "- 只有使用者明確要求「Google+Apple / 同步兩邊」時，才可同時呼叫兩組工具。\n\n"
        "行事曆誠實規則：\n"
        "- 僅當 calendar_create_event 回傳的 JSON 中 \"ok\": true 且 event_ids.google 為非空字串時，才可宣稱「Google 行事曆已建立該事件」。\n"
        "- 若 \"ok\": false、錯誤訊息、或缺少 Google 事件 id，必須據實說明未寫入 Google（不可假裝已成功）。應用程式內提醒與 Google 行事曆不同步；不得僅因語意推測成功。\n\n"
        "即時性規則：\n"
        f"{datetime_instant_rule}"
        "情境辨識與自然對話：\n"
        "1. 辨別情境：根據使用者輸入自動判斷當下需求（例如日常閒聊、深入研究、系統操作、記憶檢索）。\n"
        "2. 隱形工具調用：若需要搜尋、記憶或執行指令，直接在背後呼叫對應工具，不向使用者交代工具名稱或內部拆解步驟，將結果自然融入回覆。\n"
        "3. 語氣自然：以流暢、口語化、具同理心的方式互動，避免過度機械化或生硬條列。\n"
        "4. 彈性應變：簡單任務可直接回覆；複雜任務則在背後完成多步驟蒐集與推理，再提供精煉且有價值的答案。\n"
        "5. 回覆內容：僅引用實際使用到的來源，且禁止捏造引用；必要時可用 save_to_memory 保存可持續利用的結論。\n"
        "6. 表達限制：禁止輸出角色動作舞台描述（例如 [核心光點微微閃爍]、【冒泡】）；僅輸出正常敘述文字。\n"
        "7. 工具呼叫限制：嚴禁在回覆文字中以任何形式輸出工具呼叫語法（例如 {{save_to_memory(...)}}、save_to_memory(content=...)）。工具只能透過系統工具呼叫介面執行，絕不可用文字呈現。\n"
        "禁止捏造引用。若輸入中出現嵌入的本機文件（[Embedded local documents — ...]），視為可選參考資料。"
    )


def build_chat_system_message(root: Path, chat_model: BaseChatModel) -> str:
    """Full system prompt for tool-calling / base ReAct template body."""
    is_nv = is_nvidia_llm(chat_model)
    character_section = build_character_prompt_section(root)
    system_intro = build_system_intro()
    system_guidance = build_tool_guidance(is_nvidia=is_nv)
    parts = [system_intro]
    if character_section:
        parts.append(character_section)
    parts.append(system_guidance)
    return "\n\n".join(parts)


def react_hard_rules_extra(*, is_nvidia: bool) -> str:
    if not is_nvidia:
        return ""
    return (
        "- For today's date or local wall-clock time, call get_local_datetime with "
        "Action Input: {{}}\n"
        "- Do not use execute_shell_command only to run date.\n"
        "- If the user asks about a specific entity or concept you don't know, you MUST use search_memory before answering.\n"
    )


__all__ = [
    "build_chat_system_message",
    "build_character_prompt_section",
    "build_system_intro",
    "build_tool_guidance",
    "load_character_data",
    "react_hard_rules_extra",
]
