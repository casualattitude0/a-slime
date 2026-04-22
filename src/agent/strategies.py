"""LangChain agent construction: ReAct vs native tool-calling."""

from __future__ import annotations

from typing import Any

from langchain_classic.agents import create_react_agent, create_tool_calling_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda

from src.agent.config import nvidia_react_history_tail_limit
from src.agent.interfaces import is_nvidia_llm
from src.agent.prompts import react_hard_rules_extra
from src.agent.text_utils import format_chat_history_as_text, maybe_trim_chat_messages


def build_agent_runnable(
    chat_model: BaseChatModel,
    tools: list[Any],
    *,
    system_message: str,
    use_react: bool,
) -> Any:
    if use_react:
        nv_extra = react_hard_rules_extra(is_nvidia=is_nvidia_llm(chat_model))
        react_template = (
            system_message
            + "\n\n---\n"
            "Hard rules:\n"
            + nv_extra
            + "- If the user asks to add or change Google Calendar events, you MUST run the matching "
            "calendar tool (calendar_create_event / calendar_update_event / calendar_delete_event) "
            "and use the Observation before saying the event was created or updated.\n"
            "- Never claim a calendar action succeeded without a successful Observation from that tool.\n"
            '- Never say Google Calendar created an event unless Observation JSON has "ok": true '
            "and a non-empty event_ids.google.\n"
            "- For calendar_create_event Action Input JSON, use separate top-level keys "
            '`title`, `start_at`, `end_at` (ISO-8601 strings); never embed the whole payload only inside `title`.\n\n'
            "You can use the following tools:\n{tools}\n\n"
            "If you need a tool, use EXACTLY this format:\n"
            "Thought: decide what to do next\n"
            "Action: one of [{tool_names}]\n"
            "Action Input: valid JSON for that tool\n"
            "Observation: tool result\n\n"
            "When you have enough information, respond EXACTLY:\n"
            "Thought: I now know the final answer\n"
            "Final Answer: [must be Traditional Chinese, no tool syntax]\n\n"
            "Conversation history:\n{chat_history}\n\n"
            "Question: {input}\n"
            "Thought:{agent_scratchpad}"
        )
        react_prompt = PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad", "chat_history"],
            template=react_template,
        )
        raw_react_agent = create_react_agent(chat_model, tools, react_prompt)

        def _preprocess_react(inputs: dict[str, Any]) -> dict[str, Any]:
            out = dict(inputs)
            hist = inputs.get("chat_history")
            if is_nvidia_llm(chat_model):
                hist = maybe_trim_chat_messages(hist, nvidia_react_history_tail_limit())
            out["chat_history"] = format_chat_history_as_text(hist)
            return out

        return RunnableLambda(_preprocess_react) | raw_react_agent

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    return create_tool_calling_agent(chat_model, tools, prompt)


__all__ = ["build_agent_runnable"]
