"""Integration test: verify Gemini agent calls delegate_to_subagent."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


class TestGeminiAgentCallsSubAgent(unittest.TestCase):
    """Verify the Gemini-backed agent invokes delegate_to_subagent for delegatable tasks."""

    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv(ROOT / ".env")
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise unittest.SkipTest(
                "GOOGLE_API_KEY / GEMINI_API_KEY not set; skipping integration test."
            )

    def test_subagent_is_called(self) -> None:
        """Gemini agent should call delegate_to_subagent when instructed to delegate analysis."""
        subagent_calls: list[dict] = []

        def fake_subagent_func(
            task_description: str,
            context: str = "",
            data_path: str = "~/Developer/Agent/analysis_data",
            max_depth: int = 2,
        ) -> str:
            subagent_calls.append(
                {
                    "task_description": task_description,
                    "context": context,
                    "data_path": data_path,
                }
            )
            return json.dumps(
                {
                    "ok": True,
                    "depth": 1,
                    "data_path": data_path,
                    "result": (
                        "工具清單：make_web_search_tool, make_web_fetch_tool, "
                        "make_memory_tools, make_subagent_tool, make_shell_tool 等。"
                    ),
                },
                ensure_ascii=False,
            )

        from langchain_core.tools import StructuredTool

        from src.tools import DelegateToSubAgentArgs

        def patched_make_subagent_tool() -> StructuredTool:
            return StructuredTool.from_function(
                name="delegate_to_subagent",
                description=(
                    "Delegate a research/analysis task to a sub agent. "
                    "Use this when the task benefits from parallel decomposition, "
                    "iterative data collection, or deeper synthesis. "
                    "Provide task_description, optional context, and data_path for artifact storage."
                ),
                func=fake_subagent_func,
                args_schema=DelegateToSubAgentArgs,
            )

        with patch("src.agent.tools_registry.make_subagent_tool", patched_make_subagent_tool):
            from src.agent import build_executor, invoke_executor
            from src.agent.llm_factory import make_gemini_llm

            llm = make_gemini_llm()
            executor = build_executor(llm=llm)

            result = invoke_executor(
                executor,
                {
                    "input": (
                        "請使用 delegate_to_subagent 工具，"
                        "委派子 Agent 分析 src/tools.py 中定義了哪些工具函數，"
                        "並將分析結果整理到 analysis_data 目錄。"
                    ),
                    "chat_history": [],
                },
            )

        self.assertGreater(
            len(subagent_calls),
            0,
            "delegate_to_subagent was not called by the Gemini agent",
        )

        call = subagent_calls[0]
        self.assertIsInstance(call["task_description"], str)
        self.assertGreater(len(call["task_description"]), 0)

        from src.agent import normalize_agent_output

        raw_output = result.get("output", "")
        output = normalize_agent_output(raw_output)
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)


if __name__ == "__main__":
    unittest.main()
