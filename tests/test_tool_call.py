"""Tests: agent tool registration and output sanitization."""

from __future__ import annotations

import re
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub heavy deps so we can import src.agent without real API keys or Chrome
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__path__ = []  # type: ignore[attr-defined]
    return m


_STUBS = [
    "langchain_community",
    "langchain_community.vectorstores",
    "langchain_classic",
    "langchain_classic.agents",
    "langchain_google_genai",
    "chromadb",
]
for _s in _STUBS:
    if _s not in sys.modules:
        sys.modules[_s] = _make_stub(_s)

# Provide minimal symbol stubs
_lc_classic = sys.modules["langchain_classic"]
_lc_classic.agents = _make_stub("langchain_classic.agents")
_lc_classic.agents.AgentExecutor = MagicMock()
_lc_classic.agents.create_tool_calling_agent = MagicMock()
sys.modules["langchain_classic.agents"] = _lc_classic.agents

_lc_cv = sys.modules["langchain_community.vectorstores"]
_lc_cv.Chroma = MagicMock()
sys.modules["langchain_community.vectorstores"] = _lc_cv

_lg = sys.modules["langchain_google_genai"]
_lg.ChatGoogleGenerativeAI = MagicMock()
_lg.GoogleGenerativeAIEmbeddings = MagicMock()
sys.modules["langchain_google_genai"] = _lg

import src.agent as agent  # noqa: E402  (after stubs)
import src.tools as tools  # noqa: E402


# ---------------------------------------------------------------------------
# 1. normalize_agent_output: strips {func_name(...)} inline tool call syntax
# ---------------------------------------------------------------------------

class TestNormalizeOutputStripsInlineToolCalls(unittest.TestCase):

    def test_strips_save_to_memory_call(self):
        raw = (
            '好的，已經為您設置了明天12:00的會議提醒。有其他需要提前準備的事項嗎？\n'
            '{save_to_memory(content="明天12:00有開會", tags="提醒、會議")}'
        )
        result = agent.normalize_agent_output(raw)
        self.assertNotIn("save_to_memory", result)
        self.assertIn("會議提醒", result)

    def test_strips_search_memory_call(self):
        raw = 'Here is the answer. {search_memory(query="test")}'
        result = agent.normalize_agent_output(raw)
        self.assertNotIn("search_memory", result)
        self.assertIn("Here is the answer", result)

    def test_preserves_normal_text(self):
        raw = "明天12:00有開會，請記得準時出席。"
        result = agent.normalize_agent_output(raw)
        self.assertEqual(result, raw)

    def test_strips_call_only_output(self):
        raw = '{save_to_memory(content="test", tags="tag1")}'
        result = agent.normalize_agent_output(raw)
        self.assertEqual(result, "")

    def test_mixed_multiline(self):
        raw = (
            "好的，已儲存！\n"
            '{save_to_memory(content="會議", tags="提醒")}\n'
            "還有其他需要嗎？"
        )
        result = agent.normalize_agent_output(raw)
        self.assertNotIn("save_to_memory", result)
        self.assertIn("好的，已儲存", result)
        self.assertIn("還有其他需要嗎", result)


# ---------------------------------------------------------------------------
# 2. System prompt contains tool-call prohibition
# ---------------------------------------------------------------------------

class TestSystemPromptHasToolCallProhibition(unittest.TestCase):

    def test_prohibition_text_present(self):
        section = agent.build_executor.__code__.co_consts
        # Rebuild the prompt string via the same logic used at runtime
        with (
            patch("src.agent._make_llm", return_value=MagicMock()),
            patch("src.agent.Chroma") as mock_chroma,
            patch("src.agent.GoogleGenerativeAIEmbeddings", return_value=MagicMock()),
            patch("src.agent.run_ingest" if hasattr(agent, "run_ingest") else "src.agent._has_supported_data_files", return_value=False),
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        ):
            mock_col = MagicMock()
            mock_col.count.return_value = 0
            mock_chroma.return_value._collection = mock_col
            mock_chroma.return_value.as_retriever.return_value = MagicMock()
            mock_chroma.return_value._collection.get.return_value = {
                "ids": [], "documents": [], "metadatas": []
            }
            # We just need to check the system_guidance string constant
            # which is defined locally in build_executor; inspect source instead.
            pass

        src_text = Path(__file__).parents[1] / "src" / "agent.py"
        content = src_text.read_text(encoding="utf-8")
        self.assertIn("嚴禁在回覆文字中以任何形式輸出工具呼叫語法", content)
        self.assertIn("{{save_to_memory", content.split("嚴禁在回覆文字中以任何形式輸出工具呼叫語法")[1][:200])


# ---------------------------------------------------------------------------
# 3. make_memory_tools: returns correctly named StructuredTools
# ---------------------------------------------------------------------------

class TestMemoryToolsRegistration(unittest.TestCase):

    def setUp(self):
        mock_store = MagicMock()
        mock_store.add_texts.return_value = ["mem-abc"]
        mock_store.similarity_search.return_value = []
        self._store_patch = patch("src.tools._memory_store", return_value=mock_store)
        self._store_patch.start()

    def tearDown(self):
        self._store_patch.stop()

    def test_returns_two_tools(self):
        result = tools.make_memory_tools(Path("/tmp/fake"), MagicMock())
        self.assertEqual(len(result), 2)

    def test_tool_names(self):
        result = tools.make_memory_tools(Path("/tmp/fake"), MagicMock())
        names = {t.name for t in result}
        self.assertIn("save_to_memory", names)
        self.assertIn("search_memory", names)

    def test_save_to_memory_executes(self):
        result = tools.make_memory_tools(Path("/tmp/fake"), MagicMock())
        save = next(t for t in result if t.name == "save_to_memory")
        out = save.func("明天12:00有開會", tags="提醒、會議")
        self.assertIn("Saved memory", out)

    def test_save_to_memory_rejects_empty(self):
        result = tools.make_memory_tools(Path("/tmp/fake"), MagicMock())
        save = next(t for t in result if t.name == "save_to_memory")
        out = save.func("")
        self.assertIn("Empty", out)

    def test_search_memory_executes(self):
        result = tools.make_memory_tools(Path("/tmp/fake"), MagicMock())
        search = next(t for t in result if t.name == "search_memory")
        out = search.func("開會")
        self.assertIn("No matching", out)


# ---------------------------------------------------------------------------
# 4. _PYTHON_TOOL_CALL_RE pattern correctness
# ---------------------------------------------------------------------------

class TestPythonToolCallRegex(unittest.TestCase):

    RE = agent._PYTHON_TOOL_CALL_RE

    def test_matches_save_to_memory(self):
        s = '{save_to_memory(content="test", tags="t1")}'
        self.assertTrue(self.RE.search(s))

    def test_matches_search_memory(self):
        s = '{search_memory(query="會議")}'
        self.assertTrue(self.RE.search(s))

    def test_does_not_match_normal_braces(self):
        s = "{'key': 'value'}"
        self.assertIsNone(self.RE.search(s))

    def test_does_not_match_empty_braces(self):
        self.assertIsNone(self.RE.search("{}"))

    def test_strips_correctly(self):
        s = '回覆文字 {save_to_memory(content="x")} 繼續'
        result = self.RE.sub("", s).strip()
        self.assertEqual(result, "回覆文字  繼續".strip())


# ---------------------------------------------------------------------------
# 5. Calendar tool args: LLM sometimes stuffs JSON into a single field
# ---------------------------------------------------------------------------

class TestCalendarCreateEventArgsCoercion(unittest.TestCase):

    def test_json_object_in_title_field(self):
        m = tools.CalendarCreateEventArgs.model_validate(
            {
                "title": (
                    '{"title": "會議", "start_at": "2026-04-22T10:00:00+08:00", '
                    '"end_at": "2026-04-22T11:00:00+08:00"}'
                ),
            }
        )
        self.assertEqual(m.title, "會議")
        self.assertIn("2026-04-22T10", m.start_at)
        self.assertIn("2026-04-22T11", m.end_at)

    def test_truncated_json_uses_loose_quoted_keys(self):
        m = tools.CalendarCreateEventArgs.model_validate(
            {
                "title": (
                    '{"title": "開會", "start_at": "2026-04-22T10:00:00+08:00", '
                    '"end_at": "2026-04-22T11:00:00+08:00", "extra": "x"'
                ),
            }
        )
        self.assertEqual(m.title, "開會")
        self.assertIn("10:00:00", m.start_at)

    def test_two_iso_substrings_in_title(self):
        m = tools.CalendarCreateEventArgs.model_validate(
            {
                "title": "x 2026-04-22T10:00:00+08:00 y 2026-04-22T11:00:00+08:00",
            }
        )
        self.assertIn("2026-04-22T10", m.start_at)
        self.assertIn("2026-04-22T11", m.end_at)


class TestCalendarUpdateEventArgsCoercion(unittest.TestCase):

    def test_json_in_event_id(self):
        m = tools.CalendarUpdateEventArgs.model_validate(
            {
                "event_id": '{"event_id": "ev1", "title": "新標題"}',
            }
        )
        self.assertEqual(m.event_id, "ev1")
        self.assertEqual(m.title, "新標題")


if __name__ == "__main__":
    unittest.main()
