"""
Unit tests for LangGraph Action Intent Fallback (src/agent/graph.py).
Verifies that direct user action commands ('open notepad', 'type hello', 'press enter', math, datetime)
are reliably converted into structured tool calls even when the model outputted plain text narration.
"""

import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from src.agent.graph import create_llm_node, extract_action_intent_fallback


class TestAgentActionFallback(unittest.TestCase):
    """Test suite for extract_action_intent_fallback and narration-recovery."""

    def test_fallback_open_application(self):
        """User asking 'open notepad' synthesizes open_application tool call."""
        tc = extract_action_intent_fallback("open notepad")
        self.assertEqual(len(tc), 1)
        self.assertEqual(tc[0]["name"], "open_application")
        self.assertEqual(tc[0]["args"]["app_name"], "notepad")

        tc_launch = extract_action_intent_fallback("launch google chrome")
        self.assertEqual(len(tc_launch), 1)
        self.assertEqual(tc_launch[0]["name"], "open_application")
        self.assertEqual(tc_launch[0]["args"]["app_name"], "google chrome")

    def test_fallback_desktop_typing(self):
        """User asking 'type hello world' synthesizes type_text."""
        tc1 = extract_action_intent_fallback("type hello world")
        self.assertEqual(len(tc1), 1)
        self.assertEqual(tc1[0]["name"], "type_text")
        self.assertEqual(tc1[0]["args"]["text"], "hello world")
        self.assertFalse(tc1[0]["args"]["press_enter"])

        tc2 = extract_action_intent_fallback("type meeting notes and press enter")
        self.assertEqual(len(tc2), 1)
        self.assertEqual(tc2[0]["name"], "type_text")
        self.assertEqual(tc2[0]["args"]["text"], "meeting notes")
        self.assertTrue(tc2[0]["args"]["press_enter"])

        tc3 = extract_action_intent_fallback("type send test and send it")
        self.assertEqual(len(tc3), 1)
        self.assertEqual(tc3[0]["name"], "type_text")
        self.assertEqual(tc3[0]["args"]["text"], "send test")
        self.assertTrue(tc3[0]["args"]["press_enter"])

    def test_fallback_press_enter(self):
        """User asking 'press enter' or 'send it' synthesizes press_enter_key."""
        tc = extract_action_intent_fallback("press enter")
        self.assertEqual(len(tc), 1)
        self.assertEqual(tc[0]["name"], "press_enter_key")

        tc_send = extract_action_intent_fallback("send it")
        self.assertEqual(len(tc_send), 1)
        self.assertEqual(tc_send[0]["name"], "press_enter_key")

    def test_fallback_datetime(self):
        """User asking date or time synthesizes get_current_datetime."""
        for query in ["what time is it", "what is the date today", "what day is today"]:
            tc = extract_action_intent_fallback(query)
            self.assertEqual(len(tc), 1, f"Failed for {query}")
            self.assertEqual(tc[0]["name"], "get_current_datetime")

    def test_fallback_math(self):
        """User asking percentage or arithmetic calculation synthesizes run_python."""
        tc = extract_action_intent_fallback("what is 358% of 340")
        self.assertEqual(len(tc), 1)
        self.assertEqual(tc[0]["name"], "run_python")
        self.assertIn("340.0 * 3.58", tc[0]["args"]["code"])

    def test_fallback_ignores_conversational_queries(self):
        """Conversational facts, preferences, or questions do NOT trigger action fallbacks."""
        for query in [
            "my favorite color is blue",
            "hello jarvis",
            "what did we talk about earlier",
            "who was the first president",
            "thank you very much",
        ]:
            tc = extract_action_intent_fallback(query)
            self.assertEqual(tc, [], f"Expected empty for conversational query: {query}")

    def test_llm_node_recovers_narrated_open_notepad(self):
        """Verifies llm_node attaches open_application tool call when model narrating 'Opening Notepad.'"""
        mock_llm = MagicMock()
        # Model returns narrative text instead of structured tool call
        mock_llm.invoke.return_value = AIMessage(content="Opening Notepad.")

        node = create_llm_node(llm=mock_llm, enable_memory=False)
        state = {"messages": [HumanMessage(content="open notepad")]}

        result = node(state)
        response_msg = result["messages"][0]

        # Verify fallback attached structured tool call
        self.assertTrue(hasattr(response_msg, "tool_calls"))
        self.assertEqual(len(response_msg.tool_calls), 1)
        self.assertEqual(response_msg.tool_calls[0]["name"], "open_application")
        self.assertEqual(response_msg.tool_calls[0]["args"]["app_name"], "notepad")

    def test_system_prompt_forbids_fabricating_actions_when_no_tool_called(self):
        """Verifies that DEFAULT_SYSTEM_PROMPT strictly forbids fabricating action outcomes when no tool was called."""
        from src.agent.graph import DEFAULT_SYSTEM_PROMPT

        self.assertIn("NEVER FABRICATE ACTION OUTCOMES WHEN NO TOOL WAS CALLED", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("NEVER claim or imply that any action-related outcome occurred", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Typing was cancelled into", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("must ONLY EVER appear when a corresponding live tool result (ToolMessage) actually exists", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("If NO tool was executed in this turn, you MUST NEVER mention typing", DEFAULT_SYSTEM_PROMPT)

    def test_llm_node_redirects_run_python_datetime_to_get_current_datetime(self):
        """Verifies llm_node automatically redirects run_python with datetime import to get_current_datetime."""
        mock_llm = MagicMock()
        mock_response = AIMessage(
            content="",
            tool_calls=[{
                "name": "run_python",
                "args": {"code": "import datetime\nprint(datetime.datetime.now())"},
                "id": "call_12345",
                "type": "tool_call",
            }],
        )
        mock_llm.invoke.return_value = mock_response

        node = create_llm_node(llm=mock_llm, enable_memory=False)
        state = {"messages": [HumanMessage(content="what time is it")]}

        result = node(state)
        response_msg = result["messages"][0]

        self.assertEqual(len(response_msg.tool_calls), 1)
        self.assertEqual(response_msg.tool_calls[0]["name"], "get_current_datetime")
        self.assertEqual(response_msg.tool_calls[0]["args"], {})

    def test_fallback_redirects_json_run_python_datetime(self):
        """Verifies extract_fallback_tool_calls redirects JSON tool call with datetime code to get_current_datetime."""
        from src.agent.graph import extract_fallback_tool_calls

        raw_json = '{"name": "run_python", "args": {"code": "from datetime import datetime; datetime.now()"}}'
        cleaned, calls = extract_fallback_tool_calls(raw_json)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_current_datetime")
        self.assertEqual(calls[0]["args"], {})

    def test_system_prompt_date_time_rules(self):
        """Verifies that DEFAULT_SYSTEM_PROMPT strictly designates get_current_datetime and forbids run_python."""
        from src.agent.graph import DEFAULT_SYSTEM_PROMPT

        self.assertIn("ALWAYS call the `get_current_datetime` tool for any questions asking for the current date", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("NEVER call `run_python` or `web_search` for date or time questions", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("NEVER use `run_python` for date or time queries (use `get_current_datetime` instead)", DEFAULT_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
