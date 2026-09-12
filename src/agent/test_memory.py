"""
Unit and Integration Tests for Local Vector Memory (ChromaDB - Phase 6)

Validates:
1. Core vector memory operations: store_exchange, retrieve_relevant_context, clear_memory, format_context_for_prompt.
2. Handling edge cases: empty strings, whitespace, empty collection, top_k limits.
3. Cross-process persistence: process 1 stores a fact to disk, exits, and a completely fresh
   process 2 opens the database and retrieves the stored fact.
4. LangGraph memory integration: verifies relevant past context is injected into LLM prompt
   and completed turns are saved into vector memory.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.graph import create_llm_node
from src.agent.memory import (
    clear_memory,
    format_context_for_prompt,
    get_memory_client,
    get_memory_collection,
    retrieve_relevant_context,
    store_exchange,
)


class TestVectorMemoryCore(unittest.TestCase):
    """Tests core operations of the vector memory module."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_chroma_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_and_retrieve_basic(self):
        """Verifies that an exchange is stored with metadata and retrieved via semantic similarity."""
        doc_id = store_exchange(
            user_message="my favorite programming language is Python",
            assistant_response="Python is a fantastic language for AI and scripting.",
            metadata={"category": "preferences"},
            persist_directory=self.temp_dir,
        )
        self.assertIsNotNone(doc_id)
        self.assertTrue(doc_id.startswith("exchange_"))

        results = retrieve_relevant_context(
            query="what programming language do I prefer?",
            top_k=1,
            persist_directory=self.temp_dir,
        )
        self.assertEqual(len(results), 1)
        retrieved = results[0]
        self.assertEqual(retrieved["id"], doc_id)
        self.assertIn("Python", retrieved["text"])
        self.assertEqual(retrieved["user_message"], "my favorite programming language is Python")
        self.assertEqual(retrieved["assistant_response"], "Python is a fantastic language for AI and scripting.")
        self.assertIsNotNone(retrieved["timestamp"])
        self.assertIsNotNone(retrieved["distance"])

    def test_top_k_limits(self):
        """Verifies that retrieve_relevant_context respects the top_k parameter."""
        store_exchange("I have a golden retriever named Max", "Dogs are loyal companions.", persist_directory=self.temp_dir)
        store_exchange("I enjoy playing chess on weekends", "Chess is great for strategic thinking.", persist_directory=self.temp_dir)
        store_exchange("My favorite breakfast is oatmeal with blueberries", "Healthy and energetic start to the day.", persist_directory=self.temp_dir)

        results_k1 = retrieve_relevant_context("tell me about my dog", top_k=1, persist_directory=self.temp_dir)
        self.assertEqual(len(results_k1), 1)
        self.assertIn("Max", results_k1[0]["text"])

        results_k2 = retrieve_relevant_context("what are my hobbies and pets?", top_k=2, persist_directory=self.temp_dir)
        self.assertEqual(len(results_k2), 2)

    def test_empty_and_blank_handling(self):
        """Verifies that empty queries or blank exchanges are handled gracefully."""
        # Blank storage returns None
        self.assertIsNone(store_exchange("", "", persist_directory=self.temp_dir))
        self.assertIsNone(store_exchange("   ", "\t\n", persist_directory=self.temp_dir))

        # Blank query returns empty list
        self.assertEqual(retrieve_relevant_context("", top_k=3, persist_directory=self.temp_dir), [])
        self.assertEqual(retrieve_relevant_context("   ", top_k=3, persist_directory=self.temp_dir), [])

        # Querying an empty database returns empty list
        empty_dir = tempfile.mkdtemp(prefix="test_chroma_empty_")
        try:
            self.assertEqual(retrieve_relevant_context("hello", top_k=3, persist_directory=empty_dir), [])
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)

    def test_store_exchange_failure_filtering(self):
        """Verifies that store_exchange does not persist responses containing failure, uncertainty, corruption, or ephemeral time queries."""
        failure_cases = [
            ("what is my favorite food?", "I don't have any information about your food preferences."),
            ("what did I say earlier?", "I do not have any information on that topic."),
            ("who is my manager?", "I'm not sure who your manager is."),
            ("can you find my file?", "I couldn't locate that file on disk."),
            ("did you catch that?", "I didn't recognize what you said."),
            ("where are my keys?", "Unable to find any mentions of keys."),
            ("parse time", "Error: failed to parse time string."),
            ("what time is it", "Typing was cancelled into Current system date and time."),
            ("what time is it", "The current time is 9:18 PM."),
            ("what's the time right now", "It is 9:20 PM."),
            ("type notes", "Typing was aborted into Current system date."),
        ]
        for user_msg, ast_resp in failure_cases:
            doc_id = store_exchange(user_msg, ast_resp, persist_directory=self.temp_dir)
            self.assertIsNone(doc_id, f"Should not have stored failure/corrupted exchange: '{ast_resp}'")

        # Confirm nothing was stored in the collection
        coll = get_memory_collection(persist_directory=self.temp_dir)
        self.assertEqual(coll.count(), 0)

        # Valid exchange is stored normally
        valid_id = store_exchange("my favorite food is pizza", "Got it, I will remember that!", persist_directory=self.temp_dir)
        self.assertIsNotNone(valid_id)
        self.assertEqual(coll.count(), 1)

    def test_clear_memory(self):
        """Verifies that clear_memory wipes stored vectors."""
        store_exchange("Fact 1", "Response 1", persist_directory=self.temp_dir)
        self.assertEqual(len(retrieve_relevant_context("Fact 1", top_k=1, persist_directory=self.temp_dir)), 1)

        clear_memory(persist_directory=self.temp_dir)
        self.assertEqual(len(retrieve_relevant_context("Fact 1", top_k=1, persist_directory=self.temp_dir)), 0)

    def test_format_context_for_prompt(self):
        """Verifies the prompt formatter formats exchange dictionaries into clean text."""
        items = [
            {
                "user_message": "my favorite food is tacos",
                "assistant_response": "Tacos are delicious!",
                "text": "User: my favorite food is tacos\nAssistant: Tacos are delicious!",
            }
        ]
        formatted = format_context_for_prompt(items)
        self.assertIn("Past User: my favorite food is tacos", formatted)
        self.assertIn("Past Assistant: Tacos are delicious!", formatted)

        self.assertEqual(format_context_for_prompt([]), "")

    def test_max_distance_filtering(self):
        """Verifies that retrieve_relevant_context excludes items exceeding max_distance."""
        store_exchange("my favorite color is blue", "Blue is a calming color.", persist_directory=self.temp_dir)

        # Semantically close query -> within threshold -> returned
        close_results = retrieve_relevant_context(
            query="what color do I like",
            top_k=1,
            max_distance=0.80,
            persist_directory=self.temp_dir,
        )
        self.assertEqual(len(close_results), 1)
        self.assertIn("blue", close_results[0]["text"].lower())

        # Unrelated query -> distance exceeds 0.80 -> filtered out
        unrelated_results = retrieve_relevant_context(
            query="quantum electrodynamics and algebraic topology",
            top_k=1,
            max_distance=0.80,
            persist_directory=self.temp_dir,
        )
        self.assertEqual(len(unrelated_results), 0)

    def test_format_context_sanitization(self):
        """Verifies that format_context_for_prompt strips tool tags and excludes raw errors or corrupted fragments."""
        items = [
            {
                "user_message": "read clipboard",
                "assistant_response": "[Clipboard] Retrieved 50 characters. [Tool Invoked by Agent] read_clipboard(args={}) Here is your copied note.",
            },
            {
                "user_message": "invalid reminder",
                "assistant_response": "Error: Invalid time format specified.",
            },
            {
                "user_message": "failed check",
                "assistant_response": "I didn't recognize that as a valid time expression.",
            },
            {
                "user_message": "corrupted past turn",
                "assistant_response": "Typing was cancelled into Current system date and time.",
            },
        ]
        formatted = format_context_for_prompt(items)
        self.assertIn("Here is your copied note.", formatted)
        self.assertNotIn("[Clipboard]", formatted)
        self.assertNotIn("[Tool Invoked", formatted)
        self.assertNotIn("Error: Invalid time", formatted)
        self.assertNotIn("valid time expression", formatted)
        self.assertNotIn("Typing was cancelled into", formatted)


class TestCrossProcessPersistence(unittest.TestCase):
    """
    Validates cross-process vector memory persistence.
    Process 1 stores a fact and exits.
    Process 2 (fresh process) retrieves the stored fact from the same directory.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_chroma_proc_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cross_process_recall(self):
        """
        Requirement Test:
        Have a short conversation mentioning a specific fact (e.g. 'my favorite color is blue'),
        then in a NEW session (fresh process), ask 'what's my favorite color' and confirm
        it retrieves the stored fact correctly.
        """
        # Process 1: Store exchange and exit completely
        code_proc1 = f"""
import sys
from pathlib import Path
sys.path.insert(0, r"{PROJECT_ROOT}")
from src.agent.memory import store_exchange

store_exchange(
    user_message="my favorite color is blue",
    assistant_response="Got it, your favorite color is blue.",
    persist_directory=r"{self.temp_dir}",
)
print("STORED_OK")
"""
        res1 = subprocess.run(
            [sys.executable, "-c", code_proc1],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        self.assertIn("STORED_OK", res1.stdout)

        # Process 2: Fresh process queries the persistent directory
        code_proc2 = f"""
import sys
from pathlib import Path
sys.path.insert(0, r"{PROJECT_ROOT}")
from src.agent.memory import retrieve_relevant_context

results = retrieve_relevant_context(
    query="what's my favorite color",
    top_k=1,
    persist_directory=r"{self.temp_dir}",
)
assert len(results) > 0, "No results found in ChromaDB"
item = results[0]
assert "blue" in item["text"].lower(), f"Expected blue in text: {{item['text']}}"
assert item["user_message"] == "my favorite color is blue"
assert item["assistant_response"] == "Got it, your favorite color is blue."
print(f"RECALLED_OK: {{item['user_message']}} -> {{item['assistant_response']}}")
"""
        res2 = subprocess.run(
            [sys.executable, "-c", code_proc2],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        self.assertIn("RECALLED_OK", res2.stdout)
        self.assertIn("my favorite color is blue", res2.stdout)


class TestGraphMemoryIntegration(unittest.TestCase):
    """Tests the integration between LangGraph and vector memory."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_chroma_graph_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_llm_node_prompt_injection_and_storage(self):
        """
        Verifies that create_llm_node:
        1. Injects relevant past context into the prompt passed to the LLM.
        2. Stores the completed exchange into vector memory when the assistant responds.
        """
        # Pre-populate memory with a fact
        store_exchange(
            user_message="I work as a software engineer in Seattle",
            assistant_response="Seattle is a vibrant tech hub.",
            persist_directory=self.temp_dir,
        )

        captured_messages = []

        def mock_invoke(messages):
            captured_messages.extend(messages)
            return AIMessage(content="I recall you work as a software engineer in Seattle.")

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = mock_invoke

        llm_node = create_llm_node(
            llm=mock_llm,
            system_prompt="You are Jarvis.",
            enable_memory=True,
            memory_db_path=self.temp_dir,
        )

        state = {
            "messages": [HumanMessage(content="Where do I work and what is my job?")]
        }

        result = llm_node(state)
        self.assertEqual(len(result["messages"]), 1)
        self.assertIn("Seattle", result["messages"][0].content)

        # 1. Check that the LLM was passed the augmented system prompt with past context
        self.assertTrue(len(captured_messages) >= 1)
        system_msg = captured_messages[0]
        self.assertIsInstance(system_msg, SystemMessage)
        self.assertIn("=== Past Information for Reference Only", system_msg.content)
        self.assertIn("software engineer in Seattle", system_msg.content)

        # 2. Check that the exchange was stored in memory
        stored_items = retrieve_relevant_context(
            query="Where do I work?",
            top_k=2,
            persist_directory=self.temp_dir,
        )
        self.assertTrue(len(stored_items) >= 1)

    def test_default_system_prompt_declarative_rule(self):
        """Verifies that DEFAULT_SYSTEM_PROMPT contains explicit instructions for declarative statements."""
        from src.agent.graph import DEFAULT_SYSTEM_PROMPT
        self.assertIn("sharing personal information/preferences", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("respond conversationally acknowledging it", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("do NOT call any tool", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("explicitly asking to perform an action", DEFAULT_SYSTEM_PROMPT)

    def test_declarative_statement_stored_to_memory(self):
        """Verifies that when a user shares personal info, it is acknowledged and stored to memory."""
        captured_messages = []

        def mock_invoke(messages):
            captured_messages.extend(messages)
            return AIMessage(content="Got it, I'll remember that!")

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = mock_invoke

        llm_node = create_llm_node(
            llm=mock_llm,
            enable_memory=True,
            memory_db_path=self.temp_dir,
        )

        state = {
            "messages": [HumanMessage(content="my favorite color is blue")]
        }

        result = llm_node(state)
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0].content, "Got it, I'll remember that!")

        # Verify stored in vector memory
        stored_items = retrieve_relevant_context(
            query="what is my favorite color?",
            top_k=1,
            persist_directory=self.temp_dir,
        )
        self.assertEqual(len(stored_items), 1)
        self.assertEqual(stored_items[0]["user_message"], "my favorite color is blue")
        self.assertEqual(stored_items[0]["assistant_response"], "Got it, I'll remember that!")


def main():
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
