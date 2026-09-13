"""
Unit Tests for Local Jarvis Interaction Analytics & Logging Module

Validates:
1. SQLite database initialization and schema structure.
2. Synchronous interaction logging (`async_write=False`).
3. Asynchronous non-blocking interaction logging (`async_write=True`) with `flush_logging`.
4. Tool extraction from LangChain messages (`extract_tools_from_messages`).
5. CSV export functionality (`export_to_csv`).
6. Orchestrator integration and `--no-logging` configuration.
"""

import csv
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.analytics.logger import (
    DEFAULT_DB_PATH,
    flush_logging,
    init_db,
    log_interaction,
    shutdown_logging,
)
from src.analytics.export_csv import CSV_COLUMNS, export_to_csv
from src.orchestrator import (
    VoiceAssistantOrchestrator,
    extract_tools_from_messages,
)


class TestAnalyticsLogger(unittest.TestCase):
    """Tests for SQLite database logging and background queue."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "test_interactions.db"

    def tearDown(self):
        flush_logging(timeout=2.0)
        self.temp_dir.cleanup()

    def test_init_db_schema(self):
        """Verifies table schema and indexes are correctly initialized."""
        init_db(self.db_path)
        self.assertTrue(self.db_path.exists())

        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(interactions);")
            columns = {col[1]: col[2] for col in cursor.fetchall()}

            expected_columns = {
                "id": "INTEGER",
                "timestamp": "TEXT",
                "user_text": "TEXT",
                "assistant_response": "TEXT",
                "tools_called": "TEXT",
                "stt_seconds": "REAL",
                "brain_seconds": "REAL",
                "tts_seconds": "REAL",
                "total_seconds": "REAL",
                "success": "INTEGER",
                "trigger_mode": "TEXT",
                "is_test": "INTEGER",
            }
            for col_name, col_type in expected_columns.items():
                self.assertIn(col_name, columns)
                self.assertEqual(columns[col_name], col_type)
        finally:
            conn.close()

    def test_log_interaction_sync(self):
        """Verifies synchronous logging writes records immediately and accurately."""
        log_interaction(
            user_text="what time is it",
            assistant_response="It is 8:30 PM",
            tools_called=["get_current_datetime"],
            stt_seconds=0.45,
            brain_seconds=1.20,
            tts_seconds=0.85,
            total_seconds=2.50,
            success=True,
            trigger_mode="wake_word",
            db_path=self.db_path,
            async_write=False,
            timestamp="2026-09-13T15:00:00Z",
            is_test=False,
        )

        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM interactions;")
            rows = cursor.fetchall()
        finally:
            conn.close()

        self.assertEqual(len(rows), 1)
        row = rows[0]
        # (id, timestamp, user_text, assistant_response, tools_called, stt, brain, tts, total, success, mode, is_test)
        self.assertEqual(row[1], "2026-09-13T15:00:00Z")
        self.assertEqual(row[2], "what time is it")
        self.assertEqual(row[3], "It is 8:30 PM")
        self.assertEqual(json.loads(row[4]), ["get_current_datetime"])
        self.assertAlmostEqual(row[5], 0.45)
        self.assertAlmostEqual(row[6], 1.20)
        self.assertAlmostEqual(row[7], 0.85)
        self.assertAlmostEqual(row[8], 2.50)
        self.assertEqual(row[9], 1)
        self.assertEqual(row[10], "wake_word")
        self.assertEqual(row[11], 0)

    def test_log_interaction_async_flush(self):
        """Verifies asynchronous logging through worker queue with flush_logging."""
        for i in range(5):
            log_interaction(
                user_text=f"test utterance {i}",
                assistant_response=f"response {i}",
                tools_called=["run_python"] if i % 2 == 0 else [],
                stt_seconds=0.1 * (i + 1),
                brain_seconds=0.2 * (i + 1),
                tts_seconds=0.3 * (i + 1),
                success=True,
                trigger_mode="conversation",
                db_path=self.db_path,
                async_write=True,
            )

        flush_logging(timeout=5.0)

        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM interactions;")
            count = cursor.fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(count, 5)

    def test_log_failure_record(self):
        """Verifies failed turns are logged with success=0."""
        log_interaction(
            user_text="failing command",
            assistant_response="",
            tools_called=[],
            stt_seconds=0.5,
            brain_seconds=0.2,
            tts_seconds=0.0,
            success=False,
            trigger_mode="wake_word",
            db_path=self.db_path,
            async_write=False,
        )

        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT success FROM interactions WHERE user_text='failing command';")
            status = cursor.fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(status, 0)


class TestToolExtraction(unittest.TestCase):
    """Tests for extracting executed tools from LangChain message histories."""

    def test_empty_messages(self):
        """Returns empty list when no messages or tools exist."""
        self.assertEqual(extract_tools_from_messages([]), [])
        self.assertEqual(
            extract_tools_from_messages([
                HumanMessage(content="hello"),
                AIMessage(content="Hi there!"),
            ]),
            [],
        )

    def test_extract_from_tool_messages(self):
        """Extracts tool names from ToolMessage instances."""
        messages = [
            HumanMessage(content="what time is it"),
            AIMessage(
                content="",
                tool_calls=[{"name": "get_current_datetime", "args": {}, "id": "call_1"}],
            ),
            ToolMessage(
                content="Current date and time: 2026-09-13 20:30:00",
                name="get_current_datetime",
                tool_call_id="call_1",
            ),
            AIMessage(content="It is 8:30 PM"),
        ]
        tools = extract_tools_from_messages(messages)
        self.assertEqual(tools, ["get_current_datetime"])

    def test_extract_multiple_tools_in_order(self):
        """Extracts multiple distinct tools in call order without duplicates."""
        messages = [
            HumanMessage(content="calculate 2+2 and check clipboard"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "run_python", "args": {"code": "2+2"}, "id": "c1"},
                    {"name": "read_clipboard", "args": {}, "id": "c2"},
                ],
            ),
            ToolMessage(content="4", name="run_python", tool_call_id="c1"),
            ToolMessage(content="clipboard text", name="read_clipboard", tool_call_id="c2"),
            AIMessage(content="Result is 4 and clipboard contains text."),
        ]
        tools = extract_tools_from_messages(messages)
        self.assertEqual(tools, ["run_python", "read_clipboard"])


class TestCSVExport(unittest.TestCase):
    """Tests for export_csv.py dumping SQLite to CSV."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "interactions.db"
        self.csv_path = Path(self.temp_dir.name) / "export.csv"

        init_db(self.db_path)
        log_interaction(
            user_text="turn 1",
            assistant_response="resp 1",
            tools_called=["web_search"],
            stt_seconds=0.3,
            brain_seconds=1.0,
            tts_seconds=0.5,
            success=True,
            trigger_mode="wake_word",
            db_path=self.db_path,
            async_write=False,
            is_test=False,
        )
        log_interaction(
            user_text="turn 2",
            assistant_response="resp 2",
            tools_called=[],
            stt_seconds=0.2,
            brain_seconds=0.4,
            tts_seconds=0.3,
            success=True,
            trigger_mode="conversation",
            db_path=self.db_path,
            async_write=False,
            is_test=True,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_to_csv_success(self):
        """Verifies all rows and column headers are exported cleanly to CSV."""
        rows_exported = export_to_csv(
            db_path=self.db_path,
            output_csv_path=self.csv_path,
        )
        self.assertEqual(rows_exported, 2)
        self.assertTrue(self.csv_path.exists())

        with open(str(self.csv_path), "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        self.assertEqual(reader[0], CSV_COLUMNS)
        self.assertEqual(len(reader), 3)  # Header + 2 rows
        self.assertEqual(reader[1][2], "turn 1")
        self.assertEqual(reader[1][3], "resp 1")
        self.assertEqual(json.loads(reader[1][4]), ["web_search"])
        self.assertEqual(reader[1][11], "0")
        self.assertEqual(reader[2][2], "turn 2")
        self.assertEqual(reader[2][10], "conversation")
        self.assertEqual(reader[2][11], "1")

    def test_export_exclude_tests(self):
        """Verifies exclude_tests=True excludes automated test rows."""
        rows_exported = export_to_csv(
            db_path=self.db_path,
            output_csv_path=self.csv_path,
            exclude_tests=True,
        )
        self.assertEqual(rows_exported, 1)
        with open(str(self.csv_path), "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        self.assertEqual(len(reader), 2)  # Header + 1 row
        self.assertEqual(reader[1][2], "turn 1")
        self.assertEqual(reader[1][11], "0")

    def test_export_missing_db(self):
        """Returns 0 when target database does not exist."""
        non_existent = Path(self.temp_dir.name) / "missing.db"
        count = export_to_csv(db_path=non_existent, output_csv_path=self.csv_path)
        self.assertEqual(count, 0)


class TestOrchestratorLoggingIntegration(unittest.TestCase):
    """Tests orchestrator logging hooks and --no-logging configuration."""

    def setUp(self):
        self.orchestrator = VoiceAssistantOrchestrator(
            model="llama3.2:3b",
            enable_tray=False,
            enable_orb=False,
            enable_memory=False,
            enable_logging=True,
            continuous_mode=False,
        )
        self.orchestrator.detector = MagicMock()
        self.orchestrator.agent_app = MagicMock()

    @patch("src.orchestrator.log_interaction")
    @patch("src.orchestrator.speak")
    @patch("src.orchestrator.run_agent")
    @patch("src.orchestrator.transcribe_audio")
    def test_turn_logs_interaction_when_enabled(
        self, mock_stt, mock_agent, mock_speak, mock_log
    ):
        """Verifies log_interaction is invoked upon successful turn."""
        self.orchestrator.detector.listen_and_record.return_value = "audio.wav"
        mock_stt.return_value = "what time is it"
        tool_msg = ToolMessage(content="10:00 PM", name="get_current_datetime", tool_call_id="1")
        mock_agent.return_value = ("It is 10:00 PM", [tool_msg])

        continue_loop = self.orchestrator.run_turn()
        self.assertTrue(continue_loop)

        mock_log.assert_called_once()
        kwargs = mock_log.call_args[1]
        self.assertEqual(kwargs["user_text"], "what time is it")
        self.assertEqual(kwargs["assistant_response"], "It is 10:00 PM")
        self.assertEqual(kwargs["tools_called"], ["get_current_datetime"])
        self.assertTrue(kwargs["success"])
        self.assertEqual(kwargs["trigger_mode"], "wake_word")

    @patch("src.orchestrator.log_interaction")
    @patch("src.orchestrator.speak")
    @patch("src.orchestrator.run_agent")
    @patch("src.orchestrator.transcribe_audio")
    def test_turn_skips_logging_when_disabled(
        self, mock_stt, mock_agent, mock_speak, mock_log
    ):
        """Verifies log_interaction is NOT called when enable_logging=False."""
        self.orchestrator.enable_logging = False
        self.orchestrator.detector.listen_and_record.return_value = "audio.wav"
        mock_stt.return_value = "hello"
        mock_agent.return_value = ("Hi!", [])

        continue_loop = self.orchestrator.run_turn()
        self.assertTrue(continue_loop)
        mock_log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
