"""
Local Jarvis - Interaction Analytics & SQLite Logging Module

Records per-turn interaction metrics (transcription, response, tools executed,
latency breakdown, success flag, trigger mode) to a local SQLite database
(analysis/interactions.db, gitignored).

Performance:
- Writes are executed asynchronously in a background daemon worker thread
  via a thread-safe FIFO queue, adding 0ms noticeable latency to the voice
  pipeline.
- Synchronous write is supported via `async_write=False` for deterministic
  unit testing and verification.
"""

import json
import os
import queue
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Union

# Ensure project root is available
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "analysis" / "interactions.db"

# Internal queue & worker control
_write_queue: queue.Queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_SHUTDOWN_SENTINEL = object()


def init_db(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> Path:
    """
    Initializes the SQLite database and creates the interactions table and indexes
    if they do not already exist.

    :param db_path: Path to the SQLite database file.
    :return: Resolved Path to the initialized database file.
    """
    path = Path(db_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), timeout=10.0)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_text TEXT,
                assistant_response TEXT,
                tools_called TEXT,
                stt_seconds REAL,
                brain_seconds REAL,
                tts_seconds REAL,
                total_seconds REAL,
                success INTEGER NOT NULL DEFAULT 1,
                trigger_mode TEXT NOT NULL,
                is_test INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        # Check if is_test column exists (for backward-compatibility with existing tables)
        cursor.execute("PRAGMA table_info(interactions);")
        existing_cols = {col[1] for col in cursor.fetchall()}
        if "is_test" not in existing_cols:
            cursor.execute("ALTER TABLE interactions ADD COLUMN is_test INTEGER NOT NULL DEFAULT 0;")

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interactions_timestamp
            ON interactions(timestamp);
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interactions_trigger_mode
            ON interactions(trigger_mode);
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interactions_is_test
            ON interactions(is_test);
            """
        )
        conn.commit()
    finally:
        conn.close()

    return path


def _insert_record_sync(
    db_path: Union[str, Path],
    timestamp: str,
    user_text: str,
    assistant_response: str,
    tools_called_json: str,
    stt_seconds: float,
    brain_seconds: float,
    tts_seconds: float,
    total_seconds: float,
    success_int: int,
    trigger_mode: str,
    is_test_int: int = 0,
) -> None:
    """Direct synchronous SQLite insert helper."""
    target_path = init_db(db_path)
    conn = sqlite3.connect(str(target_path), timeout=10.0)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO interactions (
                timestamp,
                user_text,
                assistant_response,
                tools_called,
                stt_seconds,
                brain_seconds,
                tts_seconds,
                total_seconds,
                success,
                trigger_mode,
                is_test
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                timestamp,
                user_text,
                assistant_response,
                tools_called_json,
                stt_seconds,
                brain_seconds,
                tts_seconds,
                total_seconds,
                success_int,
                trigger_mode,
                is_test_int,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _worker_loop() -> None:
    """Background worker processing queue items sequentially."""
    while True:
        try:
            item = _write_queue.get()
        except Exception:
            break

        if item is _SHUTDOWN_SENTINEL:
            _write_queue.task_done()
            break

        try:
            if len(item) == 12:
                (
                    db_path,
                    timestamp,
                    user_text,
                    assistant_response,
                    tools_called_json,
                    stt_seconds,
                    brain_seconds,
                    tts_seconds,
                    total_seconds,
                    success_int,
                    trigger_mode,
                    is_test_int,
                ) = item
            else:
                (
                    db_path,
                    timestamp,
                    user_text,
                    assistant_response,
                    tools_called_json,
                    stt_seconds,
                    brain_seconds,
                    tts_seconds,
                    total_seconds,
                    success_int,
                    trigger_mode,
                ) = item
                is_test_int = 0

            _insert_record_sync(
                db_path=db_path,
                timestamp=timestamp,
                user_text=user_text,
                assistant_response=assistant_response,
                tools_called_json=tools_called_json,
                stt_seconds=stt_seconds,
                brain_seconds=brain_seconds,
                tts_seconds=tts_seconds,
                total_seconds=total_seconds,
                success_int=success_int,
                trigger_mode=trigger_mode,
                is_test_int=is_test_int,
            )
        except Exception as e:
            # Prevent logging errors from crashing the background worker
            print(f"[Analytics Logger Error] Failed to write interaction log: {e}", file=sys.stderr)
        finally:
            _write_queue.task_done()


def _ensure_worker_started() -> None:
    """Ensures the background worker thread is active."""
    global _worker_thread
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(
                target=_worker_loop,
                daemon=True,
                name="AnalyticsLoggerWorker",
            )
            _worker_thread.start()


def log_interaction(
    user_text: str,
    assistant_response: str,
    tools_called: Optional[Union[List[str], str]] = None,
    stt_seconds: float = 0.0,
    brain_seconds: float = 0.0,
    tts_seconds: float = 0.0,
    total_seconds: Optional[float] = None,
    success: bool = True,
    trigger_mode: str = "wake_word",
    db_path: Optional[Union[str, Path]] = None,
    async_write: bool = True,
    timestamp: Optional[str] = None,
    is_test: Optional[bool] = None,
) -> None:
    """
    Records a completed or failed interaction turn to the SQLite interactions table.

    :param user_text: Transcribed speech text from the user.
    :param assistant_response: Final assistant response spoken or generated.
    :param tools_called: List of tool names invoked during the turn, or JSON string.
    :param stt_seconds: Speech-to-text processing duration in seconds.
    :param brain_seconds: LangGraph agent reasoning & tool execution time in seconds.
    :param tts_seconds: Text-to-speech synthesis & playback duration in seconds.
    :param total_seconds: Total pipeline latency in seconds (defaults to sum of stages).
    :param success: Boolean indicating if the turn succeeded without critical exceptions.
    :param trigger_mode: "wake_word" or "conversation".
    :param db_path: Optional custom path to SQLite database. Defaults to analysis/interactions.db.
    :param async_write: If True, dispatches write to background queue (non-blocking).
                        If False, writes synchronously in current thread.
    :param timestamp: Optional ISO-8601 string. Defaults to current UTC timestamp.
    :param is_test: Whether this turn is from an automated test run (defaults to True if JARVIS_TEST_MODE=1).
    """
    target_db = db_path or DEFAULT_DB_PATH

    # Normalize tools_called to valid JSON string
    if tools_called is None:
        tools_called_json = "[]"
    elif isinstance(tools_called, (list, tuple)):
        tools_called_json = json.dumps(list(tools_called))
    elif isinstance(tools_called, str):
        try:
            # Check if already valid JSON list
            parsed = json.loads(tools_called)
            if isinstance(parsed, list):
                tools_called_json = tools_called
            else:
                tools_called_json = json.dumps([tools_called])
        except Exception:
            tools_called_json = json.dumps([tools_called])
    else:
        tools_called_json = json.dumps([str(tools_called)])

    # Calculate total_seconds if not explicitly provided
    if total_seconds is None:
        total_seconds = max(0.0, round(stt_seconds + brain_seconds + tts_seconds, 4))
    else:
        total_seconds = max(0.0, round(float(total_seconds), 4))

    stt_seconds = max(0.0, round(float(stt_seconds), 4))
    brain_seconds = max(0.0, round(float(brain_seconds), 4))
    tts_seconds = max(0.0, round(float(tts_seconds), 4))
    success_int = 1 if success else 0
    clean_trigger = "conversation" if "conversation" in str(trigger_mode).lower() else "wake_word"

    # Auto-detect test mode if not explicitly specified
    if is_test is None:
        is_test = os.environ.get("JARVIS_TEST_MODE") == "1"
    is_test_int = 1 if is_test else 0

    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    record_tuple = (
        target_db,
        timestamp,
        user_text or "",
        assistant_response or "",
        tools_called_json,
        stt_seconds,
        brain_seconds,
        tts_seconds,
        total_seconds,
        success_int,
        clean_trigger,
        is_test_int,
    )

    if async_write:
        _ensure_worker_started()
        _write_queue.put(record_tuple)
    else:
        _insert_record_sync(
            db_path=target_db,
            timestamp=timestamp,
            user_text=user_text or "",
            assistant_response=assistant_response or "",
            tools_called_json=tools_called_json,
            stt_seconds=stt_seconds,
            brain_seconds=brain_seconds,
            tts_seconds=tts_seconds,
            total_seconds=total_seconds,
            success_int=success_int,
            trigger_mode=clean_trigger,
            is_test_int=is_test_int,
        )


def flush_logging(timeout: float = 5.0) -> None:
    """
    Blocks until all currently enqueued interaction log records have been processed.

    :param timeout: Maximum seconds to wait for queue to drain.
    """
    deadline = threading.Event()

    def _waiter():
        _write_queue.join()
        deadline.set()

    t = threading.Thread(target=_waiter, daemon=True)
    t.start()
    deadline.wait(timeout=timeout)


def shutdown_logging(timeout: float = 3.0) -> None:
    """
    Flushes the logging queue and cleanly terminates the background worker thread.

    :param timeout: Maximum seconds to wait for worker thread to exit.
    """
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            _write_queue.put(_SHUTDOWN_SENTINEL)
            _worker_thread.join(timeout=timeout)
            _worker_thread = None
