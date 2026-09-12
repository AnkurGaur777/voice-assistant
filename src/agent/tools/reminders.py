"""
Local Jarvis - SQLite-Backed Reminders & Scheduler (Phase 4)

Provides persistent reminder management backed by SQLite (`reminders.db`) and a
background scheduler thread that checks for due reminders and dispatches alerts.

Features:
- Natural-language and relative time parsing ("in 10 minutes", "at 5pm", "tomorrow at 9am").
- Automatic rollover to the next day when an absolute time has already passed today.
- SQLite persistence with status tracking ('pending', 'completed', 'cancelled').
- Background daemon thread (`ReminderScheduler`) polling for due reminders every ~10 seconds.
- LangGraph tools: `set_reminder` and `list_reminders`.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
import sqlite3
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from langchain_core.tools import tool

# Determine project root and default database location
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = Path(os.environ.get("JARVIS_REMINDERS_DB", str(PROJECT_ROOT / "reminders.db")))


# ==============================================================================
# 1. Database Storage Layer
# ==============================================================================

def get_db_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Returns the resolved database path."""
    if custom_path is not None:
        return Path(custom_path)
    return Path(os.environ.get("JARVIS_REMINDERS_DB", str(DEFAULT_DB_PATH)))


@contextmanager
def db_connection(db_path: Optional[Union[str, Path]] = None):
    """
    Context manager yielding an open SQLite connection and guaranteeing that
    the connection is closed upon exit (essential on Windows to avoid file locks).
    """
    target_path = get_db_path(db_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """Initializes the SQLite reminders database and creates tables/indexes."""
    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                due_timestamp REAL NOT NULL,
                due_time_str TEXT NOT NULL,
                created_at REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                notified_at REAL
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_reminders_status_due ON reminders(status, due_timestamp)"
        )


def add_reminder(
    text: str,
    due_datetime: datetime,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """
    Inserts a new pending reminder into the database.

    Returns:
        The database ID of the inserted reminder.
    """
    init_db(db_path)

    due_ts = due_datetime.timestamp()
    due_str = due_datetime.strftime("%Y-%m-%d %I:%M %p")
    now_ts = time.time()

    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO reminders (text, due_timestamp, due_time_str, created_at, status)
            VALUES (?, ?, ?, ?, 'pending')
            """,
            (text.strip(), due_ts, due_str, now_ts),
        )
        return cursor.lastrowid


def get_pending_reminders(db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Retrieves all active pending reminders ordered chronologically by due time."""
    init_db(db_path)

    with db_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, text, due_timestamp, due_time_str, created_at, status
            FROM reminders
            WHERE status = 'pending'
            ORDER BY due_timestamp ASC
            """
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_due_reminders(
    current_time: Optional[float] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Retrieves all pending reminders whose due time has passed."""
    init_db(db_path)

    if current_time is None:
        current_time = time.time()

    with db_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, text, due_timestamp, due_time_str, created_at, status
            FROM reminders
            WHERE status = 'pending' AND due_timestamp <= ?
            ORDER BY due_timestamp ASC
            """,
            (current_time,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def mark_reminder_completed(
    reminder_id: int,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Marks a reminder as completed with the current notification timestamp."""
    init_db(db_path)

    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE reminders
            SET status = 'completed', notified_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (time.time(), reminder_id),
        )
        return cursor.rowcount > 0


def cancel_reminder(
    reminder_id: int,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Cancels a pending reminder."""
    init_db(db_path)

    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE reminders SET status = 'cancelled' WHERE id = ? AND status = 'pending'",
            (reminder_id,),
        )
        return cursor.rowcount > 0


# ==============================================================================
# 2. Natural-Language & Relative Time Parser
# ==============================================================================

def format_relative_delta(target: datetime, now: datetime) -> str:
    """Returns a friendly relative duration description (e.g. 'in 10 minutes')."""
    total_seconds = int((target - now).total_seconds())
    if total_seconds < 0:
        return "due now"
    if total_seconds < 60:
        return f"in {total_seconds} seconds"
    
    total_minutes = total_seconds // 60
    if total_minutes < 60:
        unit = "minute" if total_minutes == 1 else "minutes"
        return f"in {total_minutes} {unit}"
    
    total_hours = total_minutes // 60
    remaining_minutes = total_minutes % 60
    if total_hours < 24:
        hr_unit = "hour" if total_hours == 1 else "hours"
        if remaining_minutes > 0:
            return f"in {total_hours} {hr_unit} and {remaining_minutes} mins"
        return f"in {total_hours} {hr_unit}"
    
    total_days = total_hours // 24
    day_unit = "day" if total_days == 1 else "days"
    return f"in {total_days} {day_unit}"


def parse_time_expression(
    when_str: str,
    now: Optional[datetime] = None,
) -> Tuple[datetime, str]:
    """
    Parses natural language and standard time strings into a future datetime.

    Supported patterns:
    - Relative offsets: "in 10 minutes", "in 5 mins", "in 2 hours", "in 30 seconds", "in 1 day"
    - Shorthand offsets: "10m", "5 min", "2h", "30s"
    - Clock times: "at 5pm", "at 5:30pm", "at 5:30 pm", "5pm", "17:00", "at 17:30"
    - Day specifiers: "tomorrow at 9am", "at 9am tomorrow", "today at 4pm"
    - Keywords: "noon" (12:00 PM), "midnight" (12:00 AM), "tonight" (8:00 PM)

    Rollover logic:
    - If a specific clock time has already passed today and no date was specified,
      it automatically rolls over to that time tomorrow.

    Returns:
        (target_datetime, friendly_description)

    Raises:
        ValueError: If the expression cannot be parsed or evaluates to a past time.
    """
    if now is None:
        now = datetime.now()

    cleaned = when_str.strip().lower()
    if not cleaned:
        raise ValueError("Time expression cannot be empty.")

    # 1. Keywords: noon, midnight, tonight
    if cleaned in ["noon", "at noon"]:
        target = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target, target.strftime("%I:%M %p") + f" ({format_relative_delta(target, now)})"

    if cleaned in ["midnight", "at midnight"]:
        target = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return target, target.strftime("%I:%M %p tomorrow") + f" ({format_relative_delta(target, now)})"

    if cleaned in ["tonight", "at tonight"]:
        target = now.replace(hour=20, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target, target.strftime("%I:%M %p") + f" ({format_relative_delta(target, now)})"

    # 2. Relative time offsets: (e.g. "in 10 minutes", "10 mins", "in 1.5 hours", "30 seconds")
    rel_match = re.match(
        r"^(?:in\s+)?(\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$",
        cleaned,
    )
    if rel_match:
        val = float(rel_match.group(1))
        unit = rel_match.group(2)
        if unit in ["s", "sec", "secs", "second", "seconds"]:
            delta = timedelta(seconds=val)
        elif unit in ["m", "min", "mins", "minute", "minutes"]:
            delta = timedelta(minutes=val)
        elif unit in ["h", "hr", "hrs", "hour", "hours"]:
            delta = timedelta(hours=val)
        elif unit in ["d", "day", "days"]:
            delta = timedelta(days=val)
        else:
            raise ValueError(f"Unknown time unit '{unit}'")

        target = now + delta
        desc = format_relative_delta(target, now)
        return target, desc

    # 3. Clock times with optional day specifier
    # e.g. "tomorrow at 5pm", "at 5pm tomorrow", "at 5:30pm", "5pm", "17:30"
    is_tomorrow = "tomorrow" in cleaned
    cleaned_clock = cleaned.replace("tomorrow", "").replace("today", "").strip()
    cleaned_clock = re.sub(r"^at\s+", "", cleaned_clock).strip()

    # Match 12h or 24h clock: "5pm", "5:30pm", "5:30 pm", "17:00", "9"
    clock_match = re.match(
        r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$",
        cleaned_clock,
    )
    if clock_match:
        hours = int(clock_match.group(1))
        minutes = int(clock_match.group(2)) if clock_match.group(2) else 0
        meridian = clock_match.group(3)

        if minutes < 0 or minutes > 59:
            raise ValueError("Minutes must be between 00 and 59.")

        if meridian:
            if hours < 1 or hours > 12:
                raise ValueError("Hour in 12-hour format must be between 1 and 12.")
            if meridian == "pm" and hours != 12:
                hours += 12
            elif meridian == "am" and hours == 12:
                hours = 0
        else:
            if hours < 0 or hours > 23:
                raise ValueError("Hour in 24-hour format must be between 0 and 23.")

        target = now.replace(hour=hours, minute=minutes, second=0, microsecond=0)

        # Apply day logic and automatic next-day rollover
        if is_tomorrow:
            target += timedelta(days=1)
        elif target <= now:
            # If specified time already passed today, advance to tomorrow
            target += timedelta(days=1)

        day_label = "tomorrow" if target.date() > now.date() else "today"
        time_label = target.strftime("%I:%M %p").lstrip("0")
        desc = f"{day_label} at {time_label} ({format_relative_delta(target, now)})"
        return target, desc

    # 4. Fallback: ISO format attempt
    try:
        target = datetime.fromisoformat(when_str)
        if target <= now:
            raise ValueError("Specified time has already passed.")
        return target, target.strftime("%Y-%m-%d %I:%M %p") + f" ({format_relative_delta(target, now)})"
    except Exception:
        pass

    raise ValueError(
        f"Could not understand time expression '{when_str}'. "
        "Please use formats like 'in 10 minutes', 'in 1 hour', 'at 5pm', or 'tomorrow at 9am'."
    )


# ==============================================================================
# 3. Background Scheduler Daemon Thread
# ==============================================================================

class ReminderScheduler(threading.Thread):
    """
    Background daemon thread that periodically inspects SQLite for due reminders
    and dispatches console alerts.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        check_interval: float = 10.0,
        alert_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        super().__init__(daemon=True, name="JarvisReminderScheduler")
        self.db_path = get_db_path(db_path)
        self.check_interval = check_interval
        self.alert_callback = alert_callback
        self._stop_event = threading.Event()

    def _default_alert(self, reminder: Dict[str, Any]) -> None:
        """Default console alert printer."""
        border = "=" * 65
        print(f"\n{border}")
        print(f" 🔔 REMINDER: {reminder['text']}")
        print(f" Scheduled Due Time: {reminder['due_time_str']}")
        print(f"{border}\n")

    def run(self) -> None:
        """Main scheduler loop."""
        print(f"[ReminderScheduler] Started background scheduler (interval: {self.check_interval}s)")
        while not self._stop_event.is_set():
            try:
                due_items = get_due_reminders(db_path=self.db_path)
                for item in due_items:
                    if self.alert_callback:
                        self.alert_callback(item)
                    else:
                        self._default_alert(item)
                    mark_reminder_completed(item["id"], db_path=self.db_path)
            except Exception as e:
                print(f"[ReminderScheduler] Error during check: {e}", file=sys.stderr)

            # Wait for check_interval or until stop is requested
            self._stop_event.wait(timeout=self.check_interval)

        print("[ReminderScheduler] Background scheduler stopped cleanly.")

    def stop(self) -> None:
        """Signals the background thread to stop immediately."""
        self._stop_event.set()


# Module-level singleton reference
_ACTIVE_SCHEDULER: Optional[ReminderScheduler] = None


def start_reminder_scheduler(
    db_path: Optional[Union[str, Path]] = None,
    interval: float = 10.0,
    alert_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> ReminderScheduler:
    """
    Initializes and starts the background reminder scheduler thread.
    Returns the running ReminderScheduler instance.
    """
    global _ACTIVE_SCHEDULER
    if _ACTIVE_SCHEDULER is not None and _ACTIVE_SCHEDULER.is_alive():
        return _ACTIVE_SCHEDULER

    scheduler = ReminderScheduler(
        db_path=db_path,
        check_interval=interval,
        alert_callback=alert_callback,
    )
    scheduler.start()
    _ACTIVE_SCHEDULER = scheduler
    return scheduler


def stop_reminder_scheduler() -> None:
    """Stops the active reminder scheduler if running."""
    global _ACTIVE_SCHEDULER
    if _ACTIVE_SCHEDULER is not None and _ACTIVE_SCHEDULER.is_alive():
        _ACTIVE_SCHEDULER.stop()
        _ACTIVE_SCHEDULER.join(timeout=2.0)
    _ACTIVE_SCHEDULER = None


# ==============================================================================
# 4. LangGraph Agent Tools
# ==============================================================================

@tool
def set_reminder(text: str, when: str) -> str:
    """Set a future reminder for a task, event, or alert.

    Use this tool ONLY when the user explicitly asks to be reminded of something, schedule a reminder,
    or set a reminder at a future time (e.g. 'remind me to check the oven in 10 minutes',
    'set a reminder to call mom at 5pm', 'remind me tomorrow at 9am to submit taxes').
    Never call this tool for general questions, conversational queries, past facts, or personal preferences.

    Args:
        text: The reminder message or task to remember (e.g. 'check the oven', 'call mom').
        when: Natural-language or relative time string (e.g. 'in 10 minutes', 'at 5pm', 'tomorrow at 9am').
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return "Error: Reminder text cannot be empty."

    try:
        due_datetime, friendly_desc = parse_time_expression(when)
    except ValueError as e:
        return f"Error: {e}"

    reminder_id = add_reminder(cleaned_text, due_datetime)
    print(f"[Reminders] Stored reminder #{reminder_id}: '{cleaned_text}' due {due_datetime}")
    return f"Reminder set: '{cleaned_text}' for {friendly_desc}."


@tool
def list_reminders() -> str:
    """List all current pending reminders that have not yet triggered.

    Use this tool ONLY when the user explicitly asks to see, check, or list their reminders
    (e.g. 'what are my reminders', 'show my reminders', 'do I have any upcoming reminders').
    Never call this tool for general questions or personal preferences.
    """
    reminders = get_pending_reminders()
    if not reminders:
        return "You have no pending reminders."

    now = datetime.now()
    lines = [f"You have {len(reminders)} pending reminder{'s' if len(reminders) != 1 else ''}:"]
    for idx, r in enumerate(reminders, 1):
        due_dt = datetime.fromtimestamp(r["due_timestamp"])
        delta_str = format_relative_delta(due_dt, now)
        formatted_time = due_dt.strftime("%I:%M %p").lstrip("0")
        day_str = "today" if due_dt.date() == now.date() else due_dt.strftime("%b %d")
        lines.append(f"{idx}. '{r['text']}' - {day_str} at {formatted_time} ({delta_str})")

    return "\n".join(lines)
