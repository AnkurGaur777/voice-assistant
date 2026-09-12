"""
Local Jarvis - Reminders & Scheduler Unit Test Suite

Verifies:
1. Natural language and relative time parsing (relative offsets, clock times, keywords, rollovers).
2. SQLite database CRUD operations (add, pending, due, complete, cancel).
3. LangGraph tools: `set_reminder` and `list_reminders`.
4. Background scheduler thread (due reminder polling, alert dispatch, status transition, clean stop).
"""

from datetime import datetime, timedelta
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.tools.reminders import (
    ReminderScheduler,
    add_reminder,
    cancel_reminder,
    get_due_reminders,
    get_pending_reminders,
    list_reminders,
    mark_reminder_completed,
    parse_time_expression,
    set_reminder,
)


class TestTimeParser(unittest.TestCase):
    """Tests the natural language and relative time parsing engine."""

    def setUp(self):
        # Anchor fixed reference time: 2026-09-12 14:30:00 (Saturday 2:30 PM)
        self.ref_now = datetime(2026, 9, 12, 14, 30, 0)

    def test_relative_minutes(self):
        """Tests 'in 10 minutes' and 'in 5 mins'."""
        target, desc = parse_time_expression("in 10 minutes", now=self.ref_now)
        expected = self.ref_now + timedelta(minutes=10)
        self.assertEqual(target, expected)
        self.assertIn("10 minutes", desc)

        target2, _ = parse_time_expression("5 mins", now=self.ref_now)
        self.assertEqual(target2, self.ref_now + timedelta(minutes=5))

    def test_relative_hours_and_seconds(self):
        """Tests 'in 2 hours' and 'in 30 seconds'."""
        target_hr, _ = parse_time_expression("in 2 hours", now=self.ref_now)
        self.assertEqual(target_hr, self.ref_now + timedelta(hours=2))

        target_sec, _ = parse_time_expression("in 30 seconds", now=self.ref_now)
        self.assertEqual(target_sec, self.ref_now + timedelta(seconds=30))

    def test_clock_time_same_day(self):
        """Tests 'at 5pm' when current time is 2:30 PM (should stay today)."""
        target, desc = parse_time_expression("at 5pm", now=self.ref_now)
        expected = datetime(2026, 9, 12, 17, 0, 0)
        self.assertEqual(target, expected)
        self.assertIn("today", desc.lower())
        self.assertIn("5:00 PM", desc)

    def test_clock_time_with_minutes(self):
        """Tests 'at 5:30pm' and '5:30 pm'."""
        target, _ = parse_time_expression("at 5:30pm", now=self.ref_now)
        self.assertEqual(target, datetime(2026, 9, 12, 17, 30, 0))

        target2, _ = parse_time_expression("5:30 pm", now=self.ref_now)
        self.assertEqual(target2, datetime(2026, 9, 12, 17, 30, 0))

    def test_clock_time_rollover_to_tomorrow(self):
        """Tests 'at 1pm' when current time is 2:30 PM (should rollover to tomorrow 1pm)."""
        target, desc = parse_time_expression("at 1pm", now=self.ref_now)
        expected = datetime(2026, 9, 13, 13, 0, 0)
        self.assertEqual(target, expected)
        self.assertIn("tomorrow", desc.lower())
        self.assertIn("1:00 PM", desc)

    def test_explicit_tomorrow(self):
        """Tests 'tomorrow at 9am'."""
        target, desc = parse_time_expression("tomorrow at 9am", now=self.ref_now)
        expected = datetime(2026, 9, 13, 9, 0, 0)
        self.assertEqual(target, expected)
        self.assertIn("tomorrow", desc.lower())

    def test_keywords_noon_and_midnight(self):
        """Tests 'noon' and 'midnight'."""
        target_noon, _ = parse_time_expression("noon", now=self.ref_now)
        # 12:00 PM has already passed at 14:30, so rollover to tomorrow noon
        self.assertEqual(target_noon, datetime(2026, 9, 13, 12, 0, 0))

        target_midnight, _ = parse_time_expression("midnight", now=self.ref_now)
        self.assertEqual(target_midnight, datetime(2026, 9, 13, 0, 0, 0))

    def test_invalid_formats_raise_value_error(self):
        """Tests unparseable or out-of-range time strings."""
        for invalid in ["", "   ", "sometime later", "at 25:00", "in -5 minutes"]:
            with self.assertRaises(ValueError):
                parse_time_expression(invalid, now=self.ref_now)


class TestReminderDatabase(unittest.TestCase):
    """Tests SQLite database operations in an isolated temporary database."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_reminders.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_crud_lifecycle(self):
        """Tests inserting, querying pending, querying due, and completing reminders."""
        now = datetime.now()
        due_soon = now + timedelta(seconds=2)
        due_later = now + timedelta(hours=5)

        # 1. Add reminders
        id1 = add_reminder("Urgent task", due_soon, db_path=self.db_path)
        id2 = add_reminder("Later task", due_later, db_path=self.db_path)
        self.assertGreater(id1, 0)
        self.assertGreater(id2, id1)

        # 2. Query pending
        pending = get_pending_reminders(db_path=self.db_path)
        self.assertEqual(len(pending), 2)
        self.assertEqual(pending[0]["id"], id1)
        self.assertEqual(pending[0]["text"], "Urgent task")
        self.assertEqual(pending[1]["id"], id2)

        # 3. Query due before time passes -> none due
        due_now = get_due_reminders(current_time=now.timestamp(), db_path=self.db_path)
        self.assertEqual(len(due_now), 0)

        # 4. Query due after time passes -> id1 is due
        future_ts = (due_soon + timedelta(seconds=1)).timestamp()
        due_future = get_due_reminders(current_time=future_ts, db_path=self.db_path)
        self.assertEqual(len(due_future), 1)
        self.assertEqual(due_future[0]["id"], id1)

        # 5. Mark completed
        ok = mark_reminder_completed(id1, db_path=self.db_path)
        self.assertTrue(ok)

        # 6. Verify id1 is no longer pending
        pending_after = get_pending_reminders(db_path=self.db_path)
        self.assertEqual(len(pending_after), 1)
        self.assertEqual(pending_after[0]["id"], id2)

        # 7. Cancel id2
        cancelled = cancel_reminder(id2, db_path=self.db_path)
        self.assertTrue(cancelled)
        self.assertEqual(len(get_pending_reminders(db_path=self.db_path)), 0)


class TestReminderTools(unittest.TestCase):
    """Tests set_reminder and list_reminders LangGraph tools."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "tools_reminders.db"
        os.environ["JARVIS_REMINDERS_DB"] = str(self.db_path)

    def tearDown(self):
        if "JARVIS_REMINDERS_DB" in os.environ:
            del os.environ["JARVIS_REMINDERS_DB"]
        self.temp_dir.cleanup()

    def test_set_and_list_tools(self):
        """Tests setting a reminder and then listing it via the tool interface."""
        # Check initial empty list
        empty_res = list_reminders.invoke({})
        self.assertIn("no pending reminders", empty_res.lower())

        # Set a reminder
        set_res = set_reminder.invoke({"text": "Turn off stove", "when": "in 15 minutes"})
        self.assertIn("Reminder set: 'Turn off stove'", set_res)
        self.assertIn("15 minutes", set_res)

        # List reminders
        list_res = list_reminders.invoke({})
        self.assertIn("Turn off stove", list_res)
        self.assertIn("pending reminder", list_res)

    def test_invalid_inputs(self):
        """Tests empty text and invalid time string handling."""
        err_empty = set_reminder.invoke({"text": "  ", "when": "in 10 minutes"})
        self.assertIn("Error:", err_empty)

        err_time = set_reminder.invoke({"text": "Call doctor", "when": "not a real time"})
        self.assertIn("Error:", err_time)
        self.assertIn("Could not understand", err_time)


class TestReminderScheduler(unittest.TestCase):
    """Tests the background daemon thread that polls for due reminders."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "scheduler_reminders.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scheduler_alert_dispatch_and_completion(self):
        """Tests that scheduler detects a due reminder, calls alert callback, and marks completed."""
        alerts_received = []

        def mock_alert(reminder):
            alerts_received.append(reminder)

        # Create reminder due in 0.2 seconds
        due_time = datetime.now() + timedelta(seconds=0.2)
        r_id = add_reminder("Feed the cat", due_time, db_path=self.db_path)

        # Start scheduler with fast 0.1s interval
        scheduler = ReminderScheduler(
            db_path=self.db_path,
            check_interval=0.1,
            alert_callback=mock_alert,
        )
        scheduler.start()

        try:
            # Wait up to 2 seconds for the alert to trigger
            start = time.perf_counter()
            while not alerts_received and (time.perf_counter() - start < 2.0):
                time.sleep(0.05)

            self.assertEqual(len(alerts_received), 1)
            self.assertEqual(alerts_received[0]["id"], r_id)
            self.assertEqual(alerts_received[0]["text"], "Feed the cat")

            # Verify reminder was marked as completed in DB
            pending = get_pending_reminders(db_path=self.db_path)
            self.assertEqual(len(pending), 0)

        finally:
            scheduler.stop()
            scheduler.join(timeout=1.0)
            self.assertFalse(scheduler.is_alive())


def main():
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
