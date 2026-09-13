"""Local Jarvis - Agent Tool Registry Package"""

from src.agent.tools.clipboard import (
    get_clipboard_text,
    read_clipboard,
    summarize_clipboard,
)
from src.agent.tools.datetime_tool import format_current_datetime, get_current_datetime
from src.agent.tools.desktop import (
    APP_ALIASES,
    focus_window_by_name,
    get_active_window_title,
    launch_app,
    open_application,
    press_enter_in_active_window,
    press_enter_key,
    type_text,
)
from src.agent.tools.reminders import (
    ReminderScheduler,
    add_reminder,
    get_due_reminders,
    get_pending_reminders,
    list_reminders,
    mark_reminder_completed,
    parse_time_expression,
    set_reminder,
    start_reminder_scheduler,
    stop_reminder_scheduler,
)
from src.agent.tools.sandbox import (
    execute_in_sandbox,
    run_python,
    validate_code_ast,
)
from src.agent.tools.web_search import search_searxng, web_search

__all__ = [
    "web_search",
    "search_searxng",
    "get_current_datetime",
    "format_current_datetime",
    "read_clipboard",
    "summarize_clipboard",
    "get_clipboard_text",
    "open_application",
    "type_text",
    "press_enter_key",
    "press_enter_in_active_window",
    "launch_app",
    "get_active_window_title",
    "focus_window_by_name",
    "APP_ALIASES",
    "run_python",
    "execute_in_sandbox",
    "validate_code_ast",
    "set_reminder",
    "list_reminders",
    "start_reminder_scheduler",
    "stop_reminder_scheduler",
    "ReminderScheduler",
    "parse_time_expression",
    "add_reminder",
    "get_pending_reminders",
    "get_due_reminders",
    "mark_reminder_completed",
]

