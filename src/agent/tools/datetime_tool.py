"""
Local Jarvis - Current Date & Time Tool

Provides accurate, real-time local system date and time using Python's datetime module.
"""

from datetime import datetime
from langchain_core.tools import tool


def format_current_datetime() -> str:
    """Returns a formatted string containing the current local date, time, and day of week."""
    now = datetime.now().astimezone()
    day_name = now.strftime("%A")
    date_str = now.strftime("%B %d, %Y")
    time_12h = now.strftime("%I:%M %p").lstrip("0")
    time_24h = now.strftime("%H:%M")
    tz_name = now.tzname() or "Local Time"

    return (
        f"Current system date and time:\n"
        f"- Date: {day_name}, {date_str}\n"
        f"- Time: {time_12h} ({time_24h})\n"
        f"- Day of the week: {day_name}\n"
        f"- Timezone: {tz_name}"
    )


@tool
def get_current_datetime() -> str:
    """Get the current local system date, time, and day of the week.

    Use this tool for questions regarding the current date, current time,
    day of the week, month, year, or 'what day is it today'. Always use this tool
    instead of run_python or web_search for date or time questions.

    CRITICAL FOR FINAL SPEECH:
    You MUST convert the returned date/time information into a single, concise, natural
    conversational spoken sentence (e.g. "It's Sunday, September 13th, 1:46 PM" or "The time is 1:46 PM").
    NEVER recite bullet points, field labels, or multiple lines verbatim.
    """
    print("[System Clock] Fetching current system date and time...")
    result = format_current_datetime()
    summary_line = result.replace("\n", " | ")
    print(f"[System Clock] Retrieved: {summary_line}")
    return result

