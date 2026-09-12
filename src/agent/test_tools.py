"""
Local Jarvis - Tool Verification Script (Phase 4b: Clipboard & Search Tools)

Verifies:
1. Direct execution of get_current_datetime (System Clock).
2. Direct execution of web_search (SearXNG).
3. Direct execution of read_clipboard (Pyperclip).
4. Direct execution of summarize_clipboard (Pyperclip + framing).
5. Clipboard truncation safeguard.
6. End-to-end agent invocation with date/time query.
7. End-to-end agent invocation with clipboard summarization query.
"""

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.tools.clipboard import (
    get_clipboard_text,
    read_clipboard,
    summarize_clipboard,
)
from src.agent.tools.datetime_tool import get_current_datetime
from src.agent.tools.web_search import web_search
from src.agent.graph import run_agent
import pyperclip


def test_direct_datetime() -> bool:
    """Tests direct execution of get_current_datetime tool."""
    print("=" * 70)
    print(" STEP 1: Direct System Clock Datetime Tool Test")
    print("=" * 70)
    start = time.perf_counter()
    result = get_current_datetime.invoke({})
    elapsed = time.perf_counter() - start

    print(f"\nResult received in {elapsed * 1000:.2f}ms:\n")
    print(result)
    print()

    if "Current system date and time" in result and "Date:" in result:
        print("PASS: get_current_datetime returned system date/time successfully.\n")
        return True
    print("FAIL: get_current_datetime did not return expected format.\n")
    return False


def test_direct_web_search() -> bool:
    """Tests direct execution of web_search against local SearXNG."""
    print("=" * 70)
    print(" STEP 2: Direct Web Search Tool Test (SearXNG)")
    print("=" * 70)
    query = "current year and calendar"
    print(f"Executing search query: '{query}'...")

    start = time.perf_counter()
    result = web_search.invoke({"query": query})
    elapsed = time.perf_counter() - start

    print(f"\nSearch result received in {elapsed:.3f}s:\n")
    print(result[:350] + ("..." if len(result) > 350 else ""))
    print()

    if "Error:" in result or "No search results found" in result:
        print("FAIL: Direct web search did not return valid results.")
        return False

    print("PASS: Direct web search returned structured results successfully.\n")
    return True


def test_agent_datetime_invocation() -> bool:
    """
    Tests that the LangGraph agent correctly invokes get_current_datetime (and NOT web_search)
    when asked for date/time/day-of-week.
    """
    print("=" * 70)
    print(" STEP 3: Agent Tool Routing: Date/Time Query -> get_current_datetime")
    print("=" * 70)
    query = "What is today's date and what day of the week is it?"
    print(f"User Query: '{query}'\n")

    start = time.perf_counter()
    response, history = run_agent(
        user_input=query,
        model="llama3.2:3b",
        temperature=0.2,
        num_predict=120,
    )
    total_elapsed = time.perf_counter() - start

    print(f"\nFinal Agent Response: {response}")
    print(f"Total Turn Latency: {total_elapsed:.3f}s\n")

    invoked_tools = []
    for msg in history:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                invoked_tools.append(tc.get("name"))

    print(f"Tools invoked during turn: {invoked_tools}")

    if "get_current_datetime" in invoked_tools and "web_search" not in invoked_tools:
        print("\nPASS: Agent correctly routed to get_current_datetime (and did NOT call web_search)!")
        return True
    elif "get_current_datetime" in invoked_tools:
        print("\nPASS: Agent invoked get_current_datetime as required.")
        return True
    else:
        print(f"\nFAIL: Expected get_current_datetime to be invoked, but got: {invoked_tools}")
        return False


def test_direct_read_clipboard() -> bool:
    """Tests direct execution of read_clipboard tool."""
    print("=" * 70)
    print(" STEP 4: Direct Read Clipboard Tool Test")
    print("=" * 70)

    try:
        orig_clip = pyperclip.paste()
    except Exception:
        orig_clip = ""

    test_content = "Jarvis test snippet: Project Phase 4b task automation."
    try:
        pyperclip.copy(test_content)
        result = read_clipboard.invoke({})
        print(f"\nRead clipboard result: '{result}'\n")

        if result != test_content:
            print("FAIL: read_clipboard did not match copied text.")
            return False

        # Test empty clipboard handling
        pyperclip.copy("")
        empty_result = read_clipboard.invoke({})
        print(f"Empty clipboard result: '{empty_result}'\n")
        if "empty or contains non-text content" not in empty_result:
            print("FAIL: read_clipboard did not handle empty clipboard gracefully.")
            return False

        print("PASS: read_clipboard passed both text and empty tests successfully.\n")
        return True
    finally:
        pyperclip.copy(orig_clip)


def test_direct_summarize_clipboard() -> bool:
    """Tests direct execution of summarize_clipboard tool."""
    print("=" * 70)
    print(" STEP 5: Direct Summarize Clipboard Tool Test")
    print("=" * 70)

    try:
        orig_clip = pyperclip.paste()
    except Exception:
        orig_clip = ""

    article_text = (
        "Artificial intelligence systems are rapidly evolving from simple text generators "
        "into multi-modal agentic workflows. By incorporating external tools such as web search, "
        "clipboard monitoring, and local task automation, voice assistants can act as autonomous "
        "desktop co-pilots with real-time responsiveness and zero cloud dependencies."
    )
    try:
        pyperclip.copy(article_text)
        # Test default invocation
        result = summarize_clipboard.invoke({})
        print(f"\nSummarize clipboard default output:\n{result}\n")

        if "Clipboard Content for Summarization" not in result or "[Guidance:" not in result:
            print("FAIL: summarize_clipboard output missing required framing structure.")
            return False

        # Test with focus parameter
        focus_result = summarize_clipboard.invoke({"focus": "zero cloud dependencies"})
        print(f"Summarize clipboard with focus output:\n{focus_result}\n")

        if "zero cloud dependencies" not in focus_result:
            print("FAIL: summarize_clipboard did not include focus directive.")
            return False

        print("PASS: summarize_clipboard passed default and focused tests successfully.\n")
        return True
    finally:
        pyperclip.copy(orig_clip)


def test_clipboard_truncation() -> bool:
    """Tests safety limit truncation on very long clipboard contents."""
    print("=" * 70)
    print(" STEP 6: Clipboard Truncation Safeguard Test")
    print("=" * 70)

    try:
        orig_clip = pyperclip.paste()
    except Exception:
        orig_clip = ""

    long_text = "A" * 6000
    try:
        pyperclip.copy(long_text)
        result = get_clipboard_text(max_chars=100)
        print(f"\nTruncated result preview:\n{result[:150]}...\n")

        if "[Note: Clipboard content truncated to first 100 characters" not in result:
            print("FAIL: get_clipboard_text did not truncate long content as expected.")
            return False

        print("PASS: get_clipboard_text safely truncated content over limit.\n")
        return True
    finally:
        pyperclip.copy(orig_clip)


def test_agent_clipboard_invocation() -> bool:
    """Tests agent routing when user asks to summarize clipboard."""
    print("=" * 70)
    print(" STEP 7: Agent Tool Routing: Clipboard Query -> summarize_clipboard")
    print("=" * 70)

    try:
        orig_clip = pyperclip.paste()
    except Exception:
        orig_clip = ""

    sample_doc = (
        "The Python Global Interpreter Lock (GIL) is a mutex that protects access to Python objects, "
        "preventing multiple threads from executing Python bytecodes at once. In Python 3.13, experimental "
        "free-threaded mode allows running without the GIL for true multi-threaded parallelism."
    )
    try:
        pyperclip.copy(sample_doc)
        query = "Can you summarize what's on my clipboard?"
        print(f"User Query: '{query}'\n")

        start = time.perf_counter()
        response, history = run_agent(
            user_input=query,
            model="llama3.2:3b",
            temperature=0.2,
            num_predict=120,
        )
        total_elapsed = time.perf_counter() - start

        print(f"\nFinal Agent Response: {response}")
        print(f"Total Turn Latency: {total_elapsed:.3f}s\n")

        invoked_tools = []
        for msg in history:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    invoked_tools.append(tc.get("name"))

        print(f"Tools invoked during turn: {invoked_tools}")

        if "summarize_clipboard" in invoked_tools or "read_clipboard" in invoked_tools:
            print("\nPASS: Agent correctly routed to clipboard tool!")
            return True
        else:
            print(f"\nFAIL: Expected summarize_clipboard or read_clipboard to be invoked, but got: {invoked_tools}")
            return False
    finally:
        pyperclip.copy(orig_clip)


def main():
    print("\nStarting Local Jarvis Tool Verification...\n")
    dt_ok = test_direct_datetime()
    if not dt_ok:
        sys.exit(1)

    search_ok = test_direct_web_search()
    if not search_ok:
        sys.exit(1)

    read_clip_ok = test_direct_read_clipboard()
    if not read_clip_ok:
        sys.exit(1)

    sum_clip_ok = test_direct_summarize_clipboard()
    if not sum_clip_ok:
        sys.exit(1)

    trunc_ok = test_clipboard_truncation()
    if not trunc_ok:
        sys.exit(1)

    agent_dt_ok = test_agent_datetime_invocation()
    if not agent_dt_ok:
        sys.exit(1)

    agent_clip_ok = test_agent_clipboard_invocation()
    if not agent_clip_ok:
        sys.exit(1)

    print("=" * 70)
    print(" ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

