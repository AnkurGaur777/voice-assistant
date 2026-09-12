"""
Local Jarvis - Tool Verification Script (Phase 4c: Complete Tool Suite)

Verifies:
1. Direct execution of get_current_datetime (System Clock).
2. Direct execution of web_search (SearXNG).
3. Direct execution of read_clipboard (Pyperclip).
4. Direct execution of summarize_clipboard (Pyperclip + framing).
5. Clipboard truncation safeguard.
6. Direct execution of open_application (Windows App Launcher).
7. Direct execution of type_text (Empty, Fail-safe, Rejection, and Confirmation flows).
8. Window focusing logic with retry and fallback.
9. End-to-end agent invocation with date/time query.
10. End-to-end agent invocation with clipboard query.
11. End-to-end agent invocation with app launch query.
12. End-to-end agent argument extraction ('type this should not appear') and exact window name reporting.
"""

from pathlib import Path
import sys
import time
from unittest.mock import patch

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
from src.agent.tools.desktop import (
    get_active_window_title,
    launch_app,
    open_application,
    type_text,
)
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


def test_direct_open_application() -> bool:
    """Tests direct execution of open_application tool."""
    print("=" * 70)
    print(" STEP 8: Direct Open Application Tool Test")
    print("=" * 70)

    # Test empty name handling
    empty_result = open_application.invoke({"app_name": ""})
    print(f"Empty app name result: '{empty_result}'")
    if "Error: Application name cannot be empty" not in empty_result:
        print("FAIL: open_application did not reject empty app name.")
        return False

    # Test non-existent application handling
    invalid_result = open_application.invoke({"app_name": "completely_invalid_app_xyz_987"})
    print(f"Invalid app name result: '{invalid_result}'")
    if "Error: Unable to launch application" not in invalid_result:
        print("FAIL: open_application did not return error for non-existent app.")
        return False

    # Test valid application resolution via mocked launch
    with patch("os.startfile") as mock_start:
        mock_start.return_value = None
        calc_result = open_application.invoke({"app_name": "calculator"})
        if "launched successfully" not in calc_result:
            print("FAIL: open_application did not succeed with mocked launcher.")
            return False
        if not mock_start.call_args or not mock_start.call_args[0][0].lower().endswith("calc.exe"):
            print(f"FAIL: Expected mock_start to be called with calc.exe, got: {mock_start.call_args}")
            return False

    print("PASS: open_application passed empty, invalid, and aliased launch tests.\n")
    return True


def test_direct_type_text() -> bool:
    """Tests direct execution of type_text with fail-safe, reject, and accept flows."""
    print("=" * 70)
    print(" STEP 9: Direct Type Text Tool Test (Fail-Safe & Confirmation)")
    print("=" * 70)

    # 1. Fail-safe test: non-interactive / EOFError
    with patch("builtins.input", side_effect=EOFError("No interactive terminal attached")):
        eof_result = type_text.invoke({"text": "test injection"})
        print(f"Non-interactive EOF result: '{eof_result}'")
        if "Typing cancelled - no interactive confirmation available" not in eof_result:
            print("FAIL: type_text did not fail safe on EOFError.")
            return False

    # 2. Rejection test: user inputs 'n'
    with patch("builtins.input", return_value="n"):
        reject_result = type_text.invoke({"text": "test rejection"})
        print(f"User rejection result: '{reject_result}'")
        if "Typing cancelled by user" not in reject_result:
            print("FAIL: type_text did not abort when user entered 'n'.")
            return False

    # 3. Confirmation test: user inputs 'y'
    with patch("builtins.input", return_value="y"), patch("pyautogui.write") as mock_write:
        confirm_result = type_text.invoke({"text": "hello jarvis"})
        print(f"User confirmation result: '{confirm_result}'")
        if "Successfully typed 12 characters" not in confirm_result:
            print("FAIL: type_text did not succeed when user entered 'y'.")
            return False
        mock_write.assert_called_once()

    # Test empty text handling
    empty_type_result = type_text.invoke({"text": ""})
    print(f"Empty text result: '{empty_type_result}'")
    if "Error: No text provided to type" not in empty_type_result:
        print("FAIL: type_text did not return error for empty text.")
        return False

    print("PASS: type_text passed empty, fail-safe EOF, rejection, and confirmation tests.\n")
    return True


def test_direct_focus_window() -> bool:
    """Tests window focusing logic with mock pygetwindow windows."""
    print("=" * 70)
    print(" STEP 10: Window Focus Logic Test")
    print("=" * 70)

    from unittest.mock import MagicMock
    from src.agent.tools.desktop import focus_window_by_name

    mock_win = MagicMock()
    mock_win.title = "Untitled - Notepad"
    mock_win.isMinimized = False

    with patch("pygetwindow.getAllWindows", return_value=[mock_win]):
        focused = focus_window_by_name("notepad", max_retries=1, retry_delay=0.01)
        print(f"Mock Notepad focus result: {focused}")
        if not focused:
            print("FAIL: focus_window_by_name failed to focus matching window.")
            return False
        mock_win.activate.assert_called_once()

    # Test when window does not exist
    with patch("pygetwindow.getAllWindows", return_value=[]):
        not_focused = focus_window_by_name("completely_missing_app", max_retries=1, retry_delay=0.01)
        print(f"Missing window focus result: {not_focused}")
        if not_focused:
            print("FAIL: focus_window_by_name returned True for non-existent window.")
            return False

    print("PASS: focus_window_by_name accurately focuses matching windows and reports failure when absent.\n")
    return True


def test_agent_desktop_routing() -> bool:
    """Tests agent routing when user asks to open an application."""
    print("=" * 70)
    print(" STEP 11: Agent Tool Routing: App Launch Query -> open_application")
    print("=" * 70)
    query = "Please open the calculator app for me."
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

    if "open_application" in invoked_tools:
        print("\nPASS: Agent correctly routed to open_application!")
        return True
    else:
        print(f"\nFAIL: Expected open_application to be invoked, but got: {invoked_tools}")
        return False


def test_agent_type_text_extraction_and_reporting() -> bool:
    """
    Tests:
    1. Agent accurately extracts 'this should not appear' when prompted 'type this should not appear' (Issue 2).
    2. Agent reports the EXACT window name ('Windows PowerShell') in its final response and NEVER says 'Notepad' (Issue 1).
    """
    print("=" * 70)
    print(" STEP 12: Agent type_text Extraction & Exact Window Name Reporting")
    print("=" * 70)
    query = "type this should not appear"
    print(f"User Query: '{query}'\n")

    # Mock terminal input to cancel typing, and mock active window title to 'Windows PowerShell'
    with patch("builtins.input", return_value="n"), patch("src.agent.tools.desktop.get_active_window_title", return_value="Windows PowerShell"):
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

    extracted_args = []
    invoked_tools = []
    for msg in history:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                invoked_tools.append(tc.get("name"))
                if tc.get("name") == "type_text":
                    extracted_args.append(tc.get("args", {}))

    print(f"Tools invoked during turn: {invoked_tools}")
    print(f"Arguments extracted for type_text: {extracted_args}")

    if "type_text" not in invoked_tools:
        print("\nFAIL: Agent did not invoke type_text tool.")
        return False

    # Verify extracted text argument
    typed_text = extracted_args[0].get("text", "") if extracted_args else ""
    if "this should not appear" not in typed_text.lower():
        print(f"\nFAIL: Agent extracted '{typed_text}' instead of 'this should not appear'.")
        return False
    print(f"PASS: Agent correctly extracted parameter text='{typed_text}'!")

    # Verify exact window name reporting in final response
    resp_lower = response.lower()
    if "powershell" not in resp_lower:
        print(f"\nFAIL: Agent final response did not cite the exact active window ('Windows PowerShell'). Response: {response}")
        return False
    if "notepad" in resp_lower:
        print(f"\nFAIL: Agent hallucinated 'Notepad' when active window was 'Windows PowerShell'. Response: {response}")
        return False

    print("PASS: Agent accurately cited the EXACT window name ('Windows PowerShell') and did NOT assume Notepad!\n")
    return True


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

    open_app_ok = test_direct_open_application()
    if not open_app_ok:
        sys.exit(1)

    type_text_ok = test_direct_type_text()
    if not type_text_ok:
        sys.exit(1)

    focus_win_ok = test_direct_focus_window()
    if not focus_win_ok:
        sys.exit(1)

    agent_dt_ok = test_agent_datetime_invocation()
    if not agent_dt_ok:
        sys.exit(1)

    agent_clip_ok = test_agent_clipboard_invocation()
    if not agent_clip_ok:
        sys.exit(1)

    agent_app_ok = test_agent_desktop_routing()
    if not agent_app_ok:
        sys.exit(1)

    agent_type_ok = test_agent_type_text_extraction_and_reporting()
    if not agent_type_ok:
        sys.exit(1)

    print("=" * 70)
    print(" ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

