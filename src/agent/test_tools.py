"""
Local Jarvis - Tool Verification Script (Phase 4a)

Verifies:
1. Direct execution of get_current_datetime (System Clock).
2. Direct execution of web_search (SearXNG).
3. End-to-end agent invocation with date/time query, confirming get_current_datetime is invoked (and NOT web_search).
"""

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.tools.datetime_tool import get_current_datetime
from src.agent.tools.web_search import web_search
from src.agent.graph import run_agent


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


def main():
    print("\nStarting Local Jarvis Tool Verification...\n")
    dt_ok = test_direct_datetime()
    if not dt_ok:
        sys.exit(1)

    search_ok = test_direct_web_search()
    if not search_ok:
        sys.exit(1)

    agent_ok = test_agent_datetime_invocation()
    if not agent_ok:
        sys.exit(1)

    print("=" * 70)
    print(" ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
