"""
Local Jarvis - Web Search Tool Verification Script (Phase 4a)

Verifies:
1. Direct SearXNG connection & web_search tool execution.
2. End-to-end agent invocation with a live-information query ("what's today's date?").
3. Verification that the agent actually invokes the web_search tool and logs the invocation.
"""

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.tools.web_search import web_search, search_searxng
from src.agent.graph import run_agent, get_agent_app


def test_direct_web_search() -> bool:
    """Tests direct execution of web_search against local SearXNG."""
    print("=" * 70)
    print(" STEP 1: Direct Web Search Tool Test (SearXNG)")
    print("=" * 70)
    query = "current year and calendar"
    print(f"Executing search query: '{query}'...")

    start = time.perf_counter()
    result = web_search.invoke({"query": query})
    elapsed = time.perf_counter() - start

    print(f"\nSearch result received in {elapsed:.3f}s:\n")
    print(result[:400] + ("..." if len(result) > 400 else ""))
    print()

    if "Error:" in result or "No search results found" in result:
        print("FAIL: Direct web search did not return valid results.")
        return False

    print("PASS: Direct web search returned structured results successfully.\n")
    return True


def test_agent_tool_invocation() -> bool:
    """
    Tests that the LangGraph agent correctly invokes web_search
    when asked for live/current information.
    """
    print("=" * 70)
    print(" STEP 2: End-to-End Agent Tool Invocation Test")
    print("=" * 70)
    query = "What is today's date?"
    print(f"User Query: '{query}'\n")

    start = time.perf_counter()
    response, history = run_agent(
        user_input=query,
        model="llama3.2:3b",
        temperature=0.3,
        num_predict=120,
    )
    total_elapsed = time.perf_counter() - start

    print(f"\nFinal Agent Response: {response}")
    print(f"Total Turn Latency: {total_elapsed:.3f}s\n")

    # Inspect message history to verify ToolMessage presence
    has_tool_call = False
    has_tool_message = False

    for msg in history:
        msg_type = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "web_search":
                    has_tool_call = True
                    print(f"  [Verified] AIMessage contains tool_call: {tc['name']} with args {tc['args']}")
        if msg_type == "ToolMessage":
            has_tool_message = True
            print(f"  [Verified] ToolMessage received from tool: '{getattr(msg, 'name', 'unknown')}'")

    if has_tool_call and has_tool_message:
        print("\nPASS: Agent successfully decided to invoke web_search and used live results!")
        return True
    else:
        print("\nFAIL: Agent did not invoke web_search as expected.")
        return False


def main():
    print("\nStarting Local Jarvis Phase 4a Tool Verification...\n")
    direct_ok = test_direct_web_search()
    if not direct_ok:
        print("Direct search failed. Please verify that SearXNG Docker container is running.")
        sys.exit(1)

    agent_ok = test_agent_tool_invocation()
    if not agent_ok:
        print("Agent tool invocation verification failed.")
        sys.exit(1)

    print("=" * 70)
    print(" ALL PHASE 4a VERIFICATION TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
