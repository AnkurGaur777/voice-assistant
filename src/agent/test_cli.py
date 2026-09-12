"""
Local Jarvis - Interactive CLI Test Harness (Phase 3)

This script provides an interactive terminal chat loop with the Jarvis LangGraph
agent running on local Ollama (llama3.1:8b).
It tracks and logs per-turn inference latency in seconds so you can monitor
GPU inference speed on the RTX 3050.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import BaseMessage
from src.agent.graph import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    build_graph,
    run_agent,
)
from src.agent.tools.reminders import start_reminder_scheduler


def print_banner(model: str, base_url: str, max_tokens: int, enable_memory: bool = True) -> None:
    """Prints a styled startup banner with session information."""
    print("\n" + "=" * 75)
    print(" LOCAL JARVIS - INTERACTIVE CLI (PHASE 6: VECTOR MEMORY & AGENT TOOLS)")
    print("=" * 75)
    print(f" LLM Model:       {model}")
    print(f" Ollama URL:      {base_url}")
    print(f" Max Tokens:      {max_tokens} (Ollama num_predict ceiling)")
    print(" Tools Active:    get_current_datetime, web_search, read_clipboard, summarize_clipboard, open_application, type_text, run_python, set_reminder, list_reminders")
    mem_status = "ChromaDB (all-MiniLM-L6-v2, CPU) -> chroma_db/" if enable_memory else "Disabled"
    print(f" Vector Memory:   {mem_status}")
    print(" Hardware Target: NVIDIA RTX 3050 (6GB) - GPU Acceleration")
    print(" Commands:")
    print("   'exit' or 'quit' -> Exit the chat loop")
    print("   'clear'          -> Clear conversation session history (persistent memory preserved)")
    print("=" * 75 + "\n")


def chat_loop(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 150,
    enable_memory: bool = True,
) -> None:
    """Main interactive chat loop."""
    print_banner(model, base_url, max_tokens, enable_memory=enable_memory)

    # Launch the background reminder scheduler daemon thread
    scheduler = start_reminder_scheduler(interval=10.0)

    print("Initializing LangGraph workflow...")
    try:
        app = build_graph(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens,
            enable_memory=enable_memory,
        )
        print("LangGraph agent compiled and ready.\n")
    except Exception as e:
        print(f"Error initializing LangGraph with Ollama: {e}", file=sys.stderr)
        print("Please verify that Ollama is running (`ollama serve`) and the model is pulled (`ollama pull llama3.1:8b`).", file=sys.stderr)
        scheduler.stop()
        sys.exit(1)

    history: List[BaseMessage] = []
    turn_count = 0

    try:
        while True:
            try:
                user_input = input("You > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting Jarvis CLI. Goodbye!")
                break

            if not user_input:
                continue

            # Command handling
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting Jarvis CLI. Goodbye!")
                break

            if user_input.lower() in ["clear", "reset"]:
                history = []
                turn_count = 0
                print("\n[Conversation memory cleared. Starting fresh context.]\n")
                continue

            turn_count += 1
            start_time = time.perf_counter()
            try:
                response_text, history = run_agent(
                    user_input=user_input,
                    history=history,
                    app=app,
                    model=model,
                    base_url=base_url,
                    temperature=temperature,
                    num_predict=max_tokens,
                )
                latency = time.perf_counter() - start_time
                print(f"Jarvis > {response_text}\n")
                print(f"         [Latency: {latency:.3f}s | Turn: #{turn_count} | GPU: RTX 3050 | Model: {model} | MaxTokens: {max_tokens}]\n")
            except Exception as err:
                latency = time.perf_counter() - start_time
                print(f"\n[Error during inference: {err}]", file=sys.stderr)
                print(f"Make sure Ollama is responding at {base_url}.\n", file=sys.stderr)
    finally:
        scheduler.stop()


def run_single_query(
    query: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 150,
    enable_memory: bool = True,
) -> None:
    """Executes a single test query and prints the response and timing."""
    print(f"\n--- Testing Single Query with Model '{model}' ---")
    print(f"Query: {query}\n")
    start_time = time.perf_counter()
    try:
        response_text, _ = run_agent(
            user_input=query,
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens,
            enable_memory=enable_memory,
        )
        latency = time.perf_counter() - start_time
        print(f"\nJarvis > {response_text}\n")
        print(f"[Latency: {latency:.3f}s | Model: {model} | MaxTokens: {max_tokens}]\n")
    except Exception as err:
        latency = time.perf_counter() - start_time
        print(f"[Error during query: {err}]", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Local Jarvis - Interactive LangGraph Chat CLI (Phase 6: Vector Memory)"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Optional single query to run without entering interactive chat loop",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        "-u",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"Ollama API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"LLM sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum tokens to predict / num_predict ceiling (default: 150)",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable persistent vector memory retrieval and storage",
    )
    args = parser.parse_args()

    enable_memory = not args.no_memory

    if args.query:
        run_single_query(
            query=args.query,
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_memory=enable_memory,
        )
    else:
        chat_loop(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_memory=enable_memory,
        )


if __name__ == "__main__":
    main()
