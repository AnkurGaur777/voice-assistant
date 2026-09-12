"""
Local Jarvis - Clipboard Tools (Phase 4b)

Provides tools to read and summarize system clipboard contents using pyperclip.
Features safety limits (max character truncation) to prevent local LLM context overflow.
"""

from typing import Optional
import pyperclip
from langchain_core.tools import tool

DEFAULT_MAX_CLIPBOARD_CHARS = 4000


def get_clipboard_text(max_chars: int = DEFAULT_MAX_CLIPBOARD_CHARS) -> str:
    """
    Safely retrieves text from the system clipboard.

    :param max_chars: Maximum characters to return before truncating.
    :return: Clipboard text string or diagnostic message if empty or unavailable.
    """
    try:
        content = pyperclip.paste()
    except Exception as exc:
        err_msg = f"Error: Unable to access system clipboard ({exc})."
        print(f"[Clipboard] {err_msg}")
        return err_msg

    if not content or not content.strip():
        print("[Clipboard] Clipboard is empty or contains non-text data.")
        return "The clipboard is currently empty or contains non-text content."

    total_len = len(content)
    if total_len > max_chars:
        truncated = content[:max_chars]
        print(f"[Clipboard] Retrieved {total_len} characters (truncated to {max_chars}).")
        return (
            f"[Note: Clipboard content truncated to first {max_chars} characters ({total_len} total characters).]\n\n"
            f"{truncated}"
        )

    print(f"[Clipboard] Retrieved {total_len} characters.")
    return content


@tool
def read_clipboard() -> str:
    """Read the current text content from the system clipboard.

    Use this tool whenever the user asks 'what's on my clipboard', 'read my clipboard',
    wants to inspect or check copied text, or asks specific questions about the copied content.
    """
    print("[Clipboard Tool] Reading system clipboard...")
    return get_clipboard_text()


@tool
def summarize_clipboard(focus: Optional[str] = None) -> str:
    """Read the system clipboard and return its content formatted for summarization.

    Use this tool whenever the user asks to summarize, give an overview, or highlight key
    points of text currently copied to the clipboard (e.g. 'summarize my clipboard',
    'give me the gist of what I copied', 'summarize this article from my clipboard').

    Args:
        focus: Optional specific topic or aspect the user wants the summary to focus on.
    """
    print(f"[Clipboard Tool] Reading system clipboard for summarization (focus={focus})...")
    raw_content = get_clipboard_text()

    # If empty or error, return directly without summarization boilerplate
    if (
        raw_content.startswith("The clipboard is currently empty")
        or raw_content.startswith("Error:")
    ):
        return raw_content

    word_count = len(raw_content.split())
    char_count = len(raw_content)

    focus_directive = f" Focus specifically on: {focus.strip()}." if focus and focus.strip() else ""

    formatted_payload = (
        f"Clipboard Content for Summarization ({word_count} words, {char_count} characters):\n"
        f"----------------------------------------\n"
        f"{raw_content}\n"
        f"----------------------------------------\n"
        f"[Guidance: Provide a concise, spoken-friendly summary (2-3 sentences, under 40 words unless the user requested more detail).{focus_directive}]"
    )
    return formatted_payload
