"""
Local Jarvis - Desktop Automation Tools (Phase 4c)

Provides tools to launch desktop applications and safely type text into the active window.
Enforces strict fail-safe interactive confirmation before keystrokes are injected.
"""

import io
import os
import shutil
import subprocess
import sys
import time
from typing import Optional
from langchain_core.tools import tool

# Common application aliases and Windows protocols
APP_ALIASES = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "file explorer": "explorer.exe",
    "explorer": "explorer.exe",
    "terminal": "wt.exe",
    "command prompt": "cmd.exe",
    "powershell": "powershell.exe",
    "chrome": "chrome.exe",
    "edge": "msedge.exe",
    "task manager": "taskmgr.exe",
    "paint": "mspaint.exe",
    "settings": "ms-settings:",
    "vscode": "code.cmd",
    "code": "code.cmd",
}

# Window title keywords to locate and focus newly launched applications
APP_WINDOW_TITLE_KEYWORDS = {
    "notepad": ["notepad"],
    "calculator": ["calculator", "calc"],
    "calc": ["calculator", "calc"],
    "file explorer": ["file explorer", "explorer"],
    "explorer": ["file explorer", "explorer"],
    "terminal": ["terminal", "powershell", "command prompt", "cmd"],
    "command prompt": ["command prompt", "cmd"],
    "powershell": ["powershell"],
    "chrome": ["google chrome", "chrome"],
    "edge": ["microsoft edge", "edge"],
    "task manager": ["task manager"],
    "paint": ["paint"],
    "settings": ["settings"],
    "vscode": ["visual studio code", "code"],
    "code": ["visual studio code", "code"],
}


def get_active_window_title() -> str:
    """Safely retrieves the title of the currently focused desktop window."""
    try:
        import pygetwindow
        title = pygetwindow.getActiveWindowTitle()
        if title and title.strip():
            return title.strip()
    except Exception:
        pass

    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if hwnd:
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                if buf.value and buf.value.strip():
                    return buf.value.strip()
    except Exception:
        pass

    return "Unknown Window (No active focus detected)"


def focus_window_by_name(app_name: str, max_retries: int = 4, retry_delay: float = 0.35) -> bool:
    """
    Attempts to locate and bring a newly launched application's window to foreground focus.
    Retries across a short delay window since the OS takes time to initialize top-level windows.

    :param app_name: Name or alias of the application.
    :param max_retries: Number of retry attempts.
    :param retry_delay: Delay in seconds between attempts.
    :return: True if the window was located and focused, False otherwise.
    """
    clean_name = app_name.strip().lower()
    keywords = APP_WINDOW_TITLE_KEYWORDS.get(clean_name, [clean_name])

    for attempt in range(1, max_retries + 1):
        time.sleep(retry_delay)

        # Strategy 1: pygetwindow
        try:
            import pygetwindow
            all_windows = pygetwindow.getAllWindows()
            for win in all_windows:
                if not win.title or not win.title.strip():
                    continue
                win_title_lower = win.title.lower()
                if any(kw in win_title_lower for kw in keywords):
                    if hasattr(win, "isMinimized") and win.isMinimized:
                        win.restore()
                    win.activate()
                    print(f"[Desktop Automation] Successfully brought window '{win.title}' to foreground focus (attempt {attempt}).")
                    return True
        except Exception:
            pass

        # Strategy 2: Windows ctypes EnumWindows fallback
        try:
            import ctypes
            found_hwnd = [None]
            found_title = [""]

            def enum_proc(hwnd, _):
                if ctypes.windll.user32.IsWindowVisible(hwnd):
                    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                        t = buf.value.lower()
                        if any(kw in t for kw in keywords):
                            found_hwnd[0] = hwnd
                            found_title[0] = buf.value
                            return False
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
            ctypes.windll.user32.EnumWindows(WNDENUMPROC(enum_proc), 0)

            if found_hwnd[0]:
                hwnd = found_hwnd[0]
                ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                ctypes.windll.user32.SetForegroundWindow(hwnd)
                print(f"[Desktop Automation] Successfully brought window '{found_title[0]}' to foreground focus via ctypes (attempt {attempt}).")
                return True
        except Exception:
            pass

    print(f"[Desktop Automation] Window for '{app_name}' could not be brought to foreground focus after {max_retries} attempts.")
    return False


def launch_app(app_name: str) -> str:
    """
    Launches a local application on Windows using os.startfile or subprocess.Popen,
    and attempts to bring the resulting window into foreground focus.

    :param app_name: Name or alias of the application to launch.
    :return: Status string indicating launch and focus outcome.
    """
    clean_name = app_name.strip().lower()
    if not clean_name:
        return "Error: Application name cannot be empty."

    target = APP_ALIASES.get(clean_name, app_name.strip())
    print(f"[Desktop Automation] Attempting to launch application: '{clean_name}' (target: '{target}')...")

    # If it's a URI protocol (e.g. 'ms-settings:', 'calculator:')
    if ":" in target and not os.path.exists(target):
        try:
            os.startfile(target)
            print(f"[Desktop Automation] Launched URI protocol '{target}'.")
            focused = focus_window_by_name(clean_name)
            if focused:
                return f"Application '{app_name}' launched successfully and brought into foreground focus."
            return f"Application '{app_name}' launched successfully, but could not automatically bring its window to foreground focus."
        except Exception as uri_err:
            err_msg = f"Error: Unable to launch application protocol '{target}' ({uri_err})."
            print(f"[Desktop Automation] {err_msg}")
            return err_msg

    # Check if executable exists directly or is resolvable in PATH
    resolved_path = (
        target
        if os.path.exists(target)
        else (shutil.which(target) or shutil.which(f"{target}.exe"))
    )

    if not resolved_path:
        # Fallback check if os.startfile can resolve registered App Paths
        try:
            os.startfile(target)
            print(f"[Desktop Automation] Launched '{app_name}' via os.startfile.")
            focused = focus_window_by_name(clean_name)
            if focused:
                return f"Application '{app_name}' launched successfully and brought into foreground focus."
            return f"Application '{app_name}' launched successfully, but could not automatically bring its window to foreground focus."
        except Exception:
            err_msg = (
                f"Error: Unable to launch application '{app_name}'. "
                f"Please verify the application name or executable path."
            )
            print(f"[Desktop Automation] {err_msg}")
            return err_msg

    launched = False
    # Try os.startfile first (standard Windows shell launcher)
    try:
        os.startfile(resolved_path)
        print(f"[Desktop Automation] Launched '{app_name}' via os.startfile.")
        launched = True
    except Exception:
        pass

    if not launched:
        # Fall back to subprocess.Popen with executable list (no shell=True)
        try:
            subprocess.Popen([resolved_path])
            print(f"[Desktop Automation] Launched '{app_name}' via subprocess.Popen.")
            launched = True
        except Exception as popen_err:
            err_msg = (
                f"Error: Unable to launch application '{app_name}'. "
                f"Please verify the application name or executable path."
            )
            print(f"[Desktop Automation] {err_msg} ({popen_err})")
            return err_msg

    # Attempt to bring newly launched window into foreground focus
    focused = focus_window_by_name(clean_name)
    if focused:
        return f"Application '{app_name}' launched successfully and brought into foreground focus."
    return f"Application '{app_name}' launched successfully, but could not automatically bring its window to foreground focus."


def type_text_into_active_window(text: str) -> str:
    """
    Types text into the active desktop window with mandatory fail-safe confirmation.

    NOTE ON FOCUS IN TERMINAL CLI:
    `type_text` targets whatever window is ACTIVE AT THE MOMENT THE TOOL RUNS.
    In the current terminal-based CLI harness, the active window will almost always
    be the terminal itself (e.g. 'Windows PowerShell' or 'Command Prompt'), because
    sending a chat query requires focusing the terminal. This is expected behavior,
    not a bug. It will resolve naturally in Phase 7 when hands-free voice input
    (wake word + STT) replaces terminal typing, allowing the user's target application
    (such as Notepad, browser, or code editor) to retain desktop focus while speaking.

    :param text: Text string to type.
    :return: Status string confirming outcome.
    """
    clean_text = text.strip() if text else ""
    if not clean_text:
        return "Error: No text provided to type. Please specify the text you would like typed."

    window_title = get_active_window_title()

    # High-visibility security warning banner
    print("\n" + "=" * 70)
    print(" [SECURITY WARNING: DESKTOP KEYSTROKE INJECTION]")
    print(f" Target Active Window: '{window_title}'")
    preview = clean_text[:80] + ("..." if len(clean_text) > 80 else "")
    print(f" Text to Type ({len(clean_text)} chars): '{preview}'")
    print("=" * 70)

    # Fail-safe interactive prompt:
    # If input() cannot be read (e.g. no terminal attached, EOF, non-interactive mode),
    # it must DEFAULT TO NOT TYPING and abort safely.
    try:
        prompt_str = f"Allow typing into active window '{window_title}'? [y/N]: "
        user_choice = input(prompt_str).strip().lower()
    except (EOFError, OSError, io.UnsupportedOperation, Exception) as input_err:
        abort_msg = (
            f"Typing cancelled - no interactive confirmation available ({input_err}). "
            f"No keystrokes were sent to '{window_title}'."
        )
        print(f"[Desktop Automation] {abort_msg}")
        return abort_msg

    if user_choice not in ["y", "yes"]:
        cancel_msg = f"Typing cancelled by user. No text was entered into '{window_title}'."
        print(f"[Desktop Automation] {cancel_msg}")
        return cancel_msg

    print(f"[Desktop Automation] Confirmed by user. Typing {len(clean_text)} characters into '{window_title}'...")
    time.sleep(0.3)  # Brief settling pause

    try:
        import pyautogui
        pyautogui.write(clean_text, interval=0.01)
        success_msg = f"Successfully typed {len(clean_text)} characters into '{window_title}'."
        print(f"[Desktop Automation] {success_msg}")
        return success_msg
    except Exception as exc:
        err_msg = f"Error during typing: {exc}"
        print(f"[Desktop Automation] {err_msg}")
        return err_msg


@tool
def open_application(app_name: str) -> str:
    """Launch a desktop application or program on the local system and focus its window.

    Use this tool whenever the user asks to open, launch, or start an application
    such as 'open notepad', 'launch calculator', 'open chrome', 'start terminal', etc.

    Args:
        app_name: The name or common title of the application to launch.
    """
    return launch_app(app_name)


@tool
def type_text(text: str) -> str:
    """Type given text into the currently active/focused desktop window.

    Simulates keyboard keystrokes to type text into whatever application window
    currently has focus (e.g. an open text document, search bar, code editor).
    Requires explicit interactive confirmation from the user in the terminal
    before sending any keystrokes for safety.

    Args:
        text: The exact string of text to type into the active window. You MUST extract
            all words, sentences, or phrases the user requested to be typed after verbs
            like 'type', 'write', 'enter', or 'input'.
            Examples:
            - User: 'type hello world' -> text='hello world'
            - User: 'type this should not appear' -> text='this should not appear'
            - User: 'enter user@example.com' -> text='user@example.com'
            - User: 'write a quick note' -> text='a quick note'
            Do not omit any words from the user's intended phrase, and never pass an empty string
            if text to type was requested.
    """
    return type_text_into_active_window(text)
