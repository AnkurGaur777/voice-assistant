"""
Local Jarvis - Desktop Automation Tools (Phase 4c)

Provides tools to launch desktop applications and safely type text into the active window.
Enforces strict fail-safe interactive confirmation before keystrokes are injected.
"""

import difflib
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple
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

# Uninstaller, documentation, and noise keywords to exclude from application discovery
NOISE_KEYWORDS = (
    "uninstall", "unins", "remove", "install additional", "release notes",
    "whats new", "what's new", "readme", "documentation", "manual", "help",
    "license", "privacy", "web page", "website", "support", "diagnostic",
)


def ensure_interactive_desktop() -> None:
    """
    Ensures the current thread is attached to the interactive desktop ('default').
    Enables window enumeration, focus switching, and UI Automation inspection even
    when executed from background threads or non-interactive console stations.
    """
    try:
        import ctypes
        hdesk = ctypes.windll.user32.OpenDesktopW("default", 0, False, 0x01FF)
        if hdesk:
            ctypes.windll.user32.SetThreadDesktop(hdesk)
    except Exception:
        pass


def get_shortcut_directories() -> List[str]:
    """Returns accessible Start Menu and Desktop directory paths for shortcut scanning."""
    dirs = []
    # All Users Start Menu
    prog_data = os.environ.get("ProgramData", r"C:\ProgramData")
    dirs.append(os.path.join(prog_data, "Microsoft", "Windows", "Start Menu", "Programs"))
    # Current User Start Menu
    app_data = os.environ.get("APPDATA", "")
    if app_data:
        dirs.append(os.path.join(app_data, "Microsoft", "Windows", "Start Menu", "Programs"))
    # User Desktop
    user_desktop = os.path.expanduser(r"~\Desktop")
    dirs.append(user_desktop)
    # Public Desktop
    public_desktop = os.path.join(os.environ.get("PUBLIC", r"C:\Users\Public"), "Desktop")
    dirs.append(public_desktop)
    return [os.path.normpath(d) for d in dirs if os.path.isdir(d)]


def scan_shortcuts() -> Dict[str, str]:
    """
    Recursively scans Start Menu and Desktop directories for .lnk and .url shortcuts.
    Returns a dictionary mapping clean application display names to full shortcut file paths.
    Filters out uninstaller, setup, readme, and documentation links.
    """
    shortcuts: Dict[str, str] = {}
    for folder in get_shortcut_directories():
        try:
            for root, _, files in os.walk(folder):
                for file in files:
                    lower_file = file.lower()
                    if not lower_file.endswith((".lnk", ".url")):
                        continue
                    base_name, _ = os.path.splitext(file)
                    lower_base = base_name.lower().strip()
                    if any(noise in lower_base for noise in NOISE_KEYWORDS):
                        continue
                    full_path = os.path.normpath(os.path.join(root, file))
                    # If duplicate base_name, prefer .lnk over .url
                    if base_name in shortcuts:
                        existing = shortcuts[base_name]
                        if existing.lower().endswith(".url") and full_path.lower().endswith(".lnk"):
                            shortcuts[base_name] = full_path
                    else:
                        shortcuts[base_name] = full_path
        except Exception:
            pass
    return shortcuts


# Cache for Windows Store / UWP applications (Name -> AppID)
_STORE_APPS_CACHE: Optional[Dict[str, str]] = None
_STORE_APPS_LOCK = threading.Lock()


def _fetch_store_apps_from_powershell() -> Dict[str, str]:
    """Queries Windows shell via PowerShell Get-StartApps to retrieve packaged UWP/Store apps."""
    apps: Dict[str, str] = {}
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-StartApps | ConvertTo-Json",
        ]
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=6.0,
            startupinfo=startupinfo,
        )
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            if isinstance(data, dict):
                data = [data]
            for item in data:
                name = item.get("Name", "").strip()
                appid = item.get("AppID", "").strip()
                if not name or not appid:
                    continue
                lower_name = name.lower()
                if any(noise in lower_name for noise in NOISE_KEYWORDS):
                    continue
                apps[name] = appid
    except Exception as e:
        print(f"[Desktop Automation] Note retrieving Store apps: {e}")
    return apps


def get_windows_store_apps(force_refresh: bool = False) -> Dict[str, str]:
    """
    Returns discovered Windows Store / UWP applications (Name -> AppID / AUMID).
    Caches results in memory for instant subsequent lookups.
    """
    global _STORE_APPS_CACHE
    with _STORE_APPS_LOCK:
        if _STORE_APPS_CACHE is not None and not force_refresh:
            return _STORE_APPS_CACHE
        _STORE_APPS_CACHE = _fetch_store_apps_from_powershell()
        return _STORE_APPS_CACHE


def _prewarm_store_apps_cache() -> None:
    """Pre-warms Store apps cache asynchronously in the background on module import."""
    t = threading.Thread(target=get_windows_store_apps, daemon=True)
    t.start()


# Automatically pre-warm cache on Windows in background
if sys.platform == "win32":
    _prewarm_store_apps_cache()


def find_matching_app(query: str) -> Optional[Tuple[str, str, str]]:
    """
    Resolves an application name across all discovery tiers using exact,
    substring, and fuzzy matching with ambiguity resolution.

    :param query: Application name requested by the user.
    :return: (display_name, target, kind) or None.
             kind is 'alias', 'shortcut', or 'store'.
    """
    clean_q = query.strip().lower()
    if not clean_q:
        return None

    # Tier 1: Check APP_ALIASES for exact match
    if clean_q in APP_ALIASES:
        return (clean_q, APP_ALIASES[clean_q], "alias")

    # Retrieve scanned shortcuts and store apps
    shortcuts = scan_shortcuts()
    store_apps = get_windows_store_apps()

    # Tier 2: Check for exact matches (case-insensitive) in shortcuts and Store apps
    for name, path in shortcuts.items():
        if name.lower() == clean_q:
            return (name, path, "shortcut")

    for name, app_id in store_apps.items():
        if name.lower() == clean_q:
            return (name, app_id, "store")

    # Tier 3: Substring / Word / Prefix Match with scoring & ambiguity resolution
    candidates: List[Tuple[float, int, str, str, str]] = []

    all_items = [(name, path, "shortcut") for name, path in shortcuts.items()] + \
                [(name, appid, "store") for name, appid in store_apps.items()]

    for name, target, kind in all_items:
        lower_name = name.lower()
        words = lower_name.replace("-", " ").replace("_", " ").split()

        score = 0.0
        # 1. Exact full word match: e.g. "code" in ["visual", "studio", "code"]
        if clean_q in words:
            score = 0.90
        # 2. Substring match: e.g. "insta" in "instagram"
        elif clean_q in lower_name:
            score = 0.80 + (len(clean_q) / len(lower_name)) * 0.15
        elif lower_name in clean_q:
            score = 0.75 + (len(lower_name) / len(clean_q)) * 0.15
        else:
            # 3. Fuzzy similarity
            ratio = difflib.SequenceMatcher(None, clean_q, lower_name).ratio()
            if ratio >= 0.60:
                score = ratio * 0.85

        if score > 0.0:
            # Ambiguity penalty for installers/helpers/configuration utilities
            # e.g., user asks for "Visual Studio", penalize "Visual Studio Installer"
            # so "Visual Studio Code" or primary application is preferred
            penalty = 0.0
            if any(k in lower_name for k in ("installer", "setup", "update", "config", "tool", "prompt")):
                penalty = 0.20

            final_score = score - penalty
            if final_score >= 0.50:
                # Store (final_score, -len(name), name, target, kind)
                # This ensures higher score wins, and for equal scores, shorter name wins
                candidates.append((final_score, -len(name), name, target, kind))

    if candidates:
        candidates.sort(reverse=True)
        best = candidates[0]
        return (best[2], best[3], best[4])

    return None


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


def get_window_by_app_name(app_name: str) -> Optional[Tuple[int, str]]:
    """
    Locates an open top-level application window matching app_name or its aliases.
    Restores the window if minimized and brings it to foreground focus.

    :param app_name: Name or alias of the application (e.g. 'whatsapp', 'instagram', 'notepad').
    :return: (hwnd, window_title) tuple if found, or None.
    """
    ensure_interactive_desktop()
    clean_name = app_name.strip().lower()
    if not clean_name:
        return None

    keywords = list(APP_WINDOW_TITLE_KEYWORDS.get(clean_name, []))
    if clean_name not in keywords:
        keywords.append(clean_name)
    for part in clean_name.split():
        if len(part) >= 3 and part not in keywords:
            keywords.append(part)

    matched = find_matching_app(app_name)
    if matched:
        disp_lower = matched[0].lower()
        if disp_lower not in keywords:
            keywords.append(disp_lower)
        for part in disp_lower.split():
            if len(part) >= 3 and part not in keywords and part not in ("the", "app", "pro", "for", "and", "microsoft"):
                keywords.append(part)

    # Strategy 1: pygetwindow
    try:
        import pygetwindow
        for win in pygetwindow.getAllWindows():
            if not win.title or not win.title.strip():
                continue
            title_lower = win.title.lower()
            if any(kw in title_lower for kw in keywords):
                if hasattr(win, "isMinimized") and win.isMinimized:
                    win.restore()
                win.activate()
                return (win._hWnd, win.title.strip())
    except Exception:
        pass

    # Strategy 2: Windows ctypes EnumWindows (handles minimized and UWP windows)
    try:
        import ctypes
        candidates = []
        noise_prefixes = ("Default IME", "MSCTFIME UI", "GDI+ Window", "OleMainThreadWndName")

        def enum_proc(hwnd, _):
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value.strip()
                if title and not any(title.startswith(np) for np in noise_prefixes):
                    t_lower = title.lower()
                    score = 0.0
                    if clean_name == t_lower:
                        score = 1.0
                    elif any(kw == t_lower for kw in keywords):
                        score = 0.95
                    elif any(f" {kw} " in f" {t_lower} " for kw in keywords):
                        score = 0.85
                    elif any(kw in t_lower for kw in keywords):
                        score = 0.75
                    else:
                        ratio = difflib.SequenceMatcher(None, clean_name, t_lower).ratio()
                        if ratio >= 0.65:
                            score = ratio * 0.70

                    if score > 0.0:
                        candidates.append((score, hwnd, title))
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
        ctypes.windll.user32.EnumWindows(WNDENUMPROC(enum_proc), 0)

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_hwnd, best_title = candidates[0]
            try:
                ctypes.windll.user32.ShowWindow(best_hwnd, 9)  # SW_RESTORE
                ctypes.windll.user32.SetForegroundWindow(best_hwnd)
            except Exception:
                pass
            return (best_hwnd, best_title)
    except Exception:
        pass

    return None


def focus_window_by_name(app_name: str, display_name: str = "", max_retries: int = 4, retry_delay: float = 0.35) -> bool:
    """
    Attempts to locate and bring a newly launched application's window to foreground focus.
    Retries across a short delay window since the OS takes time to initialize top-level windows.

    :param app_name: Requested name or alias of the application.
    :param display_name: Optional matched display title of the application.
    :param max_retries: Number of retry attempts.
    :param retry_delay: Delay in seconds between attempts.
    :return: True if the window was located and focused, False otherwise.
    """
    ensure_interactive_desktop()
    clean_name = app_name.strip().lower()
    keywords = list(APP_WINDOW_TITLE_KEYWORDS.get(clean_name, []))
    if display_name:
        disp_lower = display_name.strip().lower()
        if disp_lower not in keywords:
            keywords.append(disp_lower)
        for part in disp_lower.split():
            if len(part) >= 3 and part not in keywords and part not in ("the", "app", "pro", "for", "and", "microsoft"):
                keywords.append(part)
    if not keywords:
        keywords = [clean_name]

    for attempt in range(1, max_retries + 1):
        ensure_interactive_desktop()
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
    Launches a local application on Windows:
    1. Checks APP_ALIASES for instant fast-path shortcuts.
    2. Searches Start Menu and Desktop shortcuts (.lnk / .url).
    3. Searches Windows Store / UWP applications (shell:AppsFolder).
    4. Falls back to direct os.startfile / shell start.
    5. Returns descriptive error naming all searched catalogs if unfound.

    Attempts to bring the resulting application window into foreground focus.

    :param app_name: Name or alias of the application to launch.
    :return: Status string indicating launch and focus outcome.
    """
    clean_name = app_name.strip().lower()
    if not clean_name:
        return "Error: Application name cannot be empty."

    print(f"[Desktop Automation] Attempting to resolve and launch application: '{app_name}'...")

    match = find_matching_app(app_name)
    launched = False
    matched_display_name = app_name

    if match:
        matched_display_name, target, kind = match
        print(f"[Desktop Automation] Resolved '{app_name}' -> '{matched_display_name}' ({kind}: '{target}')")

        if kind == "alias":
            # Protocol URI (e.g. ms-settings:)
            if ":" in target and not os.path.exists(target):
                try:
                    os.startfile(target)
                    print(f"[Desktop Automation] Launched URI protocol '{target}'.")
                    launched = True
                except Exception as uri_err:
                    err_msg = f"Error: Unable to launch application protocol '{target}' ({uri_err})."
                    print(f"[Desktop Automation] {err_msg}")
                    return err_msg
            else:
                resolved_path = (
                    target
                    if os.path.exists(target)
                    else (shutil.which(target) or shutil.which(f"{target}.exe"))
                )
                if resolved_path:
                    try:
                        os.startfile(resolved_path)
                        print(f"[Desktop Automation] Launched '{matched_display_name}' via os.startfile.")
                        launched = True
                    except Exception:
                        try:
                            subprocess.Popen([resolved_path])
                            print(f"[Desktop Automation] Launched '{matched_display_name}' via subprocess.Popen.")
                            launched = True
                        except Exception:
                            pass
                else:
                    try:
                        os.startfile(target)
                        print(f"[Desktop Automation] Launched '{matched_display_name}' via os.startfile.")
                        launched = True
                    except Exception:
                        pass

        elif kind == "shortcut":
            # Target is a .lnk or .url file path
            try:
                os.startfile(target)
                print(f"[Desktop Automation] Launched shortcut '{target}' via os.startfile.")
                launched = True
            except Exception as lnk_err:
                print(f"[Desktop Automation] os.startfile on shortcut failed: {lnk_err}")

        elif kind == "store":
            # Target is an AppID / AUMID. Launch via shell:AppsFolder
            shell_uri = f"shell:AppsFolder\\{target}"
            try:
                os.startfile(shell_uri)
                print(f"[Desktop Automation] Launched Store app '{target}' via os.startfile({shell_uri}).")
                launched = True
            except Exception as store_err:
                print(f"[Desktop Automation] os.startfile on Store app failed: {store_err}")
                try:
                    subprocess.Popen(["explorer.exe", shell_uri])
                    print(f"[Desktop Automation] Launched Store app via explorer.exe {shell_uri}.")
                    launched = True
                except Exception as exp_err:
                    print(f"[Desktop Automation] explorer.exe fallback failed: {exp_err}")

    if not launched:
        # Tier 4: Direct os.startfile or shell start fallback (checks App Paths and PATH)
        try:
            os.startfile(app_name.strip())
            print(f"[Desktop Automation] Launched '{app_name}' via direct os.startfile fallback.")
            launched = True
            matched_display_name = app_name
        except Exception:
            pass

    if not launched:
        # Tier 5: Rich error reporting naming all searched catalogs
        err_msg = (
            f"Error: Unable to launch application '{app_name}'. "
            f"Could not resolve application across fast-path aliases, Start Menu & Desktop shortcuts, "
            f"or Windows Store applications. "
            f"Use '--list-launchable-apps' to view all discoverable applications on this machine."
        )
        print(f"[Desktop Automation] {err_msg}")
        return err_msg

    # Attempt to bring newly launched window into foreground focus
    focused = focus_window_by_name(clean_name, matched_display_name)
    if focused:
        return f"Application '{matched_display_name}' launched successfully and brought into foreground focus."
    return f"Application '{matched_display_name}' launched successfully, but could not automatically bring its window to foreground focus."



def auto_focus_input_field(hwnd: int) -> bool:
    """
    Inspects the active window for Edit or Document controls using UI Automation.
    If an input field (such as a chat message box or text editor) is found,
    attempts to set focus to it (or click inside it) before keystrokes are injected.
    Returns True if an input control was targeted, False otherwise.
    """
    if not hwnd:
        return False
    try:
        ensure_interactive_desktop()
        from pywinauto.controls.uiawrapper import UIAWrapper
        from pywinauto.uia_element_info import UIAElementInfo

        wrapper = UIAWrapper(UIAElementInfo(hwnd))

        edits = []
        try:
            for ed in wrapper.descendants(control_type="Edit"):
                edits.append(ed)
                if len(edits) >= 50:
                    break
        except Exception:
            pass

        try:
            for doc in wrapper.descendants(control_type="Document"):
                edits.append(doc)
                if len(edits) >= 60:
                    break
        except Exception:
            pass

        visible_edits = []
        for ed in edits:
            try:
                if ed.is_visible():
                    r = ed.rectangle()
                    if r.width() > 0 and r.height() > 0:
                        visible_edits.append(ed)
            except Exception:
                continue

        if not visible_edits:
            return False

        # Priority matching:
        # 1. Look for primary message composer keywords
        target_ctrl = None
        primary_keywords = ("type a message", "write a message", "message", "compose")
        for ed in visible_edits:
            name = (getattr(ed.element_info, "name", "") or ed.window_text() or "").lower()
            if any(kw in name for kw in primary_keywords):
                target_ctrl = ed
                break

        # 2. If no keyword match, pick the lowest edit control on screen (standard messaging app layout)
        if target_ctrl is None:
            visible_edits.sort(key=lambda e: e.rectangle().bottom, reverse=True)
            target_ctrl = visible_edits[0]

        # Target control focus
        if target_ctrl:
            name = getattr(target_ctrl.element_info, "name", "") or target_ctrl.window_text() or "Input field"
            try:
                target_ctrl.set_focus()
                print(f"[Desktop Automation] Automatically focused input control: '{name}'")
                return True
            except Exception:
                try:
                    target_ctrl.click_input()
                    print(f"[Desktop Automation] Automatically clicked into input control: '{name}'")
                    return True
                except Exception:
                    pass
    except Exception as exc:
        print(f"[Desktop Automation] Auto-focus input field check skipped: {exc}")
    return False


def type_text_into_active_window(text: str, press_enter: bool = False) -> str:
    """
    Types text into the active desktop window with mandatory fail-safe confirmation.
    Optionally presses the Enter key afterwards (e.g. to send a message or submit)
    under the same combined confirmation prompt.

    NOTE ON FOCUS IN TERMINAL CLI:
    `type_text` targets whatever window is ACTIVE AT THE MOMENT THE TOOL RUNS.
    In the current terminal-based CLI harness, the active window will almost always
    be the terminal itself (e.g. 'Windows PowerShell' or 'Command Prompt'), because
    sending a chat query requires focusing the terminal. This is expected behavior,
    not a bug. It will resolve naturally in Phase 7 when hands-free voice input
    (wake word + STT) replaces terminal typing, allowing the user's target application
    (such as Notepad, browser, or code editor) to retain desktop focus while speaking.

    :param text: Text string to type.
    :param press_enter: Whether to press Enter after typing to send or submit.
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
    action_desc = "Type and Press Enter (Send)" if press_enter else "Type Text"
    print(f" Action: {action_desc}")
    preview = clean_text[:80] + ("..." if len(clean_text) > 80 else "")
    print(f" Text to Type ({len(clean_text)} chars): '{preview}'")
    print("=" * 70)

    # Fail-safe interactive prompt:
    # If input() cannot be read (e.g. no terminal attached, EOF, non-interactive mode),
    # it must DEFAULT TO NOT TYPING and abort safely.
    try:
        if press_enter:
            prompt_str = f"Type '{preview}' and press enter into active window '{window_title}'? [y/N]: "
        else:
            prompt_str = f"Allow typing into active window '{window_title}'? [y/N]: "
        user_choice = input(prompt_str).strip().lower()
    except (EOFError, OSError, io.UnsupportedOperation, Exception) as input_err:
        abort_action = "Typing and enter keypress" if press_enter else "Typing"
        abort_msg = (
            f"{abort_action} cancelled - no interactive confirmation available ({input_err}). "
            f"No keystrokes were sent to '{window_title}'."
        )
        print(f"[Desktop Automation] {abort_msg}")
        return abort_msg

    if user_choice not in ["y", "yes"]:
        cancel_action = "Typing and enter keypress" if press_enter else "Typing"
        cancel_msg = f"{cancel_action} cancelled by user. No text was entered into '{window_title}'."
        print(f"[Desktop Automation] {cancel_msg}")
        return cancel_msg

    action_verb = "Typing and pressing enter" if press_enter else "Typing"
    print(f"[Desktop Automation] Confirmed by user. {action_verb} {len(clean_text)} characters into '{window_title}'...")
    time.sleep(0.3)  # Brief settling pause

    # Smart targeting: Auto-focus most likely text input control in active window
    try:
        import ctypes
        active_hwnd = ctypes.windll.user32.GetForegroundWindow()
        if active_hwnd:
            auto_focus_input_field(active_hwnd)
    except Exception:
        pass

    try:
        import pyautogui
        pyautogui.write(clean_text, interval=0.01)
        if press_enter:
            time.sleep(0.05)
            pyautogui.press("enter")
            success_msg = f"Successfully typed {len(clean_text)} characters and pressed enter into '{window_title}'."
        else:
            success_msg = f"Successfully typed {len(clean_text)} characters into '{window_title}'."
        print(f"[Desktop Automation] {success_msg}")
        return success_msg
    except Exception as exc:
        err_msg = f"Error during typing: {exc}"
        print(f"[Desktop Automation] {err_msg}")
        return err_msg


def press_enter_in_active_window() -> str:
    """
    Simulates pressing the Enter key in the currently active desktop window
    with mandatory fail-safe confirmation.

    :return: Status string confirming outcome.
    """
    window_title = get_active_window_title()

    print("\n" + "=" * 70)
    print(" [SECURITY WARNING: DESKTOP KEYSTROKE INJECTION]")
    print(f" Target Active Window: '{window_title}'")
    print(" Action: Press Enter Key (Send / Submit)")
    print("=" * 70)

    try:
        prompt_str = f"Press enter in active window '{window_title}'? [y/N]: "
        user_choice = input(prompt_str).strip().lower()
    except (EOFError, OSError, io.UnsupportedOperation, Exception) as input_err:
        abort_msg = (
            f"Enter keypress cancelled - no interactive confirmation available ({input_err}). "
            f"No keystrokes were sent to '{window_title}'."
        )
        print(f"[Desktop Automation] {abort_msg}")
        return abort_msg

    if user_choice not in ["y", "yes"]:
        cancel_msg = f"Enter keypress cancelled by user. No key was sent to '{window_title}'."
        print(f"[Desktop Automation] {cancel_msg}")
        return cancel_msg

    print(f"[Desktop Automation] Confirmed by user. Pressing enter into '{window_title}'...")
    time.sleep(0.2)

    try:
        import pyautogui
        pyautogui.press("enter")
        success_msg = f"Successfully pressed enter in '{window_title}'."
        print(f"[Desktop Automation] {success_msg}")
        return success_msg
    except Exception as exc:
        err_msg = f"Error pressing enter: {exc}"
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
def type_text(text: str, press_enter: bool = False) -> str:
    """Type given text into the currently active/focused desktop window, optionally pressing enter.

    Simulates keyboard keystrokes to type text into whatever application window
    currently has focus (e.g. an open text document, WhatsApp, Messages, email compose, search bar).
    If press_enter is True, it simulates pressing the Enter key immediately after typing
    (useful to send a chat message, search, or submit a form) as a single confirmed action.
    Requires explicit interactive confirmation from the user in the terminal
    before sending any keystrokes for safety.

    Args:
        text: The exact string of text to type into the active window. You MUST extract
            all words, sentences, or phrases the user requested to be typed after verbs
            like 'type', 'write', 'enter', or 'input'.
            Examples:
            - User: 'type hello world' -> text='hello world', press_enter=False
            - User: 'type hello and send it' -> text='hello', press_enter=True
            - User: 'write how are you and press enter' -> text='how are you', press_enter=True
        press_enter: Optional boolean flag. Set to True if the user asks to send or press enter
            after typing. Default is False.
    """
    return type_text_into_active_window(text, press_enter=press_enter)


@tool
def press_enter_key() -> str:
    """Press the Enter key in the currently active/focused desktop window.

    Simulates pressing the Enter keyboard key in whatever application window
    currently has focus (e.g. sending an already-typed message in WhatsApp Web / Slack,
    submitting a focused web form, or triggering the default action).
    Requires explicit interactive confirmation from the user before sending the keystroke.
    """
    return press_enter_in_active_window()


def scroll_active_window(direction: str = "down", amount: int = 3) -> str:
    """
    Scrolls the active window up or down centered at the active window's bounding box.
    Clamps amount between 1 and 25.

    :param direction: 'up' or 'down'.
    :param amount: Number of scroll clicks.
    :return: Status string confirming outcome.
    """
    ensure_interactive_desktop()
    d = (direction or "down").strip().lower()
    if d in ("up", "top", "pageup", "scrollup", "u"):
        norm_dir = "up"
    elif d in ("down", "bottom", "pagedown", "scrolldown", "d"):
        norm_dir = "down"
    else:
        return f"Error: Invalid scroll direction '{direction}'. Must be 'up' or 'down'."

    try:
        clamped_amount = max(1, min(int(amount), 25))
    except (TypeError, ValueError):
        clamped_amount = 3

    window_title = get_active_window_title()

    cx, cy = None, None
    try:
        import ctypes
        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if hwnd:
            rect = RECT()
            if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                w = rect.right - rect.left
                h = rect.bottom - rect.top
                if w > 0 and h > 0:
                    cx = rect.left + w // 2
                    cy = rect.top + h // 2
    except Exception:
        pass

    if cx is None or cy is None:
        try:
            import pygetwindow
            win = pygetwindow.getActiveWindow()
            if win and win.width > 0 and win.height > 0:
                cx = win.left + win.width // 2
                cy = win.top + win.height // 2
        except Exception:
            pass

    clicks = clamped_amount * 120 if norm_dir == "up" else -clamped_amount * 120

    try:
        import pyautogui
        pyautogui.scroll(clicks, x=cx, y=cy)
        center_info = f" (centered at ({cx}, {cy}))" if cx is not None and cy is not None else ""
        msg = f"Successfully scrolled active window '{window_title}' {norm_dir} by {clamped_amount} clicks{center_info}."
        print(f"[Desktop Automation] {msg}")
        return msg
    except Exception as exc:
        err_msg = f"Error scrolling window: {exc}"
        print(f"[Desktop Automation] {err_msg}")
        return err_msg


@tool
def scroll_window(direction: str = "down", amount: int = 3) -> str:
    """Scroll the currently active desktop window up or down by a specified amount.

    Centers the scroll event on the active window's bounding box so events are
    delivered directly to the target application rather than whatever happens to
    be under the user's mouse cursor.

    Args:
        direction: 'up' to scroll up (earlier content, top of page) or 'down' to
            scroll down (later content, feed, reading further). Default is 'down'.
        amount: Number of scroll notches or clicks (1 to 25). Default is 3 clicks.
    """
    return scroll_active_window(direction=direction, amount=amount)


def click_element_in_app(app_name: str, element_name: str) -> str:
    """
    Locates an application window, searches its UI Automation element tree
    for an accessible control matching element_name, and triggers/clicks it.
    Prioritizes InvokePattern / TogglePattern / SelectionItemPattern without
    moving the user's cursor, falling back to click_input() if necessary.

    :param app_name: Name or alias of the application (e.g. 'Instagram', 'WhatsApp', 'Notepad').
    :param element_name: Name or label of the UI control to click (e.g. 'Reels', 'Chats', 'File').
    :return: Status string confirming outcome.
    """
    ensure_interactive_desktop()
    clean_app = app_name.strip()
    clean_elem = element_name.strip()
    if not clean_app:
        return "Error: Application name cannot be empty."
    if not clean_elem:
        return "Error: Element name cannot be empty."

    win_info = get_window_by_app_name(clean_app)
    if not win_info:
        return (
            f"Error: Could not locate an open window for application '{clean_app}'. "
            f"Please verify the application is open and running."
        )

    hwnd, window_title = win_info
    time.sleep(0.15)

    try:
        from pywinauto.controls.uiawrapper import UIAWrapper
        from pywinauto.uia_element_info import UIAElementInfo
        wrapper = UIAWrapper(UIAElementInfo(hwnd))
    except Exception as conn_err:
        return f"Error: Could not connect to UI Automation for '{window_title}': {conn_err}"

    target = clean_elem.lower()
    candidates = []

    # Phase 1: Direct title match using COM condition
    test_titles = [clean_elem, clean_elem.title(), clean_elem.capitalize(), clean_elem.lower()]
    for tt in test_titles:
        try:
            matched = wrapper.descendants(title=tt)
            for m in matched:
                candidates.append(m)
        except Exception:
            pass
        if candidates:
            break

    # Phase 2: If direct title search didn't yield enough, search common interactive control types
    if not candidates:
        interactive_types = [
            "Button", "TabItem", "MenuItem", "Hyperlink",
            "ListItem", "CheckBox", "RadioButton", "SplitButton",
            "ComboBox", "Text", "Pane",
        ]
        start_time = time.time()
        for itype in interactive_types:
            if time.time() - start_time > 3.0:
                break
            try:
                for el in wrapper.descendants(control_type=itype):
                    candidates.append(el)
                    if len(candidates) >= 250:
                        break
            except Exception:
                pass
            if len(candidates) >= 250:
                break

    # Phase 3: BFS fallback for shallow top-level controls
    if not candidates:
        try:
            queue = [(wrapper, 0)]
            t_bfs = time.time()
            while queue and len(candidates) < 100 and (time.time() - t_bfs) < 1.5:
                curr, depth = queue.pop(0)
                try:
                    name = (curr.window_text() or curr.element_info.name or "").strip()
                    if name:
                        candidates.append(curr)
                    if depth < 5:
                        for ch in curr.children():
                            queue.append((ch, depth + 1))
                except Exception:
                    continue
        except Exception:
            pass

    if not candidates:
        return f"Error: No accessible UI elements found in '{window_title}' to search."

    # Score candidates
    scored = []
    seen_ids = set()

    for ctrl in candidates:
        try:
            info = ctrl.element_info
            r_id = getattr(info, "runtime_id", None)
            if r_id is not None:
                r_key = tuple(r_id)
                if r_key in seen_ids:
                    continue
                seen_ids.add(r_key)

            name = (ctrl.window_text() or info.name or "").strip()
            if not name:
                continue

            name_lower = name.lower()
            ctype = ctrl.friendly_class_name()
            rect = ctrl.rectangle()
            w = rect.width()
            h = rect.height()

            score = 0.0
            if target == name_lower:
                score = 1.0
            elif target in name_lower.split():
                score = 0.90
            elif target in name_lower:
                score = 0.75 + 0.15 * (len(target) / max(1, len(name_lower)))
            elif name_lower in target:
                score = 0.70 + 0.15 * (len(name_lower) / max(1, len(target)))
            else:
                ratio = difflib.SequenceMatcher(None, target, name_lower).ratio()
                if ratio >= 0.65:
                    score = ratio * 0.80

            if score < 0.50:
                continue

            # Check visibility
            try:
                vis = ctrl.is_visible() and w > 0 and h > 0
            except Exception:
                vis = w > 0 and h > 0

            if vis:
                score += 0.15
            else:
                score -= 0.20

            # Control type preference
            if ctype in ("Button", "TabItem", "MenuItem", "Hyperlink", "ListItem", "SplitButton"):
                score += 0.10
            elif ctype == "Pane":
                score -= 0.10

            try:
                if ctrl.is_enabled():
                    score += 0.05
            except Exception:
                pass

            scored.append((score, ctrl, name, ctype, vis))
        except Exception:
            continue

    if not scored:
        return (
            f"Error: Could not find any element matching '{clean_elem}' in '{window_title}'. "
            f"No matching accessible controls found."
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_ctrl, best_name, best_type, best_vis = scored[0]

    # Perform activation/click: invoke first, fallback to click_input
    action_taken = None
    method = ""

    if hasattr(best_ctrl, "invoke"):
        try:
            best_ctrl.invoke()
            action_taken = "invoked"
            method = "invoked via UI Automation (cursor unmoved)"
        except Exception:
            pass

    if not action_taken and hasattr(best_ctrl, "toggle"):
        try:
            best_ctrl.toggle()
            action_taken = "toggled"
            method = "toggled via UI Automation (cursor unmoved)"
        except Exception:
            pass

    if not action_taken and hasattr(best_ctrl, "select"):
        try:
            best_ctrl.select()
            action_taken = "selected"
            method = "selected via UI Automation (cursor unmoved)"
        except Exception:
            pass

    if not action_taken:
        try:
            best_ctrl.click_input()
            action_taken = "clicked"
            method = "clicked via input simulation (click_input)"
        except Exception as click_err:
            return (
                f"Error: Located matching element '{best_name}' ({best_type}) in '{window_title}', "
                f"but clicking failed: {click_err}"
            )

    note = ""
    high_matches = [s for s in scored if s[0] >= 0.70]
    if len(high_matches) > 1:
        note = f" (Selected '{best_name}' from {len(high_matches)} matching candidates)"

    return f"Successfully {action_taken} element '{best_name}' ({best_type}) in '{window_title}' [{method}]{note}."


@tool
def find_and_click_element(app_name: str, element_name: str) -> str:
    """Find and click a specific UI element (button, tab, menu item, link) in an application.

    Uses Windows UI Automation (pywinauto) to connect to the named application's window,
    search its accessibility control tree for an element matching element_name,
    and activate/click it.

    Tries direct UI Automation invocation (InvokePattern, TogglePattern, SelectionItemPattern)
    first so your physical mouse cursor is not moved. If the control does not support direct
    invocation, it falls back to simulated mouse click (click_input).

    NOTE: Relies on standard Windows Accessibility trees. Custom-drawn canvas apps or
    games without accessibility labeling cannot be controlled with this tool.

    Args:
        app_name: Name of the application (e.g. 'Instagram', 'WhatsApp', 'Notepad', 'Chrome').
        element_name: Name or label of the control to click (e.g. 'Reels', 'Chats', 'File', 'Settings').
    """
    return click_element_in_app(app_name=app_name, element_name=element_name)


def list_ui_elements(app_name: str) -> List[Dict[str, str]]:
    """
    Lists accessible UI elements for a given running application.
    Displays control type, name, visibility, and invocation capability.

    :param app_name: Name or alias of the application.
    :return: List of element metadata dictionaries.
    """
    ensure_interactive_desktop()
    win_info = get_window_by_app_name(app_name)
    if not win_info:
        print(f"\n[Desktop Automation] Application '{app_name}' window not found.")
        return []

    hwnd, window_title = win_info
    print("\n" + "=" * 80)
    print(f" LOCAL JARVIS - ACCESSIBLE UI ELEMENTS FOR '{app_name}'")
    print(f" Target Window: '{window_title}' (HWND: {hwnd})")
    print("=" * 80)

    try:
        from pywinauto.controls.uiawrapper import UIAWrapper
        from pywinauto.uia_element_info import UIAElementInfo
        wrapper = UIAWrapper(UIAElementInfo(hwnd))
    except Exception as err:
        print(f"Error connecting to UI Automation: {err}")
        return []

    elements: List[Dict[str, str]] = []
    seen_names = set()
    queue = [(wrapper, 0)]
    t0 = time.time()

    while queue and len(elements) < 120 and (time.time() - t0) < 3.0:
        curr, depth = queue.pop(0)
        try:
            name = (curr.window_text() or curr.element_info.name or "").strip()
            ctype = curr.friendly_class_name()
            if name:
                key = (ctype, name)
                if key not in seen_names:
                    seen_names.add(key)
                    try:
                        vis = curr.is_visible()
                    except Exception:
                        vis = False
                    try:
                        enabled = curr.is_enabled()
                    except Exception:
                        enabled = False
                    has_inv = hasattr(curr, "invoke")
                    elements.append({
                        "control_type": ctype,
                        "name": name,
                        "visible": str(vis),
                        "enabled": str(enabled),
                        "invoke_supported": str(has_inv),
                    })
            if depth < 6:
                for ch in curr.children():
                    queue.append((ch, depth + 1))
        except Exception:
            continue

    if not elements:
        print("  (No accessible controls with non-empty names discovered)")
    else:
        for idx, el in enumerate(elements, 1):
            print(
                f"  {idx:>3}. [{el['control_type']:<14}] \"{el['name']}\" "
                f"(visible={el['visible']}, enabled={el['enabled']}, invoke={el['invoke_supported']})"
            )

    print("=" * 80)
    print(f" Total Elements Discovered: {len(elements)}")
    print("=" * 80 + "\n")
    return elements


def list_launchable_apps() -> List[Dict[str, str]]:
    """
    Scans and prints all discoverable applications on this system across
    Fast-Path Aliases, Start Menu & Desktop Shortcuts, and Windows Store Apps.

    :return: List of application metadata dictionaries.
    """
    shortcuts = scan_shortcuts()
    store_apps = get_windows_store_apps(force_refresh=True)

    print("\n" + "=" * 78)
    print(" LOCAL JARVIS - DISCOVERABLE LAUNCHABLE APPLICATIONS")
    print("=" * 78)

    print(f"\n[1] Fast-Path Aliases ({len(APP_ALIASES)} shortcuts)")
    print("-" * 78)
    for alias, target in sorted(APP_ALIASES.items()):
        print(f"  * {alias:<24} -> {target}")

    print(f"\n[2] Start Menu & Desktop Shortcuts ({len(shortcuts)} applications)")
    print("-" * 78)
    for name, path in sorted(shortcuts.items(), key=lambda x: x[0].lower()):
        print(f"  * {name:<30} -> {path}")

    print(f"\n[3] Windows Store / UWP Applications ({len(store_apps)} applications)")
    print("-" * 78)
    for name, appid in sorted(store_apps.items(), key=lambda x: x[0].lower()):
        print(f"  * {name:<30} -> {appid}")

    total = len(APP_ALIASES) + len(shortcuts) + len(store_apps)
    print("\n" + "=" * 78)
    print(f" Total Launchable Applications Discovered: {total}")
    print("=" * 78 + "\n")

    items = []
    for k, v in APP_ALIASES.items():
        items.append({"name": k, "target": v, "type": "alias"})
    for k, v in shortcuts.items():
        items.append({"name": k, "target": v, "type": "shortcut"})
    for k, v in store_apps.items():
        items.append({"name": k, "target": v, "type": "store"})
    return items


if __name__ == "__main__":
    if "--list-launchable-apps" in sys.argv:
        list_launchable_apps()
    elif "--list-ui-elements" in sys.argv:
        idx = sys.argv.index("--list-ui-elements")
        if idx + 1 < len(sys.argv):
            app = sys.argv[idx + 1]
            list_ui_elements(app)
        else:
            print("Error: Please specify an application name after --list-ui-elements")
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        target_app = " ".join(sys.argv[1:])
        result = launch_app(target_app)
        print(result)
    else:
        print("Usage:")
        print("  python -m src.agent.tools.desktop --list-launchable-apps")
        print("  python -m src.agent.tools.desktop --list-ui-elements <app_name>")
        print("  python -m src.agent.tools.desktop <app_name>")
