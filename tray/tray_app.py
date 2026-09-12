"""
Local Jarvis - System Tray Application (Phase 7)

Provides a Windows system tray icon using `pystray` and `Pillow`.
Supports 4 visual states corresponding to the voice assistant pipeline stages:
1. LISTENING / IDLE (Cyan/Blue circle with microphone icon)
2. PROCESSING (Amber/Orange circle with activity indicator - transcribing & thinking)
3. SPEAKING (Emerald/Green circle with audio speaker icon)
4. ERROR (Crimson/Red circle with exclamation mark)

Runs in a separate background thread alongside the orchestrator's main loop.
Provides graceful exit via right-click menu -> 'Quit' or programmatically.
"""

from enum import Enum
import logging
import threading
import time
from typing import Callable, Dict, Optional

from PIL import Image, ImageDraw
import pystray


class JarvisTrayState(str, Enum):
    LISTENING = "listening"
    IDLE = "idle"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


# State display descriptions for tooltip and menu
STATE_DESCRIPTIONS: Dict[str, str] = {
    JarvisTrayState.LISTENING: "Listening for 'Hey Jarvis'...",
    JarvisTrayState.IDLE: "Idle (Listening)...",
    JarvisTrayState.PROCESSING: "Processing (Transcribing / Thinking)...",
    JarvisTrayState.SPEAKING: "Speaking response...",
    JarvisTrayState.ERROR: "Error encountered",
}


def create_state_icon(state: str, size: int = 64) -> Image.Image:
    """
    Generates a crisp 64x64 RGBA system tray icon for the given pipeline state.

    :param state: One of 'listening', 'idle', 'processing', 'speaking', 'error'.
    :param size: Icon dimension in pixels (default 64x64).
    :return: A PIL.Image.Image in RGBA mode.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 4
    bbox = [margin, margin, size - margin, size - margin]

    state_norm = (state or "").lower().strip()
    cx, cy = size // 2, size // 2

    if state_norm in (JarvisTrayState.LISTENING.value, JarvisTrayState.IDLE.value):
        # Cyan / Electric Blue circle
        bg_color = (0, 176, 255, 255)
        draw.ellipse(bbox, fill=bg_color)
        # Microphone glyph
        draw.rounded_rectangle([cx - 4, cy - 12, cx + 4, cy + 2], radius=4, fill=(255, 255, 255, 255))
        draw.arc([cx - 8, cy - 6, cx + 8, cy + 6], start=0, end=180, fill=(255, 255, 255, 255), width=2)
        draw.line([cx, cy + 6, cx, cy + 12], fill=(255, 255, 255, 255), width=2)
        draw.line([cx - 6, cy + 12, cx + 6, cy + 12], fill=(255, 255, 255, 255), width=2)
    elif state_norm == JarvisTrayState.PROCESSING.value:
        # Amber / Orange circle
        bg_color = (255, 145, 0, 255)
        draw.ellipse(bbox, fill=bg_color)
        # 3 horizontal activity dots
        for offset in (-12, 0, 12):
            draw.ellipse([cx + offset - 3, cy - 3, cx + offset + 3, cy + 3], fill=(255, 255, 255, 255))
    elif state_norm == JarvisTrayState.SPEAKING.value:
        # Emerald Green circle
        bg_color = (0, 230, 118, 255)
        draw.ellipse(bbox, fill=bg_color)
        # Speaker cone
        draw.polygon(
            [
                (cx - 10, cy - 6),
                (cx - 4, cy - 6),
                (cx + 2, cy - 12),
                (cx + 2, cy + 12),
                (cx - 4, cy + 6),
                (cx - 10, cy + 6),
            ],
            fill=(255, 255, 255, 255),
        )
        # Sound waves arcs
        draw.arc([cx - 2, cy - 8, cx + 10, cy + 8], start=300, end=60, fill=(255, 255, 255, 255), width=2)
        draw.arc([cx + 2, cy - 14, cx + 16, cy + 14], start=300, end=60, fill=(255, 255, 255, 255), width=2)
    elif state_norm == JarvisTrayState.ERROR.value:
        # Crimson Red circle
        bg_color = (255, 23, 68, 255)
        draw.ellipse(bbox, fill=bg_color)
        # Exclamation mark
        draw.line([cx, cy - 12, cx, cy + 2], fill=(255, 255, 255, 255), width=4)
        draw.ellipse([cx - 2, cy + 8, cx + 2, cy + 12], fill=(255, 255, 255, 255))
    else:
        # Default neutral slate
        draw.ellipse(bbox, fill=(120, 144, 156, 255))

    return img


class JarvisTrayApp:
    """
    System tray icon controller for Local Jarvis.
    """

    def __init__(
        self,
        on_quit: Optional[Callable[[], None]] = None,
        initial_state: JarvisTrayState = JarvisTrayState.LISTENING,
    ):
        self._on_quit = on_quit
        self._current_state = initial_state
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Build initial icon and menu
        self._icon_image = create_state_icon(initial_state.value)
        self._menu = pystray.Menu(
            pystray.MenuItem(lambda text: f"Status: {self._get_status_text()}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit Jarvis", self._handle_quit),
        )

        self._icon = pystray.Icon(
            name="LocalJarvis",
            icon=self._icon_image,
            title=f"Local Jarvis - {STATE_DESCRIPTIONS.get(initial_state, 'Active')}",
            menu=self._menu,
        )

    @property
    def current_state(self) -> JarvisTrayState:
        with self._lock:
            return self._current_state

    def _get_status_text(self) -> str:
        with self._lock:
            state_val = self._current_state.value if isinstance(self._current_state, JarvisTrayState) else str(self._current_state)
            return state_val.capitalize()

    def set_state(self, state: JarvisTrayState | str, custom_message: Optional[str] = None) -> None:
        """
        Updates the tray icon's visual appearance and tooltip in a thread-safe manner.

        :param state: Target JarvisTrayState or string ('listening', 'processing', 'speaking', 'error').
        :param custom_message: Optional custom tooltip string.
        """
        with self._lock:
            if isinstance(state, str):
                try:
                    state_enum = JarvisTrayState(state.lower())
                except ValueError:
                    state_enum = JarvisTrayState.IDLE
            else:
                state_enum = state

            self._current_state = state_enum
            new_img = create_state_icon(state_enum.value)
            desc = custom_message or STATE_DESCRIPTIONS.get(state_enum, state_enum.value.capitalize())

            if self._icon is not None:
                try:
                    self._icon.icon = new_img
                    self._icon.title = f"Local Jarvis - {desc}"
                except Exception as e:
                    logging.debug(f"[TrayApp] Note updating icon: {e}")

    def _handle_quit(self, icon=None, item=None) -> None:
        """Callback triggered when the user clicks 'Quit' in the system tray menu."""
        print("\n[Tray] 'Quit' selected from system tray menu. Initiating graceful shutdown...")
        self.stop()
        if self._on_quit:
            try:
                self._on_quit()
            except Exception as e:
                logging.error(f"[TrayApp] Error in on_quit callback: {e}")

    def start(self) -> None:
        """Starts the system tray icon loop in a background daemon thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_icon, name="JarvisTrayThread", daemon=True)
        self._thread.start()

    def _run_icon(self) -> None:
        try:
            self._icon.run()
        except Exception as e:
            logging.debug(f"[TrayApp] Tray icon loop ended: {e}")
        finally:
            self._running = False

    def stop(self) -> None:
        """Stops and removes the system tray icon from the notification area."""
        if not self._running and self._icon is None:
            return

        self._running = False
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception as e:
                logging.debug(f"[TrayApp] Error stopping tray icon: {e}")
