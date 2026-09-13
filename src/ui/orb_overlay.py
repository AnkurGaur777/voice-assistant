"""
Local Jarvis - Floating Orb Overlay UI

Provides a borderless, always-on-top, semi-transparent glowing orb overlay using
Python's built-in Tkinter library and multiprocessing (zero third-party dependencies).

Architectural Highlights:
1. Process Isolation: Runs Tkinter inside its own dedicated child process.
   - Completely avoids GIL contention with the orchestrator, Whisper STT, and Piper TTS.
   - Completely eliminates Tcl/Tk secondary thread conflicts and Tcl_AsyncDelete panics.
2. Focus Preservation: Applies Windows WS_EX_NOACTIVATE (0x08000000) and WS_EX_TOOLWINDOW (0x00000080).
   - Mouse clicks and window redraws NEVER steal focus from user's active applications.
3. Native Transparency: Uses Windows chroma-key transparency (-transparentcolor).
   - Fully click-through on transparent areas; orb itself is draggable and clickable.
4. Visual Pipeline States:
   - IDLE / LISTENING: Soft cyan glow with gentle slow breathing pulse.
   - PROCESSING: Amber glow with faster pulse and orbiting activity dots.
   - SPEAKING: Emerald green glow with dynamic audio equalizer waveform bars.
   - CONVERSATION: Vibrant violet glow with expanding sonar ripple halo.
   - ERROR: Crimson red flashing alert.
5. Thread-safe IPC: High-performance queues for set_state(), hide(), show(), and quit callbacks.
6. Context Menu: Right-click offers "Hide Orb" and "Quit Jarvis".
"""

from enum import Enum
import logging
import math
import multiprocessing as mp
import sys
import threading
import time
from typing import Callable, Optional

# Chroma key color on Windows for pixel-level transparency
CHROMA_KEY = "#010101"


class OrbState(str, Enum):
    LISTENING = "listening"
    IDLE = "idle"
    CONVERSATION = "conversation"
    CONVERSATION_ACTIVE = "conversation_active"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


def _orb_process_entry(
    cmd_queue: mp.Queue,
    resp_queue: mp.Queue,
    ready_event: mp.Event,
    initial_state: str,
    size: int,
) -> None:
    """Entrypoint running in the isolated child process."""
    app = _OrbWindow(
        cmd_queue=cmd_queue,
        resp_queue=resp_queue,
        ready_event=ready_event,
        initial_state=initial_state,
        size=size,
    )
    app.run()


class _OrbWindow:
    """Internal Tkinter window controller running on the child process's main thread."""

    def __init__(
        self,
        cmd_queue: mp.Queue,
        resp_queue: mp.Queue,
        ready_event: mp.Event,
        initial_state: str,
        size: int,
    ):
        self.cmd_queue = cmd_queue
        self.resp_queue = resp_queue
        self.ready_event = ready_event
        self.size = size
        self.current_state = initial_state.lower().strip()
        self.is_visible = True
        self.root = None
        self.canvas = None
        self.context_menu = None

        # Drag tracking
        self._drag_offset_x = 0
        self._drag_offset_y = 0

        # Animation timing
        self._error_start_time = 0.0

    def run(self) -> None:
        """Initializes and runs the Tkinter event loop."""
        import tkinter as tk

        try:
            self.root = tk.Tk()
            self._setup_window(tk)
            self.ready_event.set()

            # Schedule polling of command queue and animation
            self.root.after(20, self._poll_commands)
            self.root.after(33, self._animate_loop)
            self.root.mainloop()
        except Exception as e:
            logging.debug(f"[OrbWindow] Mainloop terminated: {e}")
        finally:
            if self.root:
                try:
                    self.root.destroy()
                except Exception:
                    pass
                self.root = None

    def _setup_window(self, tk) -> None:
        """Configures borderless, topmost, transparent window and canvas."""
        root = self.root
        root.title("Local Jarvis Orb")
        root.overrideredirect(True)
        root.attributes("-topmost", True)

        # Set transparent chroma key background
        root.configure(bg=CHROMA_KEY)
        if sys.platform == "win32":
            try:
                root.attributes("-transparentcolor", CHROMA_KEY)
            except Exception as e:
                logging.debug(f"[OrbWindow] Note setting transparent color: {e}")

        # Compute bottom-right screen coordinates
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        init_x = max(0, screen_w - self.size - 24)
        init_y = max(0, screen_h - self.size - 70)  # offset above taskbar
        root.geometry(f"{self.size}x{self.size}+{init_x}+{init_y}")

        # Ensure window is mapped before querying HWND
        root.update_idletasks()

        # Windows focus-preservation: apply WS_EX_NOACTIVATE and WS_EX_TOOLWINDOW
        if sys.platform == "win32":
            self._apply_noactivate_style()

        # Canvas for drawing glowing orb
        self.canvas = tk.Canvas(
            root,
            width=self.size,
            height=self.size,
            bg=CHROMA_KEY,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Mouse event bindings
        self.canvas.bind("<Button-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_motion)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Button-2>", self._on_right_click)

        # Right-click context menu
        self.context_menu = tk.Menu(root, tearoff=0)
        self.context_menu.add_command(label="Hide Orb", command=self._handle_hide)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Quit Jarvis", command=self._handle_quit)

    def _apply_noactivate_style(self) -> None:
        """Applies Windows extended styles to prevent stealing window focus."""
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOOLWINDOW = 0x00000080
            cur_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd,
                GWL_EXSTYLE,
                cur_style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
            )
        except Exception as e:
            logging.debug(f"[OrbWindow] Note setting WS_EX_NOACTIVATE: {e}")

    def _on_drag_start(self, event) -> None:
        """Stores pointer offset relative to window top-left for smooth dragging."""
        if self.root:
            self._drag_offset_x = event.x_root - self.root.winfo_x()
            self._drag_offset_y = event.y_root - self.root.winfo_y()

    def _on_drag_motion(self, event) -> None:
        """Moves window to follow pointer using absolute screen coordinates."""
        if self.root:
            new_x = event.x_root - self._drag_offset_x
            new_y = event.y_root - self._drag_offset_y
            self.root.geometry(f"+{new_x}+{new_y}")

    def _on_right_click(self, event) -> None:
        """Shows the context menu at mouse cursor position."""
        if self.context_menu:
            try:
                self.context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.context_menu.grab_release()

    def _handle_hide(self) -> None:
        """Hides the window and notifies parent."""
        self.is_visible = False
        if self.root:
            self.root.withdraw()
        self.resp_queue.put("hidden")

    def _handle_quit(self) -> None:
        """Signals parent process to initiate graceful shutdown."""
        self.resp_queue.put("quit")

    def _poll_commands(self) -> None:
        """Processes IPC commands from parent process."""
        if not self.root:
            return

        try:
            while not self.cmd_queue.empty():
                cmd = self.cmd_queue.get_nowait()
                if cmd == "stop":
                    self.root.destroy()
                    return
                elif cmd == "hide":
                    self.is_visible = False
                    self.root.withdraw()
                elif cmd == "show":
                    self.is_visible = True
                    self.root.deiconify()
                elif cmd == "toggle":
                    if self.is_visible:
                        self._handle_hide()
                    else:
                        self.is_visible = True
                        self.root.deiconify()
                        self.resp_queue.put("visible")
                elif cmd.startswith("state:"):
                    new_state = cmd.split(":", 1)[1].strip()
                    self.current_state = new_state
                    if new_state == OrbState.ERROR.value:
                        self._error_start_time = time.perf_counter()
        except Exception as e:
            logging.debug(f"[OrbWindow] Command poll note: {e}")

        if self.root:
            self.root.after(20, self._poll_commands)

    def _animate_loop(self) -> None:
        """Renders glowing orb animations at ~30 FPS."""
        if not self.root or not self.canvas:
            return

        # Auto-revert error state after 2.5 seconds
        if self.current_state == OrbState.ERROR.value:
            if time.perf_counter() - self._error_start_time > 2.5:
                self.current_state = OrbState.LISTENING.value

        try:
            self._draw_state(self.current_state)
        except Exception as e:
            logging.debug(f"[OrbWindow] Draw frame note: {e}")

        if self.root:
            self.root.after(33, self._animate_loop)

    def _draw_state(self, state: str) -> None:
        """
        Draws the Arc Reactor segmented ring overlay on the Canvas.
        Consists of 10 rotating arc segments with visible gaps, inner concentric ring,
        and state-reactive central reactor core.
        """
        import tkinter as tk

        canvas = self.canvas
        if canvas is None:
            return

        canvas.delete("all")
        t = time.perf_counter()
        cx = self.size // 2
        cy = self.size // 2

        num_segments = 10
        seg_angle = 360.0 / num_segments  # 36.0 deg per segment
        arc_span = 26.0  # 26 deg visible segment
        ring_r = 38.0  # outer ring radius
        bbox = (cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r)
        inner_r = 18.0

        state_norm = state.lower().strip()

        if state_norm in (OrbState.LISTENING.value, OrbState.IDLE.value):
            # Slow uniform rotation (30 deg/s), segments evenly lit in cyan
            rot_deg = (t * 30.0) % 360.0
            cyan_color = "#00e5ff"
            seg_width = 7.5

            # Subtle outer guideline
            canvas.create_oval(cx - 47, cy - 47, cx + 47, cy + 47, outline="#00363a", width=1)

            # Draw 10 arc segments
            for i in range(num_segments):
                start = (rot_deg + i * seg_angle) % 360.0
                canvas.create_arc(bbox, start=start, extent=arc_span, style=tk.ARC, width=seg_width, outline=cyan_color)

            # Inner concentric reactor ring
            canvas.create_oval(cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r, outline="#00838f", width=1.5)

            # Gentle breathing center core
            pulse = math.sin(t * 2.2)
            core_r = 7.0 + pulse * 1.5
            canvas.create_oval(cx - core_r - 4, cy - core_r - 4, cx + core_r + 4, cy + core_r + 4, fill="#00838f", outline="")
            canvas.create_oval(cx - core_r, cy - core_r, cx + core_r, cy + core_r, fill="#00e5ff", outline="")
            canvas.create_oval(cx - core_r + 3, cy - core_r + 3, cx + core_r - 3, cy + core_r - 3, fill="#e0f7fa", outline="")

        elif state_norm == OrbState.PROCESSING.value:
            # Faster rotation (90 deg/s), amber chasing pattern across segments
            rot_deg = (t * 90.0) % 360.0
            amber_palette = [
                "#ffe082", "#ffd54f", "#ffca28", "#ffb300", "#ffa000",
                "#ff8f00", "#ff6f00", "#e65100", "#bf360c", "#5d1a00"
            ]
            chase_offset = int((t * 8.0) % num_segments)

            # Subtle outer guideline
            canvas.create_oval(cx - 47, cy - 47, cx + 47, cy + 47, outline="#3e1b00", width=1)

            # Draw 10 arc segments with chasing brightness
            for i in range(num_segments):
                pal_idx = (i - chase_offset) % num_segments
                seg_color = amber_palette[pal_idx]
                seg_width = 8.5 if pal_idx < 3 else (7.0 if pal_idx < 6 else 5.5)
                start = (rot_deg + i * seg_angle) % 360.0
                canvas.create_arc(bbox, start=start, extent=arc_span, style=tk.ARC, width=seg_width, outline=seg_color)

            # Inner concentric reactor ring
            canvas.create_oval(cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r, outline="#ffb300", width=1.5)

            # Faster amber pulsing core
            pulse = math.sin(t * 6.5)
            core_r = 7.0 + pulse * 2.0
            canvas.create_oval(cx - core_r - 4, cy - core_r - 4, cx + core_r + 4, cy + core_r + 4, fill="#e65100", outline="")
            canvas.create_oval(cx - core_r, cy - core_r, cx + core_r, cy + core_r, fill="#ff9100", outline="")
            canvas.create_oval(cx - core_r + 3, cy - core_r + 3, cx + core_r - 3, cy + core_r - 3, fill="#ffe082", outline="")

        elif state_norm == OrbState.SPEAKING.value:
            # Moderate rotation (45 deg/s), segments pulse in sync with speech amplitude
            rot_deg = (t * 45.0) % 360.0
            raw_amp = 0.4 * abs(math.sin(t * 8.0 + 0.3)) + 0.4 * abs(math.sin(t * 11.0 + 1.2)) + 0.2 * abs(math.sin(t * 13.5 + 2.1))
            amp = max(0.0, min(1.0, raw_amp * 1.35))

            if amp > 0.70:
                seg_color = "#b9f6ca"
                seg_width = 8.5
            elif amp > 0.35:
                seg_color = "#00e676"
                seg_width = 7.5
            else:
                seg_color = "#00796b"
                seg_width = 6.0

            # Subtle outer guideline
            canvas.create_oval(cx - 47, cy - 47, cx + 47, cy + 47, outline="#00251a", width=1)

            # Draw 10 arc segments pulsing in brightness
            for i in range(num_segments):
                start = (rot_deg + i * seg_angle) % 360.0
                canvas.create_arc(bbox, start=start, extent=arc_span, style=tk.ARC, width=seg_width, outline=seg_color)

            # Inner concentric reactor ring
            canvas.create_oval(cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r, outline="#00c853", width=1.5)

            # Core modulating with speech amplitude
            core_r = 6.0 + amp * 5.0
            canvas.create_oval(cx - core_r - 4, cy - core_r - 4, cx + core_r + 4, cy + core_r + 4, fill="#004d40", outline="")
            canvas.create_oval(cx - core_r, cy - core_r, cx + core_r, cy + core_r, fill="#00e676", outline="")
            canvas.create_oval(cx - core_r + 3, cy - core_r + 3, cx + core_r - 3, cy + core_r - 3, fill="#b9f6ca", outline="")

        elif state_norm in (OrbState.CONVERSATION.value, OrbState.CONVERSATION_ACTIVE.value):
            # Slow graceful rotation (22 deg/s), gentle violet breathing pulse across all segments
            rot_deg = (t * 22.0) % 360.0
            pulse = (math.sin(t * 2.5) + 1.0) / 2.0  # 0.0 to 1.0

            if pulse > 0.65:
                seg_color = "#e1bee7"
                seg_width = 8.0
            elif pulse > 0.35:
                seg_color = "#ba68c8"
                seg_width = 7.0
            else:
                seg_color = "#8e24aa"
                seg_width = 6.0

            # Expanding sonar ripple halo ring
            ripple_phase = (t * 1.3) % 1.0
            ripple_r = 20.0 + ripple_phase * 26.0
            ripple_w = max(1, int(2.5 * (1.0 - ripple_phase)))
            canvas.create_oval(cx - ripple_r, cy - ripple_r, cx + ripple_r, cy + ripple_r, outline="#ba68c8", width=ripple_w)

            # Draw 10 arc segments breathing together
            for i in range(num_segments):
                start = (rot_deg + i * seg_angle) % 360.0
                canvas.create_arc(bbox, start=start, extent=arc_span, style=tk.ARC, width=seg_width, outline=seg_color)

            # Inner concentric reactor ring
            canvas.create_oval(cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r, outline="#ba68c8", width=1.5)

            # Violet core breathing
            core_r = 6.5 + pulse * 2.5
            canvas.create_oval(cx - core_r - 4, cy - core_r - 4, cx + core_r + 4, cy + core_r + 4, fill="#4a148c", outline="")
            canvas.create_oval(cx - core_r, cy - core_r, cx + core_r, cy + core_r, fill="#ab47bc", outline="")
            canvas.create_oval(cx - core_r + 3, cy - core_r + 3, cx + core_r - 3, cy + core_r - 3, fill="#e1bee7", outline="")

        elif state_norm == OrbState.ERROR.value:
            # Stationary ring, flashing crimson red
            flash_on = (int(t * 6.0) % 2 == 0)
            seg_color = "#ff1744" if flash_on else "#7f0000"
            seg_width = 8.0 if flash_on else 6.0

            # Draw 10 arc segments flashing in unison
            for i in range(num_segments):
                start = i * seg_angle
                canvas.create_arc(bbox, start=start, extent=arc_span, style=tk.ARC, width=seg_width, outline=seg_color)

            # Inner concentric ring
            canvas.create_oval(cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r, outline="#ff1744" if flash_on else "#7f0000", width=1.5)

            # Center core with white exclamation mark
            canvas.create_oval(cx - 14, cy - 14, cx + 14, cy + 14, fill="#b71c1c" if flash_on else "#500000", outline="")
            canvas.create_line(cx, cy - 7, cx, cy + 1, fill="#ffffff", width=2.5, capstyle=tk.ROUND)
            canvas.create_oval(cx - 1.5, cy + 4, cx + 1.5, cy + 7, fill="#ffffff", outline="")

        else:
            # Neutral slate fallback
            for i in range(num_segments):
                start = i * seg_angle
                canvas.create_arc(bbox, start=start, extent=arc_span, style=tk.ARC, width=6.0, outline="#78909c")
            canvas.create_oval(cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r, outline="#78909c", width=1.5)
            canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill="#78909c", outline="")


class OrbOverlay:
    """
    Public controller interface for the Local Jarvis Floating Orb Overlay.
    Manages the child process and IPC transparently for the orchestrator.
    """

    def __init__(
        self,
        on_quit: Optional[Callable[[], None]] = None,
        initial_state: str = OrbState.LISTENING.value,
        size: int = 120,
    ):
        self.on_quit = on_quit
        self.size = size
        self._current_state = initial_state.lower().strip()
        self._visible = True
        self._lock = threading.Lock()

        # IPC Primitives
        self._cmd_queue: mp.Queue = mp.Queue()
        self._resp_queue: mp.Queue = mp.Queue()
        self._ready_event: mp.Event = mp.Event()

        self._process: Optional[mp.Process] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def current_state(self) -> str:
        with self._lock:
            return self._current_state

    @property
    def is_visible(self) -> bool:
        with self._lock:
            return self._visible

    def set_state(self, state: OrbState | str) -> None:
        """Updates the orb visual state in a thread-safe manner."""
        state_val = state.value if isinstance(state, OrbState) else str(state).lower().strip()
        with self._lock:
            self._current_state = state_val
        if self._running:
            try:
                self._cmd_queue.put(f"state:{state_val}")
            except Exception as e:
                logging.debug(f"[OrbOverlay] Note enqueueing state: {e}")

    def hide(self) -> None:
        """Hides the overlay window."""
        with self._lock:
            self._visible = False
        if self._running:
            try:
                self._cmd_queue.put("hide")
            except Exception:
                pass

    def show(self) -> None:
        """Shows the overlay window if hidden."""
        with self._lock:
            self._visible = True
        if self._running:
            try:
                self._cmd_queue.put("show")
            except Exception:
                pass

    def toggle_visibility(self) -> None:
        """Toggles orb visibility between visible and hidden."""
        with self._lock:
            self._visible = not self._visible
        if self._running:
            try:
                self._cmd_queue.put("toggle")
            except Exception:
                pass

    def start(self) -> None:
        """Launches the overlay window process and response listener thread."""
        if self._running:
            return

        self._running = True
        self._ready_event.clear()

        self._process = mp.Process(
            target=_orb_process_entry,
            args=(
                self._cmd_queue,
                self._resp_queue,
                self._ready_event,
                self._current_state,
                self.size,
            ),
            name="JarvisOrbProcess",
            daemon=True,
        )
        self._process.start()

        # Start response listener thread
        self._listener_thread = threading.Thread(
            target=self._listen_responses,
            name="JarvisOrbListener",
            daemon=True,
        )
        self._listener_thread.start()

        # Wait for Tkinter window creation
        self._ready_event.wait(timeout=3.0)

    def _listen_responses(self) -> None:
        """Listens for quit or visibility events sent by the orb child process."""
        while self._running:
            try:
                msg = self._resp_queue.get(timeout=0.2)
                if msg == "quit":
                    print("\n[OrbOverlay] 'Quit' selected from context menu. Initiating graceful shutdown...")
                    self.stop()
                    if self.on_quit:
                        try:
                            self.on_quit()
                        except Exception as e:
                            logging.error(f"[OrbOverlay] Error in on_quit callback: {e}")
                    break
                elif msg == "hidden":
                    with self._lock:
                        self._visible = False
                elif msg == "visible":
                    with self._lock:
                        self._visible = True
            except Exception:
                continue

    def stop(self) -> None:
        """Stops the overlay window process and cleans up resources."""
        if not self._running:
            return

        self._running = False
        try:
            self._cmd_queue.put("stop")
        except Exception:
            pass

        if self._process and self._process.is_alive():
            self._process.join(timeout=1.5)
            if self._process.is_alive():
                try:
                    self._process.terminate()
                    self._process.join(timeout=1.0)
                except Exception:
                    pass

        self._process = None
