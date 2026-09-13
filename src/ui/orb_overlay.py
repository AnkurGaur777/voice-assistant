"""
Local Jarvis - Floating Orb Overlay UI

Provides a borderless, always-on-top, semi-transparent glowing Tony-Stark-style
wireframe sphere overlay using Python's built-in Tkinter library and multiprocessing
(zero third-party 3D dependencies).

Architectural Highlights:
1. Process Isolation: Runs Tkinter inside its own dedicated child process.
   - Completely avoids GIL contention with orchestrator, Whisper STT, and Piper TTS.
   - Completely eliminates Tcl/Tk secondary thread conflicts and Tcl_AsyncDelete panics.
2. Focus Preservation: Applies Windows WS_EX_NOACTIVATE (0x08000000) and WS_EX_TOOLWINDOW (0x00000080).
   - Mouse clicks and window redraws NEVER steal focus from user's active applications.
3. Native Transparency: Uses Windows chroma-key transparency (-transparentcolor).
   - Fully click-through on transparent chroma areas; wireframe sphere is draggable and interactive.
4. Tony Stark Wireframe Sphere:
   - 3D sphere rendered via perspective projection and continuous 360° compound axis rotation.
   - Inner core: Dense, tangled flowing energy currents and vortex loops.
   - Middle shell: Structured latitude and longitudinal wireframe ribs.
   - Outer corona: Sparser, jagged spikes simulating sound waves / acoustic radiation.
   - Ample canvas margin (margin_factor = 2.2, cam_dist = 5.5) eliminates edge clipping at all angles and scales.
5. Dynamic Sizing & Amplitude Level Meter:
   - At rest (listening, idle, conversation): Large size (1.00x).
   - Speaking (Jarvis voice or User speaking): Large size pulsing with real-time speech amplitude.
   - Thinking / Processing: Shrinks to medium size (0.60x) with faster focal rotation.
   - Executing Task: Shrinks to smallest size (0.38x) with tight high-energy spin.
6. Dynamic Task-Load Intensity (Color):
   - Blue (cyan/ice): At rest, listening, conversation.
   - Orange (gold/amber): Thinking / processing.
   - Red (weighted load): Light coral red for fast single actions, deepening to dark blood-red
     for chained or long-running executions.
7. Interactive Resize & Session Memory:
   - Drag with left click to reposition anywhere on screen.
   - Mouse scroll wheel over orb dynamically resizes between 70px and 320px anchored to its center,
     with reliable event delivery via root.bind_all and stippled hit-testing aura.
8. Context Menu: Right-click offers "Hide Orb" and "Quit Jarvis".
"""

from enum import Enum
import logging
import math
import multiprocessing as mp
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

# Chroma key color on Windows for pixel-level transparency
CHROMA_KEY = "#010101"

# Sizing boundaries for the visual orb diameter
MIN_ORB_SIZE = 70
MAX_ORB_SIZE = 320


class OrbState(str, Enum):
    LISTENING = "listening"
    IDLE = "idle"
    CONVERSATION = "conversation"
    CONVERSATION_ACTIVE = "conversation_active"
    PROCESSING = "processing"
    EXECUTING = "executing"
    SPEAKING = "speaking"
    ERROR = "error"


def _lerp_color(hex_a: str, hex_b: str, t: float) -> str:
    """Linearly interpolates between two hex colors by factor t in [0.0, 1.0]."""
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = int(hex_a[1:3], 16), int(hex_a[3:5], 16), int(hex_a[5:7], 16)
    r2, g2, b2 = int(hex_b[1:3], 16), int(hex_b[3:5], 16), int(hex_b[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


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

        # Visual orb diameter and generous window container dimension
        self.orb_size = max(MIN_ORB_SIZE, min(MAX_ORB_SIZE, size))
        self.margin_factor = 2.4
        self.window_size = int(self.orb_size * self.margin_factor)

        self.current_state = initial_state.lower().strip()
        self.is_visible = True
        self.root = None
        self.canvas = None
        self.context_menu = None

        # Drag tracking
        self._drag_offset_x = 0
        self._drag_offset_y = 0

        # Animation & sizing state (starts immediately at full large scale)
        self._current_scale = 1.0
        self._error_start_time = 0.0

        # Task intensity tracking
        self._exec_start_time = 0.0
        self._exec_call_count = 1
        self._exec_tool_name = ""
        self._manual_intensity: Optional[float] = None

        # Precompute wireframe sphere mesh
        self._build_sphere_geometry()

    @property
    def size(self) -> int:
        """Backward-compatible size property reflecting visual orb diameter."""
        return self.orb_size

    @size.setter
    def size(self, val: int) -> None:
        self.orb_size = max(MIN_ORB_SIZE, min(MAX_ORB_SIZE, val))
        self.window_size = int(self.orb_size * self.margin_factor)

    def _build_sphere_geometry(self) -> None:
        """
        Precomputes the 3D vertex coordinates and edge connectivity for the
        Tony Stark wireframe sphere:
        1. Inner Dense Core: 4 tilted orbital great-circle loops with cross-tangled edges.
        2. Middle Shell: 5 latitude rings with longitudinal connecting ribs.
        3. Outer Corona: Radial spike nodes with jagged offsets and connecting zig-zags.
        """
        self._core_vertices: List[Tuple[float, float, float]] = []
        self._core_edges: List[Tuple[int, int]] = []

        self._shell_vertices: List[Tuple[float, float, float]] = []
        self._shell_edges: List[Tuple[int, int]] = []

        self._spike_base_indices: List[int] = []
        self._spike_heights: List[float] = []

        # -------------------------------------------------------------
        # 1. Inner Core: 4 tilted rings at radius 0.46
        # -------------------------------------------------------------
        ring_tilts = [
            (0.0, 0.0),
            (math.radians(52.0), math.radians(25.0)),
            (math.radians(-52.0), math.radians(-30.0)),
            (math.radians(85.0), math.radians(65.0)),
        ]
        pts_per_ring = 22
        core_r = 0.46

        for pitch_rad, roll_rad in ring_tilts:
            start_idx = len(self._core_vertices)
            cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
            cr, sr = math.cos(roll_rad), math.sin(roll_rad)

            for i in range(pts_per_ring):
                ang = 2.0 * math.pi * i / pts_per_ring
                # Base circle in XY plane
                bx = core_r * math.cos(ang)
                by = core_r * math.sin(ang)
                bz = 0.0

                # Tilt ring
                # Pitch around X
                x1 = bx
                y1 = by * cp - bz * sp
                z1 = by * sp + bz * cp
                # Roll around Z
                rx = x1 * cr - y1 * sr
                ry = x1 * sr + y1 * cr
                rz = z1

                self._core_vertices.append((rx, ry, rz))

            # Sequential loop edges
            for i in range(pts_per_ring):
                i_next = (i + 1) % pts_per_ring
                self._core_edges.append((start_idx + i, start_idx + i_next))
                # Tangled cross-currents (connecting to offset nodes)
                if i % 2 == 0:
                    i_cross = (i + 5) % pts_per_ring
                    self._core_edges.append((start_idx + i, start_idx + i_cross))

        # -------------------------------------------------------------
        # 2. Middle Shell: 5 latitude rings at radius 1.0
        # -------------------------------------------------------------
        lat_z_norms = [-0.75, -0.40, 0.0, 0.40, 0.75]
        pts_per_lat = 18
        shell_ring_start_indices = []

        for z_norm in lat_z_norms:
            ring_start = len(self._shell_vertices)
            shell_ring_start_indices.append(ring_start)
            r_lat = math.sqrt(max(0.0, 1.0 - z_norm * z_norm))

            for j in range(pts_per_lat):
                ang = 2.0 * math.pi * j / pts_per_lat
                sx = r_lat * math.cos(ang)
                sy = r_lat * math.sin(ang)
                sz = z_norm
                self._shell_vertices.append((sx, sy, sz))

            # Connect latitude loop
            for j in range(pts_per_lat):
                j_next = (j + 1) % pts_per_lat
                self._shell_edges.append((ring_start + j, ring_start + j_next))

        # Connect longitudinal ribs between adjacent latitudes
        for r_idx in range(len(shell_ring_start_indices) - 1):
            r1 = shell_ring_start_indices[r_idx]
            r2 = shell_ring_start_indices[r_idx + 1]
            for j in range(pts_per_lat):
                self._shell_edges.append((r1 + j, r2 + j))

        # -------------------------------------------------------------
        # 3. Outer Corona: Spikes anchored on middle latitude ring
        # -------------------------------------------------------------
        eq_start = shell_ring_start_indices[2]  # equator
        for k in range(pts_per_lat):
            self._spike_base_indices.append(eq_start + k)
            # Jagged base height pattern
            h = 0.20 + 0.16 * abs(math.sin(k * 2.8 + 0.5))
            self._spike_heights.append(h)

        # -------------------------------------------------------------
        # 4. Detached Orbital Debris Field (Asteroid belt / debris cloud)
        # -------------------------------------------------------------
        import random
        rng = random.Random(1337)  # Deterministic seed for reproducible aesthetic distribution
        self._debris_fragments: List[Dict] = []
        num_debris = 22

        for i in range(num_debris):
            # Orbit radius just outside sphere shell (1.18 to 1.46)
            r = 1.18 + rng.random() * 0.28

            # Independent orbital angular velocity: varied directions and speeds (0.60 to 1.70 rad/s)
            speed_mag = 0.60 + rng.random() * 1.10
            direction = 1.0 if rng.random() > 0.45 else -1.0
            omega = speed_mag * direction

            # Initial phase angle
            theta0 = rng.random() * 2.0 * math.pi

            # Generate random 3D orbital plane normal vector uniformly on sphere
            u_rnd = rng.random() * 2.0 - 1.0
            phi_rnd = rng.random() * 2.0 * math.pi
            r_plane = math.sqrt(max(0.0, 1.0 - u_rnd * u_rnd))
            nx = r_plane * math.cos(phi_rnd)
            ny = r_plane * math.sin(phi_rnd)
            nz = u_rnd

            # Construct two orthonormal basis vectors u_vec, v_vec spanning this plane
            if abs(nz) < 0.90:
                ux, uy, uz = ny, -nx, 0.0
            else:
                ux, uy, uz = 0.0, nz, -ny
            u_len = math.sqrt(ux * ux + uy * uy + uz * uz)
            ux, uy, uz = ux / u_len, uy / u_len, uz / u_len

            # v_vec = n x u
            vx = ny * uz - nz * uy
            vy = nz * ux - nx * uz
            vz = nx * uy - ny * ux

            # Fragment type: velocity-aligned shards (~50%) vs micro-particle nodes (~50%)
            is_shard = (i % 2 == 0)
            shard_len = 0.06 + rng.random() * 0.06

            self._debris_fragments.append({
                "r": r,
                "omega": omega,
                "theta0": theta0,
                "u_vec": (ux, uy, uz),
                "v_vec": (vx, vy, vz),
                "is_shard": is_shard,
                "shard_len": shard_len,
            })

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

        # Compute bottom-right screen coordinates based on full container window_size
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        init_x = max(0, screen_w - self.window_size - 24)
        init_y = max(0, screen_h - self.window_size - 70)  # offset above taskbar
        root.geometry(f"{self.window_size}x{self.window_size}+{init_x}+{init_y}")

        # Ensure window is mapped before querying HWND
        root.update_idletasks()

        # Windows focus-preservation: apply WS_EX_NOACTIVATE and WS_EX_TOOLWINDOW
        if sys.platform == "win32":
            self._apply_noactivate_style()

        # Canvas for drawing glowing wireframe sphere with ample margins
        self.canvas = tk.Canvas(
            root,
            width=self.window_size,
            height=self.window_size,
            bg=CHROMA_KEY,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Mouse event bindings: drag and right-click menu
        self.canvas.bind("<Button-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_motion)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Button-2>", self._on_right_click)

        # Cursor enter / leave diagnostics
        self.canvas.bind("<Enter>", self._on_mouse_enter)
        self.canvas.bind("<Leave>", self._on_mouse_leave)

        # Scroll wheel resize bindings:
        # Crucial fix for WS_EX_NOACTIVATE: On Windows, inactive windows receive WM_MOUSEWHEEL
        # dispatched to root ('.'), NOT canvas ('.!canvas') because canvas has no keyboard focus.
        # Binding to root.bind_all and root.bind guarantees events are received reliably!
        root.bind_all("<MouseWheel>", self._on_mouse_wheel)
        root.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        root.bind_all("<Button-4>", self._on_mouse_wheel)
        root.bind_all("<Button-5>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)

        print(
            f"[OrbOverlay Debug] Orb window created with initial orb_size={self.orb_size}px, "
            f"window_size={self.window_size}px",
            flush=True,
        )

        # Right-click context menu
        self.context_menu = tk.Menu(root, tearoff=0)
        self.context_menu.add_command(label="Hide Orb", command=self._handle_hide)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Quit Jarvis", command=self._handle_quit)

    def _on_mouse_enter(self, event) -> None:
        """Sets internal Tkinter focus and prints diagnostic entry."""
        if self.canvas:
            self.canvas.focus_set()
        x_r = getattr(event, "x_root", None)
        y_r = getattr(event, "y_root", None)
        print(f"[OrbOverlay Debug] Mouse ENTERED orb canvas! cursor=({x_r}, {y_r})", flush=True)

    def _on_mouse_leave(self, event) -> None:
        """Prints diagnostic exit."""
        print("[OrbOverlay Debug] Mouse LEFT orb canvas!", flush=True)

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

    def _on_mouse_wheel(self, event) -> None:
        """
        Interactive resize handler via mouse scroll wheel.
        Diagnostic instrumentation prints immediately upon receipt of ANY wheel event.
        """
        delta = getattr(event, "delta", None)
        num = getattr(event, "num", None)
        widget = getattr(event, "widget", None)
        x = getattr(event, "x", None)
        y = getattr(event, "y", None)
        x_root = getattr(event, "x_root", None)
        y_root = getattr(event, "y_root", None)

        print(
            f"[OrbOverlay Debug] MouseWheel event RECEIVED by Tkinter! "
            f"delta={delta}, num={num}, widget={widget}, local=({x}, {y}), screen=({x_root}, {y_root})",
            flush=True,
        )

        if not self.root or not self.canvas:
            return

        # Windows: event.delta is typically +/-120. Linux: Button-4 is up, Button-5 is down.
        is_up = (delta is not None and delta > 0) or num == 4
        step = 14 if is_up else -14

        old_orb = self.orb_size
        old_win = self.window_size
        new_orb_size = max(MIN_ORB_SIZE, min(MAX_ORB_SIZE, self.orb_size + step))
        if new_orb_size == self.orb_size:
            print(
                f"[OrbOverlay Debug] Orb size at boundary limit: {self.orb_size}px "
                f"(min={MIN_ORB_SIZE}px, max={MAX_ORB_SIZE}px)",
                flush=True,
            )
            return

        new_window_size = int(new_orb_size * self.margin_factor)

        # Keep center point fixed on screen
        curr_x = self.root.winfo_x()
        curr_y = self.root.winfo_y()
        center_x = curr_x + self.window_size / 2.0
        center_y = curr_y + self.window_size / 2.0

        new_x = int(round(center_x - new_window_size / 2.0))
        new_y = int(round(center_y - new_window_size / 2.0))

        self.orb_size = new_orb_size
        self.window_size = new_window_size
        self.root.geometry(f"{new_window_size}x{new_window_size}+{new_x}+{new_y}")
        self.canvas.config(width=new_window_size, height=new_window_size)
        print(
            f"[OrbOverlay Debug] Resized orb: {old_orb}px -> {self.orb_size}px "
            f"(window: {old_win}px -> {self.window_size}px)",
            flush=True,
        )

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
                    # Format: state:<name> or state:executing:<call_count>:<tool_name>
                    parts = cmd.split(":")
                    new_state = parts[1].strip()
                    self.current_state = new_state
                    if new_state == OrbState.EXECUTING.value:
                        self._exec_start_time = time.perf_counter()
                        self._exec_call_count = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
                        self._exec_tool_name = parts[3].strip() if len(parts) > 3 else ""
                        self._manual_intensity = None
                    elif new_state == OrbState.ERROR.value:
                        self._error_start_time = time.perf_counter()
                elif cmd.startswith("executing:"):
                    # Format: executing:<call_count>:<tool_name>
                    parts = cmd.split(":")
                    self.current_state = OrbState.EXECUTING.value
                    self._exec_start_time = time.perf_counter()
                    self._exec_call_count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                    self._exec_tool_name = parts[2].strip() if len(parts) > 2 else ""
                    self._manual_intensity = None
                elif cmd.startswith("task_intensity:"):
                    try:
                        val = float(cmd.split(":", 1)[1].strip())
                        self._manual_intensity = max(0.0, min(1.0, val))
                    except ValueError:
                        pass
        except Exception as e:
            logging.debug(f"[OrbWindow] Command poll note: {e}")

        if self.root:
            self.root.after(20, self._poll_commands)

    def _animate_loop(self) -> None:
        """Renders the 3D wireframe sphere animations at ~30 FPS."""
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

    def _compute_task_intensity(self, elapsed: float, call_count: int, tool_name: str) -> float:
        """
        Calculates task intensity in [0.0, 1.0]:
        - Single fast tool (like get_current_datetime, <0.5s): low intensity (~0.15-0.25) -> light red.
        - Long-running tool (like run_python near 5s timeout): high intensity (~0.85-1.0) -> dark blood-red.
        - Multiple chained tool calls: adds cumulative load weight per additional call.
        """
        if self._manual_intensity is not None:
            return self._manual_intensity

        # Weighting by tool identity
        t_name = tool_name.lower().strip()
        if "datetime" in t_name:
            tool_base = 0.05
        elif any(k in t_name for k in ("python", "sandbox")):
            tool_base = 0.35
        elif any(k in t_name for k in ("search", "web")):
            tool_base = 0.28
        elif any(k in t_name for k in ("open", "type", "press", "clipboard")):
            tool_base = 0.20
        else:
            tool_base = 0.15

        # Chained calls load penalty
        calls_weight = min(0.40, max(0, call_count - 1) * 0.20)

        # Elapsed execution duration scaling (non-linear ramp toward 4.5 seconds)
        duration_weight = min(0.60, (elapsed / 4.5) ** 1.3)

        total_intensity = min(1.0, tool_base + calls_weight + duration_weight)
        return total_intensity

    def _draw_state(self, state: str) -> None:
        """
        Renders the Tony-Stark-style wireframe sphere projected from 3D to 2D canvas.
        Includes continuous 360-degree compound rotation, state sizing, amplitude
        pulsation, and task-load color intensity.
        """
        canvas = self.canvas
        if canvas is None:
            return

        canvas.delete("all")
        t = time.perf_counter()
        cx = self.window_size // 2
        cy = self.window_size // 2

        state_norm = state.lower().strip()

        # -------------------------------------------------------------
        # 1. State-specific scale target & rotation speed
        # Bold size contrast between states:
        # - Large (1.00): Full configured orb diameter at rest
        # - Medium (0.60): 40% radius drop (64% area reduction) in processing
        # - Smallest (0.38): Compact high-energy core during task execution
        # -------------------------------------------------------------
        rot_speed = 1.2
        speech_amp = 0.0

        if state_norm in (OrbState.LISTENING.value, OrbState.IDLE.value, OrbState.CONVERSATION.value, OrbState.CONVERSATION_ACTIVE.value):
            target_scale = 1.00  # LARGE: full configured orb diameter
            rot_speed = 1.15
        elif state_norm == OrbState.PROCESSING.value:
            target_scale = 0.60  # MEDIUM: unmistakable contraction
            rot_speed = 2.40
        elif state_norm == OrbState.EXECUTING.value:
            target_scale = 0.38  # SMALLEST: tight high-energy mini-core
            rot_speed = 3.20
        elif state_norm == OrbState.SPEAKING.value:
            # Active voice: Live audio amplitude level meter
            raw_amp = (
                0.40 * abs(math.sin(t * 8.0 + 0.3))
                + 0.40 * abs(math.sin(t * 11.0 + 1.2))
                + 0.20 * abs(math.sin(t * 13.5 + 2.1))
            )
            speech_amp = max(0.0, min(1.0, raw_amp * 1.35))
            target_scale = 1.00 + speech_amp * 0.20
            rot_speed = 1.45
        elif state_norm == OrbState.ERROR.value:
            target_scale = 0.85
            rot_speed = 0.50
        else:
            target_scale = 1.00

        # Smooth scale interpolation (easing)
        self._current_scale += (target_scale - self._current_scale) * 0.16
        base_radius = (self.orb_size / 2.0) * self._current_scale

        # -------------------------------------------------------------
        # 2. Continuous 360-degree compound rotation angles
        # -------------------------------------------------------------
        yaw = t * rot_speed
        pitch = 0.38 + 0.14 * math.sin(t * 0.40)  # ~22° inclined axis with gentle wobble
        roll = t * 0.25

        cyaw, syaw = math.cos(yaw), math.sin(yaw)
        cpitch, spitch = math.cos(pitch), math.sin(pitch)
        croll, sroll = math.cos(roll), math.sin(roll)

        def rotate_point(pt: Tuple[float, float, float], r_factor: float = 1.0) -> Tuple[float, float, float]:
            x, y, z = pt[0] * r_factor, pt[1] * r_factor, pt[2] * r_factor
            # Yaw (around Y)
            x1 = x * cyaw + z * syaw
            y1 = y
            z1 = -x * syaw + z * cyaw
            # Pitch (around X)
            x2 = x1
            y2 = y1 * cpitch - z1 * spitch
            z2 = y1 * spitch + z1 * cpitch
            # Roll (around Z)
            x3 = x2 * croll - y2 * sroll
            y3 = x2 * sroll + y2 * croll
            z3 = z2
            return (x3, y3, z3)

        # Perspective projection (cam_dist = 5.5 guarantees no clipping at any angle)
        cam_dist = 5.5

        def project(p3: Tuple[float, float, float]) -> Tuple[float, float]:
            factor = cam_dist / (cam_dist + p3[2])
            px = cx + p3[0] * base_radius * factor
            py = cy + p3[1] * base_radius * factor
            return (px, py)

        # -------------------------------------------------------------
        # 3. Determine Color Palettes
        # -------------------------------------------------------------
        if state_norm in (OrbState.LISTENING.value, OrbState.IDLE.value, OrbState.CONVERSATION.value, OrbState.CONVERSATION_ACTIVE.value):
            front_color = "#00e5ff"
            back_color = "#004880"
            core_front = "#e0f7fa"
            core_halo = "#0091ea"
            spike_color = "#40c4ff"
            debris_front = "#40c4ff"
            debris_back = "#003860"
        elif state_norm == OrbState.PROCESSING.value:
            front_color = "#ffb300"
            back_color = "#663300"
            core_front = "#fff8e1"
            core_halo = "#ff6f00"
            spike_color = "#ffd54f"
            debris_front = "#ffd54f"
            debris_back = "#5a2800"
        elif state_norm == OrbState.EXECUTING.value:
            elapsed = max(0.0, time.perf_counter() - self._exec_start_time)
            intensity = self._compute_task_intensity(elapsed, self._exec_call_count, self._exec_tool_name)

            # Palette interpolation: light coral red (0.0) -> vivid red (0.5) -> deep blood red (1.0)
            if intensity < 0.5:
                sub_t = intensity / 0.5
                front_color = _lerp_color("#ff6b6b", "#e53935", sub_t)
                back_color = _lerp_color("#801515", "#500a0a", sub_t)
                core_front = _lerp_color("#ffebee", "#ff8a80", sub_t)
                core_halo = _lerp_color("#ff1744", "#d50000", sub_t)
            else:
                sub_t = (intensity - 0.5) / 0.5
                front_color = _lerp_color("#e53935", "#8a0303", sub_t)
                back_color = _lerp_color("#500a0a", "#2b0000", sub_t)
                core_front = _lerp_color("#ff8a80", "#b71c1c", sub_t)
                core_halo = _lerp_color("#d50000", "#5c0000", sub_t)
            spike_color = front_color
            debris_front = _lerp_color(front_color, "#000000", 0.15)
            debris_back = _lerp_color(back_color, "#000000", 0.20)
        elif state_norm == OrbState.SPEAKING.value:
            front_color = "#00e5ff" if speech_amp < 0.60 else "#80d8ff"
            back_color = "#003b66"
            core_front = "#ffffff"
            core_halo = "#00b0ff"
            spike_color = "#00e5ff"
            debris_front = "#80d8ff" if speech_amp > 0.50 else "#40c4ff"
            debris_back = "#002b4d"
        elif state_norm == OrbState.ERROR.value:
            flash_on = (int(t * 6.0) % 2 == 0)
            front_color = "#ff1744" if flash_on else "#7f0000"
            back_color = "#400000"
            core_front = "#ffffff" if flash_on else "#ff8a80"
            core_halo = "#b71c1c"
            spike_color = front_color
            debris_front = "#ff5252" if flash_on else "#500000"
            debris_back = "#2b0000"
        else:
            front_color = "#90a4ae"
            back_color = "#37474f"
            core_front = "#eceff1"
            core_halo = "#546e7a"
            spike_color = front_color
            debris_front = "#78909c"
            debris_back = "#263238"

        # -------------------------------------------------------------
        # 4. Project and Render 3D Wireframe Components & Debris
        # -------------------------------------------------------------
        # Rotate Core Vertices
        rot_core = [rotate_point(v, 1.0) for v in self._core_vertices]
        # Rotate Shell Vertices
        rot_shell = [rotate_point(v, 1.0) for v in self._shell_vertices]

        # Outer Corona Spikes
        spike_edges_data: List[Tuple[float, Tuple[float, float], Tuple[float, float]]] = []
        spike_tips_proj: List[Tuple[float, float]] = []

        for idx, base_idx in enumerate(self._spike_base_indices):
            base_p3 = rot_shell[base_idx]
            h = self._spike_heights[idx] + speech_amp * 0.25
            # Tip extends radially
            tip_p3 = (base_p3[0] * (1.0 + h), base_p3[1] * (1.0 + h), base_p3[2] * (1.0 + h))
            p_base_2d = project(base_p3)
            p_tip_2d = project(tip_p3)
            spike_tips_proj.append(p_tip_2d)
            avg_z = (base_p3[2] + tip_p3[2]) / 2.0
            spike_edges_data.append((avg_z, p_base_2d, p_tip_2d))

        # Compile all wireframe edges and debris with depth for true 3D depth-sorting
        drawable_edges: List[Tuple[float, Tuple[float, float], Tuple[float, float], str]] = []

        # Core edges (tangled water-like current)
        for i1, i2 in self._core_edges:
            p1, p2 = rot_core[i1], rot_core[i2]
            avg_z = (p1[2] + p2[2]) / 2.0
            drawable_edges.append((avg_z, project(p1), project(p2), "core"))

        # Shell edges (spherical ribs)
        for i1, i2 in self._shell_edges:
            p1, p2 = rot_shell[i1], rot_shell[i2]
            avg_z = (p1[2] + p2[2]) / 2.0
            drawable_edges.append((avg_z, project(p1), project(p2), "shell"))

        # Spike radial edges
        for avg_z, p_base_2d, p_tip_2d in spike_edges_data:
            drawable_edges.append((avg_z, p_base_2d, p_tip_2d, "spike"))

        # Spike connecting zig-zags (sound waves)
        for idx in range(len(spike_tips_proj)):
            next_idx = (idx + 1) % len(spike_tips_proj)
            avg_z = rot_shell[self._spike_base_indices[idx]][2]
            drawable_edges.append((avg_z, spike_tips_proj[idx], spike_tips_proj[next_idx], "wave"))

        # Detached Orbital Debris Field (freely orbiting shards & nodes)
        for deb in self._debris_fragments:
            theta = deb["theta0"] + deb["omega"] * t
            ct, st = math.cos(theta), math.sin(theta)
            r_deb = deb["r"]
            u_vec = deb["u_vec"]
            v_vec = deb["v_vec"]

            pos_3d = (
                r_deb * (ct * u_vec[0] + st * v_vec[0]),
                r_deb * (ct * u_vec[1] + st * v_vec[1]),
                r_deb * (ct * u_vec[2] + st * v_vec[2]),
            )

            if deb["is_shard"]:
                # Short line segment oriented along orbital velocity vector
                L = deb["shard_len"]
                tang_3d = (
                    r_deb * (-st * u_vec[0] + ct * v_vec[0]) * L,
                    r_deb * (-st * u_vec[1] + ct * v_vec[1]) * L,
                    r_deb * (-st * u_vec[2] + ct * v_vec[2]) * L,
                )
                head_3d = (pos_3d[0] + tang_3d[0], pos_3d[1] + tang_3d[1], pos_3d[2] + tang_3d[2])
                tail_3d = (pos_3d[0] - tang_3d[0], pos_3d[1] - tang_3d[1], pos_3d[2] - tang_3d[2])

                rot_head = rotate_point(head_3d, 1.0)
                rot_tail = rotate_point(tail_3d, 1.0)
                avg_z = (rot_head[2] + rot_tail[2]) / 2.0
                p_head_2d = project(rot_head)
                p_tail_2d = project(rot_tail)
                drawable_edges.append((avg_z, p_tail_2d, p_head_2d, "shard"))
            else:
                rot_pt = rotate_point(pos_3d, 1.0)
                p_2d = project(rot_pt)
                drawable_edges.append((rot_pt[2], p_2d, p_2d, "node"))

        # Sort all elements by depth ascending (back to front)
        drawable_edges.sort(key=lambda item: item[0])

        # Subtle stippled aura inside the sphere shell for Windows mouse hit-testing across the entire sphere
        aura_r = base_radius * 1.05
        canvas.create_oval(
            cx - aura_r, cy - aura_r,
            cx + aura_r, cy + aura_r,
            fill=core_halo, outline="",
            stipple="gray12",
        )

        # Draw Back Edges & Background Debris (z <= 0)
        for avg_z, p1_2d, p2_2d, edge_type in drawable_edges:
            if avg_z <= 0:
                if edge_type == "shard":
                    canvas.create_line(
                        p1_2d[0], p1_2d[1], p2_2d[0], p2_2d[1],
                        fill=debris_back,
                        width=1.0,
                    )
                elif edge_type == "node":
                    canvas.create_oval(
                        p1_2d[0] - 1.0, p1_2d[1] - 1.0,
                        p1_2d[0] + 1.0, p1_2d[1] + 1.0,
                        fill=debris_back,
                        outline="",
                    )
                else:
                    canvas.create_line(
                        p1_2d[0], p1_2d[1], p2_2d[0], p2_2d[1],
                        fill=back_color,
                        width=1.0,
                    )

        # Draw Glowing Center Core (Nucleus)
        core_pulse = 0.5 + 0.5 * math.sin(t * 3.0)
        core_r = max(5.0, (base_radius * 0.20) + core_pulse * 1.8 + speech_amp * 3.5)
        canvas.create_oval(
            cx - core_r * 1.8, cy - core_r * 1.8,
            cx + core_r * 1.8, cy + core_r * 1.8,
            fill=core_halo, outline="",
        )
        canvas.create_oval(
            cx - core_r, cy - core_r,
            cx + core_r, cy + core_r,
            fill=front_color, outline="",
        )
        canvas.create_oval(
            cx - core_r * 0.45, cy - core_r * 0.45,
            cx + core_r * 0.45, cy + core_r * 0.45,
            fill=core_front, outline="",
        )

        # Draw Front Edges & Foreground Debris (z > 0)
        for avg_z, p1_2d, p2_2d, edge_type in drawable_edges:
            if avg_z > 0:
                if edge_type == "shard":
                    canvas.create_line(
                        p1_2d[0], p1_2d[1], p2_2d[0], p2_2d[1],
                        fill=debris_front,
                        width=1.5,
                    )
                elif edge_type == "node":
                    canvas.create_oval(
                        p1_2d[0] - 1.8, p1_2d[1] - 1.8,
                        p1_2d[0] + 1.8, p1_2d[1] + 1.8,
                        fill=debris_front,
                        outline="",
                    )
                elif edge_type == "core":
                    # Dense core is high energy
                    width = 1.8
                    line_c = front_color
                    canvas.create_line(
                        p1_2d[0], p1_2d[1], p2_2d[0], p2_2d[1],
                        fill=line_c,
                        width=width,
                    )
                elif edge_type == "spike":
                    width = 1.6
                    line_c = spike_color
                    canvas.create_line(
                        p1_2d[0], p1_2d[1], p2_2d[0], p2_2d[1],
                        fill=line_c,
                        width=width,
                    )
                elif edge_type == "wave":
                    width = 1.0
                    line_c = spike_color
                    canvas.create_line(
                        p1_2d[0], p1_2d[1], p2_2d[0], p2_2d[1],
                        fill=line_c,
                        width=width,
                    )
                else:
                    width = 1.4
                    line_c = front_color
                    canvas.create_line(
                        p1_2d[0], p1_2d[1], p2_2d[0], p2_2d[1],
                        fill=line_c,
                        width=width,
                    )


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
        self.size = max(MIN_ORB_SIZE, min(MAX_ORB_SIZE, size))
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

    def set_executing(self, call_count: int = 1, tool_name: str = "") -> None:
        """
        Sets the orb to EXECUTING state with tool execution metadata.
        Allows the orb's task intensity logic to weight color and intensity correctly.
        """
        with self._lock:
            self._current_state = OrbState.EXECUTING.value
        if self._running:
            try:
                self._cmd_queue.put(f"state:executing:{call_count}:{tool_name}")
            except Exception as e:
                logging.debug(f"[OrbOverlay] Note enqueueing executing state: {e}")

    def set_task_intensity(self, intensity: float) -> None:
        """Manually sets the task execution intensity in [0.0, 1.0]."""
        if self._running:
            try:
                self._cmd_queue.put(f"task_intensity:{intensity:.2f}")
            except Exception as e:
                logging.debug(f"[OrbOverlay] Note enqueueing task intensity: {e}")

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
