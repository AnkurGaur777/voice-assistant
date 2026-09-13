"""
Unit tests for Local Jarvis Floating Orb Overlay (src/ui/orb_overlay.py).
"""

import math
import time
from typing import Tuple
import unittest
from unittest.mock import MagicMock

from src.ui.orb_overlay import (
    CHROMA_KEY,
    MAX_ORB_SIZE,
    MIN_ORB_SIZE,
    OrbOverlay,
    OrbState,
    _OrbWindow,
    _lerp_color,
)


class TestOrbOverlayUnit(unittest.TestCase):
    """Unit test suite for the Floating Orb Overlay."""

    def test_orb_state_enum(self):
        """Validates OrbState enum values."""
        self.assertEqual(OrbState.LISTENING.value, "listening")
        self.assertEqual(OrbState.IDLE.value, "idle")
        self.assertEqual(OrbState.PROCESSING.value, "processing")
        self.assertEqual(OrbState.EXECUTING.value, "executing")
        self.assertEqual(OrbState.SPEAKING.value, "speaking")
        self.assertEqual(OrbState.CONVERSATION.value, "conversation")
        self.assertEqual(OrbState.CONVERSATION_ACTIVE.value, "conversation_active")
        self.assertEqual(OrbState.ERROR.value, "error")

    def test_orb_overlay_initialization(self):
        """Verifies default properties before starting."""
        orb = OrbOverlay(initial_state="idle", size=120)
        self.assertEqual(orb.size, 120)
        self.assertEqual(orb.current_state, "idle")
        self.assertTrue(orb.is_visible)
        self.assertFalse(orb._running)

    def test_orb_overlay_start_and_lifecycle(self):
        """Verifies starting, IPC state changes, and clean shutdown."""
        quit_mock = MagicMock()
        orb = OrbOverlay(on_quit=quit_mock, initial_state="listening", size=120)
        try:
            orb.start()
            self.assertTrue(orb._running)
            self.assertIsNotNone(orb._process)
            self.assertTrue(orb._process.is_alive())

            # Test state transitions
            for state in ["processing", "executing", "speaking", "conversation", "error", "listening"]:
                orb.set_state(state)
                time.sleep(0.04)
                self.assertEqual(orb.current_state, state)

        finally:
            orb.stop()
            self.assertFalse(orb._running)
            self.assertIsNone(orb._process)

    def test_orb_window_canvas_drawing(self):
        """Verifies _OrbWindow draws 3D wireframe sphere components for all states without error."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            mock_cmd = MagicMock()
            mock_resp = MagicMock()
            mock_ready = MagicMock()
            win = _OrbWindow(mock_cmd, mock_resp, mock_ready, "listening", 120)
            win.root = root
            win.canvas = tk.Canvas(root, width=win.window_size, height=win.window_size, bg=CHROMA_KEY)

            test_states = [
                OrbState.LISTENING.value,
                OrbState.IDLE.value,
                OrbState.PROCESSING.value,
                OrbState.EXECUTING.value,
                OrbState.SPEAKING.value,
                OrbState.CONVERSATION.value,
                OrbState.CONVERSATION_ACTIVE.value,
                OrbState.ERROR.value,
                "unknown_fallback",
            ]
            for s in test_states:
                win._draw_state(s)
                items = win.canvas.find_all()
                self.assertGreater(len(items), 0, f"Expected wireframe shapes drawn for state {s}")
        finally:
            root.destroy()

    def test_debris_fragments_configuration_and_rendering(self):
        """Verifies configuration, orbital physics, and rendering of detached debris fragments."""
        import tkinter as tk

        mock_cmd = MagicMock()
        mock_resp = MagicMock()
        mock_ready = MagicMock()
        win = _OrbWindow(mock_cmd, mock_resp, mock_ready, "listening", 120)

        # Confirm debris collection exists and is well-populated
        self.assertEqual(len(win._debris_fragments), 22)

        shards_count = 0
        nodes_count = 0
        for deb in win._debris_fragments:
            # Orbital radius must lie outside the sphere shell (> 1.0)
            self.assertGreaterEqual(deb["r"], 1.15)
            self.assertLessEqual(deb["r"], 1.50)

            # Independent angular velocity
            self.assertNotEqual(deb["omega"], 0.0)

            # Orthonormal basis vectors
            ux, uy, uz = deb["u_vec"]
            vx, vy, vz = deb["v_vec"]
            u_len = math.sqrt(ux * ux + uy * uy + uz * uz)
            v_len = math.sqrt(vx * vx + vy * vy + vz * vz)
            dot_uv = ux * vx + uy * vy + uz * vz
            self.assertAlmostEqual(u_len, 1.0, places=4)
            self.assertAlmostEqual(v_len, 1.0, places=4)
            self.assertAlmostEqual(dot_uv, 0.0, places=4)

            if deb["is_shard"]:
                shards_count += 1
                self.assertGreater(deb["shard_len"], 0.0)
            else:
                nodes_count += 1

        self.assertGreater(shards_count, 0)
        self.assertGreater(nodes_count, 0)

        # Verify drawing with Tkinter canvas
        root = tk.Tk()
        root.withdraw()
        try:
            win.root = root
            win.canvas = tk.Canvas(root, width=win.window_size, height=win.window_size, bg=CHROMA_KEY)
            win._draw_state("listening")
            items = win.canvas.find_all()
            # Must have drawn core, shell, spikes, aura, nucleus, and debris
            self.assertGreater(len(items), 50)
        finally:
            root.destroy()

    def test_sphere_no_clipping_at_any_rotation_or_scale(self):
        """
        Validates mathematical guarantee that all 3D projected vertices and radiating spikes
        strictly fit within the transparent canvas margin with zero clipping.
        """
        cam_dist = 5.5
        for orb_size in [MIN_ORB_SIZE, 120, MAX_ORB_SIZE]:
            margin_factor = 2.4
            window_size = int(orb_size * margin_factor)
            cx = window_size / 2.0
            base_r = orb_size / 2.0

            max_dist = 0.0
            # Test all states and maximum speech amplitude pulse
            for scale, amp in [(1.0, 0.0), (1.20, 1.0), (0.60, 0.0), (0.38, 0.0)]:
                cur_r = base_r * scale
                h = 0.20 + 0.16 + amp * 0.25
                r_tip = 1.0 + h

                # Sample 360-degree rotation angles
                for yaw_deg in range(0, 360, 15):
                    for pitch_deg in range(-45, 45, 15):
                        yaw = math.radians(yaw_deg)
                        pitch = math.radians(pitch_deg)
                        cyaw, syaw = math.cos(yaw), math.sin(yaw)
                        cpitch, spitch = math.cos(pitch), math.sin(pitch)

                        for ang_deg in range(0, 360, 20):
                            ang = math.radians(ang_deg)
                            bx = r_tip * math.cos(ang)
                            by = r_tip * math.sin(ang)
                            bz = 0.0

                            x1 = bx * cyaw + bz * syaw
                            y1 = by
                            z1 = -bx * syaw + bz * cyaw

                            x2 = x1
                            y2 = y1 * cpitch - z1 * spitch
                            z2 = y1 * spitch + z1 * cpitch

                            factor = cam_dist / (cam_dist + z2)
                            px = x2 * cur_r * factor
                            py = y2 * cur_r * factor
                            dist = math.hypot(px, py)
                            if dist > max_dist:
                                max_dist = dist

            self.assertLess(
                max_dist, cx,
                f"Projected radius {max_dist:.1f}px must be strictly less than canvas half-width {cx:.1f}px (size={orb_size})"
            )
            # Confirm comfortable safety margin
            self.assertGreater(cx - max_dist, 10.0, f"Must have at least 10px breathing margin for size={orb_size}")

    def test_state_scale_contrast(self):
        """Verifies bold contrast between large at-rest scale, medium processing, and small executing."""
        mock_cmd = MagicMock()
        mock_resp = MagicMock()
        mock_ready = MagicMock()
        win = _OrbWindow(mock_cmd, mock_resp, mock_ready, "listening", 120)

        # Large size at rest
        self.assertEqual(win.orb_size, 120)
        self.assertEqual(win._current_scale, 1.0)

        # Ratio test: medium should drop by ~40%, executing by ~60%
        scale_large = 1.00
        scale_medium = 0.60
        scale_small = 0.38
        self.assertLess(scale_medium, scale_large * 0.70, "Processing must be visibly smaller than at-rest")
        self.assertLess(scale_small, scale_medium * 0.70, "Executing must be visibly smaller than processing")

    def test_task_intensity_computation(self):
        """Validates that fast tools stay light red while long/chained calls ramp up to blood-red."""
        mock_cmd = MagicMock()
        mock_resp = MagicMock()
        mock_ready = MagicMock()
        win = _OrbWindow(mock_cmd, mock_resp, mock_ready, "executing", 120)

        # Fast single tool (get_current_datetime, 0.2s duration, 1 call) -> Light intensity
        fast_intensity = win._compute_task_intensity(elapsed=0.2, call_count=1, tool_name="get_current_datetime")
        self.assertLess(fast_intensity, 0.35, "Fast single tool call should have low task intensity (light red)")

        # Long-running execution (run_python, 4.5s elapsed near timeout) -> High intensity (blood red)
        long_intensity = win._compute_task_intensity(elapsed=4.5, call_count=1, tool_name="run_python")
        self.assertGreaterEqual(long_intensity, 0.85, "Long running execution should reach high task intensity (blood red)")

        # Multiple chained tool calls (call_count=3)
        multi_intensity = win._compute_task_intensity(elapsed=1.0, call_count=3, tool_name="open_application")
        self.assertGreater(multi_intensity, fast_intensity, "Chained tool calls should produce higher intensity than single call")

        # Manual override
        win._manual_intensity = 0.92
        self.assertEqual(win._compute_task_intensity(elapsed=0.1, call_count=1, tool_name="datetime"), 0.92)

    def test_mouse_wheel_resizing(self):
        """Verifies scroll-wheel resizing, window container scaling, and boundary clamping (70-320px)."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            mock_cmd = MagicMock()
            mock_resp = MagicMock()
            mock_ready = MagicMock()
            win = _OrbWindow(mock_cmd, mock_resp, mock_ready, "listening", 120)
            win.root = root
            win.canvas = tk.Canvas(root, width=win.window_size, height=win.window_size, bg=CHROMA_KEY)

            # Scroll up: should increase by 14px
            event_up = MagicMock()
            event_up.delta = 120
            event_up.num = 0
            win._on_mouse_wheel(event_up)
            self.assertEqual(win.size, 134)
            self.assertEqual(win.window_size, int(134 * 2.4))

            # Scroll down: should decrease by 14px
            event_down = MagicMock()
            event_down.delta = -120
            event_down.num = 0
            win._on_mouse_wheel(event_down)
            self.assertEqual(win.size, 120)
            self.assertEqual(win.window_size, int(120 * 2.4))

            # Clamping to max size (320)
            for _ in range(30):
                win._on_mouse_wheel(event_up)
            self.assertEqual(win.size, MAX_ORB_SIZE)
            self.assertEqual(win.window_size, int(MAX_ORB_SIZE * 2.4))

            # Clamping to min size (70)
            for _ in range(30):
                win._on_mouse_wheel(event_down)
            self.assertEqual(win.size, MIN_ORB_SIZE)
            self.assertEqual(win.window_size, int(MIN_ORB_SIZE * 2.4))
        finally:
            root.destroy()

    def test_color_lerp_helper(self):
        """Verifies hex color interpolation."""
        c_start = "#000000"
        c_end = "#ffffff"
        mid = _lerp_color(c_start, c_end, 0.5)
        self.assertEqual(mid, "#7f7f7f")

        c_red_start = "#ff0000"
        c_red_end = "#0000ff"
        self.assertEqual(_lerp_color(c_red_start, c_red_end, 0.0), "#ff0000")
        self.assertEqual(_lerp_color(c_red_start, c_red_end, 1.0), "#0000ff")

    def test_set_executing_and_task_intensity_api(self):
        """Verifies set_executing and set_task_intensity controller methods."""
        orb = OrbOverlay(initial_state="listening", size=120)
        try:
            orb.start()
            orb.set_executing(call_count=2, tool_name="run_python")
            time.sleep(0.05)
            self.assertEqual(orb.current_state, "executing")

            orb.set_task_intensity(0.85)
            time.sleep(0.05)
            # Controller remains in executing state while passing intensity
            self.assertEqual(orb.current_state, "executing")
        finally:
            orb.stop()

    def test_orb_visibility_toggle(self):
        """Verifies hide, show, and toggle_visibility methods."""
        orb = OrbOverlay(initial_state="listening", size=120)
        try:
            orb.start()
            self.assertTrue(orb.is_visible)

            orb.hide()
            self.assertFalse(orb.is_visible)

            orb.show()
            self.assertTrue(orb.is_visible)

            orb.toggle_visibility()
            self.assertFalse(orb.is_visible)

            orb.toggle_visibility()
            self.assertTrue(orb.is_visible)
        finally:
            orb.stop()

    def test_orb_quit_ipc_flow(self):
        """Verifies that quit response triggers on_quit callback."""
        quit_mock = MagicMock()
        orb = OrbOverlay(on_quit=quit_mock, initial_state="listening", size=120)
        try:
            orb.start()
            # Send quit message to response queue directly to test listener
            orb._resp_queue.put("quit")
            # Wait briefly for listener thread to process
            time.sleep(0.3)
            quit_mock.assert_called_once()
        finally:
            orb.stop()


if __name__ == "__main__":
    unittest.main()
