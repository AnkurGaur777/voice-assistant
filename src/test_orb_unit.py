"""
Unit tests for Local Jarvis Floating Orb Overlay (src/ui/orb_overlay.py).
"""

import time
import unittest
from unittest.mock import MagicMock

from src.ui.orb_overlay import OrbOverlay, OrbState, CHROMA_KEY, _OrbWindow


class TestOrbOverlayUnit(unittest.TestCase):
    """Unit test suite for the Floating Orb Overlay."""

    def test_orb_state_enum(self):
        """Validates OrbState enum values."""
        self.assertEqual(OrbState.LISTENING.value, "listening")
        self.assertEqual(OrbState.IDLE.value, "idle")
        self.assertEqual(OrbState.PROCESSING.value, "processing")
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
            for state in ["processing", "speaking", "conversation", "error", "listening"]:
                orb.set_state(state)
                time.sleep(0.04)
                self.assertEqual(orb.current_state, state)

        finally:
            orb.stop()
            self.assertFalse(orb._running)
            self.assertIsNone(orb._process)

    def test_orb_window_canvas_drawing(self):
        """Verifies _OrbWindow draws shapes for all states without error."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            mock_cmd = MagicMock()
            mock_resp = MagicMock()
            mock_ready = MagicMock()
            win = _OrbWindow(mock_cmd, mock_resp, mock_ready, "listening", 120)
            win.root = root
            win.canvas = tk.Canvas(root, width=120, height=120, bg=CHROMA_KEY)

            test_states = [
                OrbState.LISTENING.value,
                OrbState.IDLE.value,
                OrbState.PROCESSING.value,
                OrbState.SPEAKING.value,
                OrbState.CONVERSATION.value,
                OrbState.CONVERSATION_ACTIVE.value,
                OrbState.ERROR.value,
                "unknown_fallback",
            ]
            for s in test_states:
                win._draw_state(s)
                items = win.canvas.find_all()
                self.assertGreater(len(items), 0, f"Expected shapes drawn for state {s}")
        finally:
            root.destroy()

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
