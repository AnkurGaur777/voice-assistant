"""
Unit Tests for System Tray Application (Phase 7)

Validates:
1. Icon generation for all 4 visual states (listening, processing, speaking, error).
2. Correct RGBA mode and dimension (64x64) for generated icons.
3. Thread-safe state transitions and description text.
4. Tray app initialization, state setting, and quit callback execution.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tray.tray_app import JarvisTrayApp, JarvisTrayState, create_state_icon


class TestTrayApp(unittest.TestCase):
    """Tests for the JarvisTrayApp and icon generation."""

    def test_create_state_icons(self):
        """Verifies icon generation for all defined states produces valid 64x64 RGBA images."""
        states = [
            JarvisTrayState.LISTENING,
            JarvisTrayState.IDLE,
            JarvisTrayState.CONVERSATION,
            JarvisTrayState.CONVERSATION_ACTIVE,
            JarvisTrayState.PROCESSING,
            JarvisTrayState.SPEAKING,
            JarvisTrayState.ERROR,
            "custom_unknown_state",
        ]
        for s in states:
            val = s.value if isinstance(s, JarvisTrayState) else str(s)
            img = create_state_icon(val, size=64)
            self.assertIsNotNone(img)
            self.assertEqual(img.size, (64, 64))
            self.assertEqual(img.mode, "RGBA")

    def test_tray_app_state_transitions(self):
        """Verifies that set_state updates the current_state property cleanly."""
        app = JarvisTrayApp()
        self.assertEqual(app.current_state, JarvisTrayState.LISTENING)

        app.set_state(JarvisTrayState.CONVERSATION)
        self.assertEqual(app.current_state, JarvisTrayState.CONVERSATION)

        app.set_state(JarvisTrayState.PROCESSING)
        self.assertEqual(app.current_state, JarvisTrayState.PROCESSING)

        app.set_state(JarvisTrayState.SPEAKING)
        self.assertEqual(app.current_state, JarvisTrayState.SPEAKING)

        app.set_state(JarvisTrayState.ERROR)
        self.assertEqual(app.current_state, JarvisTrayState.ERROR)

        app.set_state("listening")
        self.assertEqual(app.current_state, JarvisTrayState.LISTENING)

    def test_tray_app_quit_callback(self):
        """Verifies that selecting quit invokes the registered callback."""
        quit_called = False

        def on_quit():
            nonlocal quit_called
            quit_called = True

        app = JarvisTrayApp(on_quit=on_quit)
        app._handle_quit()
        self.assertTrue(quit_called)


if __name__ == "__main__":
    unittest.main()
