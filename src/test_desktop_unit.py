"""
Unit Tests for Desktop Automation Tools (src/agent/tools/desktop.py)

Validates:
1. type_text without press_enter: confirmation prompt, EOF abort, reject flow, confirm typing.
2. type_text with press_enter=True: combined "Type and press enter" prompt, EOF abort, reject flow, confirm typing + enter press.
3. press_enter_key: standalone Enter tool with explicit confirmation prompt, reject, and confirm flows.
4. Window title reporting and empty text handling.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.tools.desktop import (
    get_active_window_title,
    press_enter_in_active_window,
    press_enter_key,
    type_text,
    type_text_into_active_window,
)


class TestDesktopTools(unittest.TestCase):
    """Unit tests for desktop typing and keystroke tools."""

    def setUp(self):
        # Mock pyautogui in sys.modules so local 'import pyautogui' uses the mock
        self.mock_pyautogui = MagicMock()
        self.mock_pyautogui.write = MagicMock()
        self.mock_pyautogui.press = MagicMock()
        self.patcher_pyautogui = patch.dict("sys.modules", {"pyautogui": self.mock_pyautogui})
        self.patcher_pyautogui.start()

        # Mock window title for deterministic testing
        self.patcher_win = patch(
            "src.agent.tools.desktop.get_active_window_title",
            return_value="WhatsApp Web - Google Chrome",
        )
        self.mock_win = self.patcher_win.start()

    def tearDown(self):
        self.patcher_pyautogui.stop()
        self.patcher_win.stop()

    def test_empty_text(self):
        """Verifies empty text returns an error message without prompting."""
        result = type_text.invoke({"text": ""})
        self.assertIn("Error: No text provided", result)
        self.mock_pyautogui.write.assert_not_called()
        self.mock_pyautogui.press.assert_not_called()

    def test_type_text_without_enter_confirmed(self):
        """Verifies type_text with press_enter=False types text and does NOT press enter."""
        with patch("builtins.input", return_value="y") as mock_input:
            result = type_text.invoke({"text": "Hello world", "press_enter": False})

            mock_input.assert_called_once()
            prompt_arg = mock_input.call_args[0][0]
            self.assertIn("Allow typing into active window", prompt_arg)
            self.assertIn("WhatsApp Web - Google Chrome", prompt_arg)

            self.mock_pyautogui.write.assert_called_once_with("Hello world", interval=0.01)
            self.mock_pyautogui.press.assert_not_called()
            self.assertIn("Successfully typed 11 characters", result)
            self.assertIn("WhatsApp Web - Google Chrome", result)

    def test_type_text_without_enter_rejected(self):
        """Verifies rejecting type_text aborts without sending keystrokes."""
        with patch("builtins.input", return_value="n"):
            result = type_text.invoke({"text": "Test rejection", "press_enter": False})

            self.mock_pyautogui.write.assert_not_called()
            self.mock_pyautogui.press.assert_not_called()
            self.assertIn("Typing cancelled by user", result)
            self.assertIn("WhatsApp Web - Google Chrome", result)

    def test_type_text_with_enter_confirmed(self):
        """Verifies type_text with press_enter=True presents combined prompt, types, and presses enter."""
        with patch("builtins.input", return_value="y") as mock_input:
            result = type_text.invoke({"text": "Send this message", "press_enter": True})

            mock_input.assert_called_once()
            prompt_arg = mock_input.call_args[0][0]
            self.assertIn("and press enter into active window", prompt_arg)
            self.assertIn("Send this message", prompt_arg)
            self.assertIn("WhatsApp Web - Google Chrome", prompt_arg)

            self.mock_pyautogui.write.assert_called_once_with("Send this message", interval=0.01)
            self.mock_pyautogui.press.assert_called_once_with("enter")
            self.assertIn("Successfully typed 17 characters and pressed enter", result)
            self.assertIn("WhatsApp Web - Google Chrome", result)

    def test_type_text_with_enter_rejected(self):
        """Verifies rejecting combined type and enter aborts completely."""
        with patch("builtins.input", return_value="n"):
            result = type_text.invoke({"text": "Do not send", "press_enter": True})

            self.mock_pyautogui.write.assert_not_called()
            self.mock_pyautogui.press.assert_not_called()
            self.assertIn("Typing and enter keypress cancelled by user", result)

    def test_type_text_with_enter_eof_failsafe(self):
        """Verifies EOFError (non-interactive mode) fails safe without typing or sending."""
        with patch("builtins.input", side_effect=EOFError("No TTY attached")):
            result = type_text.invoke({"text": "Automated text", "press_enter": True})

            self.mock_pyautogui.write.assert_not_called()
            self.mock_pyautogui.press.assert_not_called()
            self.assertIn("Typing and enter keypress cancelled", result)

    def test_press_enter_key_confirmed(self):
        """Verifies standalone press_enter_key asks for confirmation and presses enter."""
        with patch("builtins.input", return_value="yes") as mock_input:
            result = press_enter_key.invoke({})

            mock_input.assert_called_once()
            prompt_arg = mock_input.call_args[0][0]
            self.assertIn("Press enter in active window", prompt_arg)
            self.assertIn("WhatsApp Web - Google Chrome", prompt_arg)

            self.mock_pyautogui.press.assert_called_once_with("enter")
            self.assertIn("Successfully pressed enter in 'WhatsApp Web - Google Chrome'", result)

    def test_press_enter_key_rejected(self):
        """Verifies rejecting standalone press_enter_key aborts safely."""
        with patch("builtins.input", return_value="no"):
            result = press_enter_key.invoke({})

            self.mock_pyautogui.press.assert_not_called()
            self.assertIn("Enter keypress cancelled by user", result)

    def test_press_enter_key_eof_failsafe(self):
        """Verifies standalone press_enter_key fails safe on EOF."""
        with patch("builtins.input", side_effect=EOFError("EOF")):
            result = press_enter_key.invoke({})

            self.mock_pyautogui.press.assert_not_called()
            self.assertIn("Enter keypress cancelled - no interactive confirmation", result)


if __name__ == "__main__":
    unittest.main()
