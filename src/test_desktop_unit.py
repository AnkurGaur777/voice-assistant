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


class TestOpenApplicationLauncher(unittest.TestCase):
    """Unit tests for universal Windows application discovery and open_application launcher."""

    def test_empty_app_name(self):
        """Verifies empty application name returns clear error."""
        from src.agent.tools.desktop import open_application
        result = open_application.invoke({"app_name": "  "})
        self.assertIn("Error: Application name cannot be empty", result)

    def test_alias_fast_path_launch(self):
        """Verifies known aliases launch via fast-path."""
        from src.agent.tools.desktop import find_matching_app, launch_app
        match = find_matching_app("calc")
        self.assertIsNotNone(match)
        self.assertEqual(match[0], "calc")
        self.assertEqual(match[1], "calc.exe")
        self.assertEqual(match[2], "alias")

        with patch("os.startfile") as mock_start, patch("src.agent.tools.desktop.focus_window_by_name", return_value=True):
            res = launch_app("calc")
            self.assertIn("launched successfully", res)
            mock_start.assert_called_once()
            self.assertTrue(mock_start.call_args[0][0].lower().endswith("calc.exe"))

    def test_shortcut_scanning_and_noise_filtering(self):
        """Verifies Start Menu/Desktop scanning indexes valid apps and ignores uninstall/noise files."""
        from src.agent.tools.desktop import scan_shortcuts

        mock_walk_data = [
            (
                r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs",
                [],
                [
                    "Word.lnk",
                    "Microsoft Edge.lnk",
                    "Uninstall Razer.lnk",
                    "Release Notes.lnk",
                    "readme.txt",
                ],
            )
        ]

        with patch("src.agent.tools.desktop.get_shortcut_directories", return_value=[r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs"]), \
             patch("os.walk", return_value=mock_walk_data):
            shortcuts = scan_shortcuts()

            self.assertIn("Word", shortcuts)
            self.assertIn("Microsoft Edge", shortcuts)
            # Noise should be excluded
            self.assertNotIn("Uninstall Razer", shortcuts)
            self.assertNotIn("Release Notes", shortcuts)
            self.assertNotIn("readme", shortcuts)

    def test_store_app_query_and_caching(self):
        """Verifies PowerShell Get-StartApps queries and caches packaged UWP applications."""
        import json
        from src.agent.tools.desktop import get_windows_store_apps

        mock_output = json.dumps([
            {"Name": "Instagram", "AppID": "Facebook.InstagramBeta_8xx8rvfyw5nnt!App"},
            {"Name": "WhatsApp", "AppID": "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App"},
            {"Name": "Uninstall Node.js", "AppID": "Microsoft.AutoGenerated.12345"},
        ])

        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = mock_output

        with patch("subprocess.run", return_value=mock_res):
            apps = get_windows_store_apps(force_refresh=True)
            self.assertIn("Instagram", apps)
            self.assertEqual(apps["Instagram"], "Facebook.InstagramBeta_8xx8rvfyw5nnt!App")
            self.assertIn("WhatsApp", apps)
            # Uninstaller entry filtered out
            self.assertNotIn("Uninstall Node.js", apps)

    def test_fuzzy_matching_precedence(self):
        """Verifies exact match, prefix match, and ambiguity resolution."""
        from src.agent.tools.desktop import find_matching_app

        mock_shortcuts = {
            "Visual Studio Code": r"C:\Programs\Visual Studio Code.lnk",
            "Visual Studio Installer": r"C:\Programs\Visual Studio Installer.lnk",
            "VLC media player": r"C:\Programs\VLC media player.lnk",
        }
        mock_store = {
            "Instagram": "Facebook.InstagramBeta_8xx8rvfyw5nnt!App",
            "WhatsApp": "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App",
        }

        with patch("src.agent.tools.desktop.scan_shortcuts", return_value=mock_shortcuts), \
             patch("src.agent.tools.desktop.get_windows_store_apps", return_value=mock_store):
            # 1. Exact match for Store app
            match_ig = find_matching_app("Instagram")
            self.assertIsNotNone(match_ig)
            self.assertEqual(match_ig[0], "Instagram")
            self.assertEqual(match_ig[1], "Facebook.InstagramBeta_8xx8rvfyw5nnt!App")
            self.assertEqual(match_ig[2], "store")

            # 2. Prefix / partial match
            match_insta = find_matching_app("insta")
            self.assertIsNotNone(match_insta)
            self.assertEqual(match_insta[0], "Instagram")

            # 3. Ambiguity resolution: "visual studio" must prefer "Visual Studio Code" over "Visual Studio Installer"
            match_vs = find_matching_app("visual studio")
            self.assertIsNotNone(match_vs)
            self.assertEqual(match_vs[0], "Visual Studio Code")

            # 4. Word match
            match_vlc = find_matching_app("vlc")
            self.assertIsNotNone(match_vlc)
            self.assertEqual(match_vlc[0], "VLC media player")

    def test_launch_store_app(self):
        """Verifies launching a Windows Store app via shell:AppsFolder URI."""
        from src.agent.tools.desktop import launch_app

        mock_store = {
            "Instagram": "Facebook.InstagramBeta_8xx8rvfyw5nnt!App",
        }

        with patch("src.agent.tools.desktop.scan_shortcuts", return_value={}), \
             patch("src.agent.tools.desktop.get_windows_store_apps", return_value=mock_store), \
             patch("os.startfile") as mock_start, \
             patch("src.agent.tools.desktop.focus_window_by_name", return_value=True):
            res = launch_app("Instagram")
            self.assertIn("launched successfully", res)
            self.assertIn("Instagram", res)
            mock_start.assert_called_once_with(r"shell:AppsFolder\Facebook.InstagramBeta_8xx8rvfyw5nnt!App")

    def test_launch_shortcut_app(self):
        """Verifies launching a desktop shortcut (.lnk) via os.startfile."""
        from src.agent.tools.desktop import launch_app

        mock_shortcuts = {
            "Excel": r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Excel.lnk",
        }

        with patch("src.agent.tools.desktop.scan_shortcuts", return_value=mock_shortcuts), \
             patch("src.agent.tools.desktop.get_windows_store_apps", return_value={}), \
             patch("os.startfile") as mock_start, \
             patch("src.agent.tools.desktop.focus_window_by_name", return_value=True):
            res = launch_app("Excel")
            self.assertIn("launched successfully", res)
            self.assertIn("Excel", res)
            mock_start.assert_called_once_with(r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Excel.lnk")

    def test_unresolvable_app_rich_error(self):
        """Verifies non-existent app returns descriptive error mentioning all searched catalogs."""
        from src.agent.tools.desktop import launch_app

        with patch("src.agent.tools.desktop.scan_shortcuts", return_value={}), \
             patch("src.agent.tools.desktop.get_windows_store_apps", return_value={}), \
             patch("os.startfile", side_effect=FileNotFoundError("Not found")):
            res = launch_app("completely_unknown_fake_app_999")
            self.assertIn("Error: Unable to launch application", res)
            self.assertIn("fast-path aliases", res)
            self.assertIn("Start Menu & Desktop shortcuts", res)
            self.assertIn("Windows Store applications", res)
            self.assertIn("--list-launchable-apps", res)

    def test_list_launchable_apps_function(self):
        """Verifies list_launchable_apps returns discovered applications across all categories."""
        from src.agent.tools.desktop import list_launchable_apps

        mock_shortcuts = {"TestApp": r"C:\TestApp.lnk"}
        mock_store = {"TestStoreApp": "TestStoreApp.AppId"}

        with patch("src.agent.tools.desktop.scan_shortcuts", return_value=mock_shortcuts), \
             patch("src.agent.tools.desktop.get_windows_store_apps", return_value=mock_store):
            items = list_launchable_apps()
            names = [item["name"] for item in items]
            self.assertIn("notepad", names)
            self.assertIn("TestApp", names)
            self.assertIn("TestStoreApp", names)


class TestUIAAndDesktopControls(unittest.TestCase):
    """Unit tests for Windows UI Automation and scroll controls."""

    def test_scroll_window_down(self):
        """Verifies scrolling down calculates center and sends negative wheel delta."""
        from src.agent.tools.desktop import scroll_window
        mock_pyautogui = MagicMock()
        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}), \
             patch("src.agent.tools.desktop.get_active_window_title", return_value="Test App"), \
             patch("pygetwindow.getActiveWindow") as mock_get_win:
            mock_win = MagicMock()
            mock_win.left = 100
            mock_win.top = 200
            mock_win.width = 400
            mock_win.height = 600
            mock_get_win.return_value = mock_win

            res = scroll_window.invoke({"direction": "down", "amount": 3})
            self.assertIn("Successfully scrolled active window 'Test App' down by 3 clicks", res)
            self.assertIn("(300, 500)", res)
            mock_pyautogui.scroll.assert_called_once_with(-360, x=300, y=500)

    def test_scroll_window_up_clamped(self):
        """Verifies scrolling up with amount clamping (e.g. 50 -> 25) and positive delta."""
        from src.agent.tools.desktop import scroll_window
        mock_pyautogui = MagicMock()
        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}), \
             patch("src.agent.tools.desktop.get_active_window_title", return_value="Browser"), \
             patch("pygetwindow.getActiveWindow", return_value=None):

            res = scroll_window.invoke({"direction": "up", "amount": 50})
            self.assertIn("Successfully scrolled active window 'Browser' up by 25 clicks", res)
            mock_pyautogui.scroll.assert_called_once_with(3000, x=None, y=None)

    def test_scroll_window_invalid_direction(self):
        """Verifies invalid scroll direction returns descriptive error."""
        from src.agent.tools.desktop import scroll_window
        res = scroll_window.invoke({"direction": "sideways", "amount": 3})
        self.assertIn("Error: Invalid scroll direction 'sideways'", res)

    def test_find_and_click_app_not_found(self):
        """Verifies error when target application window cannot be found."""
        from src.agent.tools.desktop import find_and_click_element
        with patch("src.agent.tools.desktop.get_window_by_app_name", return_value=None):
            res = find_and_click_element.invoke({"app_name": "NonExistentApp", "element_name": "Button"})
            self.assertIn("Error: Could not locate an open window for application 'NonExistentApp'", res)

    def test_find_and_click_invoke_success(self):
        """Verifies element is activated via invoke() without cursor movement."""
        from src.agent.tools.desktop import find_and_click_element

        mock_ctrl = MagicMock()
        mock_ctrl.window_text.return_value = "Reels"
        mock_ctrl.element_info.name = "Reels"
        mock_ctrl.element_info.runtime_id = [1, 2, 3]
        mock_ctrl.friendly_class_name.return_value = "TabItem"
        rect = MagicMock()
        rect.width.return_value = 100
        rect.height.return_value = 50
        mock_ctrl.rectangle.return_value = rect
        mock_ctrl.is_visible.return_value = True
        mock_ctrl.is_enabled.return_value = True
        mock_ctrl.invoke = MagicMock()

        mock_wrapper = MagicMock()
        mock_wrapper.descendants.return_value = [mock_ctrl]

        with patch("src.agent.tools.desktop.get_window_by_app_name", return_value=(12345, "Instagram")), \
             patch("pywinauto.controls.uiawrapper.UIAWrapper", return_value=mock_wrapper), \
             patch("pywinauto.uia_element_info.UIAElementInfo", return_value=MagicMock()):
            res = find_and_click_element.invoke({"app_name": "Instagram", "element_name": "Reels"})
            self.assertIn("Successfully invoked element 'Reels' (TabItem)", res)
            self.assertIn("invoked via UI Automation (cursor unmoved)", res)
            mock_ctrl.invoke.assert_called_once()
            mock_ctrl.click_input.assert_not_called()

    def test_find_and_click_fallback_click_input(self):
        """Verifies fallback to click_input() when invoke() raises exception."""
        from src.agent.tools.desktop import find_and_click_element

        mock_ctrl = MagicMock()
        mock_ctrl.window_text.return_value = "Chats"
        mock_ctrl.element_info.name = "Chats"
        mock_ctrl.element_info.runtime_id = [4, 5, 6]
        mock_ctrl.friendly_class_name.return_value = "Button"
        rect = MagicMock()
        rect.width.return_value = 80
        rect.height.return_value = 40
        mock_ctrl.rectangle.return_value = rect
        mock_ctrl.is_visible.return_value = True
        mock_ctrl.is_enabled.return_value = True
        mock_ctrl.invoke.side_effect = RuntimeError("NoPatternInterfaceError")
        del mock_ctrl.toggle
        del mock_ctrl.select
        mock_ctrl.click_input = MagicMock()

        mock_wrapper = MagicMock()
        mock_wrapper.descendants.return_value = [mock_ctrl]

        with patch("src.agent.tools.desktop.get_window_by_app_name", return_value=(67890, "WhatsApp")), \
             patch("pywinauto.controls.uiawrapper.UIAWrapper", return_value=mock_wrapper), \
             patch("pywinauto.uia_element_info.UIAElementInfo", return_value=MagicMock()):
            res = find_and_click_element.invoke({"app_name": "WhatsApp", "element_name": "Chats"})
            self.assertIn("Successfully clicked element 'Chats' (Button)", res)
            self.assertIn("click_input", res)
            mock_ctrl.invoke.assert_called_once()
            mock_ctrl.click_input.assert_called_once()

    def test_auto_focus_input_field_prioritizes_composer(self):
        """Verifies auto_focus_input_field finds and focuses message composer."""
        from src.agent.tools.desktop import auto_focus_input_field

        search_box = MagicMock()
        search_box.element_info.name = "Search or start a new chat"
        search_box.window_text.return_value = "Search or start a new chat"
        search_box.is_visible.return_value = True
        r1 = MagicMock()
        r1.width.return_value = 200
        r1.height.return_value = 30
        r1.bottom = 180
        search_box.rectangle.return_value = r1

        composer_box = MagicMock()
        composer_box.element_info.name = "Type a message to Alice"
        composer_box.window_text.return_value = "Type a message to Alice"
        composer_box.is_visible.return_value = True
        r2 = MagicMock()
        r2.width.return_value = 600
        r2.height.return_value = 40
        r2.bottom = 900
        composer_box.rectangle.return_value = r2
        composer_box.set_focus = MagicMock()

        mock_wrapper = MagicMock()
        mock_wrapper.descendants.side_effect = lambda control_type: [search_box, composer_box] if control_type == "Edit" else []

        with patch("pywinauto.controls.uiawrapper.UIAWrapper", return_value=mock_wrapper), \
             patch("pywinauto.uia_element_info.UIAElementInfo", return_value=MagicMock()):
            focused = auto_focus_input_field(12345)
            self.assertTrue(focused)
            composer_box.set_focus.assert_called_once()
            search_box.set_focus.assert_not_called()

    def test_list_ui_elements(self):
        """Verifies list_ui_elements returns metadata dictionaries."""
        from src.agent.tools.desktop import list_ui_elements

        btn = MagicMock()
        btn.window_text.return_value = "Submit"
        btn.element_info.name = "Submit"
        btn.friendly_class_name.return_value = "Button"
        btn.is_visible.return_value = True
        btn.is_enabled.return_value = True
        btn.children.return_value = []

        mock_wrapper = MagicMock()
        mock_wrapper.window_text.return_value = "Root"
        mock_wrapper.element_info.name = "Root"
        mock_wrapper.friendly_class_name.return_value = "Dialog"
        mock_wrapper.children.return_value = [btn]

        with patch("src.agent.tools.desktop.get_window_by_app_name", return_value=(999, "Test App")), \
             patch("pywinauto.controls.uiawrapper.UIAWrapper", return_value=mock_wrapper), \
             patch("pywinauto.uia_element_info.UIAElementInfo", return_value=MagicMock()):
            elements = list_ui_elements("Test App")
            self.assertTrue(len(elements) >= 1)
            names = [e["name"] for e in elements]
            self.assertIn("Submit", names)


if __name__ == "__main__":
    unittest.main()

