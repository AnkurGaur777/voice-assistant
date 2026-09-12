"""
Local Jarvis - Sandbox Unit Test Suite

Verifies:
1. Normal math calculations (e.g. 15% of 340, arithmetic, math module).
2. Blocked module imports are rejected via pre-execution AST static checks:
   - OS/Process modules: os, sys, subprocess, shutil, pathlib, io
   - Network modules: socket, urllib, requests, http
3. Prohibited dynamic calls are rejected: __import__, eval, exec, compile.
4. Timeout enforcement: infinite loop triggers 5-second subprocess termination.
5. Confinement & no filesystem access: open() is disabled and raises PermissionError;
   cannot read/write files anywhere.
6. Strict output capping: outputs exceeding 2000 chars are truncated.
7. Temporary directory cleanup: confirmed cleaned up on both normal execution AND on timeout.
8. LangGraph @tool invocation interface.
"""

import os
from pathlib import Path
import sys
import time
import unittest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.tools.sandbox import (
    DEFAULT_TIMEOUT_SECONDS,
    execute_in_sandbox,
    run_python,
    validate_code_ast,
)


class TestSandboxMath(unittest.TestCase):
    """Tests normal mathematical calculations and safe standard libraries."""

    def test_quick_percentage_calculation(self):
        """Calculates 'what's 15% of 340' -> 51.0."""
        result = execute_in_sandbox("340 * 0.15")
        self.assertEqual(result, "51.0")

    def test_multiline_arithmetic(self):
        """Tests multiline script calculating total and printing result."""
        code = (
            "base_price = 250\n"
            "tax_rate = 0.08\n"
            "discount = 20\n"
            "final_price = (base_price - discount) * (1 + tax_rate)\n"
            "print(f'Total: {final_price:.2f}')\n"
        )
        result = execute_in_sandbox(code)
        self.assertEqual(result, "Total: 248.40")

    def test_allowed_math_modules(self):
        """Tests that safe math modules (math, statistics, decimal) work properly."""
        code = (
            "import math\n"
            "import statistics\n"
            "vals = [10, 20, 30, 40, 50]\n"
            "mean_val = statistics.mean(vals)\n"
            "sqrt_val = math.sqrt(144)\n"
            "print(f'Mean: {mean_val}, Sqrt: {sqrt_val}')\n"
        )
        result = execute_in_sandbox(code)
        self.assertEqual(result, "Mean: 30, Sqrt: 12.0")

    def test_trailing_expression_evaluated(self):
        """Verifies trailing expressions without explicit print() are captured."""
        code = "a = 7\nb = 6\na * b"
        result = execute_in_sandbox(code)
        self.assertEqual(result, "42")

    def test_langgraph_tool_invocation(self):
        """Verifies direct LangGraph @tool .invoke() works as expected."""
        result = run_python.invoke({"code": "100 / 4"})
        self.assertEqual(result, "25.0")


class TestSandboxSecurityAndImports(unittest.TestCase):
    """Tests rejection of blocked modules, network modules, and dynamic evaluators."""

    def test_blocked_os_import(self):
        """Rejects 'import os'."""
        err = validate_code_ast("import os")
        self.assertIsNotNone(err)
        self.assertIn("SecurityError", err)
        self.assertIn("os", err)

        res = execute_in_sandbox("import os")
        self.assertTrue(res.startswith("Error: SecurityError"))

    def test_blocked_sys_import(self):
        """Rejects 'import sys'."""
        res = execute_in_sandbox("import sys")
        self.assertTrue(res.startswith("Error: SecurityError"))

    def test_blocked_subprocess_import(self):
        """Rejects 'import subprocess'."""
        res = execute_in_sandbox("import subprocess")
        self.assertTrue(res.startswith("Error: SecurityError"))

    def test_blocked_shutil_import(self):
        """Rejects 'import shutil'."""
        res = execute_in_sandbox("import shutil")
        self.assertTrue(res.startswith("Error: SecurityError"))

    def test_blocked_socket_import(self):
        """Rejects 'import socket'."""
        res = execute_in_sandbox("import socket")
        self.assertTrue(res.startswith("Error: SecurityError"))

    def test_blocked_urllib_import(self):
        """Rejects 'import urllib.request'."""
        res = execute_in_sandbox("import urllib.request")
        self.assertTrue(res.startswith("Error: SecurityError"))

    def test_blocked_from_urllib_import(self):
        """Rejects 'from urllib.request import urlopen'."""
        res = execute_in_sandbox("from urllib.request import urlopen")
        self.assertTrue(res.startswith("Error: SecurityError"))

    def test_blocked_requests_import(self):
        """Rejects 'import requests'."""
        res = execute_in_sandbox("import requests")
        self.assertTrue(res.startswith("Error: SecurityError"))

    def test_blocked_http_import(self):
        """Rejects 'from http import client'."""
        res = execute_in_sandbox("from http import client")
        self.assertTrue(res.startswith("Error: SecurityError"))

    def test_blocked_pathlib_and_io_imports(self):
        """Rejects 'import pathlib' and 'import io'."""
        res1 = execute_in_sandbox("import pathlib")
        self.assertTrue(res1.startswith("Error: SecurityError"))
        res2 = execute_in_sandbox("import io")
        self.assertTrue(res2.startswith("Error: SecurityError"))

    def test_blocked_dynamic_calls(self):
        """Rejects __import__, eval, exec, and compile invocations."""
        for expr in ["__import__('os')", "eval('1 + 1')", "exec('x = 1')", "compile('1', '', 'eval')"]:
            res = execute_in_sandbox(expr)
            self.assertTrue(res.startswith("Error: SecurityError"), f"Failed for {expr}: {res}")


class TestSandboxFilesystemConfinement(unittest.TestCase):
    """Tests that filesystem access is blocked and cannot read/write outside temp dir."""

    def test_open_is_disabled(self):
        """Direct open() call raises PermissionError."""
        res = execute_in_sandbox("open('test.txt', 'w')")
        self.assertTrue(
            "PermissionError" in res or "Filesystem access is disabled" in res,
            f"Expected PermissionError, got: {res}",
        )

    def test_read_outside_file_blocked(self):
        """Attempting to read outside workspace file raises PermissionError."""
        res = execute_in_sandbox("with open('../../requirements.txt', 'r') as f: data = f.read()")
        self.assertTrue(
            "PermissionError" in res or "Filesystem access is disabled" in res,
            f"Expected PermissionError, got: {res}",
        )

    def test_write_outside_file_blocked(self):
        """Attempting to write outside temp dir raises PermissionError."""
        outside_path = str(PROJECT_ROOT / "escaped_sandbox_test.txt").replace("\\", "/")
        res = execute_in_sandbox(f"with open('{outside_path}', 'w') as f: f.write('bad')")
        self.assertTrue(
            "PermissionError" in res or "Filesystem access is disabled" in res,
            f"Expected PermissionError, got: {res}",
        )
        self.assertFalse(os.path.exists(outside_path), "Outside file must not have been created!")


class TestSandboxTimeoutAndCleanup(unittest.TestCase):
    """Tests timeout enforcement and temporary directory cleanup."""

    def test_timeout_on_infinite_loop(self):
        """Infinite loop triggers timeout error and terminates subprocess."""
        start = time.perf_counter()
        # Use a 1.5s timeout for fast automated testing while verifying subprocess kill
        res = execute_in_sandbox("while True: pass", timeout=1.5)
        elapsed = time.perf_counter() - start

        self.assertIn("Execution timed out", res)
        self.assertGreaterEqual(elapsed, 1.4)
        self.assertLess(elapsed, 3.5)

    def test_temp_dir_cleaned_up_on_normal_execution(self):
        """Temporary directory is guaranteed to be deleted after normal execution."""
        captured = []
        res = execute_in_sandbox("print('hello')", captured_temp_dirs=captured)
        self.assertEqual(res, "hello")
        self.assertEqual(len(captured), 1)
        temp_dir = captured[0]
        self.assertFalse(os.path.exists(temp_dir), f"Temp dir {temp_dir} was not cleaned up!")

    def test_temp_dir_cleaned_up_on_timeout(self):
        """
        Confirms the temp directory is cleaned up even when the subprocess times out
        (while True: pass case), not just on normal completion.
        """
        captured = []
        res = execute_in_sandbox("while True: pass", timeout=1.0, captured_temp_dirs=captured)
        self.assertIn("Execution timed out", res)
        self.assertEqual(len(captured), 1)
        temp_dir = captured[0]
        self.assertFalse(
            os.path.exists(temp_dir),
            f"Temp dir {temp_dir} must be cleaned up even on timeout!",
        )


class TestSandboxOutputCap(unittest.TestCase):
    """Tests output length capping to 2000 characters."""

    def test_output_capped_at_2000_chars(self):
        """Large stdout is capped at 2000 characters with truncation notice."""
        code = "print('A' * 5000)"
        res = execute_in_sandbox(code)
        self.assertIn("Output truncated at 2000 characters", res)
        # Content before the truncation message should be at most 2000 chars
        content_part = res.split("\n... [Output truncated")[0]
        self.assertEqual(len(content_part), 2000)


def main():
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
