"""
Local Jarvis - Sandboxed Python Execution Tool (Phase 4)

Provides a secure execution sandbox for quick mathematical calculations,
string transformations, and basic data processing.

Security Architecture:
1. Dual-Layer Isolation:
   - Layer 1 (Static AST Analysis): Pre-execution AST inspection strictly rejects
     import statements targeting banned modules (network, filesystem, OS, processes)
     as well as dangerous dynamic evaluators (eval, exec, compile, __import__).
   - Layer 2 (Subprocess & Builtins Isolation): Executes code in an isolated child
     process (`sys.executable -I`) inside a throwaway temporary directory (`tempfile.TemporaryDirectory()`).
     File-related builtins (`open`) are replaced with permission errors, and runtime
     module imports are restricted to a safe math/logic whitelist.
2. Resource Limits & Windows Considerations:
   - Strict execution timeout: 5.0 seconds (child process terminated upon expiry).
   - Strict output cap: 2000 characters to prevent flooding LLM conversation context.
   - NOTE: This sandbox does not limit memory or CPU usage beyond the 5-second timeout
     (Windows lacks easy resource-limit primitives like POSIX ulimit) — this is an accepted
     known limitation for personal local use, not something to silently miss.
"""

import ast
import os
import subprocess
import sys
import tempfile
from typing import List, Optional, Set

from langchain_core.tools import tool

# --- Configuration & Security Constants ---
DEFAULT_TIMEOUT_SECONDS: float = 5.0
MAX_OUTPUT_LENGTH: int = 2000

# Modules prohibited from being imported in sandbox code
BANNED_MODULES: Set[str] = {
    # Operating system, process, low-level system access
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "io",
    "ctypes",
    "posix",
    "nt",
    "builtins",
    "importlib",
    "multiprocessing",
    "threading",
    "signal",
    "pty",
    "platform",
    "inspect",
    "winreg",
    "_winapi",
    # Network modules
    "socket",
    "urllib",
    "requests",
    "http",
    "aiohttp",
    "httpx",
    "ftplib",
    "poplib",
    "imaplib",
    "smtplib",
    "telnetlib",
    "xmlrpc",
    "asyncio",
    "webbrowser",
}

# Dangerous dynamic builtins banned from AST call invocations
BANNED_BUILTIN_CALLS: Set[str] = {"eval", "exec", "compile", "__import__"}

# Modules permitted in the runtime sandbox whitelist
SAFE_MATH_MODULES: Set[str] = {
    "math",
    "cmath",
    "decimal",
    "fractions",
    "random",
    "statistics",
    "itertools",
    "functools",
    "string",
    "re",
    "collections",
    "operator",
}

# Subprocess runner script executed in isolated mode (-I)
_RUNNER_SCRIPT: str = '''
import ast
import sys

user_code = sys.stdin.read()

SAFE_MATH_MODULES = {
    "math", "cmath", "decimal", "fractions", "random", "statistics",
    "itertools", "functools", "string", "re", "collections", "operator"
}

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in SAFE_MATH_MODULES:
        raise ImportError(f"Importing '{name}' is not allowed in sandbox.")
    return __import__(name, globals, locals, fromlist, level)

def forbidden_open(*args, **kwargs):
    raise PermissionError("Filesystem access is disabled in sandbox.")

safe_builtins = {
    "abs": abs, "all": all, "any": any, "ascii": ascii, "bin": bin,
    "bool": bool, "bytearray": bytearray, "bytes": bytes, "chr": chr,
    "complex": complex, "dict": dict, "divmod": divmod, "enumerate": enumerate,
    "filter": filter, "float": float, "format": format, "frozenset": frozenset,
    "hash": hash, "hex": hex, "int": int, "isinstance": isinstance,
    "issubclass": issubclass, "iter": iter, "len": len, "list": list,
    "map": map, "max": max, "min": min, "next": next, "oct": oct,
    "ord": ord, "pow": pow, "print": print, "range": range, "repr": repr,
    "reversed": reversed, "round": round, "set": set, "slice": slice,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "type": type,
    "zip": zip, "True": True, "False": False, "None": None,
    "open": forbidden_open,
    "__import__": safe_import,
    "ArithmeticError": ArithmeticError, "AssertionError": AssertionError,
    "AttributeError": AttributeError, "BaseException": BaseException,
    "Exception": Exception, "FloatingPointError": FloatingPointError,
    "ImportError": ImportError, "IndexError": IndexError, "KeyError": KeyError,
    "LookupError": LookupError, "NameError": NameError, "OverflowError": OverflowError,
    "PermissionError": PermissionError, "TypeError": TypeError, "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
}

safe_globals = {"__builtins__": safe_builtins, "__name__": "__main__"}

try:
    tree = ast.parse(user_code)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = tree.body.pop()
        if tree.body:
            exec(compile(tree, "<sandbox>", "exec"), safe_globals)
        expr_val = eval(compile(ast.Expression(body=last_expr.value), "<sandbox>", "eval"), safe_globals)
        if expr_val is not None:
            print(repr(expr_val))
    else:
        exec(compile(tree, "<sandbox>", "exec"), safe_globals)
except Exception as e:
    print(f"{type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
'''


def validate_code_ast(code: str) -> Optional[str]:
    """
    Performs pre-execution static analysis on code using Python's AST parser.

    Rejects:
    - Syntax errors
    - Imports of banned modules (network, filesystem, OS, process)
    - Calls to dynamic execution builtins (__import__, eval, exec, compile)

    Returns:
        None if validation succeeds, or an error message string if rejected.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    for node in ast.walk(tree):
        # Check standard imports (e.g. 'import os', 'import socket.error')
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0].lower()
                if root_module in BANNED_MODULES:
                    return f"SecurityError: Importing '{alias.name}' is prohibited in sandbox."

        # Check from-imports (e.g. 'from os import system', 'from urllib.request import urlopen')
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split(".")[0].lower()
                if root_module in BANNED_MODULES:
                    return f"SecurityError: Importing from '{node.module}' is prohibited in sandbox."

        # Check banned function calls (e.g. __import__('os'), eval(...))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_BUILTIN_CALLS:
                return f"SecurityError: Calling '{node.func.id}()' is prohibited in sandbox."

    return None


def execute_in_sandbox(
    code: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_length: int = MAX_OUTPUT_LENGTH,
    captured_temp_dirs: Optional[List[str]] = None,
) -> str:
    """
    Executes Python code in a dedicated, isolated child process.

    Safety measures:
    - Pre-execution AST check for blocked imports and calls.
    - Child process runs in isolated mode (-I) using `sys.executable`.
    - Working directory is bound to a throwaway temporary directory that is
      guaranteed to be deleted upon exit or timeout.
    - No filesystem access: `open` raises PermissionError.
    - Subprocess execution governed by strict timeout (terminates child process if exceeded).
    - Output capped at `max_output_length` characters.

    Note:
        This sandbox does not limit memory or CPU usage beyond the 5-second timeout
        (Windows lacks easy resource-limit primitives like POSIX ulimit) — this is an accepted
        known limitation for personal local use, not something to silently miss.

    Args:
        code: Python source code or arithmetic expression to run.
        timeout: Maximum execution time in seconds before terminating child process.
        max_output_length: Maximum number of characters returned in output.
        captured_temp_dirs: Optional list to append the temporary directory path to,
            allowing verification that directories are deleted even on timeouts.

    Returns:
        Output string (stdout, stderr, or error description).
    """
    if not code or not code.strip():
        return "Error: No Python code provided for execution."

    # 1. Pre-execution static analysis check
    validation_error = validate_code_ast(code)
    if validation_error:
        return f"Error: {validation_error}"

    # 2. Setup isolated throwaway temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        if captured_temp_dirs is not None:
            captured_temp_dirs.append(temp_dir)

        # Minimal environment for child process
        safe_env = {
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
            "WINDIR": os.environ.get("WINDIR", "C:\\Windows"),
            "TEMP": temp_dir,
            "TMP": temp_dir,
            "PYTHONPATH": "",
        }

        try:
            # Launch isolated child process
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER_SCRIPT],
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=temp_dir,
                env=safe_env,
            )

            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()

            if proc.returncode != 0:
                err_msg = stderr if stderr else stdout
                return f"Error: {err_msg}" if err_msg else f"Error: Process exited with returncode {proc.returncode}"

            if not stdout:
                result = "Result: Code executed successfully with no output."
            else:
                result = stdout

        except subprocess.TimeoutExpired:
            return f"Error: Execution timed out (exceeded {timeout:.1f} seconds limit)."
        except Exception as e:
            return f"Error executing sandbox code: {type(e).__name__}: {e}"

    # 3. Enforce strict output length cap
    if len(result) > max_output_length:
        result = result[:max_output_length] + "\n... [Output truncated at 2000 characters]"

    return result


@tool
def run_python(code: str) -> str:
    """Execute Python code in a secure sandbox for mathematical calculations, arithmetic, and percentages.

    Use this tool whenever you need to:
    - Calculate percentages (e.g. "what is 358% of 340" -> code: '340 * 3.58', "15% of 80" -> code: '80 * 0.15').
    - Perform mathematical calculations and arithmetic (e.g. addition, multiplication, division, powers, square roots).
    - Evaluate numerical expressions, statistics, or unit conversions.
    - Format strings or perform programmatic transformations.

    Code executes in an isolated child subprocess without filesystem or network access.
    Returns stdout/stderr and expression results capped at 2000 characters.

    Note: This sandbox does not limit memory or CPU usage beyond the 5-second timeout
    (Windows lacks easy resource-limit primitives like POSIX ulimit) — this is an accepted
    known limitation for personal local use.

    Args:
        code: Python source code or expression to execute (e.g. '340 * 3.58' or 'import math; math.sqrt(144)').
    """
    print(f"[Sandbox] Executing code: {code!r}")
    result = execute_in_sandbox(code)
    summary = result.replace("\n", " ")
    if len(summary) > 100:
        summary = summary[:100] + "..."
    print(f"[Sandbox] Result: {summary}")
    return result
