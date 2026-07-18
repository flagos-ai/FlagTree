"""One-click test runner: execute every standalone script in this directory."""

import os
import subprocess
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SKIP = {"conftest.py", "__init__.py", "test_all_scripts.py"}


def _collect_scripts():
    return sorted(f for f in os.listdir(_THIS_DIR) if f.endswith(".py") and f not in _SKIP)


@pytest.mark.parametrize("script", _collect_scripts(), ids=lambda s: s.removesuffix(".py"))
def test_edsl_script(script):
    result = subprocess.run(
        [sys.executable, os.path.join(_THIS_DIR, script)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (f"{script} failed (exit {result.returncode}):\n"
                                    f"--- stdout ---\n{result.stdout[-2000:]}\n"
                                    f"--- stderr ---\n{result.stderr[-2000:]}")
