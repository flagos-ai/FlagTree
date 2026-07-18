# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
