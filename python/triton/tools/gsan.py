# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
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
"""Run a Python script under Triton Global Memory Sanitizer.

This module provides a small command-line wrapper runs the wrapped script with GSan enabled, e.g.
    python -m triton.tools.gsan my_script.py
"""

from __future__ import annotations

import argparse
import contextlib
import runpy
import sys
from pathlib import Path
from typing import Sequence

import triton
import torch
from triton.experimental.gsan._allocator import create_mem_pool


def _parse_args(argv: Sequence[str] | None = None) -> tuple[Path, list[str], str]:
    parser = argparse.ArgumentParser(description="Run a Python script with Triton Global Memory Sanitizer.")
    parser.add_argument("script", help="Python script to execute")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the target script")
    args = parser.parse_args(argv)

    script_args = list(args.script_args)
    if script_args[:1] == ["--"]:
        script_args = script_args[1:]

    return Path(args.script), script_args


@contextlib.contextmanager
def _script_context(script_path: Path, script_args: Sequence[str]):
    original_argv = sys.argv[:]
    original_path = sys.path[:]

    sys.argv = [str(script_path), *script_args]
    sys.path.insert(0, str(script_path.parent))
    try:
        yield script_path
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path


def main(argv: Sequence[str] | None = None) -> int:
    script_path, script_args = _parse_args(argv)

    script_path = script_path.resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_path}")

    triton.knobs.compilation.instrumentation_mode = "gsan"

    with torch.cuda.use_mem_pool(create_mem_pool()), \
        _script_context(script_path, script_args):
        runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
