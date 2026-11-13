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
"""TLE lit registration follows available tools after root configuration."""

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_TOOLS = ("triton-opt", "triton-tensor-layout", "triton-llvm-opt")


def _llvm_cmake_dir():
    candidates = [os.environ.get("LLVM_DIR")]
    syspath = os.environ.get("LLVM_SYSPATH")
    if syspath:
        candidates.append(str(Path(syspath) / "lib/cmake/llvm"))
    for cache in sorted((_ROOT / "build").glob("cmake.*/CMakeCache.txt")):
        match = re.search(r"^LLVM_DIR:[^=]*=(.*)$", cache.read_text(), re.MULTILINE)
        if match:
            candidates.append(match.group(1))
    for candidate in candidates:
        if candidate and (Path(candidate) / "LLVMConfig.cmake").is_file():
            return candidate
    pytest.skip("requires the LLVM CMake package used for a local build")


@pytest.mark.parametrize("tool_count", range(4))
def test_lit_registration_with_late_tool_targets(tmp_path, tool_count):
    cmake = shutil.which("cmake")
    if not cmake or not shutil.which("ninja"):
        pytest.skip("requires CMake and Ninja")
    llvm_dir = _llvm_cmake_dir()
    source = _ROOT / "third_party/tle/test"
    lines = [
        "cmake_minimum_required(VERSION 3.20)",
        "project(TleLitConfiguration LANGUAGES C CXX)",
        "find_package(LLVM CONFIG REQUIRED)",
        'list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")',
        "include(AddLLVM)",
        "set(TRITON_BUILD_UT OFF)",
        f'add_subdirectory("{source.as_posix()}" "${{CMAKE_BINARY_DIR}}/tle-test")',
    ]
    # Match the root build order: TLE is configured before native tools.
    lines.extend(f"add_custom_target({tool})" for tool in _TOOLS[:tool_count])
    (tmp_path / "CMakeLists.txt").write_text("\n".join(lines) + "\n")
    build = tmp_path / "build"
    result = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B",
         str(build), "-G", "Ninja", f"-DLLVM_DIR={llvm_dir}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    registered = "build check-triton-tle-lit-tests:" in (build / "build.ninja").read_text()
    assert registered == (tool_count == len(_TOOLS))
