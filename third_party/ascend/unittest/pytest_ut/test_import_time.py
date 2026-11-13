# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Regression test for triton import time.

Guards against accidental regressions from PR #1200 (lazy init of
``is_compile_on_910_95`` and moving the check out of module import time).
Import is measured in a fresh subprocess so the measurement is not masked by
triton already being imported in the pytest worker process.
"""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.backend("native")

# Upper limit (ms) for ``import triton``. The baseline after PR #1200 is
# ~240ms on the dev container; 1000ms gives ~4x headroom to absorb CI jitter
# while still catching real regressions (e.g. re-introducing eager init).
IMPORT_TIME_LIMIT_MS = 1000

# Measure only the import call; interpreter startup is excluded by reading the
# elapsed time from inside the subprocess.
_TIMING_SCRIPT = ("import time\n"
                  "t = time.perf_counter()\n"
                  "import triton\n"
                  "print((time.perf_counter() - t) * 1000)\n")


@pytest.mark.parametrize("run_id", range(3))
def test_import_triton_time_under_limit(run_id):
    result = subprocess.run(
        [sys.executable, "-c", _TIMING_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
    )
    elapsed_ms = float(result.stdout.strip())
    msg = (f"run {run_id}: import triton took {elapsed_ms:.1f} ms "
           f"(limit {IMPORT_TIME_LIMIT_MS} ms)")
    assert elapsed_ms < IMPORT_TIME_LIMIT_MS, msg
