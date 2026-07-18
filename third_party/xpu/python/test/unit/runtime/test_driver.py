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

import sys
import os

import pytest
import triton


def test_is_lazy():
    from importlib import reload
    reload(sys.modules["triton.runtime.driver"])
    reload(sys.modules["triton.runtime"])
    assert triton.runtime.driver._active is None
    assert triton.runtime.driver._default is None
    utils = triton.runtime.driver.active.utils  # noqa: F841
    assert issubclass(triton.runtime.driver.active.__class__, getattr(triton.backends.driver, "DriverBase"))


@pytest.mark.skipif(os.environ.get("FLAGTREE_BACKEND") != "xpu", reason="XPU backend is not active")
def test_xpu_driver_smoke():
    from triton.runtime import driver

    assert set(triton.backends.backends) == {"xpu"}
    assert type(driver.active).__name__ == "XPUDriver"
    assert driver.active.get_current_target().backend == "xpu"
