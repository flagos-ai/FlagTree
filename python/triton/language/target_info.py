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

from triton.runtime import driver
from triton.runtime.jit import constexpr_function

__all__ = ["current_target"]


def current_target():
    try:
        active_driver = driver.active
    except RuntimeError:
        # If there is no active driver, return None
        return None
    return active_driver.get_current_target()


current_target.__triton_builtin__ = True


@constexpr_function
def is_cuda():
    target = current_target()
    return target is not None and target.backend == "cuda"


@constexpr_function
def cuda_capability_geq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    target = current_target()
    if target is None or target.backend != "cuda":
        return False
    assert isinstance(target.arch, int)
    return target.arch >= major * 10 + minor


@constexpr_function
def is_hip():
    target = current_target()
    return target is not None and target.backend == "hip"


@constexpr_function
def is_hip_cdna3():
    target = current_target()
    return target is not None and target.arch == "gfx942"


@constexpr_function
def is_hip_cdna4():
    target = current_target()
    return target is not None and target.arch == "gfx950"
