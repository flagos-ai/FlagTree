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

import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language._layouts import SwizzledSharedLayout
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

__all__ = ["MBarrierLayout", "init", "wait", "arrive"]


class MBarrierLayout(SwizzledSharedLayout):
    """
    Layout for mbarrier synchronization.

    Args:
        cga_layout (List[List[int]]): CGA layout bases. Defaults to [].
    """

    def __init__(self, cga_layout=None):
        super().__init__(vec=1, per_phase=1, max_phase=1, order=[0], cga_layout=cga_layout or [])


@builtin
def init(mbarrier, count, _semantic=None):
    """
    Initialize an mbarrier with a specified count. An mbarrier consists of an init count, a pending count and a phase.
    At initialization, the init count and pending count are initialized with the given 'count' and the phase is initialized to 0.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to initialize.
        count (int): The initial count for the barrier. Must be a positive integer.
    """
    count = _unwrap_if_constexpr(count)
    _semantic.builder.create_lds_barrier_init(mbarrier.handle, count)


@builtin
def wait(mbarrier, phase, _semantic=None):
    """
    Wait until the mbarrier's phase differs from the provided phase value.
    This means that the given 'phase' has completed.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to wait on.
        phase (int): The phase value to compare against. The wait completes when
        the barrier's phase becomes different from this value.
    """
    phase = _semantic.to_tensor(phase)

    _semantic.builder.create_lds_barrier_wait(mbarrier.handle, phase.handle)


@builtin
def arrive(mbarrier, *, count=1, _semantic=None):
    """
    Arrive at an mbarrier with a specified count. The operation requires a `count` attribute
    of at least 1, and decreases the pending arrival count of the mbarrier by the specific count.
    If the pending count reaches zero, the phase changes (is decremented in a wraparound manner) and the
    pending count is reloaded with the init count value. Returns the mbarrier's phase parity (0 for even, 1 for odd) prior to the "arrive" operation.

    Args:
        mbarrier (shared_memory_descriptor): Barrier to be signalled.
        count (int): Count to arrive with. Defaults to 1.

    Returns:
        prior phase (int): phase of mbarrier, prior to "arrive" operation.
    """
    count = _unwrap_if_constexpr(count)
    handle = _semantic.builder.create_lds_barrier_arrive(mbarrier.handle, count)
    return ttgl.tensor(handle, ttgl.int32)
