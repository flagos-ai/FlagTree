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

from ..._core import _unwrap_if_constexpr, builtin
from ..hopper.mbarrier import (
    MBarrierLayout,
    allocate_mbarrier,
    expect,
    fence_init_release_cluster,
    init,
    invalidate,
    wait,
)

__all__ = [
    "allocate_mbarrier",
    "arrive",
    "expect",
    "fence_init_release_cluster",
    "init",
    "invalidate",
    "MBarrierLayout",
    "wait",
]


@builtin
def arrive(mbarrier, *, count=1, cta_mask=0, pred=True, _semantic=None):
    """
    Arrive at an mbarrier with a specified count.

    When ``cta_mask`` is non-zero, the arrive is multicast across the cluster.
    Each bit set in the mask identifies a CTA ID dimension to multicast
    along. CTA IDs ``a`` and ``b`` belong to the same equivalence class iff
    ``a & ~cta_mask == b & ~cta_mask``; all CTAs in a class multicast to each
    other. Multicast requires ``num_ctas > 1``, ``0 < cta_mask <= num_ctas - 1``,
    and the barrier must have the identity CGA layout ``[[1], [2], ...]``. The
    default value of ``cta_mask`` is 0 (no multicast).

    Args:
        mbarrier (shared_memory_descriptor): Barrier to be signalled.
        count (int): Count to arrive with. Defaults to 1.
        cta_mask (int): CTA broadcast dimension bits (see above). Defaults
            to 0 (no multicast). Must satisfy ``0 < cta_mask <= num_ctas - 1``
            when non-zero.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
    """
    count = _unwrap_if_constexpr(count)
    cta_mask = _unwrap_if_constexpr(cta_mask)
    if not isinstance(cta_mask, int) or isinstance(cta_mask, bool):
        raise TypeError(f"cta_mask must be an int, got {type(cta_mask).__name__}")
    if cta_mask:
        num_ctas = _semantic.builder.options.num_ctas
        if cta_mask < 0:
            raise ValueError(f"cta_mask must be positive, got {cta_mask}")
        if cta_mask > num_ctas - 1:
            raise ValueError(f"cta_mask must be <= num_ctas - 1 ({num_ctas - 1}), got {cta_mask}")
    pred = _semantic.to_tensor(pred)
    _semantic.builder.create_mbarrier_arrive(mbarrier.handle, count, cta_mask, pred.handle)
