# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Iluvatar remapping of TLE ``nv_mma_shared_layout=True``.

``True`` never materializes NVIDIA ``#ttg.nvmma_shared``. Eligible 2D TCU
tiles become ``#ttg.swizzled_shared`` with ``useTcu=true`` and
``vec=512/bitwidth``. Everything else uses the generic swizzled encoding.

The default is row-major. A col-major global operand needs a buffer of
matching order, which the stride-agnostic ``alloc`` defaults cannot infer, so
pass ``make_tcu_swizzled_layout(..., col_major=True)`` as ``alloc(layout=...)``
for those.
"""

import triton.language as tl

from .. import types as tle

_TCU_DTYPES = (tl.int8, tl.float16, tl.bfloat16, tl.float32)
_SME_SEGMENT_BYTES = 64


class IluvatarTcuSwizzledSharedLayout(tle.swizzled_shared_layout):
    """A swizzled shared layout consumed by the Iluvatar TCU.

    ``use_tcu`` is a class attribute rather than a field, so the generic layout
    reconstructions in ``types.py`` (slot, permute) keep the TCU encoding just
    by rebuilding through ``type(...)``.
    """

    use_tcu = True

    def to_ir(self, builder):
        return builder.make_swizzled_shared_encoding_attr(
            self.vectorSize,
            self.perPhase,
            self.maxPhase,
            self.order,
            self.numCTAsPerCGA,
            self.numCTASplit,
            self.numCTAOrder,
            True,
        )


def _unwrap(value):
    return value.value if isinstance(value, tl.constexpr) else value


def is_tcu_eligible(shape, dtype, col_major=False) -> bool:
    shape = [_unwrap(dim) for dim in shape]
    dtype = _unwrap(dtype)
    if len(shape) < 2 or dtype not in _TCU_DTYPES:
        return False
    try:
        contig = int(shape[-2] if col_major else shape[-1])
        bitwidth = int(dtype.primitive_bitwidth)
    except (TypeError, ValueError):
        return False
    if bitwidth not in (8, 16, 32) or contig <= 0:
        return False
    return contig * bitwidth // 8 >= _SME_SEGMENT_BYTES


def make_tcu_swizzled_layout(shape, dtype, col_major=False) -> IluvatarTcuSwizzledSharedLayout:
    # The TCU layout only differs from the default row-major one in vectorSize.
    layout = IluvatarTcuSwizzledSharedLayout.make_default(rank=len(shape))
    layout.vectorSize = 512 // int(_unwrap(dtype).primitive_bitwidth)
    if col_major:
        order = list(layout.order)
        order[0], order[1] = order[1], order[0]
        layout.order = order
    return layout


def select_default_smem_layout(builder, shape, dtype, nv_mma_shared_layout):
    """Pick the default Iluvatar SMEM encoding when ``layout`` is None."""
    if bool(_unwrap(nv_mma_shared_layout)) and is_tcu_eligible(shape, dtype):
        layout = make_tcu_swizzled_layout(shape, dtype)
    else:
        layout = tle.swizzled_shared_layout.make_default(rank=len(shape))
    return layout, layout.to_ir(builder)
