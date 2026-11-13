"""Allocator and native-to-linear conversion must agree on no-op layouts."""

import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _scale(x, out, L: tl.constexpr):
    row = tle.gpu.set_layout(tl.arange(0, 64), tle.gpu.SlicedEncoding(1, L))
    col = tle.gpu.set_layout(tl.arange(0, 128), tle.gpu.SlicedEncoding(0, L))
    scale = tl.load(x + col)
    value = tle.gpu.set_layout(tl.full((64, 128), 1.0, tl.float32), L)
    value = value * scale[None, :]
    tl.store(out + row[:, None] * 128 + col[None, :], value)


def test_pointer_carrier_identity_does_not_reserve_scratch():
    k = compile_musa(
        _scale,
        {"x": "*fp32", "out": "*fp32", "L": "constexpr"},
        {"L": tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [8, 1], [32, 128, 128])},
        options={"num_warps": 8},
    )
    assert k.metadata.shared == 0
