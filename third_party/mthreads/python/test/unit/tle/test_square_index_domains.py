"""Equal-length row/column indices must keep independent explicit domains."""

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _square(X, Y, O, L: tl.constexpr, B: tl.constexpr, SAME_BASE: tl.constexpr):
    m = tle.gpu.set_layout(tl.arange(0, B), tle.gpu.SlicedEncoding(1, L))
    n = tle.gpu.set_layout(tl.arange(0, B), tle.gpu.SlicedEncoding(0, L))
    a = tl.load(X + m)
    if SAME_BASE:
        b = tl.load(X + n)
    else:
        b = tl.load(Y + n)
    tl.store(O + m[:, None] * B + n[None, :], a[:, None] + b[None, :])


@pytest.mark.parametrize("same_base", [False, True])
def test_shared_cse_leaves_do_not_override_square_row_or_column(tmp_path, same_base):
    layout = tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [16, 1], [32, 128, 128])
    k = compile_musa(
        _square,
        {"X": "*fp32", "Y": "*fp32", "O": "*fp32", "L": "constexpr", "B": "constexpr", "SAME_BASE": "constexpr"},
        {"L": layout, "B": 128, "SAME_BASE": same_base},
        options={"num_warps": 16},
    )
    assert "musa_tle.set_layout" not in k.asm["ttgir"]
    assert "dim = 0" in k.asm["ttgir"] and "dim = 1" in k.asm["ttgir"]
    assert k.asm["mubin"]
