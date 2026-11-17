"""Equal-length row/column indices must keep independent explicit domains."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _square(X, Y, O, L: tl.constexpr, B: tl.constexpr, SAME_BASE: tl.constexpr, TAG_ROW: tl.constexpr):
    m = tle.gpu.set_layout(tl.arange(0, B), tle.gpu.SlicedEncoding(1, L))
    n = tle.gpu.set_layout(tl.arange(0, B), tle.gpu.SlicedEncoding(0, L))
    a = tl.load(X + m)
    if SAME_BASE:
        b = tl.load(X + n)
    else:
        b = tl.load(Y + n)
    if TAG_ROW:
        a = 2 * a
    tl.store(O + m[:, None] * B + n[None, :], a[:, None] + b[None, :])


@pytest.mark.parametrize("same_base", [False, True])
def test_shared_cse_leaves_do_not_override_square_row_or_column(tmp_path, same_base):
    layout = tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [16, 1], [32, 128, 128])
    k = compile_musa(
        _square,
        {"X": "*fp32", "Y": "*fp32", "O": "*fp32", "L": "constexpr", "B": "constexpr",
         "SAME_BASE": "constexpr", "TAG_ROW": "constexpr"},
        {"L": layout, "B": 128, "SAME_BASE": same_base, "TAG_ROW": False},
        options={"num_warps": 16},
    )
    assert "musa_tle.set_layout" not in k.asm["ttgir"]
    assert "dim = 0" in k.asm["ttgir"] and "dim = 1" in k.asm["ttgir"]
    assert k.asm["mubin"]


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA device required")
@pytest.mark.parametrize("same_base", [False, True])
def test_shared_cse_leaves_runtime(same_base):
    layout = tle.gpu.mthreads.MusaSqmmaEncoding([3, 1], [16, 1], [32, 128, 128])
    x = torch.arange(128, dtype=torch.float32) * 16
    y = torch.arange(128, dtype=torch.float32) * -3
    expected = 2 * x[:, None] + (x if same_base else y)[None, :]
    dx, dy = x.to("musa"), y.to("musa")
    output = torch.empty((128, 128), dtype=torch.float32, device="musa")
    # Asymmetric weights make mistaken row/column selections visible, even
    # when the shared-pointer path lets the producer be CSE'd.
    for _ in range(3):
        output.fill_(float("nan"))
        _square[(1, )](dx, dy, output, layout, 128, same_base, True, num_warps=16)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
