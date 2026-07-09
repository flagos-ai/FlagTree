import pytest
import torch

import triton
import triton.language as tl

from triton.language.extra import libdevice
from triton.language.extra.libdevice import fast_dividef as my_fast_dividef


def test_libdevice_rename(device):
    # mark the import as used by this test
    _ = my_fast_dividef

    @triton.jit
    def triton_copy(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        data = tl.load(in_ptr + offsets)
        tl.store(out_ptr + offsets, data)

    BLOCK_SIZE = 256
    inp = torch.randn(BLOCK_SIZE, device=device)
    out = torch.empty_like(inp)

    triton_copy[(1, )](inp, out, BLOCK_SIZE)


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
def test_isinf(device, dtype_str):

    @triton.jit
    def triton_isinf(in_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        in_tile = tl.load(in_ptr + offsets, mask=mask)
        if in_ptr.dtype.element_ty == tl.float32:
            out_tile = libdevice.finitef(in_tile)
        else:
            out_tile = libdevice.isfinited(in_tile)
        tl.store(out_ptr + offsets, out_tile, mask=mask)

    x = torch.tensor(
        [float(1), -float(1),
         float(0), -float(0),
         float("inf"), -float("inf"),
         float("nan"), -float("nan")], device=device, dtype=getattr(torch, dtype_str))
    res = torch.tensor([True, True, True, True, False, False, False, False])
    numel = x.numel()
    y = torch.empty_like(x, dtype=torch.bool)
    BLOCK_SIZE = 256
    triton_isinf[(triton.cdiv(numel, BLOCK_SIZE), )](x, y, numel, BLOCK_SIZE)
    assert torch.equal(y.cpu(), res)
