"""Matrix writeback must preserve holes, masked tails and side consumers."""
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _writeback(A, B, Out, Side, VALID, ROW: tl.constexpr, COL: tl.constexpr, M: tl.constexpr, SIDE: tl.constexpr):
    batch = tl.program_id(0)
    row, k, col = tl.arange(0, M), tl.arange(0, 16), tl.arange(0, 16)
    a = tl.load(A + batch * M * 16 + row[:, None] * 16 + k[None, :])
    b = tl.load(B + k[:, None] * 16 + col[None, :])
    product = tl.dot(a, b, input_precision='ieee')
    if SIDE:
        tl.store(Side + batch * M * 16 + row[:, None] * 16 + col[None, :], product)
    offset = batch * M * 512 + row[:, None] * ROW + col[None, :] * COL + 15
    tl.store(Out + offset, product, row[:, None] < VALID)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('auto', [False, True])
@pytest.mark.parametrize('side', [False, True])
@pytest.mark.parametrize('strides', [(256, 16), (511, 3), (16, 1), (1, 32)])
def test_dot_store_holes(strides, side, auto, dtype):
    m, batch, guard, sentinel = 32, 3, 128, 31415.25
    torch.manual_seed(101101)
    a, b = torch.randn(batch, m, 16).to(dtype), torch.randn(16, 16).to(dtype)
    da, db = a.npu(), b.npu()
    allocation = torch.full((batch * m * 512 + 2 * guard, ), sentinel, dtype=dtype, device='npu')
    out = allocation[guard:-guard]
    side_allocation = torch.full((batch * m * 16 + 2 * guard, ), sentinel, dtype=dtype, device='npu')
    side_out = side_allocation[guard:-guard]
    row_stride, col_stride = strides
    indices = (torch.arange(batch)[:, None, None] * m * 512 + torch.arange(m)[None, :, None] * row_stride +
               torch.arange(16)[None, None, :] * col_stride + 15)
    live = torch.zeros(out.numel(), dtype=torch.bool)
    compiled = None
    for repeat, valid in enumerate((m - 3, m, 1, 0)):
        changed = a * (2.0**(repeat - 1))
        da.copy_(changed)
        allocation.fill_(sentinel)
        side_allocation.fill_(sentinel)
        expected = (changed.double() @ b.double()).to(dtype)
        if repeat == 0:
            compiled = _writeback[(batch, )](da, db, out, side_out, valid, row_stride, col_stride, m, side, num_warps=4,
                                             num_stages=1, multibuffer=False, enable_auto_blockify=auto,
                                             enable_fp_fusion=False)
        else:
            _writeback[(batch, )](da, db, out, side_out, valid, row_stride, col_stride, m, side, num_warps=4,
                                  num_stages=1, multibuffer=False, enable_auto_blockify=auto, enable_fp_fusion=False)
        torch.npu.synchronize()
        wanted = indices[:, :valid].flatten()
        live.zero_()
        live[wanted] = True
        got = out.cpu()
        assert torch.equal(got[~live], torch.full_like(got[~live], sentinel))
        torch.testing.assert_close(got[wanted], expected[:, :valid].flatten(),
                                   rtol=2e-5 if dtype == torch.float32 else 1e-2,
                                   atol=2e-5 if dtype == torch.float32 else 1e-2)
        if side:
            torch.testing.assert_close(side_out.cpu(), expected.flatten(),
                                       rtol=2e-5 if dtype == torch.float32 else 1e-2,
                                       atol=2e-5 if dtype == torch.float32 else 1e-2)
        for buf in (allocation, side_allocation):
            assert (buf[:guard].cpu() == sentinel).all()
            assert (buf[-guard:].cpu() == sentinel).all()
        assert torch.equal(da.cpu(), changed)
        assert torch.equal(db.cpu(), b)
    # The dense side output must retain its original Cube path.
    assert 'linalg.matmul' in compiled.asm['ttadapter']
