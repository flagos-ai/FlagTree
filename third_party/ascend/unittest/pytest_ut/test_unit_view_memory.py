"""Unit-only pointer and mask views must retain memory geometry."""
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _copy(X, Y, N, B: tl.constexpr, VIEW: tl.constexpr, STEP: tl.constexpr, VIEW_MASK: tl.constexpr):
    q = tl.program_id(0) * B + tl.arange(0, B)
    off = q * STEP
    mask = q < N
    if VIEW == 1:
        off = off[None, :].reshape((B, ))
    elif VIEW == 2:
        off = off[:, None].reshape((B, ))
    elif VIEW == 3:
        off = off[None, :, None].reshape((B, ))
    if VIEW_MASK:
        mask = mask[None, :, None].reshape((B, ))
    if VIEW == 4:
        ptr = (X + off)[None, :].reshape((B, ))
    else:
        ptr = X + off
    x = tl.load(ptr, mask, 0.)
    tl.store(Y + q, x * 1.125 + 0.25, q < N)


@pytest.mark.parametrize('view', [1, 2, 3, 4])
@pytest.mark.parametrize('step', [1, 2])
@pytest.mark.parametrize('view_mask', [False, True])
def test_unit_pointer_and_mask_view(view, step, view_mask):
    torch.manual_seed(92813)
    n, block = 4079, 256
    host = torch.randn(n * step)
    src = host.npu()
    backing = torch.full((n + 128, ), 31415.25, device='npu')
    out = backing[64:-64]
    for size in (n, n - 17):
        backing.fill_(31415.25)
        k = _copy[(triton.cdiv(size, block), )](src, out, size, block, view, step, view_mask, num_warps=4, num_stages=1,
                                                multibuffer=False, enable_fp_fusion=False, enable_auto_blockify=True)
        torch.npu.synchronize()
        assert torch.equal(out[:size].cpu(), host[:size * step:step] * 1.125 + 0.25)
        assert (backing[:64].cpu() == 31415.25).all()
        assert (backing[64 + size:].cpu() == 31415.25).all()
        assert torch.equal(src.cpu(), host)
    # Successful values alone would also pass with the slow scalar fallback.
    assert 'DiscreteMemAccess' not in k.asm['ttadapter']
