"""Live dot prefixes, including vector consumers and overlapping outputs.

Requires the CANN batched accumulator correctness fix. A shared-loop optimization
may fire only when memory dependencies permit it; every case must remain exact.
"""
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _live_prefix(A, B, Out, PG: tl.constexpr, TERMS: tl.constexpr, VECTOR: tl.constexpr, STRIDE: tl.constexpr):
    batch = tl.program_id(0) * PG + tl.arange(0, PG)[:, None, None]
    row = tl.arange(0, 32)[None, :, None]
    k = tl.arange(0, 16)[None, None, :]
    col = tl.arange(0, 32)[None, None, :]
    a0 = tl.load(A + batch * 2048 + row * 64 + k)
    a1 = tl.load(A + batch * 2048 + row * 64 + 16 + k)
    b = tl.load(B + batch * 512 + tl.arange(0, 16)[None, :, None] * 32 + col)
    if TERMS == 4:
        a2 = tl.load(A + batch * 2048 + row * 64 + 32 + k)
        a3 = tl.load(A + batch * 2048 + row * 64 + 48 + k)
    y0 = tl.dot(a0, b, input_precision='ieee')
    y1 = tl.dot(a1, b, acc=y0, input_precision='ieee')
    if TERMS == 4:
        y2 = tl.dot(a2, b, acc=y1, input_precision='ieee')
        y3 = tl.dot(a3, b, acc=y2, input_precision='ieee')
    if VECTOR:
        y0 = y0 + 1.
        y1 = y1 + 2.
        if TERMS == 4:
            y2 = y2 + 3.
            y3 = y3 + 4.
    offset = batch * 1024 + row * 32 + col
    tl.store(Out + offset, y0)
    tl.store(Out + STRIDE + offset, y1)
    if TERMS == 4:
        tl.store(Out + 2 * STRIDE + offset, y2)
        tl.store(Out + 3 * STRIDE + offset, y3)


def _inputs(batch):
    a = ((torch.arange(batch * 2048).remainder(31) - 15).float() / 8).reshape(batch, 32, 64)
    b = ((torch.arange(batch * 512).remainder(23) - 11).float() / 8).reshape(batch, 16, 32)
    return a, b


@pytest.mark.parametrize('auto', [False, True])
@pytest.mark.parametrize('pg', [1, 4])
@pytest.mark.parametrize('terms,vector,overlap', [(2, False, False), (2, True, False), (4, True, False),
                                                  (2, True, True)])
def test_live_batched_prefixes(auto, pg, terms, vector, overlap):
    a, b = _inputs(pg)
    source, weight = a.npu(), b.npu()
    stride = 0 if overlap else pg * 1024
    size = pg * 1024 if overlap else terms * pg * 1024
    storage = torch.full((size + 128, ), 31415.25, device='npu')
    out = storage[64:-64]
    options = dict(num_warps=4, num_stages=1, multibuffer=False, enable_auto_blockify=auto, enable_fp_fusion=False)
    for scale in (1, -1):
        source.copy_((a * scale).npu())
        out.fill_(float('nan'))
        _live_prefix[(1, )](source, weight, out, pg, terms, vector, stride, **options)
        torch.npu.synchronize()
        expected = torch.zeros(size, dtype=torch.float64)
        value = torch.zeros((pg, 32, 32), dtype=torch.float64)
        for i in range(terms):
            value += a[:, :, i * 16:(i + 1) * 16].double() * scale @ b.double()
            result = value + i + 1 if vector else value
            expected[i * stride:i * stride + pg * 1024] = result.flatten()
        assert torch.equal(out.cpu().double(), expected)
        assert torch.all(storage[:64].cpu() == 31415.25)
        assert torch.all(storage[-64:].cpu() == 31415.25)
    assert torch.equal(source.cpu(), -a)
    assert torch.equal(weight.cpu(), b)
