"""Numerical regressions for masked batched inputs and dot accumulators.

Requires CANN fixes for batched ND2NZ initialization and batch matmul tiling.
The integer / 8 operands make the CPU float64 reference exact in FP32.
"""
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _batched_dot(A, B, Seed, C, Side, PG: tl.constexpr, KA: tl.constexpr, KB: tl.constexpr, PAD: tl.constexpr,
                 MODE: tl.constexpr, PREC: tl.constexpr):
    batch = tl.arange(0, PG)[:, None, None]
    row = tl.arange(0, 32)[None, :, None]
    k = tl.arange(0, 16)[None, None, :]
    kk = tl.arange(0, 16)[None, :, None]
    col = tl.arange(0, 64)[None, None, :]
    a = tl.load(A + batch * 2048 + row * 64 + k, k < KA, PAD)
    b = tl.load(B + batch * KB * 64 + kk * 64 + col, kk < KB, PAD)
    out = batch * 2048 + row * 64 + col
    if MODE == 'seeded':
        initial = tl.load(Seed + out)
        y = tl.dot(a, b, acc=initial, input_precision=PREC)
    else:
        y = tl.dot(a, b, input_precision=PREC)
    if MODE == 'live':
        tl.store(Side + out, y)
    if MODE != 'single':
        d = tl.load(A + batch * 2048 + row * 64 + 16 + k, k < KA, PAD)
        if MODE == 'sum':
            y = y + tl.dot(d, b, input_precision=PREC)
        else:
            y = tl.dot(d, b, acc=y, input_precision=PREC)
    if MODE == 'three':
        e = tl.load(A + batch * 2048 + row * 64 + 32 + k, k < KA, PAD)
        y = tl.dot(e, b, acc=y, input_precision=PREC)
    tl.store(C + out, y)


def _check(pg, ka, kb, pad, mode, dtype, precision, auto):
    a = ((torch.arange(pg * 2048).remainder(31) - 15).float() / 8).reshape(pg, 32, 64).to(dtype)
    b = ((torch.arange(pg * kb * 64).remainder(23) - 11).float() / 8).reshape(pg, kb, 64).to(dtype)
    seed = ((torch.arange(pg * 2048).remainder(7) - 3).float() / 8).reshape(pg, 32, 64)
    source, weight, initial = a.npu(), b.npu(), seed.npu()
    storage = torch.full((pg * 2048 + 128, ), 31415.25, device='npu')
    side_storage = storage.clone()
    output, side = storage[64:-64], side_storage[64:-64]
    options = dict(num_warps=4, num_stages=1, multibuffer=False, enable_auto_blockify=auto, enable_fp_fusion=False)
    for scale in (1, -1):
        source.copy_((a * scale).npu())
        output.fill_(float('nan'))
        side.fill_(float('nan'))
        _batched_dot[(1, )](source, weight, initial, output, side, pg, ka, kb, pad, mode, precision, **options)
        torch.npu.synchronize()
        right = torch.full((pg, 16, 64), pad, dtype=torch.float64)
        right[:, :kb] = b.double()
        count = 1 if mode == 'single' else 3 if mode == 'three' else 2
        expected = seed.double() if mode == 'seeded' else torch.zeros_like(seed, dtype=torch.float64)
        first = None
        for i in range(count):
            left = torch.full((pg, 32, 16), pad, dtype=torch.float64)
            left[:, :, :ka] = a[:, :, i * 16:i * 16 + ka].double() * scale
            term = left @ right
            if i == 0:
                first = term
            expected += term
        actual = output.cpu().reshape_as(expected).double()
        assert torch.equal(actual,
                           expected), (pg, ka, kb, mode, dtype, precision, auto, (actual - expected).abs().max().item())
        if mode == 'live':
            assert torch.equal(side.cpu().reshape_as(first).double(), first)
        for backing in (storage, side_storage):
            assert torch.all(backing[:64].cpu() == 31415.25)
            assert torch.all(backing[-64:].cpu() == 31415.25)
    assert torch.equal(source.cpu(), -a)
    assert torch.equal(weight.cpu(), b)


_TYPES = [(torch.float32, 'ieee'), (torch.float32, 'hf32'), (torch.float16, 'ieee'), (torch.bfloat16, 'ieee')]


@pytest.mark.parametrize('auto', [False, True])
@pytest.mark.parametrize('pg', [1, 8])
@pytest.mark.parametrize('dtype,precision', _TYPES)
@pytest.mark.parametrize('ka,kb,pad', [(16, 16, 0.), (16, 15, 0.), (15, 16, 0.)])
def test_masked_batched_dot(pg, ka, kb, pad, dtype, precision, auto):
    _check(pg, ka, kb, pad, 'single', dtype, precision, auto)


@pytest.mark.parametrize('auto', [False, True])
@pytest.mark.parametrize('pg', [1, 8])
@pytest.mark.parametrize('dtype,precision', _TYPES)
@pytest.mark.parametrize('mode', ['acc', 'sum', 'three', 'seeded', 'live'])
def test_batched_dot_accumulator(pg, mode, dtype, precision, auto):
    _check(pg, 16, 16, 0., mode, dtype, precision, auto)
