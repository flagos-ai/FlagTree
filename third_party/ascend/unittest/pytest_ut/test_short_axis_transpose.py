"""Exact-value regressions for permutations retaining a sub-block final axis."""
import math

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
import triton
import triton.language as tl


@triton.jit
def _permutation(X, Y, N: tl.constexpr, SHAPE: tl.constexpr, ORDER: tl.constexpr, COMPUTED_JOIN: tl.constexpr):
    program = tl.program_id(0)
    i = tl.arange(0, N)
    raw = tl.load(X + program * N + i)
    if COMPUTED_JOIN:
        a, b = tl.split(tl.reshape(raw, (N // 2, 2)))
        value = tl.reshape(tl.join(a + 1, b + 1), SHAPE)
    else:
        value = tl.reshape(raw + 1, SHAPE)
    value = tl.permute(value, ORDER)
    tl.store(Y + program * N + i, tl.reshape(value, (N, )))


@pytest.mark.parametrize('shape,order', [
    ((16, 4, 2), (1, 0, 2)),
    ((8, 8, 2), (1, 0, 2)),
    ((4, 2, 8, 2), (2, 1, 0, 3)),
    ((2, 4, 8, 2, 1), (2, 0, 1, 3, 4)),
])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.int16, torch.int32])
@pytest.mark.parametrize('computed_join', [False, True])
def test_unchanged_short_final_axis(shape, order, dtype, computed_join):
    n, batch, guard = math.prod(shape), 41, 32
    cpu = torch.arange(batch * n, dtype=torch.int32).remainder(97).to(dtype)
    source = cpu.npu()
    storage = torch.full((batch * n + 2 * guard, ), -111, dtype=dtype, device='npu')
    output = storage[guard:-guard]
    _permutation[(batch, )](source, output, n, shape, order, computed_join, num_warps=8, num_stages=1,
                            multibuffer=False)
    torch.npu.synchronize()
    expected = (cpu + 1).reshape(batch, *shape).permute(0, *[i + 1 for i in order]).contiguous().reshape(-1)
    got = storage.cpu()
    assert torch.equal(got[guard:-guard], expected)
    assert torch.all(got[:guard] == -111) and torch.all(got[-guard:] == -111)


@triton.jit
def _matrix_transpose(X, Y, JOIN: tl.constexpr, OFFSET: tl.constexpr):
    base = tl.program_id(0) * 4
    r = tl.arange(0, 16)[:, None]
    c = base * 2 + tl.arange(0, 8)[None, :]
    raw = tl.load(X + (r + OFFSET) * 442 + c, (r < 11) & (c < 442), other=0.)
    if JOIN:
        a, b = tl.split(tl.reshape(raw, (64, 2)))
        value = tl.reshape(tl.join(a + 1., b + 1.), (16, 4, 2))
    else:
        value = tl.reshape(raw + 1., (16, 4, 2))
    value = tl.reshape(tl.permute(value, 1, 0, 2), (4, 32))
    rr = base + tl.arange(0, 4)[:, None]
    cc = tl.arange(0, 32)[None, :]
    tl.store(Y + rr * 418 + cc + 2 * OFFSET, value, (rr < 221) & (cc < 22))


@pytest.mark.parametrize('join', [False, True])
@pytest.mark.parametrize('offset', [0, 1, 7])
@pytest.mark.parametrize('late', [False, True])
def test_masked_matrix_transport(join, offset, late):
    cpu = torch.arange(209 * 221 * 2, dtype=torch.float32).reshape(209, 221, 2)
    source = cpu.npu()
    output = torch.full((221, 209, 2), -17., device='npu')
    _matrix_transpose[(triton.cdiv(221,
                                   4), )](source, output, join, offset, num_warps=8, num_stages=1, multibuffer=False,
                                          structured_address_hoisting=late, structured_store_sync_optimization=late)
    torch.npu.synchronize()
    expected = torch.full_like(cpu, -17.)
    expected[offset:offset + 11] = cpu[offset:offset + 11] + 1.
    assert torch.equal(output.cpu(), expected.transpose(0, 1).contiguous())
