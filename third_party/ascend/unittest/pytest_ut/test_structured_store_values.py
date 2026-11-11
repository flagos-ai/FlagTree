"""Device regression for grouped outputs, small-loop dispatch, and GM offsets."""
import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _outputs(X, Y, T, J, Q: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, 4 * Q)
    v = tl.load(X + row * 4 * Q + i)
    t = tl.trans(tl.reshape(v, (4, Q)))
    tl.store(T + row * 4 * Q + i, tl.reshape(t, (4 * Q, )))
    lo, hi = tl.split(tl.reshape(t, (Q, 2, 2)))
    a, b = tl.split(lo)
    c, d = tl.split(hi)
    k = tl.arange(0, Q)
    tl.store(Y + row * 4 * Q + k, a)
    tl.store(Y + row * 4 * Q + Q + k, b)
    tl.store(Y + row * 4 * Q + 2 * Q + k, c)
    tl.store(Y + row * 4 * Q + 3 * Q + k, d)
    joined = tl.join(tl.join(a, b), tl.join(c, d))
    tl.store(J + row * 4 * Q + i, tl.reshape(joined, (4 * Q, )))


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('offset', [0, 1, 4, 7])
@pytest.mark.parametrize('batch', [1, 41, 81])
def test_grouped_store_guards_and_short_loops(dtype, offset, batch):
    q = 32
    cpu = torch.arange(batch * 4 * q, dtype=torch.int32).remainder(97).to(dtype)
    x = cpu.npu()
    reference = cpu.reshape(batch, 4, q)
    expected_y = reference[:, [0, 2, 1, 3], :].reshape(-1)
    expected_t = reference.transpose(1, 2).contiguous().reshape(-1)
    size = cpu.numel()
    # Keep a full 32-byte guard on both ends while varying the GM address mod 32.
    start = 32 + offset
    storages = [torch.full((start + size + 32, ), -111, dtype=dtype, device='npu') for _ in range(3)]
    views = [storage[start:start + size] for storage in storages]
    for enabled in [False, True]:
        for storage in storages:
            storage.fill_(-111)
        _outputs[(batch, )](x, *views, Q=q, structured_address_hoisting=enabled,
                            structured_store_sync_optimization=enabled)
        torch.npu.synchronize()
        for storage, expected in zip(storages, [expected_y, expected_t, expected_t]):
            actual = storage.cpu()
            assert torch.equal(actual[start:start + size], expected)
            assert torch.all(actual[:start] == -111)
            assert torch.all(actual[start + size:] == -111)
