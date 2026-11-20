"""Admission, tail, launch-boundary and buffer-reuse regressions.

Run this entire file in separate processes with TRITON_ALL_BLOCKS_PARALLEL=0/1.
All host references use logical elements; guard checks include storage holes.
"""
import pytest
import random
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _batched_channel(X, Y, SIDE, Q: tl.constexpr, VALID: tl.constexpr, XS: tl.constexpr, YS: tl.constexpr,
                     CASE: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, 2 * Q)
    value = (tl.load(X + (row * 2 * Q + i) * XS, i < VALID, other=0) + 1).to(X.dtype.element_ty)
    tl.store(SIDE + row * 2 * Q + i, value)
    a, b = tl.split(tl.reshape(value, (Q, 2)))
    j = tl.arange(0, Q)
    if CASE == 0:
        tl.store(Y + (row * 2 * Q + j) * YS, a)
    elif CASE == 1:
        tl.store(Y + (row * 2 * Q + j) * YS, b)
    else:
        joined = tl.reshape(tl.join(b, a), (2 * Q, ))
        tl.store(Y + (row * 2 * Q + i) * YS, joined)


@pytest.mark.parametrize('batch', [1, 39, 40, 41, 2047, 2049])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('case', [0, 1, 2])
def test_launch_boundaries_and_single_result(batch, dtype, case):
    q, valid, xs, ys = 16, 31, 3, 2
    n = batch * 2 * q
    cpu = (torch.arange(n * xs + 2, dtype=torch.int32).remainder(97) - 48).to(dtype)
    x = cpu.npu()[1:]
    storage = torch.full((n * ys + 2, ), -111, dtype=dtype, device='npu')
    side = torch.empty(n, dtype=dtype, device='npu')
    _batched_channel[(batch, )](x, storage[1:], side, Q=q, VALID=valid, XS=xs, YS=ys, CASE=case)
    ref = cpu[1:1 + n * xs:xs].reshape(batch, 2 * q).clone()
    ref[:, valid:] = 0
    ref += 1
    assert torch.equal(side.cpu(), ref.reshape(-1))
    expected = torch.full((batch, 2 * q), -111, dtype=dtype)
    if case < 2:
        expected[:, :q] = ref[:, case::2]
    else:
        expected = ref.reshape(batch, q, 2).flip(-1).reshape(batch, 2 * q)
    assert torch.equal(storage.cpu()[1:1 + n * ys:ys], expected.reshape(-1))
    guard = torch.ones(storage.numel(), dtype=torch.bool)
    guard[1:1 + n * ys:ys] = False
    assert torch.all(storage.cpu()[guard] == -111)


@triton.jit
def _join_admission(X, Z, Y, SIDE, Q: tl.constexpr, ROUTE: tl.constexpr, XS: tl.constexpr, YS: tl.constexpr):
    i = tl.arange(0, Q)
    if ROUTE == 'mask':
        a = tl.load(X + i * XS, i < Q - 1, other=0)
        b = tl.load(Z + i, i < Q - 3, other=0)
    else:
        a = tl.load(X + i * XS)
        b = tl.load(Z + i)
    if ROUTE == 'computed':
        a = (a + 1).to(X.dtype.element_ty)
        b = (b - 1).to(Z.dtype.element_ty)
    if ROUTE == 'shared':
        tl.store(SIDE + i, a)
    y = tl.reshape(tl.join(a, b), (2 * Q, ))
    tl.store(Y + tl.arange(0, 2 * Q) * YS, y)


@pytest.mark.parametrize('policy', ['auto', 'native', 'packed'])
@pytest.mark.parametrize('route', ['dense', 'computed', 'shared', 'mask', 'xstride', 'ystride'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_dense_native_admission(policy, route, dtype):
    q = 32
    xs, ys = (3 if route == 'xstride' else 1), (2 if route == 'ystride' else 1)
    cpu = torch.arange(q * xs + 2, dtype=torch.int32).remainder(37).to(dtype)
    zcpu = torch.arange(q, dtype=torch.int32).flip(0).remainder(29).to(dtype)
    x, z = cpu.npu()[1:], zcpu.npu()
    storage = torch.full((2 * q * ys + 2, ), -111, dtype=dtype, device='npu')
    side = torch.full((q, ), -111, dtype=dtype, device='npu')
    kernel = _join_admission[(1, )](x, z, storage[1:], side, Q=q, ROUTE=route, XS=xs, YS=ys,
                                    structured_join_lowering=policy)
    a, b = cpu[1:1 + q * xs:xs].clone(), zcpu.clone()
    if route == 'mask':
        a[-1:] = 0
        b[-3:] = 0
    elif route == 'computed':
        a += 1
        b -= 1
    assert torch.equal(storage.cpu()[1:1 + 2 * q * ys:ys], torch.stack((a, b), -1).reshape(-1))
    assert torch.equal(side.cpu(), a if route == 'shared' else torch.full_like(a, -111))
    guard = torch.ones(storage.numel(), dtype=torch.bool)
    guard[1:1 + 2 * q * ys:ys] = False
    assert torch.all(storage.cpu()[guard] == -111)
    assert kernel.metadata.uses_structured_permutation == (route != 'dense' or policy != 'native')


@pytest.mark.parametrize('q', [1, 2, 4, 8, 16, 1024, 2048, 4096])
@pytest.mark.parametrize('dtype', [torch.float32, torch.int8])
def test_join_resource_and_block_boundaries(q, dtype, record_property):
    cpu = torch.arange(q + 2, dtype=torch.int32).remainder(71).to(dtype)
    zcpu = torch.arange(q, dtype=torch.int32).remainder(53).to(dtype)
    x, z = cpu.npu()[1:], zcpu.npu()
    out = torch.full((2 * q + 2, ), -111, dtype=dtype, device='npu')
    side = torch.full((q, ), -111, dtype=dtype, device='npu')
    route = 'dense' if q > 2048 else 'computed'
    try:
        kernel = _join_admission[(1, )](x, z, out[1:], side, Q=q, ROUTE=route, XS=1, YS=1)
    except Exception as error:
        # The retained native FP32 route currently needs 224 KiB > 192 KiB.
        # Reject only this known oversized case, explicitly at compile time.
        assert q == 4096 and dtype == torch.float32
        assert type(error).__name__ == 'MLIRCompilationError', type(error)
        assert 'ub overflow, requires' in str(error) and 'bits available' in str(error)
        record_property('budget_outcome', 'explicit_compile_time_ub_overflow')
        return
    expected = torch.stack((cpu[1:1 + q], zcpu), -1).reshape(-1) if route == 'dense' else torch.stack(
        (cpu[1:1 + q] + 1, zcpu - 1), -1).reshape(-1)
    assert torch.equal(out.cpu()[1:-1], expected)
    assert out.cpu()[[0, -1]].tolist() == [-111, -111]
    if q > 2048:
        assert not kernel.metadata.uses_structured_permutation


@pytest.mark.parametrize('valid', [0, 1, 7, 8, 9, 15, 16, 17, 31, 32])
def test_masked_logical_byte_boundaries(valid):
    q = 16
    cpu = torch.arange(2 * q, dtype=torch.int32).to(torch.int8)
    x = cpu.npu()
    out = torch.full((2 * q, ), -111, dtype=torch.int8, device='npu')
    side = torch.empty_like(out)
    _batched_channel[(1, )](x, out, side, Q=q, VALID=valid, XS=1, YS=1, CASE=2)
    ref = cpu.clone()
    ref[valid:] = 0
    ref += 1
    assert torch.equal(out.cpu(), ref.reshape(q, 2).flip(-1).reshape(-1))
    assert torch.equal(side.cpu(), ref)


def test_computed_large_join_budget_contract(record_property):
    # CANN 9.0 currently needs 224 KiB for this combination on 192 KiB UB.
    # A future compiler may tile it successfully. Both legal outcomes are
    # checked: exact outputs, or an explicit compile-time resource diagnostic.
    q = 4096
    cpu = torch.arange(q, dtype=torch.int32).remainder(71).float()
    x, z = cpu.npu(), cpu.flip(0).npu()
    out, side = torch.empty(2 * q, device='npu'), torch.empty(q, device='npu')
    try:
        _join_admission[(1, )](x, z, out, side, Q=q, ROUTE='computed', XS=1, YS=1)
    except Exception as error:
        assert type(error).__name__ == 'MLIRCompilationError', type(error)
        assert 'ub overflow, requires' in str(error) and 'bits available' in str(error)
        record_property('budget_outcome', 'explicit_compile_time_ub_overflow')
    else:
        torch.npu.synchronize()
        expected = torch.stack((cpu + 1, cpu.flip(0) - 1), -1).reshape(-1)
        assert torch.equal(out.cpu(), expected)
        record_property('budget_outcome', 'compiled_and_bit_exact')


def test_dispatch_change_in_same_process(monkeypatch):
    batch, q = 41, 16
    cpu = torch.arange(batch * 2 * q, dtype=torch.int32).remainder(97).float()
    x = cpu.npu()
    y, side = [torch.full_like(x, -111) for _ in range(2)]
    kernels = []
    for mode in [0, 1, 0]:
        monkeypatch.setenv('TRITON_ALL_BLOCKS_PARALLEL', str(mode))
        y.fill_(-111)
        kernel = _batched_channel[(batch, )](x, y, side, Q=q, VALID=2 * q, XS=1, YS=1, CASE=2)
        expected = (cpu + 1).reshape(batch, q, 2).flip(-1).reshape(-1)
        assert torch.equal(y.cpu(), expected)
        assert kernel.metadata.enable_auto_blockify is bool(mode)
        kernels.append(kernel)
    assert kernels[0].hash != kernels[1].hash
    assert kernels[0].hash == kernels[2].hash


def _graph_cases():
    rng = random.Random(20261027)
    cases = []
    while len(cases) < 16:
        rank = rng.randrange(2, 6)
        shape = [1 << rng.randrange(3) for _ in range(rank)]
        split_axis = rng.randrange(rank)
        shape[split_axis] = 2
        order = [i for i in range(rank) if i != split_axis]
        rng.shuffle(order)
        case = (tuple(shape), tuple(order + [split_axis]), bool(len(cases) % 2))
        if case not in cases:
            cases.append(case)
    return cases


@triton.jit
def _generated_graph(X, CHANNELS, TRANSPOSED, JOINED, N: tl.constexpr, SHAPE: tl.constexpr, ORDER: tl.constexpr,
                     SWAP: tl.constexpr):
    i = tl.arange(0, N)
    v = (tl.load(X + i) + 1).to(X.dtype.element_ty)
    t = tl.permute(tl.reshape(v, SHAPE), ORDER)
    tl.store(TRANSPOSED + i, tl.reshape(t, (N, )))
    a, b = tl.split(t)
    j = tl.arange(0, N // 2)
    tl.store(CHANNELS + j, tl.reshape(a, (N // 2, )))
    tl.store(CHANNELS + N // 2 + j, tl.reshape(b, (N // 2, )))
    joined = tl.join(b, a) if SWAP else tl.join(a, b)
    tl.store(JOINED + i, tl.reshape(joined, (N, )))


@pytest.mark.parametrize('shape,order,swap', _graph_cases())
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_seeded_rank2_to_rank5_graphs(shape, order, swap, dtype):
    n = 1
    for d in shape:
        n *= d
    cpu = torch.arange(n, dtype=torch.int32).to(dtype)
    x = cpu.npu()
    buffers = [torch.full((n + 2, ), -111, dtype=dtype, device='npu') for _ in range(3)]
    _generated_graph[(1, )](x, *(b[1:] for b in buffers), N=n, SHAPE=shape, ORDER=order, SWAP=swap)
    t = (cpu + 1).reshape(shape).permute(order).contiguous()
    a, b = t[..., 0], t[..., 1]
    expected = [
        torch.cat((a.reshape(-1), b.reshape(-1))),
        t.reshape(-1),
        torch.stack((b, a) if swap else (a, b), -1).reshape(-1)
    ]
    for storage, ref in zip(buffers, expected):
        assert torch.equal(storage.cpu()[1:-1], ref)
        assert storage.cpu()[[0, -1]].tolist() == [-111, -111]
