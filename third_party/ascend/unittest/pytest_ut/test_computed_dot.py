"""Vector values must reach Cube through valid GM/L1 buffers."""
import math

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
import triton
import triton.language as tl
from triton.backends.ascend.compiler import get_auto_bind_sub_block_option
import triton.language.extra.cann.extension as al


@triton.jit
def _rounded(x):
    bits = x.to(tl.int32, bitcast=True)
    return ((bits + 4095 + ((bits >> 13) & 1)) & -8192).to(tl.float32, bitcast=True)


@triton.jit
def _compensated(X, T, W, Y, Side, K: tl.constexpr, N: tl.constexpr, R: tl.constexpr, LEFT: tl.constexpr,
                 TWIDDLE: tl.constexpr, MATERIALIZE: tl.constexpr):
    batch = tl.program_id(1)
    r = tl.arange(0, K)[:, None]
    c = tl.arange(0, N)[None, :]
    offset = r * N + c
    xr = tl.load(X + batch * 2 * K * N + offset, r < R, 0.)
    xi = tl.load(X + batch * 2 * K * N + K * N + offset, r < R, 0.)
    if TWIDDLE:
        wr = tl.load(T + offset)
        wi = tl.load(T + K * N + offset)
        xr, xi = xr * wr - xi * wi, xr * wi + xi * wr
    hi_r = _rounded(xr)
    lo_r = _rounded(xr - hi_r)
    hi_i = _rounded(xi)
    lo_i = _rounded(xi - hi_i)
    # A live Vector consumer must keep its UB value, independent of L1 copies.
    tl.store(Side + batch * K * N + offset, hi_r)
    if MATERIALIZE:
        hi_r, lo_r = hi_r + 0., lo_r + 0.
        hi_i, lo_i = hi_i + 0., lo_i + 0.
    row = tl.arange(0, 2 * K)
    k = tl.arange(0, K)
    if LEFT:
        hr, lr = tl.trans(hi_r), tl.trans(lo_r)
        hi, li = tl.trans(hi_i), tl.trans(lo_i)
        a = tl.load(W + k[:, None] + row[None, :] * 2 * K)
        b = tl.load(W + 4 * K * K + k[:, None] + row[None, :] * 2 * K)
        y = tl.dot(hr, a, input_precision='hf32')
        y += tl.dot(lr, a, input_precision='hf32')
        y += tl.dot(hr, b, input_precision='hf32')
        a = tl.load(W + K + k[:, None] + row[None, :] * 2 * K)
        b = tl.load(W + 4 * K * K + K + k[:, None] + row[None, :] * 2 * K)
        v = tl.dot(hi, a, input_precision='hf32')
        v += tl.dot(li, a, input_precision='hf32')
        v += tl.dot(hi, b, input_precision='hf32')
        y += v
        yr = tl.trans(al.extract_slice(y, [0, 0], [N, K], [1, 1]))
        yi = tl.trans(al.extract_slice(y, [0, K], [N, K], [1, 1]))
    else:
        a = tl.load(W + row[:, None] * 2 * K + k[None, :])
        b = tl.load(W + 4 * K * K + row[:, None] * 2 * K + k[None, :])
        y = tl.dot(a, hi_r, input_precision='hf32')
        y += tl.dot(a, lo_r, input_precision='hf32')
        y += tl.dot(b, hi_r, input_precision='hf32')
        a = tl.load(W + row[:, None] * 2 * K + K + k[None, :])
        b = tl.load(W + 4 * K * K + row[:, None] * 2 * K + K + k[None, :])
        v = tl.dot(a, hi_i, input_precision='hf32')
        v += tl.dot(a, lo_i, input_precision='hf32')
        v += tl.dot(b, hi_i, input_precision='hf32')
        y += v
        yr = al.extract_slice(y, [0, 0], [K, N], [1, 1])
        yi = al.extract_slice(y, [K, 0], [K, N], [1, 1])
    tl.store(Y + batch * 2 * K * N + offset, yr, r < R)
    tl.store(Y + batch * 2 * K * N + K * N + offset, yi, r < R)


def _cpu_round(x):
    bits = x.contiguous().view(torch.int32)
    return ((bits + 4095 + ((bits >> 13) & 1)) & -8192).view(torch.float32)


@pytest.mark.parametrize('radix,n,twiddle,materialize', [(15, 256, True, False), (17, 128, False, False),
                                                         (8, 64, True, False), (15, 128, True, True)])
@pytest.mark.parametrize('left', [False, True])
@pytest.mark.parametrize('auto', [False, True])
def test_computed_compensated_dot(radix, n, twiddle, materialize, left, auto):
    torch.manual_seed(20429)
    k, batch, guard = max(16, triton.next_power_of_2(radix)), 41, 32
    x = torch.randn(batch, 2, k, n)
    x[:, :, radix:] = 0
    phase = torch.randn(2, k, n)
    real, imag = x[:, 0], x[:, 1]
    if twiddle:
        real, imag = real * phase[0] - imag * phase[1], real * phase[1] + imag * phase[0]
    weight = torch.randn(2, radix, radix) / math.sqrt(radix)
    block = torch.zeros(2 * k, 2 * k)
    block[:radix, :radix] = weight[0]
    block[:radix, k:k + radix] = -weight[1]
    block[k:k + radix, :radix] = weight[1]
    block[k:k + radix, k:k + radix] = weight[0]
    high = _cpu_round(block)
    weights = torch.stack((high, _cpu_round(block - high))).npu()
    expected = torch.einsum('ok,bkn->bon', block.double(), torch.cat((real, imag), dim=1).double())
    size = x.numel()
    storage = torch.full((size + 2 * guard, ), -111., device='npu')
    output = storage[guard:-guard]
    side_storage = torch.full((batch * k * n + 2 * guard, ), -111., device='npu')
    side = side_storage[guard:-guard]
    source, tw = x.npu(), phase.npu()
    options = dict(num_warps=4, num_stages=1, multibuffer=False, enable_auto_blockify=auto, enable_fp_fusion=False)

    def launch():
        return _compensated[(1, batch)](source, tw, weights, output, side, k, n, radix, left, twiddle, materialize,
                                        **options)

    def verify(scale):
        torch.npu.synchronize()
        got = output.cpu().reshape(batch, 2, k, n)
        want = expected.reshape(batch, 2, k, n) * scale
        actual = got[:, :, :radix].double()
        reference = want[:, :, :radix]
        error = torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference)
        assert error < 5e-7, error
        assert torch.all(got[:, :, radix:] == -111.)
        values = side.cpu().reshape(batch, k, n)
        # Recompute the scaled Vector result with the same FP32 arithmetic.
        raw = x * scale
        rr, ii = raw[:, 0], raw[:, 1]
        if twiddle:
            rr = rr * phase[0] - ii * phase[1]
        assert torch.equal(values.view(torch.int32), _cpu_round(rr).view(torch.int32))
        assert torch.all(storage[:guard].cpu() == -111.) and torch.all(storage[-guard:].cpu() == -111.)
        assert torch.all(side_storage[:guard].cpu() == -111.) and torch.all(side_storage[-guard:].cpu() == -111.)

    compiled = launch()
    verify(1.)
    assert str(compiled.asm['ttadapter']).count('memref_ext.alloc_workspace') == 4
    assert compiled.metadata.uses_computed_dot_transfer
    assert get_auto_bind_sub_block_option(compiled.metadata._asdict()) is auto
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch()
    for scale in (1e-6, 1e6, 1.):
        source.copy_((x * scale).npu())
        storage.fill_(-111.)
        side_storage.fill_(-111.)
        graph.replay()
        verify(scale)
    assert torch.equal(source.cpu(), x)


@triton.jit
def _both_inputs(A, B, C):
    r = tl.arange(0, 32)[:, None]
    k = tl.arange(0, 16)[None, :]
    a = _rounded(tl.load(A + r * 16 + k) * 1.125)
    b = _rounded(tl.load(B + tl.trans(k) * 32 + tl.trans(r)) * 0.875)
    value = tl.dot(a, b, input_precision='hf32')
    tl.store(C + r * 32 + tl.arange(0, 32)[None, :], value)


@pytest.mark.parametrize('auto', [False, True])
def test_both_dot_inputs_computed(auto):
    torch.manual_seed(1429)
    a, b = torch.randn(32, 16), torch.randn(16, 32)
    output = torch.empty((32, 32), device='npu')
    compiled = _both_inputs[(1, )](a.npu(), b.npu(), output, num_warps=4, num_stages=1, multibuffer=False,
                                   enable_auto_blockify=auto, enable_auto_bind_sub_block=True, enable_fp_fusion=False)
    torch.npu.synchronize()
    expected = _cpu_round(a * 1.125).double() @ _cpu_round(b * 0.875).double()
    assert torch.linalg.vector_norm(output.cpu().double() - expected) / torch.linalg.vector_norm(expected) < 5e-7
    assert str(compiled.asm['ttadapter']).count('memref_ext.alloc_workspace') == 2


@triton.jit
def _loaded_inputs(A, B, C):
    r = tl.arange(0, 32)[:, None]
    k = tl.arange(0, 16)[None, :]
    a = tl.load(A + r * 16 + k)
    b = tl.load(B + tl.trans(k) * 32 + tl.trans(r))
    tl.store(C + r * 32 + tl.arange(0, 32)[None, :], tl.dot(a, b, input_precision='hf32'))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('auto', [False, True])
def test_loaded_dot_keeps_native_path(dtype, auto):
    # Exactly HF32-representable inputs also provide a strict FP32 oracle.
    a = (torch.arange(512).remainder(31).float().reshape(32, 16) / 8).to(dtype)
    b = (torch.arange(512).remainder(23).float().reshape(16, 32) / 8).to(dtype)
    output = torch.empty((32, 32), device='npu')
    compiled = _loaded_inputs[(1, )](a.npu(), b.npu(), output, num_warps=4, num_stages=1, multibuffer=False,
                                     enable_auto_blockify=auto, enable_fp_fusion=False)
    torch.npu.synchronize()
    assert torch.equal(output.cpu().double(), a.double() @ b.double())
    assert 'memref_ext.alloc_workspace' not in str(compiled.asm['ttadapter'])
    assert not compiled.metadata.uses_computed_dot_transfer


@triton.jit
def _batched_input(A, B, C):
    batch = tl.arange(0, 2)[:, None, None]
    r = tl.arange(0, 32)[None, :, None]
    k = tl.arange(0, 16)[None, None, :]
    a = _rounded(tl.load(A + batch * 512 + r * 16 + k) * 1.125)
    b = tl.load(B + batch * 512 + tl.arange(0, 16)[None, :, None] * 32 + tl.arange(0, 32)[None, None, :])
    value = tl.dot(a, b, input_precision='hf32')
    tl.store(C + batch * 1024 + r * 32 + tl.arange(0, 32)[None, None, :], value)


@pytest.mark.parametrize('auto', [False, True])
def test_batched_computed_dot(auto):
    torch.manual_seed(2429)
    a = torch.randn(2, 32, 16)
    b = _cpu_round(torch.randn(2, 16, 32))
    output = torch.empty((2, 32, 32), device='npu')
    compiled = _batched_input[(1, )](a.npu(), b.npu(), output, num_warps=4, num_stages=1, multibuffer=False,
                                     enable_auto_blockify=auto, enable_fp_fusion=False)
    torch.npu.synchronize()
    expected = _cpu_round(a * 1.125).double() @ b.double()
    assert torch.linalg.vector_norm(output.cpu().double() - expected) / torch.linalg.vector_norm(expected) < 5e-7
    assert str(compiled.asm['ttadapter']).count('memref_ext.alloc_workspace') == 1


@triton.jit
def _split_dot(X, W, Y, Side, K: tl.constexpr, N: tl.constexpr, LEFT: tl.constexpr, MATERIALIZE: tl.constexpr,
               PRECISION: tl.constexpr):
    batch = tl.program_id(0)
    r = tl.arange(0, K)[:, None]
    c = tl.arange(0, 2 * N)[None, :]
    raw = tl.load(X + batch * 2 * K * N + r * 2 * N + c)
    first, second = tl.split(tl.reshape(raw, (K * N, 2)))
    first, second = tl.reshape(first, (K, N)), tl.reshape(second, (K, N))
    if MATERIALIZE:
        first, second = first + 0., second + 0.
    # A live Vector consumer must retain the original UB value.
    offset = r * N + tl.arange(0, N)[None, :]
    tl.store(Side + batch * K * N + offset, first)
    row = tl.arange(0, 2 * K)
    k = tl.arange(0, K)
    if LEFT:
        a = tl.load(W + row[None, :] * 2 * K + k[:, None])
        b = tl.load(W + row[None, :] * 2 * K + K + k[:, None])
        first, second = tl.trans(first), tl.trans(second)
        value = tl.dot(first, a, input_precision=PRECISION)
        value = tl.dot(second, b, acc=value, input_precision=PRECISION)
        # The repeated SSA input must reuse its first transfer.
        value = tl.dot(first, a, acc=value, input_precision=PRECISION)
        value = tl.trans(value)
    else:
        a = tl.load(W + row[:, None] * 2 * K + k[None, :])
        b = tl.load(W + row[:, None] * 2 * K + K + k[None, :])
        value = tl.dot(a, first, input_precision=PRECISION)
        value = tl.dot(b, second, acc=value, input_precision=PRECISION)
        value = tl.dot(a, first, acc=value, input_precision=PRECISION)
    tl.store(Y + batch * 2 * K * N + row[:, None] * N + tl.arange(0, N)[None, :], value)


@pytest.mark.parametrize('k,n', [(16, 64), (32, 128)])
@pytest.mark.parametrize('left,materialize', [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize('precision', ['ieee', 'hf32'])
@pytest.mark.parametrize('auto', [False, True])
def test_split_dot_accumulation(k, n, left, materialize, precision, auto):
    torch.manual_seed(3429)
    batch, guard = 41, 32
    # Exactly representable values keep the IEEE and HF32 oracles identical.
    x = torch.randint(-31, 32, (batch, k, n, 2)).float() / 8
    w = torch.randint(-15, 16, (2 * k, 2 * k)).float() / 16
    source, weights = x.npu(), w.npu()
    storage = torch.full((batch * 2 * k * n + 2 * guard, ), -111., device='npu')
    side_storage = torch.full((batch * k * n + 2 * guard, ), -111., device='npu')
    output, side = storage[guard:-guard], side_storage[guard:-guard]
    expected = 2 * (w[:, :k].double() @ x[..., 0].double()) + w[:, k:].double() @ x[..., 1].double()
    options = dict(num_warps=4, num_stages=1, multibuffer=False, enable_auto_blockify=auto, enable_fp_fusion=False)

    def launch():
        return _split_dot[(batch, )](source, weights, output, side, k, n, left, materialize, precision, **options)

    def verify(scale):
        torch.npu.synchronize()
        assert torch.equal(output.cpu().reshape(batch, 2 * k, n).double(), expected * scale)
        assert torch.equal(side.cpu().reshape(batch, k, n), x[..., 0] * scale)
        for buffer in (storage, side_storage):
            assert torch.all(buffer[:guard].cpu() == -111.)
            assert torch.all(buffer[-guard:].cpu() == -111.)

    compiled = launch()
    verify(1.)
    assert str(compiled.asm['ttadapter']).count('memref_ext.alloc_workspace') == 2
    assert compiled.metadata.uses_computed_dot_transfer
    assert get_auto_bind_sub_block_option(compiled.metadata._asdict()) is auto
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch()
    for scale in (2.**-20, 2.**20, 1.):
        source.copy_((x * scale).npu())
        storage.fill_(-111.)
        side_storage.fill_(-111.)
        graph.replay()
        verify(scale)
    assert torch.equal(source.cpu(), x)
