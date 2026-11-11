"""Cube results must reach Vector permutations through a live GM/UB junction."""
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def _split_phase(A, B, Phase, Out, Side, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, SIDE: tl.constexpr):
    batch = tl.program_id(0)
    m, k, n = tl.arange(0, M), tl.arange(0, K), tl.arange(0, N)
    a = tl.load(A + batch * M * K + m[:, None] * K + k[None, :])
    b = tl.load(B + batch * K * N + k[:, None] * N + n[None, :])
    product = tl.dot(a, b, input_precision='ieee')
    if SIDE:
        tl.store(Side + batch * M * N + m[:, None] * N + n[None, :], product)
    first = al.extract_slice(product, [0, 0], [M // 2, N], [1, 1])
    second = al.extract_slice(product, [M // 2, 0], [M // 2, N], [1, 1])
    cr, ci = tl.split(tl.reshape(first, (M * N // 4, 2)))
    sr, si = tl.split(tl.reshape(second, (M * N // 4, 2)))
    real = tl.reshape(cr - si, (M // 2, N // 2))
    imag = tl.reshape(sr + ci, (M // 2, N // 2))
    row, col = tl.arange(0, M // 2), tl.arange(0, N // 2)
    offset = row[:, None] * (N // 2) + col[None, :]
    wr = tl.load(Phase + offset)
    wi = tl.load(Phase + M * N // 4 + offset)
    real, imag = real * wr - imag * wi, real * wi + imag * wr
    dst = col[:, None] * (M // 2) + row[None, :]
    tl.store(Out + batch * M * N // 2 + dst, tl.trans(real))
    tl.store(Out + batch * M * N // 2 + M * N // 4 + dst, tl.trans(imag))


@triton.jit
def _join_accumulator(A, B, C, Out, Side, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, CHUNK: tl.constexpr,
                      SIDE: tl.constexpr):
    batch = tl.program_id(0)
    m, k, n = tl.arange(0, M), tl.arange(0, K), tl.arange(0, N)
    a = tl.load(A + batch * M * K + m[:, None] * K + k[None, :])
    b = tl.load(B + batch * K * N + k[:, None] * N + n[None, :])
    c = tl.load(C + batch * M * K + m[:, None] * K + k[None, :])
    product = tl.dot(a, b, input_precision='ieee')
    product = tl.dot(c, b, acc=product, input_precision='ieee')
    if SIDE:
        tl.store(Side + batch * M * N + m[:, None] * N + n[None, :], product)
    real = tl.reshape(al.extract_slice(product, [0, 0], [M // 2, N], [1, 1]), (M * N // 2, ))
    imag = tl.reshape(al.extract_slice(product, [M // 2, 0], [M // 2, N], [1, 1]), (M * N // 2, ))
    for start in tl.static_range(0, M * N // 2, CHUNK):
        r = al.extract_slice(real, [start], [CHUNK], [1])
        i = al.extract_slice(imag, [start], [CHUNK], [1])
        packed = tl.reshape(tl.join(r, i), (CHUNK // N, 2 * N))
        row = start // N + tl.arange(0, CHUNK // N)
        word = tl.arange(0, 2 * N)
        dst = row[:, None] * (2 * N) + word[None, :]
        tl.store(Out + batch * M * N + dst, packed)


@triton.jit
def _roundtrip(A, B, C, Out):
    m, k, n = tl.arange(0, 32), tl.arange(0, 16), tl.arange(0, 64)
    a = tl.load(A + m[:, None] * 16 + k[None, :])
    b = tl.load(B + k[:, None] * 64 + n[None, :])
    product = tl.dot(a, b, input_precision='ieee')
    even, odd = tl.split(tl.reshape(product, (1024, 2)))
    selected = tl.reshape(even, (32, 32))
    q = tl.arange(0, 32)
    c = tl.load(C + q[:, None] * 16 + k[None, :])
    result = tl.dot(selected, c, input_precision='ieee')
    tl.store(Out + m[:, None] * 16 + k[None, :], result)


def _guarded(count):
    guard = torch.full((count + 128, ), 31415.25, device='npu')
    out = guard[64:-64]
    out.fill_(float('nan'))
    return guard, out


def _check(guard, out, expected):
    torch.npu.synchronize()
    got = out.cpu().double()
    assert torch.isfinite(got).all()
    assert (guard[:64].cpu() == 31415.25).all()
    assert (guard[-64:].cpu() == 31415.25).all()
    error = torch.linalg.vector_norm(got - expected.double().flatten()) / torch.linalg.vector_norm(expected.double())
    assert error < 1e-6, float(error)


@pytest.mark.parametrize('auto', [True, False])
@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('side', [False, True])
@pytest.mark.parametrize('shape', [(32, 16, 64), (64, 32, 128)])
def test_dot_slices_split_phase(shape, side, batch, auto):
    m, k, n = shape
    torch.manual_seed(172009)
    a, b = torch.randn(batch, m, k), torch.randn(batch, k, n)
    phase = torch.randn(2, m // 2, n // 2)
    product = (a.double() @ b.double()).float()
    first, second = product[:, :m // 2], product[:, m // 2:]
    real = first[..., 0::2] - second[..., 1::2]
    imag = second[..., 0::2] + first[..., 1::2]
    real, imag = real * phase[0] - imag * phase[1], real * phase[1] + imag * phase[0]
    expected = torch.stack((real.transpose(1, 2), imag.transpose(1, 2)), dim=1)
    ga, out = _guarded(batch * m * n // 2)
    gs, side_out = _guarded(batch * m * n)
    compiled = _split_phase[(batch, )](a.npu(), b.npu(), phase.npu(), out, side_out, m, k, n, side, num_warps=4,
                                       num_stages=1, multibuffer=False, enable_auto_blockify=auto,
                                       enable_fp_fusion=False)
    assert 'flagtree.cube_to_vector' in compiled.asm['ttadapter']
    _check(ga, out, expected)
    if side:
        _check(gs, side_out, product)


@pytest.mark.parametrize('auto', [True, False])
@pytest.mark.parametrize('side', [False, True])
@pytest.mark.parametrize('shape', [(32, 16, 64, 512), (32, 16, 128, 1024)])
def test_dot_accumulator_join(shape, side, auto):
    m, k, n, chunk = shape
    torch.manual_seed(172010)
    a, b, c = torch.randn(2, m, k), torch.randn(2, k, n), torch.randn(2, m, k)
    product = a.double() @ b.double() + c.double() @ b.double()
    expected = torch.stack((product[:, :m // 2], product[:, m // 2:]), dim=-1)
    guard, out = _guarded(2 * m * n)
    side_guard, side_out = _guarded(2 * m * n)
    compiled = _join_accumulator[(2, )](a.npu(), b.npu(), c.npu(), out, side_out, m, k, n, chunk, side, num_warps=4,
                                        num_stages=1, multibuffer=False, enable_auto_blockify=auto,
                                        enable_fp_fusion=False)
    assert 'flagtree.cube_to_vector' in compiled.asm['ttadapter']
    _check(guard, out, expected)
    if side:
        _check(side_guard, side_out, product)


@pytest.mark.parametrize('auto', [True, False])
def test_cube_vector_cube(auto):
    torch.manual_seed(172011)
    a, b, c = torch.randn(32, 16), torch.randn(16, 64), torch.randn(32, 16)
    expected = (a.double() @ b.double())[:, ::2] @ c.double()
    guard, out = _guarded(32 * 16)
    compiled = _roundtrip[(1, )](a.npu(), b.npu(), c.npu(), out, num_warps=4, num_stages=1, multibuffer=False,
                                 enable_auto_blockify=auto, enable_fp_fusion=False)
    assert 'flagtree.cube_to_vector' in compiled.asm['ttadapter']
    assert 'flagtree.vector_to_cube' in compiled.asm['ttadapter']
    _check(guard, out, expected)
