import numpy as np
import pytest
import torch

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError


@triton.jit
def _copy_packed(X, Y, N: tl.constexpr, XI: tl.constexpr, XS: tl.constexpr, YI: tl.constexpr, YS: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    value = tl.load(X + XI + i * XS, i < N, other=0)
    tl.store(Y + YI + i * YS, value, i < N)


@triton.jit
def _copy_packed_words(X, Y, N: tl.constexpr):
    # Only complete, aligned words use int32 pointers. Copy the tail as bytes.
    _copy_packed(X.to(tl.pointer_type(tl.int32)), Y.to(tl.pointer_type(tl.int32)), N // 4, 0, 1, 0, 1)
    i = N // 4 * 4 + tl.program_id(0) * 256 + tl.arange(0, 256)
    value = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, value, i < N)


@triton.jit
def _unpack_storage(X, Y, WORD: tl.constexpr):
    count: tl.constexpr = 64 if WORD else 256
    slots: tl.constexpr = 8 if WORD else 2
    i, j = tl.arange(0, count), tl.arange(0, slots)
    value = tl.load(X + i).to(tl.uint32)
    codes = (value[:, None] >> (4 * j[None, :])) & 15
    tl.store(Y + i[:, None] * slots + j[None, :], codes)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.float4_e2m1fn_x2])
def test_metax_fp4_host_device_storage(dtype, metax_device):
    raw = torch.arange(4096, dtype=torch.int64).remainder(256).to(torch.uint8)
    source = raw.view(dtype)
    device = source.to(metax_device)
    copied = device.clone()
    assert copied.data_ptr() != device.data_ptr()
    assert torch.equal(device.cpu().view(torch.uint8), raw)
    assert torch.equal(copied.cpu().view(torch.uint8), raw)


@pytest.mark.parametrize("mode", ["uint8", "int32", "x2_view", "x2_reinterpret"])
@pytest.mark.parametrize("n,xi,xs,yi,ys", [
    (1024, 32, 1, 32, 1),
    (1024, 33, 1, 35, 1),
    (513, 33, 2, 34, 3),
    (513, 544, -1, 32, 1),
    (259, 32, 1, 35, 1),
])
def test_metax_fp4_packed_copy(mode, n, xi, xs, yi, ys, metax_device):
    # Offsets, strides and masks count containers, not logical FP4 values.
    dtype = torch.int32 if mode == "int32" else torch.uint8
    elem_bytes = dtype.itemsize
    bits = torch.arange(2048 * elem_bytes, dtype=torch.int64).remainder(256).to(torch.uint8)
    x = bits.view(dtype).to(metax_device)
    raw = torch.full_like(x.view(torch.uint8), 165)
    y = raw.view(dtype)
    if mode == "x2_view":
        x, y = x.view(torch.float4_e2m1fn_x2).view(torch.uint8), y.view(torch.float4_e2m1fn_x2).view(torch.uint8)
    elif mode == "x2_reinterpret":
        x = triton.reinterpret(x.view(torch.float4_e2m1fn_x2), tl.uint8)
        y = triton.reinterpret(y.view(torch.float4_e2m1fn_x2), tl.uint8)
    kernel = _copy_packed[(triton.cdiv(n, 256), )](x, y, n, xi, xs, yi, ys)
    expected = torch.full_like(bits, 165).view(dtype)
    i = torch.arange(n)
    expected[yi + i * ys] = bits.view(dtype)[xi + i * xs]
    assert torch.equal(raw.cpu(), expected.view(torch.uint8))
    assert "tt.fp_to_fp" not in kernel.asm["ttir"]


@pytest.mark.parametrize("n", [0, 1, 3, 4, 130, 259, 1024])
def test_metax_fp4_packed_word_tail(n, metax_device):
    source = torch.arange(1088, dtype=torch.int64).remainder(256).to(torch.uint8)
    x = source.to(metax_device)
    raw = torch.full_like(x, 165)
    kernel = _copy_packed_words[(max(1, triton.cdiv(n, 1024)), )](x[32:], raw[32:], n)
    expected = torch.full_like(source, 165)
    expected[32:32 + n] = source[32:32 + n]
    assert torch.equal(raw.cpu(), expected)
    # 130 bytes can hold 259 logical values. Copying preserves the spare high
    # nibble; it does not silently replace it with an encoding-time zero pad.
    if n == 130:
        assert int(raw.cpu()[161]) >> 4 == int(source[161]) >> 4 != 0
    assert "tt.fp_to_fp" not in kernel.asm["ttir"]


@pytest.mark.parametrize("word", [False, True])
def test_metax_fp4_storage_nibble_order(word, metax_device):
    raw = torch.arange(256, dtype=torch.int64).to(torch.uint8)
    x = raw.view(torch.int32) if word else raw
    output = torch.full((576, ), 165, dtype=torch.uint8, device=metax_device)
    _unpack_storage[(1, )](x.to(metax_device), output[32:544], word)
    expected = torch.full((576, ), 165, dtype=torch.uint8)
    expected[32:544:2], expected[33:544:2] = raw & 15, raw >> 4
    assert torch.equal(output.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32])
def test_metax_fp4_shared_gather(dtype, metax_device):
    from test_core import gather_test_kernel_1d

    i = torch.arange(1024 * dtype.itemsize, dtype=torch.int64)
    raw = (i * 73 + (i // 256) * 19).remainder(256).to(torch.uint8)
    source = raw.view(dtype)
    indices = (torch.arange(1024, dtype=torch.int32) * 73 + 17).remainder(1024)
    output = torch.full((1088 * dtype.itemsize, ), 165, dtype=torch.uint8, device=metax_device)
    kernel = gather_test_kernel_1d[(1, )](source.to(metax_device), indices.to(metax_device),
                                          output.view(dtype)[32:1056], 0, 1024, 1024, 1024, num_warps=4)
    expected = torch.full((1088 * dtype.itemsize, ), 165, dtype=torch.uint8)
    expected.view(dtype)[32:1056] = source[indices.long()]
    assert torch.equal(output.cpu(), expected)
    assert kernel.metadata.shared > 0
    assert "addrspace(3)" in kernel.asm["llir"]
    assert "tt.fp_to_fp" not in kernel.asm["ttir"]


@triton.jit
def _encode(X, Y, N: tl.constexpr, ROUND: tl.constexpr, AXIS: tl.constexpr = -1, BLOCK: tl.constexpr = 512):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    if ROUND == "default":
        packed = tl.fp_to_fp4(x, axis=AXIS)
    else:
        packed = tl.fp_to_fp4(x, axis=AXIS, fp_downcast_rounding=ROUND)
    j = tl.program_id(0) * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    tl.store(Y + j, packed, j < (N + 1) // 2)


@triton.jit
def _encode_2d(X, Y, AXIS: tl.constexpr, ROUND: tl.constexpr):
    i, j = tl.arange(0, 16), tl.arange(0, 32)
    x = tl.load(X + i[:, None] * 32 + j[None, :])
    y = tl.fp_to_fp4(x, axis=AXIS, fp_downcast_rounding=ROUND)
    r, c = tl.arange(0, 8 if AXIS == 0 else 16), tl.arange(0, 32 if AXIS == 0 else 16)
    tl.store(Y + r[:, None] * (32 if AXIS == 0 else 16) + c[None, :], y)


def _reference_codes(values, rounding):
    x = values.to(torch.float64).numpy()
    a = np.abs(x)
    boundaries = np.array([.5, 1., 1.5, 2., 3., 4., 6.] if rounding == "rtz" else [.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
    indices = np.searchsorted(boundaries, a, side="right" if rounding == "rtz" else "left")
    if rounding != "rtz":
        exact_ties = (indices < 7) & (a == boundaries[np.minimum(indices, 6)])
        indices += exact_ties & (indices % 2 == 1)
    indices = np.where(np.isnan(a), 7, indices)
    return torch.from_numpy((indices | (np.signbit(x).astype(np.uint8) << 3)).astype(np.uint8))


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32", "fp64"])
@pytest.mark.parametrize("rounding", ["default", "rtne", "rtz"])
def test_metax_fp4_encode(dtype, rounding, device):
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}[dtype]
    if dtype in ("fp16", "bf16"):
        values = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch_dtype)
    else:
        boundaries = torch.tensor([.25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 3.5, 4., 5., 6.], dtype=torch_dtype)
        boundaries = torch.cat((boundaries, -boundaries))
        lo = torch.nextafter(boundaries, torch.full_like(boundaries, -float("inf")))
        hi = torch.nextafter(boundaries, torch.full_like(boundaries, float("inf")))
        extra = torch.tensor([
            0., -0.,
            float("inf"), -float("inf"),
            float("nan"), -float("nan"),
            torch.finfo(torch_dtype).max, -torch.finfo(torch_dtype).max
        ], dtype=torch_dtype)
        rng = np.random.default_rng(20261026)
        raw = rng.integers(0, 2**32, size=4096, dtype=np.uint32)
        random = raw.view(np.float32) if dtype == "fp32" else rng.integers(0, 2**64, size=4096, dtype=np.uint64).view(
            np.float64)
        values = torch.cat((lo, boundaries, hi, extra, torch.from_numpy(random)))
    codes = _reference_codes(values, rounding)
    if codes.numel() % 2:
        codes = torch.cat((codes, torch.zeros(1, dtype=torch.uint8)))
    expected = codes[::2] | (codes[1::2] << 4)
    output = torch.full((expected.numel() + 64, ), 165, dtype=torch.uint8, device=device)
    _encode[(triton.cdiv(values.numel(), 512), )](values.to(device), output[32:-32], values.numel(), rounding)
    reference = torch.full((expected.numel() + 64, ), 165, dtype=torch.uint8)
    reference[32:-32] = expected
    assert torch.equal(output.cpu(), reference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("rounding", ["rtne", "rtz"])
def test_metax_fp4_encode_axis(dtype, axis, rounding, metax_device):
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=dtype)
    codes = torch.arange(512).remainder(16).reshape(16, 32)
    values = table[codes]
    shape = (8, 32) if axis == 0 else (16, 16)
    expected = torch.empty(shape, dtype=torch.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            a, b = ((2 * i, j), (2 * i + 1, j)) if axis == 0 else ((i, 2 * j), (i, 2 * j + 1))
            expected[i, j] = int(codes[a]) | (int(codes[b]) << 4)
    output = torch.full((320, ), 165, dtype=torch.uint8, device=metax_device)
    _encode_2d[(1, )](values.to(metax_device), output[32:288], axis, rounding)
    reference = torch.full((320, ), 165, dtype=torch.uint8)
    reference[32:288] = expected.flatten()
    assert torch.equal(output.cpu(), reference)


@triton.jit
def _scaled(X, S, Y, M: tl.constexpr, N: tl.constexpr, AXIS: tl.constexpr, GROUPED: tl.constexpr, DTYPE: tl.constexpr):
    BM: tl.constexpr = 16 if AXIS == 1 else 32
    BN: tl.constexpr = 32 if AXIS == 1 else 16
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.program_id(1) * BN + tl.arange(0, BN)
    x = tl.load(X + r[:, None] * N + c[None, :], (r[:, None] < M) & (c[None, :] < N), other=0)
    if GROUPED:
        if AXIS == 1:
            g = tl.program_id(1) * 2 + tl.arange(0, 2)
            s = tl.load(S + r[:, None] * tl.cdiv(N, 16) + g[None, :], (r[:, None] < M) & (g[None, :] < tl.cdiv(N, 16)),
                        other=127)
        else:
            g = tl.program_id(0) * 2 + tl.arange(0, 2)
            s = tl.load(S + g[:, None] * N + c[None, :], (g[:, None] < tl.cdiv(M, 16)) & (c[None, :] < N), other=127)
    else:
        s = tl.load(S + r[:, None] * N + c[None, :], (r[:, None] < M) & (c[None, :] < N), other=127)
    y = tl.fp4_to_fp_scaled(x, s, DTYPE, AXIS - 2)
    out_r = tl.program_id(0) * (BM * 2 if AXIS == 0 else BM) + tl.arange(0, BM * 2 if AXIS == 0 else BM)
    out_c = tl.program_id(1) * (BN * 2 if AXIS == 1 else BN) + tl.arange(0, BN * 2 if AXIS == 1 else BN)
    out_m: tl.constexpr = M * 2 if AXIS == 0 else M
    out_n: tl.constexpr = N * 2 if AXIS == 1 else N
    tl.store(Y + out_r[:, None] * out_n + out_c[None, :], y, (out_r[:, None] < out_m) & (out_c[None, :] < out_n))


@pytest.mark.parametrize("dtype", [tl.float16, tl.bfloat16])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("signed", [False, True])
def test_metax_fp4_scaled(dtype, axis, grouped, signed, metax_device):
    raw = torch.arange(259, dtype=torch.int64).remainder(256).to(torch.uint8).repeat(256, 1)
    width = 17 if grouped else 259
    scales = (torch.arange(256)[:, None] + 7 * torch.arange(width)[None, :]).remainder(256).to(torch.uint8)
    expanded = scales.repeat_interleave(16, 1)[:, :259] if grouped else scales
    codes = torch.stack((raw & 15, raw >> 4), dim=2).flatten(1, 2).long()
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                         dtype=torch.float64)
    factors = torch.from_numpy(np.ldexp(np.ones(expanded.shape), expanded.numpy().astype(np.int16) - 127))
    factors[expanded == 255] = float("nan")
    reference = table[codes] * factors.repeat_interleave(2, 1)
    if axis == 0:
        raw, scales, reference = raw.T.contiguous(), scales.T.contiguous(), reference.T.contiguous()
    out_dtype = torch.float16 if dtype == tl.float16 else torch.bfloat16
    expected = reference.to(out_dtype).contiguous().flatten()
    x, s = raw.to(metax_device), scales.to(metax_device)
    if signed:
        x, s = x.view(torch.int8), s.view(torch.int8)
    output = torch.full((expected.numel() + 64, ), 11., dtype=out_dtype, device=metax_device)
    m, n = raw.shape
    grid = (triton.cdiv(m, 16 if axis == 1 else 32), triton.cdiv(n, 32 if axis == 1 else 16))
    kernel = _scaled[grid](x, s, output[32:-32], m, n, axis, grouped, dtype)
    actual = output.cpu()
    guard = torch.full((32, ), 11., dtype=out_dtype).view(torch.uint8)
    assert torch.equal(actual[:32].view(torch.uint8), guard) and torch.equal(actual[-32:].view(torch.uint8), guard)
    actual = actual[32:-32]
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual[~nan].view(torch.uint8), expected[~nan].view(torch.uint8))
    assert "llvm.mxc.cvt.pk4.f4" not in kernel.asm["llir"]


def test_metax_fp4_public_exports():
    assert tl.fp_to_fp4 is tl.core.fp_to_fp4
    assert tl.fp4_to_fp_scaled is tl.core.fp4_to_fp_scaled


@triton.jit
def _scaled_documented(X, S, Y, DTYPE: tl.constexpr, RAW: tl.constexpr, WIDTH: tl.constexpr = 1,
                       AXIS: tl.constexpr = 1):
    r, c = tl.arange(0, 32), tl.arange(0, 16)
    x = tl.load(X + r[:, None] * 16 + c[None, :])
    s = tl.load(S + r[:, None] * WIDTH + tl.arange(0, WIDTH)[None, :])
    if RAW:
        y = tl.core.fp4_to_fp_scaled(x, s, DTYPE, AXIS)
    else:
        y = tl.fp4_to_fp_scaled(x, s, DTYPE, AXIS)
    j = tl.arange(0, 32)
    tl.store(Y + r[:, None] * 32 + j[None, :], y)


@pytest.mark.parametrize("dtype", [tl.float16, tl.bfloat16])
@pytest.mark.parametrize("raw_api", [False, True])
def test_metax_fp4_scaled_documented(dtype, raw_api, metax_device):
    x = torch.arange(512, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(32, 16)
    s = torch.arange(96, 128, dtype=torch.uint8)
    codes = torch.stack((x & 15, x >> 4), 2).flatten(1, 2).long()
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                         dtype=torch.float64)
    values = table[codes] * torch.from_numpy(np.ldexp(np.ones((32, 1)), s.numpy().astype(np.int16)[:, None] - 127))
    out_dtype = torch.float16 if dtype == tl.float16 else torch.bfloat16
    output = torch.full((1088, ), 11., dtype=out_dtype, device=metax_device)
    kernel = _scaled_documented[(1, )](x.to(metax_device), s.to(metax_device), output[32:1056], dtype, raw_api)
    expected = torch.full((1088, ), 11., dtype=out_dtype)
    expected[32:1056] = values.to(out_dtype).flatten()
    assert torch.equal(output.cpu().view(torch.uint8), expected.view(torch.uint8))
    assert "llvm.mxc.cvt.pk4.f4" not in kernel.asm["llir"]


@pytest.mark.parametrize("src,scale,dtype,width,axis,error", [
    ("fp16", "u8", tl.bfloat16, 1, 1, "src must have int8 or uint8 dtype"),
    ("u8", "fp32", tl.bfloat16, 1, 1, "scale must have int8 or uint8 dtype"),
    ("u8", "u8", tl.float32, 1, 1, "elem_type must be tl.float16 or tl.bfloat16"),
    ("u8", "u8", tl.bfloat16, 2, 1, "scale packing dimension must be"),
    ("u8", "u8", tl.bfloat16, 1, 2, "axis 2 must be"),
])
def test_metax_fp4_scaled_invalid(src, scale, dtype, width, axis, error):
    source = ASTSource(_scaled_documented, {"X": "*" + src, "S": "*" + scale, "Y": "*bf16"},
                       {"DTYPE": dtype, "RAW": False, "WIDTH": width, "AXIS": axis})
    with pytest.raises(CompilationError, match=error):
        triton.compile(source, target=GPUTarget("maca", 80, 64))


@pytest.mark.parametrize("dtype,rounding,axis,block,error", [
    ("i32", "rtne", -1, 512, "requires FP16/BF16/FP32/FP64"),
    ("fp32", "bad", -1, 512, "rounding must be rtne or rtz"),
    ("fp32", "rtne", 1, 512, "axis 1 must be"),
    ("fp32", "rtne", 0, 1, "packing dimension must be even"),
])
def test_metax_fp4_encode_invalid(dtype, rounding, axis, block, error):
    source = ASTSource(_encode, {"X": "*" + dtype, "Y": "*u8"},
                       {"N": 512, "ROUND": rounding, "AXIS": axis, "BLOCK": block})
    with pytest.raises(CompilationError, match=error):
        triton.compile(source, target=GPUTarget("maca", 80, 64))


@pytest.mark.parametrize("arch", [80, 86])
def test_metax_fp4_scaled_diagnostic(arch, fresh_triton_cache, monkeypatch, capfd):
    monkeypatch.setenv("MLIR_ENABLE_DIAGNOSTICS", "warnings")
    source = ASTSource(_scaled, {"X": "*u8", "S": "*u8", "Y": "*bf16"},
                       {"M": 32, "N": 16, "AXIS": 1, "GROUPED": True, "DTYPE": tl.bfloat16})
    kernel = triton.compile(source, target=GPUTarget("maca", arch, 64))
    assert "non-native scaled FP4 conversion on MetaX" in capfd.readouterr().err
    assert "llvm.mxc.cvt.pk4.f4" not in kernel.asm["llir"]


def test_metax_fp4_encode_diagnostic(fresh_triton_cache):
    source = ASTSource(_encode, {"X": "*fp32", "Y": "*u8"}, {"N": 512, "ROUND": "rtne", "AXIS": -1, "BLOCK": 512})
    with pytest.warns(UserWarning, match="non-native GPU software encoding"):
        kernel = triton.compile(source, target=GPUTarget("maca", 80, 64))
    assert "llvm.mxc.cvt.pk4.f32tof4" not in kernel.asm["llir"]


@pytest.mark.parametrize("rounding", ["rtne", "rtz"])
def test_metax_fp4_encode_odd_tail(rounding, metax_device):
    values = torch.arange(259, dtype=torch.float32).remainder(17) - 8
    values[-1] = -0.
    codes = _reference_codes(values, rounding)
    codes = torch.cat((codes, torch.zeros(1, dtype=torch.uint8)))
    reference = torch.full((194, ), 165, dtype=torch.uint8)
    reference[32:162] = codes[::2] | (codes[1::2] << 4)
    output = torch.full_like(reference, 165, device=metax_device)
    _encode[(1, )](values.to(metax_device), output[32:162], 259, rounding)
    assert torch.equal(output.cpu(), reference)
    # The final low nibble retains -0; the padded high nibble is +0.
    assert int(output.cpu()[161]) == 8


@triton.jit
def _dot_scaled(A, B, SA, SB, C, Y, A0: tl.constexpr, A1: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr,
                M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, STEP: tl.constexpr, FA: tl.constexpr,
                FB: tl.constexpr, PACK_A: tl.constexpr = True, PACK_B: tl.constexpr = True,
                ACC_TYPE: tl.constexpr = tl.float32, OUT_TYPE: tl.constexpr = tl.float32, BATCH: tl.constexpr = False,
                SCALE_WIDTH: tl.constexpr = 0):
    r, c = tl.arange(0, M), tl.arange(0, N)
    ka = tl.arange(0, STEP // 2 if FA == "e2m1" else STEP)
    kb = tl.arange(0, STEP // 2 if FB == "e2m1" else STEP)
    g = tl.arange(0, SCALE_WIDTH if SCALE_WIDTH else STEP // 32)
    acc = tl.load(C + r[:, None] * N + c[None, :]).to(ACC_TYPE)
    for start in range(0, K, STEP):
        a = tl.load(A + r[:, None] * A0 + (start // (2 if FA == "e2m1" else 1) + ka[None, :]) * A1,
                    start + ka[None, :] * (2 if FA == "e2m1" else 1) < K, other=0)
        b = tl.load(B + (start // (2 if FB == "e2m1" else 1) + kb[:, None]) * B0 + c[None, :] * B1,
                    start + kb[:, None] * (2 if FB == "e2m1" else 1) < K, other=0)
        if SA is not None:
            sa = tl.load(SA + r[:, None] * (K // 32) + start // 32 + g[None, :], start // 32 + g[None, :] < K // 32,
                         other=127.0)
        else:
            sa = None
        if SB is not None:
            sb = tl.load(SB + c[:, None] * (K // 32) + start // 32 + g[None, :], start // 32 + g[None, :] < K // 32,
                         other=127.0)
        else:
            sb = None
        if BATCH:
            a, b = a[None, :, :], b[None, :, :]
        acc = tl.dot_scaled(a, sa, FA, b, sb, FB, acc, lhs_k_pack=PACK_A, rhs_k_pack=PACK_B, out_dtype=OUT_TYPE)
    tl.store(Y + r[:, None] * N + c[None, :], acc)


def _dot_operand(fmt, rows, k, offset, seed=None):
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    if seed is None:
        codes = (torch.arange(rows * k).reshape(rows, k) * 7 + offset).remainder(16)
    else:
        codes = torch.from_numpy(np.random.default_rng(seed).integers(0, 16, size=(rows, k), dtype=np.int64))
    values = table[codes]
    if fmt == "e2m1":
        return (codes[:, ::2] | (codes[:, 1::2] << 4)).to(torch.uint8), values
    dtype = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2, "fp16": torch.float16, "bf16": torch.bfloat16}[fmt]
    return values.to(dtype), values


@pytest.mark.parametrize("fa,fb", [("e2m1", "e2m1")] + [(a, b)
                                                        for t in ("e4m3", "e5m2", "fp16", "bf16")
                                                        for a, b in (("e2m1", t), (t, "e2m1"))])
@pytest.mark.parametrize("scale_a,scale_b,k,step", [(True, True, 128, 64), (False, True, 64, 64), (True, False, 64, 64),
                                                    (False, False, 64, 64), (False, False, 16, 16)])
def test_metax_fp4_dot_scaled(fa, fb, scale_a, scale_b, k, step, metax_device):
    m, n = (16, 16) if k == 16 else (32, 16)
    a, av = _dot_operand(fa, m, k, 1)
    b, bv = _dot_operand(fb, n, k, 3)
    if scale_a:
        a = a.T.contiguous().T
    if not scale_b:
        b = b.T.contiguous().T
    sa = ((torch.arange(m * (k // 32)).reshape(m, -1) * 3).remainder(7) + 124).to(torch.uint8)
    sb = ((torch.arange(n * (k // 32)).reshape(n, -1) * 5).remainder(7) + 124).to(torch.uint8)
    compute_type = torch.float16 if "fp16" in (fa, fb) else torch.bfloat16
    av, bv = av.double(), bv.double()
    if scale_a:
        av *= torch.from_numpy(np.ldexp(np.ones(sa.shape), sa.numpy().astype(np.int16) - 127)).repeat_interleave(32, 1)
    if scale_b:
        bv *= torch.from_numpy(np.ldexp(np.ones(sb.shape), sb.numpy().astype(np.int16) - 127)).repeat_interleave(32, 1)
    av, bv = av.to(compute_type).double(), bv.to(compute_type).double()
    acc = torch.arange(m * n, dtype=torch.float32).reshape(m, n).remainder(13) / 8
    expected = av @ bv.T + acc.double()
    # FP32 accumulation error, relative to the sum of absolute products.
    bound = (k + 2) * 2**-24 * (av.abs() @ bv.abs().T + acc.double().abs())
    x, y = a.to(metax_device), b.T.to(metax_device)
    # Signed byte scales exercise identical E8M0 raw bits above code 127.
    s = sa.view(torch.int8).to(metax_device) if scale_a else None
    t = sb.to(metax_device) if scale_b else None
    out = torch.full((m * n + 64, ), 11., dtype=torch.float32, device=metax_device)
    kernel = _dot_scaled[(1, )](x, y, s, t, acc.to(metax_device), out[32:-32], *x.stride(), *y.stride(), m, n, k, step,
                                fa, fb, SCALE_WIDTH=max(1, step // 32), num_warps=4, num_stages=2)
    actual = out.cpu()
    assert torch.equal(actual[:32], torch.full((32, ), 11.))
    assert torch.equal(actual[-32:], torch.full((32, ), 11.))
    assert torch.all((actual[32:-32].reshape(m, n).double() - expected).abs() <= bound)
    assert "tt.dot_scaled" not in kernel.asm["ttgir"]
    assert "llvm.mxc.mma.f32." in kernel.asm["llir"]


@pytest.mark.parametrize("block", [(64, 128, 64, 8, 3), (64, 64, 64, 4, 2), (64, 128, 32, 4, 2), (64, 64, 64, 4, 4),
                                   (32, 32, 32, 8, 2), (32, 32, 32, 8, 3), (32, 32, 32, 8, 4)])
@pytest.mark.parametrize("k", [32, 96, 160])
@pytest.mark.parametrize("scenario", ["", "fullstage"])
@pytest.mark.parametrize("fa,fb", [("e2m1", "e2m1"), ("e2m1", "fp16"), ("e2m1", "e5m2")])
def test_metax_fp4_dot_scaled_pipeline(block, k, scenario, fa, fb, metax_device):
    m, n, step, warps, stages = block
    a, av = _dot_operand(fa, m, k, 1, seed=3101)
    b, bv = _dot_operand(fb, n, k, 3, seed=3102)
    sa = (torch.arange(m * (k // 32)).reshape(m, -1).remainder(5) + 125).to(torch.uint8)
    sb = (torch.arange(n * (k // 32)).reshape(n, -1).remainder(7) + 124).to(torch.uint8)
    av = av.double() * torch.from_numpy(np.ldexp(np.ones(sa.shape),
                                                 sa.numpy().astype(np.int16) - 127)).repeat_interleave(32, 1)
    bv = bv.double() * torch.from_numpy(np.ldexp(np.ones(sb.shape),
                                                 sb.numpy().astype(np.int16) - 127)).repeat_interleave(32, 1)
    compute_type = torch.float16 if "fp16" in (fa, fb) else torch.bfloat16
    av, bv = av.to(compute_type).double(), bv.to(compute_type).double()
    acc = torch.arange(m * n, dtype=torch.float32).reshape(m, n).remainder(13) / 8
    expected = av @ bv.T + acc.double()
    bound = (k + 2) * 2**-24 * (av.abs() @ bv.abs().T + acc.double().abs())
    x, y = a.to(metax_device), b.T.contiguous().to(metax_device)
    out = torch.full((m * n + 64, ), 11., dtype=torch.float32, device=metax_device)
    args = (x, y, sa.view(torch.int8).to(metax_device), sb.to(metax_device), acc.to(metax_device), out[32:-32],
            *x.stride(), *y.stride(), m, n, k, step, fa, fb)
    # Reuse the same inputs and compiled kernel to expose intermittent races.
    repeats = 500 if warps > 4 and stages == 2 and k > step else 3
    for iteration in range(repeats):
        out.fill_(11.)
        kernel = _dot_scaled[(1, )](*args, num_warps=warps, num_stages=stages, scenario=scenario)
        actual = out.cpu()
        assert torch.equal(actual[:32], torch.full((32, ), 11.)), iteration
        assert torch.equal(actual[-32:], torch.full((32, ), 11.)), iteration
        assert torch.all((actual[32:-32].reshape(m, n).double() - expected).abs() <= bound), iteration
    assert "llvm.mxc.mma.f32." in kernel.asm["llir"]
    if warps > 4 and k > step:
        assert "memdesc<2x" in kernel.asm["ttgir"]


@triton.jit
def _dot_control(A, B, Y):
    r, k, c = tl.arange(0, 16), tl.arange(0, 64), tl.arange(0, 16)
    a = tl.load(A + r[:, None] * 64 + k[None, :])
    b = tl.load(B + k[:, None] * 16 + c[None, :])
    y = tl.dot(a, b)
    tl.store(Y + r[:, None] * 16 + c[None, :], y)


@triton.jit
def _dot_scaled_alias_store(A, B, Y):
    r, k, c = tl.arange(0, 16), tl.arange(0, 32), tl.arange(0, 16)
    ap = A + r[:, None] * 32 + k[None, :]
    bp = B + k[:, None] * 16 + c[None, :]
    acc = tl.zeros((16, 16), tl.float32)
    for _ in range(2):
        a, b = tl.load(ap), tl.load(bp)
        acc = tl.dot_scaled(a, None, "e2m1", b, None, "e2m1", acc)
        tl.store(ap, tl.full((16, 32), 0x44, tl.uint8))
    tl.store(Y + r[:, None] * 16 + c[None, :], acc)


def test_metax_fp4_dot_scaled_alias_store(metax_device):
    a = torch.full((16, 32), 0x22, dtype=torch.uint8, device=metax_device)
    b = torch.full((32, 16), 0x22, dtype=torch.uint8, device=metax_device)
    out = torch.empty((16, 16), dtype=torch.float32, device=metax_device)
    _dot_scaled_alias_store[(1, )](a, b, out, num_warps=4, num_stages=3)
    # Iteration 0 reads ones; iteration 1 must read the twos just stored.
    assert torch.equal(out.cpu(), torch.full((16, 16), 192.))
    assert torch.equal(a.cpu(), torch.full((16, 32), 0x44, dtype=torch.uint8))


@triton.jit
def _dot_scaled_dependent_load(A, B, S, Y):
    r, k, c = tl.arange(0, 16), tl.arange(0, 16), tl.arange(0, 16)
    acc = tl.zeros((16, 16), tl.float32)
    for i in range(2):
        scale = tl.load(S + i * 16 + r[:, None])
        offset = (scale.to(tl.int32) - 127) * 256
        a = tl.load(A + offset + r[:, None] * 16 + k[None, :])
        b = tl.load(B + k[:, None] * 16 + c[None, :])
        acc = tl.dot_scaled(a, scale, "e2m1", b, None, "e2m1", acc)
    tl.store(Y + r[:, None] * 16 + c[None, :], acc)


def test_metax_fp4_dot_scaled_dependent_load(metax_device):
    a = torch.stack((torch.full((16, 16), 0x22, dtype=torch.uint8), torch.full((16, 16), 0x44,
                                                                               dtype=torch.uint8))).to(metax_device)
    b = torch.full((16, 16), 0x22, dtype=torch.uint8, device=metax_device)
    s = torch.stack((torch.full((16, 1), 127, dtype=torch.uint8), torch.full((16, 1), 128,
                                                                             dtype=torch.uint8))).to(metax_device)
    out = torch.empty((16, 16), dtype=torch.float32, device=metax_device)
    _dot_scaled_dependent_load[(1, )](a, b, s, out, num_warps=4, num_stages=3)
    assert torch.equal(out.cpu(), torch.full((16, 16), 160.))


@triton.jit
def _dot_scaled_scale_codes(Y):
    p = tl.program_id(0)
    code, scale = p // 256, p % 256
    packed = (code | (code << 4)).to(tl.uint8)
    a = tl.full((16, 16), 0, tl.uint8) | packed
    b = tl.full((16, 16), 0x22, tl.uint8)
    sa = tl.full((16, 1), 0, tl.uint8) | scale.to(tl.uint8)
    sb = tl.full((16, 1), 0, tl.uint8) | tl.where(scale == 255, 127, 254 - scale).to(tl.uint8)
    result = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1")
    i, j = tl.arange(0, 16), tl.arange(0, 16)
    tl.store(Y + p * 256 + i[:, None] * 16 + j[None, :], result)


def test_metax_fp4_dot_scaled_scale_codes(metax_device):
    # Every E2M1 value and E8M0 code, with compensating RHS scales so large
    # finite LHS values remain observable rather than overflowing the sum.
    codes = torch.arange(4096) // 256
    scales = torch.arange(4096).remainder(256)
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                         dtype=torch.float64)
    a = table[codes] * torch.from_numpy(np.ldexp(np.ones(4096), scales.numpy() - 127))
    a[scales == 255] = float("nan")
    inverse = torch.where(scales == 255, 127, 254 - scales)
    b = torch.from_numpy(np.ldexp(np.ones(4096), inverse.numpy() - 127))
    a, b = a.to(torch.bfloat16).double(), b.to(torch.bfloat16).double()
    # Use ordinary MMA as the reference for the device's denormal handling.
    tiny = torch.full((16, 64), 2**-127, dtype=torch.bfloat16).to(metax_device)
    large = torch.full((64, 16), float(2**126), dtype=torch.bfloat16).to(metax_device)
    control = torch.empty((16, 16), dtype=torch.float32, device=metax_device)
    _dot_control[(1, )](tiny, large, control, num_warps=4, num_stages=2)
    if torch.equal(control.cpu(), torch.zeros((16, 16))):
        minimum = torch.finfo(torch.bfloat16).tiny
        a[a.abs() < minimum] = 0.
        b[b.abs() < minimum] = 0.
    else:
        assert torch.equal(control.cpu(), torch.full((16, 16), 32.))
    expected = (32 * a * b).float()
    out = torch.empty((4096, 16, 16), dtype=torch.float32, device=metax_device)
    _dot_scaled_scale_codes[(4096, )](out, num_warps=4, num_stages=2)
    actual = out.cpu()
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan[:, None, None].expand_as(actual))
    assert torch.equal(actual[~nan], expected[~nan, None, None].expand_as(actual[~nan]))


@pytest.mark.parametrize("case", [
    "scale_zero", "bf16_subnormal", "large_scale", "scale_nan", "overflow", "fp16_small_scale", "fp8_inf", "fp8_nan",
    "fp8_scale_zero"
])
def test_metax_fp4_dot_scaled_special(case, metax_device):
    m, n, k = 16, 16, 64
    a = torch.full((m, k // 2), 0x22, dtype=torch.uint8)
    b = torch.full((k // 2, n), 0x22, dtype=torch.uint8)
    sa = torch.full((m, k // 32), 127, dtype=torch.uint8)
    sb = torch.full((n, k // 32), 127, dtype=torch.uint8)
    fb, expected = "e2m1", 64.
    if case == "scale_zero":
        a.fill_(0x44)
        sa.fill_(0)
        sb.fill_(253)
    elif case == "bf16_subnormal":
        sa.fill_(0)
        sb.fill_(253)
        expected = None
    elif case == "large_scale":
        a.fill_(0x11)
        sa.fill_(254)
        sb.fill_(1)
    elif case == "scale_nan":
        sa.fill_(255)
        expected = float("nan")
    elif case == "overflow":
        a.fill_(0x77)
        sa.fill_(254)
        expected = float("inf")
    elif case == "fp16_small_scale":
        a.fill_(0x77)
        sa.fill_(102)
        fb, b, sb, expected = "fp16", torch.full((k, n), 2**15, dtype=torch.float16), None, .375
    else:
        fb = "e5m2"
        value = {"fp8_inf": float("inf"), "fp8_nan": float("nan"), "fp8_scale_zero": 2**15}[case]
        b = torch.full((k, n), value).to(torch.float8_e5m2)
        if case == "fp8_scale_zero":
            sb.fill_(0)
            expected = 2**-106
        else:
            sb = None
            expected = value
    x, y = a.to(metax_device), b.to(metax_device)
    s, t = sa.to(metax_device), sb.to(metax_device) if sb is not None else None
    out = torch.empty((m, n), dtype=torch.float32, device=metax_device)
    _dot_scaled[(1, )](x, y, s, t, torch.zeros_like(out), out, *x.stride(), *y.stride(), m, n, k, k, "e2m1", fb,
                       num_warps=4, num_stages=2)
    actual = out.cpu()
    if expected is None:
        # Compare with hardware MMA independently of FP4 decoding.
        # C550 BF16 MMA flushes subnormal inputs; FP16 MMA preserves them.
        reference = torch.empty_like(out)
        tiny = torch.full((m, k), 2**-127, dtype=torch.bfloat16).to(metax_device)
        large = torch.full((k, n), float(2**126), dtype=torch.bfloat16).to(metax_device)
        _dot_control[(1, )](tiny, large, reference, num_warps=4, num_stages=2)
        assert torch.equal(actual, reference.cpu())
    elif np.isnan(expected):
        assert torch.isnan(actual).all()
    else:
        assert torch.equal(actual, torch.full_like(actual, expected))


def _dot_scaled_source(**kwargs):
    constants = dict(A0=32, A1=1, B0=16, B1=1, M=16, N=16, K=64, STEP=64, FA="e2m1", FB="e2m1", PACK_A=True,
                     PACK_B=True, ACC_TYPE=tl.float32, OUT_TYPE=tl.float32, BATCH=False, SCALE_WIDTH=0)
    scale_type = kwargs.pop("scale_type", "u8")
    constants.update(kwargs)
    return ASTSource(_dot_scaled,
                     {"A": "*u8", "B": "*u8", "SA": "*" + scale_type, "SB": "*u8", "C": "*fp32", "Y": "*fp32"},
                     constants)


@pytest.mark.parametrize("options,error", [
    ({"PACK_A": False}, "requires packing along K"),
    ({"PACK_B": False}, "requires packing along K"),
    ({"BATCH": True}, "requires 2D inputs"),
    ({"scale_type": "fp8e4nv"}, "must contain E8M0 int8/uint8 codes"),
    ({"scale_type": "i32"}, "must contain E8M0 int8/uint8 codes"),
    ({"ACC_TYPE": tl.float16}, "expected accumulator dtype fp32"),
    ({"OUT_TYPE": tl.float16}, "Only float32 is supported for out_dtype"),
    ({"SCALE_WIDTH": 1}, "lhs_scale must be a tensor of shape"),
    ({"K": 16, "STEP": 16, "SCALE_WIDTH": 1}, "requires K divisible by 32"),
    ({"M": 1, "N": 1}, "requires M >= 16, N >= 16 and K >= 16; got M=1, N=1, K=64"),
    ({"M": 8}, "requires M >= 16, N >= 16 and K >= 16; got M=8, N=16, K=64"),
    ({"N": 8}, "requires M >= 16, N >= 16 and K >= 16; got M=16, N=8, K=64"),
    *[({"K": k, "STEP": k, "SCALE_WIDTH": 1, "SA": None, "SB": None
        }, f"requires M >= 16, N >= 16 and K >= 16; got M=16, N=16, K={k}") for k in (2, 4, 8)],
])
@pytest.mark.parametrize("arch", [80, 86])
def test_metax_fp4_dot_scaled_invalid(options, error, arch):
    with pytest.raises(CompilationError, match=error):
        triton.compile(_dot_scaled_source(**options), target=GPUTarget("maca", arch, 64))


@pytest.mark.parametrize("arch", [80, 86])
@pytest.mark.parametrize("diagnostics", ["", "warnings"])
def test_metax_fp4_dot_scaled_diagnostic(arch, diagnostics, fresh_triton_cache, monkeypatch, capfd):
    if diagnostics:
        monkeypatch.setenv("MLIR_ENABLE_DIAGNOSTICS", diagnostics)
    else:
        monkeypatch.delenv("MLIR_ENABLE_DIAGNOSTICS", raising=False)
    with pytest.warns(UserWarning, match="MetaX FP4 dot_scaled uses non-native GPU software decoding/scaling"):
        kernel = triton.compile(_dot_scaled_source(), target=GPUTarget("maca", arch, 64))
    stderr = capfd.readouterr().err
    assert ("non-native FP4 dot_scaled on MetaX" in stderr) == bool(diagnostics)
    assert "tt.dot_scaled" not in kernel.asm["ttgir"]
    assert "llvm.mxc.mma.f32." in kernel.asm["llir"]
    assert "llvm.mxc.scmma" not in kernel.asm["llir"]
