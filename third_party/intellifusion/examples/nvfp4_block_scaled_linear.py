"""E06: input quantization + NVFP4-format block16 linear on IFWA."""
import math
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _encode_fp4(x):
    a = tl.abs(x)
    code = tl.where(a > 0.25, 1, 0)
    code = tl.where(a >= 0.75, 2, code)
    code = tl.where(a > 1.25, 3, code)
    code = tl.where(a >= 1.75, 4, code)
    code = tl.where(a > 2.5, 5, code)
    code = tl.where(a >= 3.5, 6, code)
    code = tl.where(a > 5.0, 7, code)
    return (code | tl.where(x < 0, 8, 0)).to(tl.uint8)


@triton.jit
def _quantize(X, PACKED, SCALES, K: tl.constexpr, GLOBAL: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    i = tl.arange(0, 8)
    lo = tl.load(X + row * K + block * 16 + i * 2) / GLOBAL
    hi = tl.load(X + row * K + block * 16 + i * 2 + 1) / GLOBAL
    amax = tl.maximum(tl.max(tl.abs(lo), 0), tl.max(tl.abs(hi), 0))
    # Clamp to the finite E4M3 scale range; all-zero blocks use a unit scale.
    scale = tl.where(amax == 0, 1.0, tl.minimum(tl.maximum(amax / 6.0, 0.001953125), 448.0))
    rounded_scale = scale.to(tl.float8e4nv)
    scale_fp32 = rounded_scale.to(tl.float32)
    packed = _encode_fp4(lo / scale_fp32) | (_encode_fp4(hi / scale_fp32) << 4)
    tl.store(PACKED + row * (K // 2) + block * 8 + i, packed)
    tl.store(SCALES + row * (K // 16) + block, rounded_scale)


@triton.jit
def _nvfp4_linear(A, B, SA, SB, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GLOBAL_PRODUCT: tl.constexpr):
    m = tl.program_id(0) * 128 + tl.arange(0, 128)
    n = tl.program_id(1) * 128 + tl.arange(0, 128)
    p = tl.arange(0, 32)
    s = tl.arange(0, 4)
    acc = tl.zeros((128, 128), tl.float32)
    scale_one = tl.full((128, 4), 0x38, tl.uint8).to(tl.float8e4nv, bitcast=True)
    for block in range(K // 64):
        kp = block * 32 + p
        ks = block * 4 + s
        a = tl.load(A + m[:, None] * (K // 2) + kp[None, :], m[:, None] < M, 0)
        b = tl.load(B + n[None, :] * (K // 2) + kp[:, None], n[None, :] < N, 0)
        sa = tl.load(SA + m[:, None] * (K // 16) + ks[None, :], m[:, None] < M, scale_one)
        sb = tl.load(SB + n[:, None] * (K // 16) + ks[None, :], n[:, None] < N, scale_one)
        acc = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc, lhs_k_pack=True, rhs_k_pack=True)
    tl.store(Y + m[:, None] * N + n[None, :], acc * GLOBAL_PRODUCT, (m[:, None] < M) & (n[None, :] < N))


def quantize_nvfp4(x, *, global_scale=1.0):
    """Contiguous finite FP32 IFWA x[rows,K], K multiple64 -> packed, E4M3 scales.

    For tiny/large blocks scale is clamped to E4M3's finite range and values
    saturate to E2M1 +/-6. Caller must supply finite data (no device scan here).
    """
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 64 or x.shape[1] % 64:
        raise ValueError("expected [rows,K], rows>0, K positive multiple of 64")
    if x.device.type != "ifwa" or x.dtype != torch.float32 or not x.is_contiguous() or x.requires_grad:
        raise ValueError("expected contiguous inference FP32 IFWA input")
    if not math.isfinite(global_scale) or global_scale <= 0:
        raise ValueError("global_scale must be finite and positive")
    rows, k = x.shape
    packed = torch.empty((rows, k // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((rows, k // 16), dtype=torch.float8_e4m3fn, device=x.device)
    _quantize[(rows, k // 16)](x, packed, scales, k, global_scale, num_warps=1, enable_fp_fusion=False)
    return packed, scales


def nvfp4_block_scaled_linear(x, weight, *, x_global=1.0, weight_global=1.0):
    """Quantize FP32 x[M,K], weight[N,K] on IFWA, then native scaled dot -> FP32.

    Returned operands allow a caller to inspect the actual packed device data.
    No unquantized reference GEMM is substituted for the quantized computation.
    """
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1] or x.device != weight.device:
        raise ValueError("expected same-device x[M,K], weight[N,K]")
    a, sa = quantize_nvfp4(x, global_scale=x_global)
    b, sb = quantize_nvfp4(weight, global_scale=weight_global)
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _nvfp4_linear[(triton.cdiv(m, 128), triton.cdiv(n, 128))](a, b, sa, sb, out, m, n, k, x_global * weight_global,
                                                              num_warps=4, num_stages=1)
    return out, (a, sa, b, sb)


def reference_quantize(x, global_scale):
    """Independent CPU nearest-even E2M1 quantization by enumerating values."""
    rows, k = x.shape
    blocks = (x / global_scale).reshape(rows, k // 16, 16)
    amax = blocks.abs().amax(dim=-1)
    scales = torch.where(amax == 0, 1.0, (amax / 6).clamp(2**-9, 448)).to(torch.float8_e4m3fn)
    values = (blocks / scales.float()[..., None]).reshape(rows, k)
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    distance = (values.abs()[..., None] - levels).abs()
    minimum = distance.amin(dim=-1, keepdim=True)
    # Resolve exact ties toward a code whose low mantissa bit is zero.
    code_ids = torch.arange(8)
    priority = code_ids + (code_ids % 2) * 8
    candidates = torch.where(distance == minimum, priority, 100)
    codes = candidates.argmin(dim=-1).to(torch.uint8) | ((values < 0).to(torch.uint8) << 3)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    decoded = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    decoded *= scales.float().repeat_interleave(16, dim=1) * global_scale
    return packed, scales, decoded


@pytest.mark.parametrize("shape", [(7, 19, 64), (9, 17, 128)])
@pytest.mark.parametrize("pattern", ["random", "zero", "ties"])
def test_nvfp4_linear(shape, pattern):
    m, n, k = shape
    generator = torch.Generator().manual_seed(61)
    x = torch.randn((m, k), generator=generator) * 2
    w = torch.randn((n, k), generator=generator) * 3
    x_global, w_global = 0.5, 2.0
    if pattern == "zero":
        x.zero_()
        w.zero_()
    elif pattern == "ties":
        # max=6 gives scale=1, exposing all seven nearest-even boundaries.
        values = torch.tensor([0, .25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6, -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5])
        x = values.repeat(m, k // 16) * x_global
        w = values.flip(0).repeat(n, k // 16) * w_global
    ap, sa, aq = reference_quantize(x, x_global)
    bp, sb, bq = reference_quantize(w, w_global)
    out, actual_operands = nvfp4_block_scaled_linear(x.to('ifwa:0'), w.to('ifwa:0'), x_global=x_global,
                                                     weight_global=w_global)
    for actual, expected in zip(actual_operands, (ap, sa, bp, sb)):
        # PyTorch CPU float8 comparison operators are not universally defined.
        torch.testing.assert_close(actual.cpu().view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(out.cpu(), aq @ bq.T, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    print("E06 quantize + native E2M1/E4M3 block16 dot on", triton.runtime.driver.active.get_current_target(),
          flush=True)
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
