"""E07: native MXFP4/MXFP8 block32 linear using tl.dot_scaled."""
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _mx_linear(A, B, SA, SB, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, FORMAT: tl.constexpr):
    PACK: tl.constexpr = 2 if FORMAT == "e2m1" else 1
    m = tl.program_id(0) * 128 + tl.arange(0, 128)
    n = tl.program_id(1) * 128 + tl.arange(0, 128)
    p = tl.arange(0, 64 // PACK)
    s = tl.arange(0, 2)
    acc = tl.zeros((128, 128), tl.float32)
    for block in range(K // 64):
        kp = block * (64 // PACK) + p
        ks = block * 2 + s
        a = tl.load(A + m[:, None] * (K // PACK) + kp[None, :], m[:, None] < M, 0)
        b = tl.load(B + n[None, :] * (K // PACK) + kp[:, None], n[None, :] < N, 0)
        sa = tl.load(SA + m[:, None] * (K // 32) + ks[None, :], m[:, None] < M, 127)
        sb = tl.load(SB + n[:, None] * (K // 32) + ks[None, :], n[:, None] < N, 127)
        acc = tl.dot_scaled(a, sa, FORMAT, b, sb, FORMAT, acc, lhs_k_pack=True, rhs_k_pack=True)
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def mxfp_linear(x, weight, x_scale, weight_scale, *, format="mxfp4"):
    """Prequantized IFWA operands and uint8 E8M0 scales -> FP32[M,N].

    MXFP4: uint8 packed x[M,K/2], weight[N,K/2], low nibble first.
    MXFP8: uint8 E4M3FN bit patterns x[M,K], weight[N,K].
    Scales have shape [rows,K/32], values 0..254; 255 is a reserved NaN
    encoding and the caller must exclude it. K must be a multiple of 64.
    """
    if format not in ("mxfp4", "mxfp8"):
        raise ValueError("format must be mxfp4 or mxfp8")
    pack = 2 if format == "mxfp4" else 1
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("expected equally sized reduction dimensions")
    m, storage_k = x.shape
    n = weight.shape[0]
    k = storage_k * pack
    if min(m, n, k) < 1 or k % 64:
        raise ValueError("positive dimensions and K multiple64 are required")
    if x_scale.shape != (m, k // 32) or weight_scale.shape != (n, k // 32):
        raise ValueError("scales must have shape [rows,K/32]")
    tensors = (x, weight, x_scale, weight_scale)
    if x.device.type != "ifwa" or any(t.device != x.device or t.dtype != torch.uint8 or not t.is_contiguous()
                                      for t in tensors):
        raise ValueError("requires contiguous uint8 IFWA operands and scales on the same device")
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _mx_linear[(triton.cdiv(m, 128), triton.cdiv(n, 128))](x, weight, x_scale, weight_scale, out, m, n, k,
                                                           "e2m1" if pack == 2 else "e4m3", num_warps=4, num_stages=1)
    return out


def make_operand(rows, k, format, generator, zero=False):
    """CPU fixture encoding; independent CPU dequantized matmul is the oracle."""
    if format == "mxfp4":
        codes = torch.randint(0, 16, (rows, k), generator=generator, dtype=torch.uint8)
        if zero:
            codes.zero_()
        storage = codes[:, 0::2] | (codes[:, 1::2] << 4)
        levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
        values = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    else:
        values = torch.randn((rows, k), generator=generator).to(torch.float8_e4m3fn)
        if zero:
            values = torch.zeros((rows, k)).to(torch.float8_e4m3fn)
        storage, values = values.view(torch.uint8), values.float()
    scales = torch.randint(125, 130, (rows, k // 32), generator=generator, dtype=torch.uint8)
    decoded = values * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)
    return storage.contiguous(), scales, decoded


@pytest.mark.parametrize("format", ["mxfp4", "mxfp8"])
@pytest.mark.parametrize("shape,zero", [((7, 19, 64), False), ((9, 17, 128), False), ((3, 5, 64), True)])
def test_mx_linear(format, shape, zero):
    m, n, k = shape
    generator = torch.Generator().manual_seed(71)
    a, sa, aq = make_operand(m, k, format, generator, zero)
    b, sb, bq = make_operand(n, k, format, generator, zero)
    out = mxfp_linear(a.to('ifwa:0'), b.to('ifwa:0'), sa.to('ifwa:0'), sb.to('ifwa:0'), format=format)
    torch.testing.assert_close(out.cpu(), aq @ bq.T, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    print("E07 native E2M1/E4M3 + E8M0 block32 dot on", triton.runtime.driver.active.get_current_target(), flush=True)
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
