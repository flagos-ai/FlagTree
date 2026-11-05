"""E05: E4M3FN FP8 linear with tensor, row, or K-block128 scales."""
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_linear(X, W, SX, SW, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr):
    m = tl.program_id(0) * 64 + tl.arange(0, 64)
    n = tl.program_id(1) * 32 + tl.arange(0, 32)
    acc = tl.zeros((64, 32), tl.float32)
    for block in range(tl.cdiv(K, 128)):
        k = block * 128 + tl.arange(0, 128)
        x_zero = tl.full((64, 128), 0, tl.uint8).to(tl.float8e4nv, bitcast=True)
        w_zero = tl.full((128, 32), 0, tl.uint8).to(tl.float8e4nv, bitcast=True)
        x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), x_zero)
        w = tl.load(W + n[None, :] * K + k[:, None], (n[None, :] < N) & (k[:, None] < K), w_zero)
        partial = tl.dot(x, w)
        if MODE == "tensor":
            sx = tl.load(SX)
            sw = tl.load(SW)
            acc += partial * sx * sw
        else:
            if MODE == "row":
                sx = tl.load(SX + m, m < M, 0)
                sw = tl.load(SW + n, n < N, 0)
            else:
                blocks = tl.cdiv(K, 128)
                sx = tl.load(SX + m * blocks + block, m < M, 0)
                sw = tl.load(SW + n * blocks + block, n < N, 0)
            acc += partial * sx[:, None] * sw[None, :]
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def fp8_linear(x, weight, x_scale, weight_scale, *, mode="tensor"):
    """Contiguous IFWA E4M3FN x[M,K], weight[N,K]; return FP32[M,N].

    Scales: tensor=[1], row=[rows], block128=[rows,ceil(K/128)].
    All scale tensors must be contiguous FP32 on the input device.
    """
    if mode not in ("tensor", "row", "block128"):
        raise ValueError("mode must be tensor, row, or block128")
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("expected x[M,K], weight[N,K]")
    m, k = x.shape
    n = weight.shape[0]
    if min(m, n, k) < 1:
        raise ValueError("dimensions must be positive")
    if x.device.type != "ifwa" or x.dtype != torch.float8_e4m3fn or weight.dtype != x.dtype:
        raise ValueError("requires IFWA E4M3FN operands")
    for t in (x, weight, x_scale, weight_scale):
        if t.device != x.device or not t.is_contiguous() or t.requires_grad:
            raise ValueError("inputs must be contiguous on the same device, inference only")
    for scales, rows in ((x_scale, m), (weight_scale, n)):
        expected = (1, ) if mode == "tensor" else (rows, ) if mode == "row" else (rows, triton.cdiv(k, 128))
        if scales.dtype != torch.float32 or scales.shape != expected:
            raise ValueError(f"expected FP32 scales of shape {expected}")
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _fp8_linear[(triton.cdiv(m, 64), triton.cdiv(n, 32))](x, weight, x_scale, weight_scale, out, m, n, k, mode,
                                                          num_warps=4, num_stages=1, enable_fp_fusion=False)
    return out


@pytest.mark.parametrize("mode", ["tensor", "row", "block128"])
@pytest.mark.parametrize("shape", [(7, 19, 96), (9, 35, 160)])
def test_fp8_linear(mode, shape):
    m, n, k = shape
    generator = torch.Generator().manual_seed(51)
    x = torch.randn((m, k), generator=generator).to(torch.float8_e4m3fn)
    w = torch.randn((n, k), generator=generator).to(torch.float8_e4m3fn)
    scales = []
    expanded = []
    for rows in (m, n):
        shape_scale = (1, ) if mode == "tensor" else (rows, ) if mode == "row" else (rows, triton.cdiv(k, 128))
        scale = torch.rand(shape_scale, generator=generator) * 1.5 + 0.25
        scales.append(scale)
        expanded.append(scale if mode == "tensor" else scale[:, None] if mode ==
                        "row" else scale.repeat_interleave(128, dim=1)[:, :k])
    expected = (x.float() * expanded[0]) @ (w.float() * expanded[1]).T
    actual = fp8_linear(x.to('ifwa:0'), w.to('ifwa:0'), scales[0].to('ifwa:0'), scales[1].to('ifwa:0'), mode=mode)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    print("E05 native FP8 tl.dot on", triton.runtime.driver.active.get_current_target(), flush=True)
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
