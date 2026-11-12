"""E04: persistent tiled linear projection, optional bias, FP32 accumulation."""
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _linear(X, W, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BIAS: tl.constexpr, PROGRAMS: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tiles_n = tl.cdiv(N, BN)
    for tile in range(tl.program_id(0), tl.cdiv(M, BM) * tiles_n, PROGRAMS):
        m = (tile // tiles_n) * BM + tl.arange(0, BM)
        n = (tile % tiles_n) * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(K, BK)):
            k = block * BK + tl.arange(0, BK)
            x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0)
            w = tl.load(W + n[None, :] * K + k[:, None], (n[None, :] < N) & (k[:, None] < K), 0)
            acc = tl.dot(x, w, acc)
        if BIAS:
            acc += tl.load(B + n, n < N, 0)[None, :].to(tl.float32)
        tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def dense_linear_persistent(x, weight, bias=None, *, programs=2):
    """x[M,K], weight[N,K], optional bias[N]; contiguous IFWA FP16/BF16; FP32 output.

    programs caps the resident grid. Each program handles multiple output tiles
    when necessary. This is a correctness example, not an autotuned GEMM.
    """
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("expected x[M,K] and weight[N,K]")
    m, k = x.shape
    n = weight.shape[0]
    if min(m, n, k) < 1 or not isinstance(programs, int) or programs < 1:
        raise ValueError("dimensions and programs must be positive")
    if bias is not None and bias.shape != (n, ):
        raise ValueError("bias must have shape [N]")
    tensors = [x, weight] + ([] if bias is None else [bias])
    if x.device.type != "ifwa" or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("requires FP16/BF16 IFWA inputs")
    if any(t.device != x.device or t.dtype != x.dtype or not t.is_contiguous() or t.requires_grad for t in tensors):
        raise ValueError("inputs must be contiguous, same dtype/device, inference only")
    count = min(programs, triton.cdiv(m, 64) * triton.cdiv(n, 32))
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _linear[(count, )](x, weight, x if bias is None else bias, out, m, n, k, bias is not None, count, 64, 32, 32,
                       num_warps=4, num_stages=1)
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,bias,programs", [((7, 19, 37), False, 1), ((65, 35, 64), True, 1),
                                                 ((8, 33, 96), True, 2)])
def test_dense_linear(dtype, shape, bias, programs):
    m, n, k = shape
    generator = torch.Generator().manual_seed(41)
    x = (torch.randn((m, k), generator=generator) * 0.25).to(dtype)
    w = (torch.randn((n, k), generator=generator) * 0.25).to(dtype)
    b = torch.randn(n, generator=generator).to(dtype) if bias else None
    expected = x.float() @ w.float().T
    if b is not None:
        expected += b.float()
    actual = dense_linear_persistent(x.to('ifwa:0'), w.to('ifwa:0'), None if b is None else b.to('ifwa:0'),
                                     programs=programs).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    print("E04 persistent tl.dot linear on", triton.runtime.driver.active.get_current_target(), flush=True)
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
