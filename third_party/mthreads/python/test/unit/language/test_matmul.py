import numpy as np
import torch
import triton
import triton.language as tl
from numpy.random import RandomState


def test_chained_matmul_tf32(device):
    M, N, K = 32, 32, 32
    num_warps = 4
    rs = RandomState(17)

    x = rs.randn(M, K).astype(np.float32) * 0.1
    y = rs.randn(K, N).astype(np.float32) * 0.1
    w = rs.randn(N, N).astype(np.float32) * 0.1

    mask = np.uint32(0xffffe000)
    x = (x.view('uint32') & mask).view('float32')
    y = (y.view('uint32') & mask).view('float32')
    w = (w.view('uint32') & mask).view('float32')

    x_tri = torch.tensor(x, device=device)
    y_tri = torch.tensor(y, device=device)
    w_tri = torch.tensor(w, device=device)
    z_tri = torch.empty((M, N), dtype=torch.float32, device=device)

    @triton.jit
    def kernel(X, stride_xm, stride_xk, Y, stride_yk, stride_yn, W, stride_wn, stride_wl, Z, stride_zm, stride_zn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, INPUT_PRECISION: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_l = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        xv = tl.load(X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk)
        yv = tl.load(Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn)
        wv = tl.load(W + off_n[:, None] * stride_wn + off_l[None, :] * stride_wl)
        z = tl.dot(xv, yv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        z = tl.dot(z.to(wv.dtype), wv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        tl.store(Z + off_m[:, None] * stride_zm + off_l[None, :] * stride_zn, z)

    kernel[(1, 1)](
        x_tri,
        x_tri.stride(0),
        x_tri.stride(1),
        y_tri,
        y_tri.stride(0),
        y_tri.stride(1),
        w_tri,
        w_tri.stride(0),
        w_tri.stride(1),
        z_tri,
        z_tri.stride(0),
        z_tri.stride(1),
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=K,
        INPUT_PRECISION='tf32',
        num_warps=num_warps,
        num_ctas=1,
    )

    z_ref = x @ y
    z_ref = z_ref @ w
    np.testing.assert_allclose(z_tri.cpu().numpy(), z_ref, rtol=0.01, atol=1e-3)
