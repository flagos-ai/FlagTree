import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import numpy as np


@triton.jit
def minimum(a, b):
    ret = tl.minimum(a, b, tl.PropagateNan.ALL)
    if a.dtype == tl.bfloat16:
        ret = ret.to(tl.bfloat16)
    return ret


@triton.jit
def triton_pw_rdc5d(in_ptr0, in_ptr1, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    Z: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    kblk_idx = tl.arange(0, K)
    zblk_idx = tl.arange(0, Z)
    idx = (lblk_idx[:, None, None, None, None] * Z * K * N * M + mblk_idx[None, :, None, None, None] * Z * K * N +
           nblk_idx[None, None, :, None, None] * Z * K + kblk_idx[None, None, None, :, None] * Z +
           zblk_idx[None, None, None, None, :])
    x0 = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret0 = x0 * x1
    ret = tl.reduce(ret0, 4, minimum, keep_dims=True)
    zblk_idx = tl.arange(0, 1)
    odx = (lblk_idx[:, None, None, None, None] * K * N * M + mblk_idx[None, :, None, None, None] * K * N +
           nblk_idx[None, None, :, None, None] * K + kblk_idx[None, None, None, :, None] +
           zblk_idx[None, None, None, None, :])
    tl.store(out_ptr0 + odx, ret)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 1, 1, 1, 1)])  # L=16, others=1
def test_pw_rdc5d(dtype, shape):
    L, M, N, K, Z = shape
    a = torch.randn(*shape, dtype=dtype, device='npu')
    b = torch.randn(*shape, dtype=dtype, device='npu')
    out = torch.empty(*shape, dtype=dtype, device='npu')

    expected = (a * b).to(dtype)

    triton_pw_rdc5d[(1, )](a, b, out, L=L, M=M, N=N, K=K, Z=Z)

    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
