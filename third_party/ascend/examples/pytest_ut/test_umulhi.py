import torch
import torch_npu
import numpy as np
import triton
import triton.language as tl
from numpy.random import RandomState

# inp the two 32 bit signed integers.
@triton.jit
def umulhi_kernel(X, Y, Z, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    z = tl.umulhi(x, y)
    tl.store(Z + tl.arange(0, N), z)


# accuracy reference
def umulhi32(a, b):
    a_64 = a.astype(np.int64)
    b_64 = b.astype(np.int64)
    product_64 = a_64 * b_64
    # get the high part
    result_high_32 = product_64 >> 32
    return result_high_32.astype(np.int32)


def test_umulhi():
    N = 128
    x = torch.randint(low=0, high=2000, size=(N,), dtype=torch.int32)
    y = torch.randint(low=0, high=2000, size=(N,), dtype=torch.int32)
    xx = x.npu()
    yy = y.npu()
    z_tri = torch.zeros(size=(N,), dtype=torch.int32).npu()
    umulhi_kernel[(1,)](xx, yy, z_tri, N=N)

    xxx = x.numpy()
    yyy = y.numpy()
    z_ref = umulhi32(xxx, yyy)
    z_ref1 = torch.from_numpy(z_ref).npu()
    torch.equal(z_tri, z_ref1)