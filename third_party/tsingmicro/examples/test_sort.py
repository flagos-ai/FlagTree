import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import numpy_random

# @pytest.mark.interpreter
# @pytest.mark.parametrize("M, N", [[1, 512], [8, 64], [256, 16], [512, 8]])
@pytest.mark.parametrize("M, N", [[8, 64]])
# @pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("descending", [False])
# @pytest.mark.parametrize("dtype_str", ['int32', 'float16', 'float32', 'bfloat16'])
@pytest.mark.parametrize("dtype_str", ['float32'])
def test_sort(M, N, descending, dtype_str, device):

    @triton.jit
    def sort_kernel(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
        offx = tl.arange(0, M)
        offy = tl.arange(0, N) * M
        off2d = offx[None, :] + offy[:, None]
        x = tl.load(X + off2d)
        x = tl.sort(x, descending=descending)
        tl.store(Z + off2d, x)

    x = numpy_random((N, M), dtype_str=dtype_str)
    x = torch.from_numpy(x).to(device)
    y = torch.sort(x, descending=descending)[0]
    z = torch.empty_like(x)
    sort_kernel[(1, )](x, z, N, M, descending, num_warps=8)
    assert (y == z).all(), (y, z)
