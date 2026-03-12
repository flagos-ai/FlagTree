import pytest
import torch
import triton
import triton.language as tl
import benchmark

# benchmark.select_cpu_backend()
from triton._internal_testing import numpy_random


# @pytest.mark.interpreter
# @pytest.mark.parametrize("M, N", [[1, 512], [8, 64], [256, 16], [512, 8]])
@pytest.mark.parametrize("M, N", [[8, 64]])
# @pytest.mark.parametrize("dtype_str", ['int32', 'float16', 'float32', 'bfloat16'])
@pytest.mark.parametrize("dtype_str", ['float32'])
def test_flip(M, N, dtype_str, device):

    @triton.jit
    def flip_kernel(X, Z, N: tl.constexpr, M: tl.constexpr):
        offx = tl.arange(0, M)
        offy = tl.arange(0, N) * M
        off2d = offx[None, :] + offy[:, None]
        x = tl.load(X + off2d)
        x = tl.flip(x)
        tl.store(Z + off2d, x)

    x = numpy_random((N, M), dtype_str=dtype_str)
    x = torch.from_numpy(x).to(device)
    y = torch.flip(x, (1, ))
    z = torch.empty_like(x, device=device)
    flip_kernel[(1, )](x, z, N, M, num_warps=8)
    assert (y == z).all(), (y, z)


if __name__ == "__main__":
    # Test with different sizes and data types
    test_sizes = [(8, 64)]
    dtypes = ['float32']

    for M, N in test_sizes:
        for dtype in dtypes:
            print(f"\nTesting flip with M={M}, N={N}, dtype={dtype}")
            try:
                test_flip(M, N, dtype, "cpu")
                print("Test passed!")
            except Exception as e:
                print(f"Test failed: {str(e)}")
