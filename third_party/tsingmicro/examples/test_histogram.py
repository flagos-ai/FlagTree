import pytest
import torch
import triton
import triton.language as tl


# @pytest.mark.interpreter
# @pytest.mark.parametrize("M, N", [[2048, 2], [1024, 8], [1024, 128], [256, 512], [32, 512], [8, 512], [8, 2]])
@pytest.mark.parametrize("M, N", [[1024, 8]])
def test_histogram(M, N, device):

    @triton.jit
    def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
        offset1 = tl.arange(0, M)
        offset2 = tl.arange(0, N)
        x = tl.load(x_ptr + offset1)
        z = tl.histogram(x, N)
        bias = tl.full([M, N], 1, dtype=tl.int32)
        # check that histogram produces object compatible with broadcasting
        biased = z + bias
        tl.store(z_ptr + offset2, z)

    torch.manual_seed(17)
    x = torch.randint(0, N, (M, ), device=device, dtype=torch.int32)
    z = torch.empty(N, dtype=torch.int32, device=device)
    # torch.histc does not work when the input type is not float and the device is CPU
    # https://github.com/pytorch/pytorch/issues/74236
    # This is a workload by converting the input to float
    z_torch = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    histogram_kernel[(1, )](x, z, M=M, N=N)
    assert (z_torch == z).all()


if __name__ == "__main__":
    import benchmark
    benchmark.select_cpu_backend()
    test_histogram(1024, 8, 'cpu')
    # test_histogram_2d(32, 16, 8, 'cpu')
    # test_histogram(2048, 2, 'cpu')
    # test_histogram(1024, 128, 'cpu')
    # test_histogram(256, 512, 'cpu')
    # test_histogram(32, 512, 'cpu')
    # test_histogram(8, 512, 'cpu')
    # test_histogram(8, 2, 'cpu')
