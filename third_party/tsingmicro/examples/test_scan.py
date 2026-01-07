import pytest
import torch
import triton
import triton.language as tl
import benchmark


# @pytest.mark.parametrize("M, N", [(1, 64), (2, 32), (4, 16), (8, 8), (16, 4), (32, 2), (64, 1)])
@pytest.mark.parametrize("M, N", [(32, 2)])
@pytest.mark.parametrize("reversed", [True, False])
def test_scan_1d(M, N, reversed, device):

    @triton.jit
    def scan_kernel(out_ptr, in_ptr, n_elements, M: tl.constexpr, N: tl.constexpr):
        offsets = tl.arange(0, M)
        mask = offsets < n_elements
        input = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        output = tl.cumsum(input).reshape([1, M]).broadcast_to([N, M])
        # tl.store(out_ptr + tl.arange(0, M * N), output.reshape([M * N]))
        offs_cm = tl.arange(0, N)
        offs_cn = tl.arange(0, M)
        c_ptrs = out_ptr + M * offs_cm[:, None] + offs_cn[None, :]
        c_mask = (offs_cn[None, :] < n_elements)
        tl.store(c_ptrs, output, mask=c_mask)

    @triton.jit
    def scan_kernel_reverse(out_ptr, in_ptr, n_elements, M: tl.constexpr, N: tl.constexpr):
        offsets = tl.arange(0, M)
        mask = offsets < n_elements
        input = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        output = tl.cumsum(input, reverse=True).reshape([1, M]).broadcast_to([N, M])
        # tl.store(out_ptr + tl.arange(0, M * N), output.reshape([M * N]))
        offs_cm = tl.arange(0, N)
        offs_cn = tl.arange(0, M)
        c_ptrs = out_ptr + M * offs_cm[:, None] + offs_cn[None, :]
        c_mask = (offs_cn[None, :] < n_elements)
        tl.store(c_ptrs, output, mask=c_mask)

    # x = torch.randint(-100, 100, (M, ), dtype=torch.int32, device=device)
    # output = torch.empty(M * N, dtype=torch.int32, device=device)

    # x = torch.rand((M, ), dtype=torch.float32, device=device
    n_elements = 32

    x = torch.arange(0, n_elements, dtype=torch.float32, device=device)
    output = torch.empty(M * N, dtype=torch.float32, device=device)

    if reversed:
        scan_kernel = scan_kernel_reverse
        ref_x = torch.flip(x, dims=[0])
    else:
        scan_kernel = scan_kernel
        ref_x = x
    scan_kernel[(1, )](output, x, n_elements, M, N)

    ref = torch.cumsum(ref_x, dim=0).reshape([1, M]).broadcast_to([N, M]).reshape([M * N])
    if reversed:
        ref = torch.flip(ref, dims=[0])

    print(f"The maximum difference between torch and triton is "
          f"{torch.max(torch.abs(ref - output))}")
    torch.testing.assert_close(ref.to(torch.float32), output, atol=0, rtol=0)


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    test_scan_1d(32, 2, False, "cpu")
    test_scan_1d(32, 2, True, "cpu")
