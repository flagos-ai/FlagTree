import os
import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256}, num_warps=32, num_stages=1),
    ], key=['M', 'N'])
@triton.jit
def transpose_kernel(input_ptr, output_ptr, stride_am, stride_an, stride_bn, stride_bm, M, N, BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)

    # num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    input = input_ptr + (rm[:, None] * stride_am + rn[None, :] * stride_an)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    output = output_ptr + (rm[:, None] * stride_bm + rn[None, :] * stride_bn)

    tl.store(output, tl.load(input, mask=mask), mask=mask)


def transpose(input):
    M, N = input.shape
    output = torch.empty((N, M), device=input.device, dtype=input.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    transpose_kernel[grid](input, output, input.stride(0), input.stride(1), output.stride(0), output.stride(1), M, N)
    return output


def check_accuracy(M_list):
    """
    Check that triton results are allclose to torch results.
    """
    print("Check accuracy: Triton vs Torch")
    for M in M_list:
        x = torch.randn(M, M, device='cuda', dtype=torch.float32)
        y_triton = transpose(x)
        y_torch = torch.transpose(x, 0, 1).contiguous()
        if torch.allclose(y_triton, y_torch):
            print(f"Test shape = ({M}, {M}), test passed.")
        else:
            print(f"Test shape = ({M}, {M}), test failed")


def run_performance(M_list, save_path):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['M'],
            x_vals=[M for M in M_list],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=["Triton", "Torch"],
            styles=[('blue', '-'), ('green', '-')],
            ylabel="GB/s",
            plot_name="transpose-performance",
            args={},
        ))
    def benchmark(M, provider):
        x = torch.randn(M, M, device='cuda', dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.transpose(x, 0, 1).contiguous(),
                                                         quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: transpose(x), quantiles=quantiles)

        perf = lambda ms: 8 * M * M / ms * 1e-6
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=True, print_data=True, save_path='.')


if __name__ == "__main__":
    M_list = [32, 64, 128, 256, 512, 1024]
    check_accuracy(M_list)
    run_performance(M_list)
