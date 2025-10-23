import time

import numpy as np
import torch
import torch_npu

import triton
import triton.language as tl
from triton.runtime.libentry import libentry

device = torch.npu.current_device()
stream = torch.npu.current_stream(device)
stream_id = stream.npu_stream


def benchmark(func):
    warmup = 10
    repeat = 100

    def wrapper(*args, **kwargs):
        #
        for _ in range(warmup):
            result = func(*args, **kwargs)
        stream.synchronize()
        #
        start_time = time.perf_counter_ns()
        for _ in range(repeat):
            result = func(*args, **kwargs)
        stream.synchronize()
        end_time = time.perf_counter_ns()
        #
        start_time = start_time * 1e-3
        end_time = end_time * 1e-3
        elapsed_time = (end_time - start_time) / repeat
        return (result, elapsed_time)

    return wrapper


def plot_performance_comparison(sizes, times_torch, times_triton, fname):
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Maple Mono NF CN"
    plt.style.use('ggplot')
    #
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(sizes, times_torch, 'o-', label='Torch')
    ax.plot(sizes, times_triton, 's-', label='Triton')
    ax.set_title('Torch vs Triton Time Cost', fontsize=16)
    ax.set_xlabel('Batch Size', fontsize=14)
    ax.set_ylabel('Kernel Time (us)', fontsize=14)
    ax.set_xlim([0, 2e4])
    ax.set_ylim([0, 500])
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    plt.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"{fname} is saved")


def save_print_data(sizes, times_torch, times_triton, fname):
    perf_data = np.zeros((len(sizes), 3))
    perf_data[:, 0] = sizes
    perf_data[:, 1] = times_torch
    perf_data[:, 2] = times_triton
    np.savetxt(fname, perf_data, delimiter=",", header="batch, torch(us), triton(us)")
    print("batch, torch(us), triton(us)")
    for idx, size in enumerate(sizes):
        print(f"{int(size)}, {times_torch[idx]}, {times_triton[idx]}")


def load_data(fname):
    perf_data = np.loadtxt(fname, delimiter=",", skiprows=1)
    sizes = perf_data[:, 0].astype(np.float32)
    times_torch = perf_data[:, 1]
    times_triton = perf_data[:, 2]
    return sizes, times_torch, times_triton


@libentry()
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * XBLOCK
    rblk_idx = tl.arange(0, XBLOCK_SUB)
    col_idx = tl.arange(0, RBLOCK)
    for row_idx in tl.range(0, XBLOCK, XBLOCK_SUB):
        row_offsets = row_start + row_idx + rblk_idx[:, None]
        col_offsets = col_idx[None, :]
        xmask = row_offsets < n_rows
        ymask = col_offsets < n_cols
        mask = xmask & ymask
        input_idx = row_offsets * input_row_stride + col_offsets
        input_ptrs = input_ptr + input_idx
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=1).reshape(XBLOCK_SUB, 1)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=1).reshape(XBLOCK_SUB, 1)
        softmax_output = numerator / denominator
        output_ptrs = output_ptr + (row_offsets * output_row_stride + col_offsets)
        tl.store(output_ptrs, softmax_output, mask=mask)


@benchmark
def torch_func(x0: torch.Tensor):
    m = torch.nn.Softmax(dim=1)
    return m(x0)


@benchmark
def triton_func(y0: torch.Tensor, x0: torch.Tensor, stream_id0: int):
    n_rows, n_cols = x0.shape
    ncore = 40
    xs = (n_rows + ncore - 1) // ncore
    xss = min(xs, 5)
    softmax_kernel[(ncore, 1, 1)](
        y0,
        x0,
        x0.stride(0),
        y0.stride(0),
        n_rows,
        n_cols,
        XBLOCK=xs,
        XBLOCK_SUB=xss,
        RBLOCK=n_cols,
        stream=stream_id0,
    )
    return y0


torch.manual_seed(0)
DEV = "npu"
DTYPE = torch.float32
seq_len = 2 * 1024

batch_sizes = []
torch_times = []
triton_times = []
for i in range(1, 16 + 1):
    batch = i * 1000
    batch_sizes.append(batch)
    x = torch.rand((batch, seq_len), dtype=DTYPE, device=DEV)
    y = torch.empty_like(x)
    torch_out, torch_time = torch_func(x)
    triton_out, triton_time = triton_func(y, x, stream_id)
    torch.testing.assert_close(triton_out, torch_out)
    torch_times.append(torch_time)
    triton_times.append(triton_time)

data_fname = "compare_perf_softmax_triton_torch.csv"
save_print_data(batch_sizes, torch_times, triton_times, data_fname)
# In case of you already have csv file, you can directly run load_data(data_fname)
# to load the data.
figname = "compare_perf_softmax_triton_torch.png"
plot_performance_comparison(batch_sizes, torch_times, triton_times, figname)
