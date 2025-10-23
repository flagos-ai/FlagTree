"""
Fused Softmax
=============

In this tutorial, you will write a fused softmax operation that is significantly faster
than PyTorch's native op for a particular class of matrices: those whose rows can fit in
the NPU's SRAM.

In doing so, you will learn about:

* The benefits of kernel fusion for bandwidth-bound operations.

* Reduction operators in Triton.

"""

# %%
# Motivations
# -----------
#
# Custom NPU kernels for elementwise additions are educationally valuable but won't get you very far in practice.
# Let us consider instead the case of a simple (numerically stabilized) softmax operation:

import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime import driver
import time

def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


# %%
# When implemented naively in PyTorch, computing :code:`y = naive_softmax(x)` for :math:`x \in R^{M \times N}`
# requires reading :math:`5MN + 2M` elements from DRAM and writing back :math:`3MN + 2M` elements.
# This is obviously wasteful; we'd prefer to have a custom "fused" kernel that only reads
# X once and does all the necessary computations on-chip.
# Doing so would require reading and writing back only :math:`MN` bytes, so we could
# expect a theoretical speed-up of ~4x (i.e., :math:`(8MN + 4M) / 2MN`).
# The `torch.jit.script` flags aims to perform this kind of "kernel fusion" automatically
# but, as we will see later, it is still far from ideal.

# %%
# Compute Kernel
# --------------
#
# Our softmax kernel works as follows: each program loads a set of rows of the input matrix X strided by number of programs,
# normalizes it and writes back the result to the output Y.
#
# Note that one important limitation of Triton is that each block must have a
# power-of-two number of elements, so we need to internally "pad" each row and guard the
# memory operations properly if we want to handle any possible input shapes:

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   XBLOCK:tl.constexpr, num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0) * XBLOCK
    XBLOCK_SUB : tl.constexpr = 8
    #for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
    for row_idx in tl.range(0, XBLOCK, XBLOCK_SUB) :
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_offsets = row_start + row_idx + tl.arange(0, XBLOCK_SUB)[:,None]
        col_offsets = tl.arange(0, BLOCK_SIZE)[None,:]
        xmask = (row_offsets < n_rows)
        ymask = (col_offsets < n_cols)
        mask = xmask & ymask
        input_ptrs = input_ptr + (row_offsets * input_row_stride + col_offsets )
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=1).reshape(XBLOCK_SUB,1).broadcast_to(XBLOCK_SUB,BLOCK_SIZE)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=1).reshape(XBLOCK_SUB,1).broadcast_to(XBLOCK_SUB,BLOCK_SIZE)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_ptrs = output_ptr + (row_offsets * output_row_stride + col_offsets )
        tl.store(output_ptrs, softmax_output, mask=mask)


# %%
# We can create a helper function that enqueues the kernel and its (meta-)arguments for any given input tensor.
# NUM_SM = properties["multiprocessor_count"]
# NUM_REGS = properties["max_num_regs"]
# SIZE_SMEM = properties["max_shared_mem"]
# WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

device = torch.npu.current_device()
stream = torch.npu.current_stream(device).npu_stream

def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    num_programs = 32
    
    XBLOCK = (n_rows + num_programs -1) // num_programs
    BLOCK_SIZE = n_cols
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software piepling stages.
    #num_stages = 4 if SIZE_SMEM > 200000 else 2
    num_stages =4

    # Allocate output
    y = torch.empty_like(x)


    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, num_programs))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       XBLOCK=XBLOCK, num_stages=num_stages, num_warps=num_warps, grid=(32, ))
        kernel._init_handles()
        # n_regs = kernel.n_regs
        # size_smem = kernel.metadata.shared
        # occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        # occupancy = min(occupancy, SIZE_SMEM // size_smem)
        # num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(32, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        stream=stream
    )
    return y


# %%
# Unit Test
# ---------

# %%
# We make sure that we test our kernel on a matrix with an irregular number of rows and columns.
# This will allow us to verify that our padding mechanism works.

def torch_softmax(x):
    return torch.softmax(x, axis=-1)

torch.manual_seed(0)
# x = torch.randn(1823, 781, device='npu')
x = torch.randn(4096, 1024, device='npu')
y_triton = softmax(x)
y_torch = torch_softmax(x)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
#torch.testing.assert_close(y_triton, y_torch, rtol=1e-3, atol=1e-3)
# %%
# As expected, the results are identical.
def benchmark_test(fn, fn_triton, args =(), name="gen_fn", times=100, repeat=10):
    print(f"--------------------benchmark_{name} for {times * repeat} times--------------------")
    stream = torch.npu.current_stream()
    # warm_up
    stream.synchronize()
    for _ in range(10) :
        fn_triton(args)
    stream.synchronize()

    start = time.perf_counter()
    for _ in range(times * repeat) :
        fn_triton(args)
    stream.synchronize()
    end = time.perf_counter()

    time_compiled = (end - start) / (times * repeat)
    time_compiled *= 1000000
    print(f"time_triton:{time_compiled:.6f}")


    print(f"Runing eager {name} for {times * repeat} times")
    
    # warm_up
    stream.synchronize()
    for _ in range(10) :
        std = fn(args)
    stream.synchronize()

    start = time.perf_counter()
    for _ in range(times * repeat) :
        std = fn(args)
    stream.synchronize()
    end = time.perf_counter()
    time_eager = (end - start) / (times * repeat)
    time_eager *= 1000000
    print(f"time_eager:{time_eager:.6f}")

    accelerated = (time_eager - time_compiled)/time_compiled*100
    print(f"Accelerated: {accelerated:.4f}% eager takes {time_eager:.3f} us, triton takes {time_compiled:.3f} us")

    return accelerated, time_eager, time_compiled

# x = torch.randn(4096, 1024, device='npu')
benchmark_test(torch_softmax,softmax,args=x)
# %%
# Benchmark
# ---------
#
# Here we will benchmark our operation as a function of the number of columns in the input matrix -- assuming 4096 rows.
# We will then compare its performance against (1) :code:`torch.softmax` and (2) the :code:`naive_softmax` defined above.


# @triton.testing.perf_report(
#     triton.testing.Benchmark(
#         x_names=['N'],  # argument names to use as an x-axis for the plot
#         x_vals=[128 * i for i in range(2, 8)],  # different possible values for `x_name`
#         line_arg='provider',  # argument name whose value corresponds to a different line in the plot
#         line_vals=['triton', 'torch'],  # possible values for `line_arg``
#         line_names=[
#             "Triton",
#             "Torch",
#         ],  # label name for the lines
#         styles=[('blue', '-'), ('green', '-')],  # line styles
#         ylabel="GB/s",  # label name for the y-axis
#         plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
#         args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
#     ))
# def benchmark(M, N, provider):
#     x = torch.randn(M, N, device='npu', dtype=torch.float32)
#     #stream = torch.npu.Stream()
#     #torch.npu.set_stream(stream)

#     if provider == 'torch':
#         ms = triton.testing.do_bench(lambda: torch_softmax(x))
#     if provider == 'triton':
#         ms = triton.testing.do_bench(lambda: softmax(x))
#     # gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
#     gbps = lambda ms: ms*1000
#     return gbps(ms)


# benchmark.run(show_plots=True, print_data=True)

# %%
# In the above plot, we can see that:
#  - Triton is 4x faster than the Torch JIT. This confirms our suspicions that the Torch JIT does not do any fusion here.
#  - Triton is noticeably faster than :code:`torch.softmax` -- in addition to being **easier to read, understand and maintain**.
#    Note however that the PyTorch `softmax` operation is more general and will work on tensors of any shape.
