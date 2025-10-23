"""
Layer Normalization
====================
In this tutorial, you will write a high-performance layer normalization
kernel that runs faster than the PyTorch implementation.

In doing so, you will learn about:

* Implementing backward pass in Triton.

* Implementing parallel reduction in Triton.

"""

# %%
# Motivations
# -----------
#
# The *LayerNorm* operator was first introduced in [BA2016]_ as a way to improve the performance
# of sequential models (e.g., Transformers) or neural networks with small batch size.
# It takes a vector :math:`x` as input and produces a vector :math:`y` of the same shape as output.
# The normalization is performed by subtracting the mean and dividing by the standard deviation of :math:`x`.
# After the normalization, a learnable linear transformation with weights :math:`w` and biases :math:`b` is applied.
# The forward pass can be expressed as follows:
#
# .. math::
#    y = \frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} } * w + b
#
# where :math:`\epsilon` is a small constant added to the denominator for numerical stability.
# Let’s first take a look at the forward pass implementation.

import torch
import torch_npu

import triton
import triton.language as tl

import time

HAS_APEX = False
DEVICE = "npu"


@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,
    M,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    XBLOCK_SIZE: tl.constexpr,
    RBLOCK_SIZE: tl.constexpr
):
    # Map the program id to the row of X and Y it should compute.
    row_begin = tl.program_id(0) * RBLOCK_SIZE
    row_idx = row_begin + tl.arange(0,RBLOCK_SIZE)
    row_mask = row_idx < M
    row_offsets = row_idx[:,None]*stride
    # Compute mean

    _mean = tl.zeros((RBLOCK_SIZE, XBLOCK_SIZE), dtype=tl.float32)
    for off in range(0, N, XBLOCK_SIZE):
        col_idx = off + tl.arange(0, XBLOCK_SIZE)
        col_mask = col_idx < N
        mask = row_mask[:,None] & col_mask[None,:]
        a = tl.load(X + row_offsets + col_idx[None,:], mask=mask, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1, keep_dims = True) / N

    # Compute variance
    _var = tl.zeros((RBLOCK_SIZE, XBLOCK_SIZE), dtype=tl.float32)
    for off in range(0, N, XBLOCK_SIZE):
        col_idx = off + tl.arange(0, XBLOCK_SIZE)
        col_mask = col_idx < N
        mask = row_mask[:,None] & col_mask[None,:]
        x = tl.load(X + row_offsets + col_idx[None,:], mask=mask, other=0.).to(tl.float32)
        x = tl.where(mask, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=1, keep_dims=True) / N

    rstd = 1 / tl.sqrt(var + eps)
    
    # Write mean / rstd
    tl.store(Mean + row_idx[:,None], mean, mask = row_mask[:,None])
    tl.store(Rstd + row_idx[:,None], rstd, mask = row_mask[:,None])
    # mean = mean.broadcast_to((RBLOCK_SIZE, XBLOCK_SIZE))
    # rstd = rstd.broadcast_to((RBLOCK_SIZE, XBLOCK_SIZE))
    # Normalize and apply linear transformation
    for off in range(0, N, XBLOCK_SIZE):
        col_idx = off + tl.arange(0, XBLOCK_SIZE)
        col_mask = col_idx < N
        mask = row_mask[:,None] & col_mask[None,:]
        w = tl.load(W + col_idx, mask=col_mask).reshape((1,XBLOCK_SIZE))
        b = tl.load(B + col_idx, mask=col_mask).reshape((1,XBLOCK_SIZE))
        x = tl.load(X + row_offsets + col_idx[None,:], mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + row_offsets + col_idx[None,:], y, mask=mask)


# %%
# Backward pass
# -------------
#
# The backward pass for the layer normalization operator is a bit more involved than the forward pass.
# Let :math:`\hat{x}` be the normalized inputs :math:`\frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} }` before the linear transformation,
# the Vector-Jacobian Products (VJP) :math:`\nabla_{x}` of :math:`x` are given by:
#
# .. math::
#    \nabla_{x} = \frac{1}{\sigma}\Big( \nabla_{y} \odot w - \underbrace{ \big( \frac{1}{N} \hat{x} \cdot (\nabla_{y} \odot w) \big) }_{c_1} \odot \hat{x} - \underbrace{ \frac{1}{N} \nabla_{y} \cdot w }_{c_2} \Big)
#
# where :math:`\odot` denotes the element-wise multiplication, :math:`\cdot` denotes the dot product, and :math:`\sigma` is the standard deviation.
# :math:`c_1` and :math:`c_2` are intermediate constants that improve the readability of the following implementation.
#
# For the weights :math:`w` and biases :math:`b`, the VJPs :math:`\nabla_{w}` and :math:`\nabla_{b}` are more straightforward:
#
# .. math::
#    \nabla_{w} = \nabla_{y} \odot \hat{x} \quad \text{and} \quad \nabla_{b} = \nabla_{y}
#
# Since the same weights :math:`w` and biases :math:`b` are used for all rows in the same batch, their gradients need to sum up.
# To perform this step efficiently, we use a parallel reduction strategy: each kernel instance accumulates
# partial :math:`\nabla_{w}` and :math:`\nabla_{b}` across certain rows into one of :math:`\text{GROUP_SIZE_M}` independent buffers.
# These buffers stay in the L2 cache and then are further reduced by another function to compute the actual :math:`\nabla_{w}` and :math:`\nabla_{b}`.
#
# Let the number of input rows :math:`M = 4` and :math:`\text{GROUP_SIZE_M} = 2`,
# here's a diagram of the parallel reduction strategy for :math:`\nabla_{w}` (:math:`\nabla_{b}` is omitted for brevity):
#
#   .. image:: parallel_reduction.png
#
# In Stage 1, the rows of X that have the same color share the same buffer and thus a lock is used to ensure that only one kernel instance writes to the buffer at a time.
# In Stage 2, the buffers are further reduced to compute the final :math:`\nabla_{w}` and :math:`\nabla_{b}`.
# In the following implementation, Stage 1 is implemented by the function :code:`_layer_norm_bwd_dx_fused` and Stage 2 is implemented by the function :code:`_layer_norm_bwd_dwdb`.


@triton.jit
def _layer_norm_bwd_dx_fused(DX,  # pointer to the input gradient
                             DY,  # pointer to the output gradient
                             DW,  # pointer to the partial sum of weights gradient
                             DB,  # pointer to the partial sum of biases gradient
                             X,  # pointer to the input
                             W,  # pointer to the weights
                             Mean,  # pointer to the mean
                             Rstd,  # pointer to the 1/std
                             Lock,  # pointer to the lock
                             stride,  # how much to increase the pointer when moving by 1 row
                             N,  # number of columns in X
                             GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    # Release the lock
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient
                         DB,  # pointer to the partial sum of biases gradient
                         FINAL_DW,  # pointer to the weights gradient
                         FINAL_DB,  # pointer to the biases gradient
                         M,  # GROUP_SIZE_M
                         N,  # number of columns
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


# %%
# Benchmark
# ---------
#
# We can now compare the performance of our kernel against that of PyTorch.
# Here we focus on inputs that have Less than 64KB per feature.
# Specifically, one can set :code:`'mode': 'backward'` to benchmark the backward pass.


device = torch.npu.current_device()
stream = torch.npu.current_stream(device).npu_stream
kernels = {}

class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        # Less than 64KB per feature: enqueue fused kernel

        MAX_FUSED_SIZE = 65536 // x.element_size()
        XBLOCK_SIZE = 256
        RBLOCK_SIZE = 32
        NUM_CORE = (M -1) // RBLOCK_SIZE + 1
        num_warps = min(max((N - 1) // XBLOCK_SIZE + 1, 1), 8)
        # enqueue kernel

        kernel, num_programs = kernels.get(XBLOCK_SIZE^RBLOCK_SIZE, (None, NUM_CORE))
        if kernel is None:
            kernel = _layer_norm_fwd_fused.warmup( x_arg, y, weight, bias, mean, rstd,  #
                                            x_arg.stride(0), N, M, eps,  #
                                            XBLOCK_SIZE = XBLOCK_SIZE,
                                            RBLOCK_SIZE = RBLOCK_SIZE,
                                            grid=(NUM_CORE,))
            kernel._init_handles()
            kernels[XBLOCK_SIZE^RBLOCK_SIZE] = (kernel, num_programs)

        kernel[(num_programs,1,1 )](  #
            x_arg, y, weight, bias, mean, rstd,  #
            x_arg.stride(0), N, M, eps,  #
            stream=stream,
        )

        # _layer_norm_fwd_fused[(NUM_CORE, )](  #
        #     x_arg, y, weight, bias, mean, rstd,  #
        #     x_arg.stride(0), N, M, eps,  #
        #     XBLOCK_SIZE = XBLOCK_SIZE,
        #     RBLOCK_SIZE = RBLOCK_SIZE,
        #     num_warps=num_warps,
        #     num_ctas=1)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        # ctx.BLOCK_SIZE = XBLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors
        # heuristics for amount of parallel reduction stream for DW/DB
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256
        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[(M, )](  #
            dx, dy, _dw, _db, x, w, m, v, locks,  #
            x_arg.stride(0), N,  #
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
            GROUP_SIZE_M=GROUP_SIZE_M,  #
            num_warps=ctx.num_warps)
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        # accumulate partial sums in separate kernel
        _layer_norm_bwd_dwdb[grid](
            _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,  #
            BLOCK_SIZE_M=32,  #
            BLOCK_SIZE_N=128, num_ctas=1)
        return dx, None, dw, db, None


layer_norm = LayerNorm.apply


def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)

    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(20, 30)],
        line_arg='provider',
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
        line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 3072, 'dtype': torch.float16, 'mode': 'forward'}, # 4096 better
    ))
def bench_layer_norm(M, N, dtype, provider, mode='forward', eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            return layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "apex":
            apex_layer_norm = (apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype))
            return apex_layer_norm(x)  # noqa: F811, E704

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: ms*1000
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: ms*1000
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

def benchmark_test(fn, fn_triton, args =(), name="gen_fn", times=100, repeat=10):
    print(f"--------------------benchmark_{name} for {times * repeat} times--------------------")
    stream = torch.npu.current_stream()
    # warm_up
    stream.synchronize()
    for _ in range(10) :
        fn_triton(*args)
    stream.synchronize()

    start = time.perf_counter()
    for _ in range(times * repeat) :
        fn_triton(*args)
    stream.synchronize()
    end = time.perf_counter()

    time_compiled = (end - start) / (times * repeat)
    time_compiled *= 1000000
    print(f"time_triton:{time_compiled:.6f}")


    print(f"Runing eager {name} for {times * repeat} times")
    
    # warm_up
    stream.synchronize()
    for _ in range(10) :
        std = fn(*args)
    stream.synchronize()

    start = time.perf_counter()
    for _ in range(times * repeat) :
        std = fn(*args)
    stream.synchronize()
    end = time.perf_counter()
    time_eager = (end - start) / (times * repeat)
    time_eager *= 1000000
    print(f"time_eager:{time_eager:.6f}")

    accelerated = (time_eager - time_compiled)/time_compiled*100
    print(f"Accelerated: {accelerated:.4f}% eager takes {time_eager:.3f} us, triton takes {time_compiled:.3f} us")

    return accelerated, time_eager, time_compiled

test_layer_norm(1151, 8192, torch.float16)

M = 2048
N = 8192 # 12288 12800  13312 13000
x_shape = (M, N)
w_shape = (x_shape[-1], )
weight = torch.rand(w_shape, dtype=torch.float16, device='npu', requires_grad=True)
bias = torch.rand(w_shape, dtype=torch.float16, device='npu', requires_grad=True)
x = -2.3 + 0.5 * torch.randn(x_shape, dtype=torch.float16, device='npu')
eps = 1e-5
benchmark_test(torch.nn.functional.layer_norm,layer_norm,args=(x, w_shape, weight, bias, eps))

# %%
# References
# ----------
#
# .. [BA2016] Jimmy Lei Ba and Jamie Ryan Kiros and Geoffrey E. Hinton, "Layer Normalization", Arxiv 2016
