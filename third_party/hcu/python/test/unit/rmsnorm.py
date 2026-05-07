"""
RMS Normalization
"""

import math
import argparse
import sys
import pytest
from itertools import product

import torch
import triton
import triton.language as tl

torch_to_triton_dtype = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def get_num_sms():
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    num_sms = current_device.multi_processor_count
    return num_sms


def get_hip_autotune_config():
    return [triton.Config({'waves_per_eu': we}, num_warps=nw) for (we, nw) in product([0, 1, 2, 4], [4, 8, 16])]


# @triton.autotune(configs=get_hip_autotune_config(), key=['n_rows', 'n_cols'])
@triton.jit
def _rms_norm_fwd_rocm(Y_ptr, Y_row_stride, X_ptr, X_row_stride, W_ptr, RSTD_ptr, n_rows, n_cols, eps,
                       BLOCK_SIZE: tl.constexpr, USE_BLOCKED: tl.constexpr, NUM_PRGMS: tl.constexpr):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # as older version Triton doesn't support tl.assume and BUFF OPS, comment out for now
    tl.assume(X_row_stride >= 0)
    tl.assume(Y_row_stride >= 0)
    tl.assume(row_start >= 0)

    if USE_BLOCKED:

        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = X_ptr + row_idx * X_row_stride
            row_output_ptr = Y_ptr + row_idx * Y_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            # older version of triton doesn't accept below init
            # sum_squares: tl.float32 = 0.
            # however, with type promoting rule in triton, sum_squares should be always fp32 with below init
            sum_squares = 0.
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                x = tl.load(input_ptrs).to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16, ))
            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            sum_squares += tl.sum(x * x, axis=0)

            # Compute normalization factor
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + eps)

            # Store rstd (norm_factor)
            tl.store(RSTD_ptr + row_idx, norm_factor)

            # Normalize and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                x = tl.load(input_ptrs).to(tl.float32)
                W_ptrs = W_ptr + cols
                w = tl.load(W_ptrs).to(tl.float32)
                rms_norm = x * norm_factor * w
                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, rms_norm.to(Y_ptr.type.element_ty))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            W_ptrs = W_ptr + cols
            w = tl.load(W_ptrs, mask=mask, other=0.0).to(tl.float32)
            rms_norm = x * norm_factor * w
            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, rms_norm.to(Y_ptr.type.element_ty), mask=mask)

    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = X_ptr + row_idx * X_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16, ))
            row = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.rsqrt((row_norm / n_cols) + eps)

            # Store rstd (norm_factor)
            rstd_output_ptr = RSTD_ptr + row_idx
            tl.store(rstd_output_ptr, norm_factor)

            rms_norm = row * norm_factor * w

            output_ptrs = Y_ptr + row_idx * Y_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16, ))
            tl.store(output_ptrs, rms_norm.to(Y_ptr.type.element_ty), mask=mask)


# @triton.autotune(configs=get_hip_autotune_config(), key=['n_rows', 'n_cols'])
@triton.jit
def _rms_norm_bwd(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    RSTD_ptr,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    dx = (1 / RMS) * [dy * (w + offset) - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over M dimension
    """

    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dW_row = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)

    dY_ptr += row_start * dY_row_stride
    dX_ptr += row_start * dX_row_stride

    X_ptr += row_start * X_row_stride
    RSTD_ptr += row_start

    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    W_row = W_row + offset

    for row_idx in range(row_start, row_end):
        dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)

        # Get cached rms
        rstd_row = tl.load(RSTD_ptr)
        X_row = X_row.to(tl.float32)
        m = dY_row * W_row
        dX_row = rstd_row * m
        dX_row += (rstd_row) * (-(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row)

        # here X_row is already in fp32 (see previous if block)
        dW_row += dY_row * (X_row * rstd_row)

        tl.store(dX_ptr + col_offsets, dX_row.to(X_dtype), mask=mask)

        dY_ptr += dY_row_stride
        dX_ptr += dX_row_stride
        X_ptr += X_row_stride
        RSTD_ptr += 1

    tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)


class _RMSNormFunction(torch.autograd.Function):
    """
    Performs RMSNorm (Root Mean Square Normalization), which normalizes the input tensor `X` using the
    weight tensor `W`, with an optional offset.

    Some models use an 'offset' to shift the weight tensor `W` by a constant value. For example, Gemma
    uses an offset of 1.0, so the computation becomes `(X / RMS(X)) * (W + 1.0)` instead of the usual
    `(X / RMS(X)) * W`. You can pass the offset value as an argument to the forward function.

    `in_place` option means whether to in_place modify dY to store dX. This is default to `True` to save memory. However, under certain cases, it can produce incorrect inputs.
        For example, gemma2 uses two rmsnorm sequentially with residual in between. The resesidual part needs dY so it cannot be modified in-place.
        Therefore, for the patching of RMSNorm in gemma2, we set `in_place` to `False`
    """

    @staticmethod
    def forward(ctx, X, W, eps, offset=0.0, out_dtype=torch.float16):
        """
        X: (M, N)
        W: (N,)
        """
        n_rows, n_cols = X.shape
        MAX_FUSED_SIZE = 65536 // X.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))

        Y = torch.zeros_like(X, device=X.device, dtype=out_dtype, requires_grad=True)
        RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

        # Check constraints.
        assert (X.shape[1] == W.shape[0]
                ), "Incompatible hidden size dimension between tensor1.shape[1] and tensor2.shape[0]"

        assert (offset == 0.0), "rocm triton don't support offset"
        NUM_PRGMS = min(n_rows, get_num_sms())
        USE_BLOCKED = n_cols > BLOCK_SIZE
        grid = lambda meta: (NUM_PRGMS, )
        _rms_norm_fwd_rocm[grid](Y, Y.stride(0), X, X.stride(0), W, RSTD, n_rows, n_cols, eps, BLOCK_SIZE, USE_BLOCKED,
                                 NUM_PRGMS)
        ctx.offset = offset
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.save_for_backward(X, W, RSTD)
        return Y

    @staticmethod
    def backward(ctx, dY):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        """
        Y: (M, N)
        """
        X, W, RSTD = ctx.saved_tensors
        offset, BLOCK_SIZE = ctx.offset, ctx.BLOCK_SIZE

        n_rows, n_cols = dY.shape

        sm_count = get_num_sms()

        # fp32 for numerical stability especially.
        _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)

        if n_cols > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        rows_per_program = math.ceil(n_rows / sm_count)
        grid = (sm_count, )

        dX = torch.zeros_like(dY)

        _rms_norm_bwd[grid](dY, dY.stride(0), dX, dX.stride(0), X, X.stride(0), torch_to_triton_dtype[X.dtype], W, RSTD,
                            _dW, _dW.stride(0), n_rows, n_cols, offset, rows_per_program, BLOCK_SIZE=BLOCK_SIZE)
        dW = _dW.sum(dim=0).to(W.dtype)

        return dX, dW, None, None, None


triton_rmsnorm = _RMSNormFunction.apply

arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def torch_rmsnorm(x, w, epsilon=1e-6, offset=0.0, out_dtype=torch.float16):
    M, N = x.shape
    # cast to float32 as the triton kernel
    x_f32 = x.float()
    w_f32 = w.float()
    rms = torch.sqrt(torch.sum(x_f32 * x_f32, dim=-1) * 1 / N + epsilon)
    rsigma = 1.0 / rms
    rms_norm_f32 = x_f32 * rsigma.unsqueeze(1) * (w_f32 + offset)
    rms_norm = rms_norm_f32.to(out_dtype)
    return rms_norm, rsigma


@pytest.mark.parametrize("out_dtype_str", ["fp16"])
@pytest.mark.parametrize("in_dtype_str", ["fp16"])
@pytest.mark.parametrize('M, N', [
    (1, 4),
    (4, 8),
    (2, 10),
    (63, 41),
    (256, 512),
    (615, 123),
    (1, 16384),
    (873, 1245),
    (8192, 4096),
    (4096, 8192),
])
def test_rmsnorm(M, N, in_dtype_str, out_dtype_str):
    in_dtype = arg_to_torch_dtype[in_dtype_str]
    out_dtype = arg_to_torch_dtype[out_dtype_str]
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda', dtype=in_dtype, requires_grad=True)
    w = torch.ones(N, device='cuda', dtype=in_dtype, requires_grad=True)
    dy = torch.randn(M, N, device='cuda', dtype=out_dtype)
    esp = 1e-6
    offset = 0.0
    ref_y, ref_rsigma = torch_rmsnorm(x, w, esp, offset, out_dtype)
    ref_y.backward(dy)
    ref_dx, x.grad = x.grad.clone(), None  # None is to clear gradient for next step
    ref_dw, w.grad = w.grad.clone(), None
    triton_y = triton_rmsnorm(x, w, esp, offset, out_dtype)
    triton_y.backward(dy)
    triton_dx, x.grad = x.grad.clone(), None
    triton_dw, w.grad = w.grad.clone(), None
    if out_dtype in (torch.float16, torch.bfloat16) or \
       in_dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-3, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-5, 1e-3

    assert triton_y.dtype == out_dtype, f"triton_y has dtype={triton_y.dtype}, expected {out_dtype}"
    assert ref_y.dtype == out_dtype, f"ref_y has dtype={ref_y.dtype}, expected {out_dtype}"

    torch.testing.assert_close(triton_y, ref_y, atol=atol, rtol=rtol)
    torch.testing.assert_close(triton_dx, ref_dx, atol=atol, rtol=rtol)
    torch.testing.assert_close(triton_dw, ref_dw, atol=atol, rtol=rtol)


#Benchmark
def run_benchmark(args):
    config = []
    for mode in ['fwd', 'bwd']:
        if (args.M_benchmark):
            val = args.M_start
            x_vals_list = []
            while val <= args.M_end:
                x_vals_list.append(val)
                val *= args.M_step
            mn_args = {'N': args.N_start}
            plot_name = str("rmsnorm-performance_" + args.dtype + "_" + mode + "_N" + str(args.N_start) + "_M" +
                            str(args.M_start) + "-" + str(args.M_end) + "-" + str(args.M_step))
            x_names = ['M']
        else:
            x_vals_list = [i for i in range(args.N_start, args.N_end, args.N_step)]
            mn_args = {'M': args.M_start}
            x_names = ['N']
            plot_name = str("rmsnorm-performance_" + args.dtype + "_" + mode + "_M" + str(args.M_start) + "_N" +
                            str(args.N_start) + "-" + str(args.N_end) + "-" + str(args.N_step))
        _args = mn_args
        _args['mode'] = mode
        _args['dtype'] = arg_to_torch_dtype[args.dtype]
        config.append(
            triton.testing.Benchmark(x_names=x_names, x_vals=x_vals_list, line_arg='provider', line_vals=['triton'],
                                     line_names=["Triton"], styles=[('green', '-')], ylabel="GB/s", plot_name=plot_name,
                                     args=_args))

    @triton.testing.perf_report(config)
    def benchmark(M, N, provider, mode, dtype):
        x = torch.randn(M, N, device='cuda', dtype=dtype, requires_grad=True)
        w = torch.ones(N, device='cuda', dtype=dtype, requires_grad=True)
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        esp = 1e-6
        offset = 0.0
        if provider == 'torch':
            fn = lambda: torch.rms_norm(x, [N], w, esp)
            if mode == 'bwd':
                dy = torch.randn_like(x)
                y = fn()
                fn = lambda: y.backward(dy, retain_graph=True)
            ms = triton.testing.do_bench(fn)
        if provider == 'triton':
            fn = lambda: triton_rmsnorm(x, w, esp, offset, dtype)
            if mode == 'bwd':
                dy = torch.randn_like(x)
                y = fn()
                fn = lambda: y.backward(dy, retain_graph=True)
            ms = triton.testing.do_bench(fn)
            global verbose
            if verbose:
                if mode == 'fwd':
                    print(f'SIZE: {M, N} Forward best tuning config: ({_rms_norm_fwd_rocm.best_config})')
                    print(f'time: {ms}ms')
                if mode == 'bwd':
                    print(f'SIZE: {M, N} Backward best tuning config: ({_rms_norm_bwd.best_config})')
                    print(f'time: {ms} ms')
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark RMSNorm",
        allow_abbrev=False,
    )
    parser.add_argument('-M', "--M_start", default="1", type=int)
    parser.add_argument('-Ms', "--M_step", default="2", type=int)  #This is multiplicative step
    parser.add_argument('-Me', "--M_end", default="512", type=int)
    parser.add_argument('-Mb', "--M_benchmark", default=False, type=bool)

    parser.add_argument('-N', "--N_start", default="8192", type=int)
    parser.add_argument('-Ns', "--N_step", default="1024", type=int)
    parser.add_argument('-Ne', "--N_end", default="32768", type=int)

    parser.add_argument('-d', "--dtype", default="fp16")
    parser.add_argument("-v", action='store_true', default=False, help="Print out the best tuning config")

    return parser.parse_args()


def main():
    args = parse_args()
    global verbose
    verbose = args.v
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
