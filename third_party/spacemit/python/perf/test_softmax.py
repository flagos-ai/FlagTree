import time
import torch

import triton
import triton.language as tl
from triton.language.extra.cpu import libdevice as tl_extra_shim
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

SOFTMAX_TUNING_CONFIGS = [
    triton.Config({"ROW_SIZE": 1, "COL_SIZE": 32}, num_warps=1),
    triton.Config({"ROW_SIZE": 1, "COL_SIZE": 64}, num_warps=1),
    triton.Config({"ROW_SIZE": 2, "COL_SIZE": 64}, num_warps=1),
    triton.Config({"ROW_SIZE": 4, "COL_SIZE": 64}, num_warps=1),
    triton.Config({"ROW_SIZE": 1, "COL_SIZE": 128}, num_warps=1),
]


def _best_of_repeats(run_once, num_warmup=5, num_iterations=100, num_repeats=3):
    for _ in range(num_warmup):
        run_once()

    best_ms = float("inf")
    for _ in range(num_repeats):
        start = time.time()
        for _ in range(num_iterations):
            run_once()
        elapsed_ms = (time.time() - start) / num_iterations * 1000
        best_ms = min(best_ms, elapsed_ms)

    return best_ms


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, ROW_SIZE: tl.constexpr,
                   COL_SIZE: tl.constexpr):
    row_start = tl.program_id(0) * ROW_SIZE
    element_ty = input_ptr.type.element_ty

    for row_idx in range(row_start, row_start + ROW_SIZE):
        if row_idx < n_rows:
            denominator = tl.zeros((1, ), dtype=element_ty)
            row_max = tl.full((COL_SIZE, ), -float('inf'), dtype=element_ty)

            for col_idx in range(0, n_cols, COL_SIZE):
                input_block_ptr = tl.make_block_ptr(base=input_ptr + row_idx * input_row_stride, shape=(n_cols),
                                                    strides=(1, ), offsets=(col_idx, ), block_shape=(COL_SIZE, ),
                                                    order=(0, ))
                row = tl.load(input_block_ptr, boundary_check=(0, ), padding_option="neg_inf")
                row_max = tl.maximum(row, row_max)

            # tl.max promotes fp16/bf16 to fp32 in this frontend.
            # Cast back to element_ty so downstream ops match block element type.
            row_max_total = tl.max(row_max, axis=0).to(element_ty)

            for col_idx in range(0, n_cols, COL_SIZE):
                input_block_ptr = tl.make_block_ptr(base=input_ptr + row_idx * input_row_stride, shape=(n_cols),
                                                    strides=(1, ), offsets=(col_idx, ), block_shape=(COL_SIZE, ),
                                                    order=(0, ))
                output_block_ptr = tl.make_block_ptr(base=output_ptr + row_idx * output_row_stride, shape=(n_cols, ),
                                                     strides=(1, ), offsets=(col_idx, ), block_shape=(COL_SIZE, ),
                                                     order=(0, ))
                row = tl.load(input_block_ptr, boundary_check=(0, ), padding_option="neg_inf")
                row_minus_max = row - row_max_total
                numerator = tl_extra_shim.exp(row_minus_max).to(element_ty)
                denominator += tl.sum(numerator, axis=0)
                tl.store(output_block_ptr, numerator, boundary_check=(0, ))

            one = tl.full((1, ), 1, dtype=element_ty)
            inv_denom = one / denominator

            for col_idx in range(0, n_cols, COL_SIZE):
                output_block_ptr = tl.make_block_ptr(base=output_ptr + row_idx * output_row_stride, shape=(n_cols, ),
                                                     strides=(1, ), offsets=(col_idx, ), block_shape=(COL_SIZE, ),
                                                     order=(0, ))
                exp_out = tl.load(output_block_ptr, boundary_check=(0, ))
                softmax_output = exp_out * inv_denom
                tl.store(output_block_ptr, softmax_output, boundary_check=(0, ))


def _launch_softmax(x, y, row_size, col_size):
    n_rows, n_cols = x.shape

    def grid(META):
        return (triton.cdiv(n_rows, META["ROW_SIZE"]), )

    softmax_kernel[grid](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        ROW_SIZE=row_size,
        COL_SIZE=col_size,
    )


def tune_softmax_config(
    x,
    configs=SOFTMAX_TUNING_CONFIGS,
    num_warmup=5,
    num_iterations=100,
    num_repeats=3,
    verbose=False,
):
    best_time_ms = float("inf")
    best_config = None
    last_error = None

    for config in configs:
        row_size = config.kwargs["ROW_SIZE"]
        col_size = config.kwargs["COL_SIZE"]
        y = torch.empty_like(x)

        try:
            config_time_ms = _best_of_repeats(
                lambda: _launch_softmax(x, y, row_size=row_size, col_size=col_size),
                num_warmup=num_warmup,
                num_iterations=num_iterations,
                num_repeats=num_repeats,
            )
        except Exception as e:
            last_error = (config, e)
            if verbose:
                print(f"[tune_softmax_config] config failed: ROW_SIZE={row_size}, COL_SIZE={col_size}, "
                      f"dtype={x.dtype}, shape={tuple(x.shape)}, err={type(e).__name__}: {e}")
            continue

        if config_time_ms < best_time_ms:
            best_time_ms = config_time_ms
            best_config = config

    if best_config is None:
        if last_error is None:
            raise RuntimeError(f"No valid softmax Triton config found for shape={tuple(x.shape)}, dtype={x.dtype}.")
        failed_cfg, failed_exc = last_error
        raise RuntimeError("No valid softmax Triton config found for "
                           f"shape={tuple(x.shape)}, dtype={x.dtype}. "
                           f"Last error at ROW_SIZE={failed_cfg.kwargs['ROW_SIZE']}, "
                           f"COL_SIZE={failed_cfg.kwargs['COL_SIZE']}: "
                           f"{type(failed_exc).__name__}: {failed_exc}") from failed_exc

    return best_config, best_time_ms


def softmax(x, row_size=None, col_size=None):
    n_rows, _ = x.shape
    y = torch.empty_like(x)

    row_size = row_size if row_size is not None else min(4, n_rows)
    col_size = col_size if col_size is not None else 64

    _launch_softmax(x, y, row_size=row_size, col_size=col_size)

    return y


def benchmark_softmax(
    x,
    num_warmup=5,
    num_iterations=100,
    num_repeats=3,
    configs=SOFTMAX_TUNING_CONFIGS,
):
    best_config, _ = tune_softmax_config(
        x,
        configs=configs,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
        num_repeats=num_repeats,
    )

    y = torch.empty_like(x)
    best_time_ms = _best_of_repeats(
        lambda: _launch_softmax(
            x,
            y,
            row_size=best_config.kwargs["ROW_SIZE"],
            col_size=best_config.kwargs["COL_SIZE"],
        ),
        num_warmup=num_warmup,
        num_iterations=num_iterations,
        num_repeats=num_repeats,
    )

    return best_time_ms, best_config


if __name__ == "__main__":
    torch.manual_seed(0)

    test_shape_list = [(1024, 1024), (512, 512), (256, 256), (128, 128)]
    test_dtype_list = [torch.float32, torch.float16]
    test_warm_up = 5
    test_iterations = 100
    test_repeats = 3

    # ====================================================================
    # Phase 1: Correctness Validation
    # ====================================================================
    print("=" * 80)
    print("Phase 1: Correctness Validation")
    print("=" * 80)

    for test_dtype in test_dtype_list:
        for test_shape in test_shape_list:
            M, N = test_shape
            x = torch.randn(M, N, device='cpu', dtype=test_dtype)
            cfg, _ = tune_softmax_config(
                x,
                num_warmup=test_warm_up,
                num_iterations=max(10, test_iterations // 2),
                num_repeats=test_repeats,
            )
            y_triton = softmax(
                x,
                row_size=cfg.kwargs["ROW_SIZE"],
                col_size=cfg.kwargs["COL_SIZE"],
            )
            y_torch = torch.softmax(x, dim=1)
            is_correct = torch.allclose(y_triton, y_torch, atol=1e-2, rtol=1e-2)
            max_diff = torch.max(torch.abs(y_triton - y_torch)).item()
            status = "✅ PASS" if is_correct else f"❌ FAIL (max_diff={max_diff:.2e})"
            print(f"  softmax      | {str(test_dtype):15} | {str(test_shape):15} | {status}")

    # ====================================================================
    # Phase 2: Triton Performance
    # ====================================================================
    print()
    print("=" * 80)
    print("Phase 2: Triton Kernel Performance")
    print("=" * 80)

    for test_dtype in test_dtype_list:
        print(f"\n  dtype: {test_dtype}")
        header = f"  {'Op':12}"
        for shape in test_shape_list:
            header += f" | {str(shape):>22}"
        print(f"  {'-' * (16 + 25 * len(test_shape_list))}")
        print(header)
        print(f"  {'-' * (16 + 25 * len(test_shape_list))}")

        row = f"  {'softmax':12}"
        for test_shape in test_shape_list:
            M, N = test_shape
            x = torch.randn(M, N, device='cpu', dtype=test_dtype)
            triton_time, best_config = benchmark_softmax(
                x,
                num_warmup=test_warm_up,
                num_iterations=test_iterations,
                num_repeats=test_repeats,
            )

            n_elements = x.numel()
            throughput = n_elements / (triton_time / 1000)  # elements/sec
            config_str = f"r={best_config.kwargs['ROW_SIZE']},c={best_config.kwargs['COL_SIZE']}"
            if throughput >= 1e9:
                tp_str = f"{triton_time:.2f}ms {throughput/1e9:.2f}G/s {config_str}"
            elif throughput >= 1e6:
                tp_str = f"{triton_time:.2f}ms {throughput/1e6:.2f}M/s {config_str}"
            else:
                tp_str = f"{triton_time:.2f}ms {throughput:.0f}/s {config_str}"
            row += f" | {tp_str:>22}"
        print(row)

        print()

    print("=" * 80)
