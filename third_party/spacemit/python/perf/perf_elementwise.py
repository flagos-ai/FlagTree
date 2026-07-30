import time
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
from triton.language.extra.cpu import libdevice as tl_extra_shim

# ============================================================================
# Elementwise Math Function Kernels
# ============================================================================


@triton.jit
def exp_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """Exponential function: exp(x)"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )

        x = tl.load(in_block_ptr, boundary_check=(0, ))
        y = tl_extra_shim.exp(x)
        tl.store(out_block_ptr, y, boundary_check=(0, ))


@triton.jit
def cos_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """Cosine function: cos(x)"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )

        x = tl.load(in_block_ptr, boundary_check=(0, ))
        y = tl_extra_shim.cos(x)
        tl.store(out_block_ptr, y, boundary_check=(0, ))


@triton.jit
def sin_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """Sine function: sin(x)"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )

        x = tl.load(in_block_ptr, boundary_check=(0, ))
        y = tl_extra_shim.sin(x)
        tl.store(out_block_ptr, y, boundary_check=(0, ))


@triton.jit
def tanh_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """Hyperbolic tangent function: tanh(x)"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )

        x = tl.load(in_block_ptr, boundary_check=(0, ))
        y = tl_extra_shim.tanh(x)
        tl.store(out_block_ptr, y, boundary_check=(0, ))


@triton.jit
def erf_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """Error function: erf(x)"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )

        x = tl.load(in_block_ptr, boundary_check=(0, ))
        y = tl_extra_shim.erf(x)
        tl.store(out_block_ptr, y, boundary_check=(0, ))


# ============================================================================
# Activation Function Kernels
# ============================================================================


@triton.jit
def silu_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """SiLU: x * sigmoid(x)"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )

        x = tl.load(in_block_ptr, boundary_check=(0, ))
        y = tl_extra_shim.silu(x)
        tl.store(out_block_ptr, y, boundary_check=(0, ))


@triton.jit
def relu_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """ReLU: max(0, x)"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )

        x = tl.load(in_block_ptr, boundary_check=(0, ))

        # ReLU = max(0, x)
        y = tl.where(x > 0, x, 0)

        tl.store(out_block_ptr, y, boundary_check=(0, ))


@triton.jit
def gelu_none_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """GeLU using erf: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )

        x = tl.load(in_block_ptr, boundary_check=(0, ))
        y = tl_extra_shim.gelu_none(x)
        tl.store(out_block_ptr, y, boundary_check=(0, ))


@triton.jit
def gelu_tanh_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """GeLU using tanh approximation"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements, ),
            strides=(1, ),
            offsets=(block_start, ),
            block_shape=(BLOCK_SIZE, ),
            order=(0, ),
        )

        x = tl.load(in_block_ptr, boundary_check=(0, ))
        y = tl_extra_shim.gelu_tanh(x)

        tl.store(out_block_ptr, y, boundary_check=(0, ))


# ============================================================================
# Performance Benchmark Functions
# ============================================================================

ELEMENTWISE_TUNING_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 64, "num_ctas": 8}, num_warps=1),
    triton.Config({"BLOCK_SIZE": 128, "num_ctas": 8}, num_warps=1),
    triton.Config({"BLOCK_SIZE": 128, "num_ctas": 16}, num_warps=1),
    triton.Config({"BLOCK_SIZE": 256, "num_ctas": 8}, num_warps=1),
    triton.Config({"BLOCK_SIZE": 256, "num_ctas": 16}, num_warps=1),
]


def _best_of_repeats(run_once, num_warmup=10, num_iterations=100, num_repeats=3):
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


def _launch_elementwise_kernel(kernel, x, output, n_elements, config):
    meta = config.kwargs
    kernel[(meta["num_ctas"], )](
        x,
        output,
        n_elements,
        BLOCK_SIZE=meta["BLOCK_SIZE"],
        num_ctas=meta["num_ctas"],
    )


def tune_best_elementwise_config(
    kernel,
    x,
    configs=ELEMENTWISE_TUNING_CONFIGS,
    num_iterations=100,
    num_warmup=10,
    num_repeats=3,
):
    output = torch.empty_like(x)
    n_elements = x.numel()
    best_time_ms = float("inf")
    best_config = None

    for config in configs:
        try:
            config_time_ms = _best_of_repeats(
                lambda: _launch_elementwise_kernel(kernel, x, output, n_elements, config),
                num_warmup=num_warmup,
                num_iterations=num_iterations,
                num_repeats=num_repeats,
            )
        except Exception:
            continue

        if config_time_ms < best_time_ms:
            best_time_ms = config_time_ms
            best_config = config

    if best_config is None:
        raise RuntimeError("No valid Triton config found for the current input.")

    return best_config, best_time_ms


def benchmark_triton_kernel(
    kernel,
    x,
    num_iterations=100,
    num_warmup=10,
    num_repeats=3,
    configs=ELEMENTWISE_TUNING_CONFIGS,
):
    """Tune Triton configs and return best measured latency."""
    output = torch.empty_like(x)
    n_elements = x.numel()

    best_config, _ = tune_best_elementwise_config(
        kernel,
        x,
        configs=configs,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
        num_repeats=num_repeats,
    )
    best_time_ms = _best_of_repeats(
        lambda: _launch_elementwise_kernel(kernel, x, output, n_elements, best_config),
        num_warmup=num_warmup,
        num_iterations=num_iterations,
        num_repeats=num_repeats,
    )

    return best_time_ms, best_config


def validate_kernel(kernel, torch_func, x, num_ctas=16, atol=1e-4, rtol=1e-4):
    """Validate Triton kernel output against PyTorch reference"""
    output_triton = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 32

    kernel[(num_ctas, )](x, output_triton, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_ctas=num_ctas)
    output_torch = torch_func(x)

    is_close = torch.allclose(output_triton, output_torch, atol=atol, rtol=rtol)
    max_diff = torch.max(torch.abs(output_triton - output_torch)).item()

    return is_close, max_diff


if __name__ == "__main__":
    test_warm_up = 5
    test_iterations = 100
    test_repeats = 3
    validate_num_ctas = 16

    test_shape_list = [(1024, 1024), (512, 512), (256, 256), (128, 128), (32, 32)]
    test_dtype_list = [torch.float32, torch.float16]

    # (kernel, torch_func, atol, rtol)
    test_ops = {
        # Math functions
        "exp": (exp_kernel, torch.exp, 1e-4, 1e-4),
        "cos": (cos_kernel, torch.cos, 1e-4, 1e-4),
        "sin": (sin_kernel, torch.sin, 1e-4, 1e-4),
        "tanh": (tanh_kernel, torch.tanh, 1e-4, 1e-4),
        "erf": (erf_kernel, torch.erf, 1e-4, 1e-4),
        # Activation functions
        "silu": (silu_kernel, torch.nn.functional.silu, 1e-4, 1e-4),
        "relu": (relu_kernel, torch.nn.functional.relu, 1e-4, 1e-4),
        "gelu_none": (gelu_none_kernel, lambda x: torch.nn.functional.gelu(x, approximate="none"), 1e-4, 1e-4),
        "gelu_tanh": (gelu_tanh_kernel, lambda x: torch.nn.functional.gelu(x, approximate="tanh"), 1e-3, 1e-3),
    }

    # ====================================================================
    # Phase 1: Correctness Validation
    # ====================================================================
    print("=" * 80)
    print("Phase 1: Correctness Validation")
    print("=" * 80)

    for op_name, (triton_kernel, torch_func, atol, rtol) in test_ops.items():
        for test_dtype in test_dtype_list:
            x = torch.randn(test_shape_list[0], dtype=test_dtype, device="cpu")
            is_correct, max_diff = validate_kernel(
                triton_kernel,
                torch_func,
                x,
                validate_num_ctas,
                atol,
                rtol,
            )
            status = "✅ PASS" if is_correct else f"❌ FAIL (max_diff={max_diff:.2e})"
            print(f"  {op_name:12} | {str(test_dtype):15} | {status}")

    # ====================================================================
    # Phase 2: Triton Performance
    # ====================================================================
    print()
    print("=" * 80)
    print("Phase 2: Triton Kernel Performance")
    print("=" * 80)

    for test_dtype in test_dtype_list:
        print(f"\n  dtype: {test_dtype}")
        # Table header
        header = f"  {'Op':12}"
        for shape in test_shape_list:
            header += f" | {str(shape):>22}"
        print(f"  {'-' * (16 + 25 * len(test_shape_list))}")
        print(header)
        print(f"  {'-' * (16 + 25 * len(test_shape_list))}")

        for op_name, (triton_kernel, torch_func, atol, rtol) in test_ops.items():
            row = f"  {op_name:12}"
            for test_shape in test_shape_list:
                x = torch.randn(test_shape, dtype=test_dtype, device="cpu")
                triton_time, best_config = benchmark_triton_kernel(
                    triton_kernel,
                    x,
                    num_iterations=test_iterations,
                    num_warmup=test_warm_up,
                    num_repeats=test_repeats,
                )
                n_elements = x.numel()
                throughput = n_elements / (triton_time / 1000)  # elements/sec
                config_str = f"bs={best_config.kwargs['BLOCK_SIZE']},cta={best_config.kwargs['num_ctas']}"
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
