import math
import os
import statistics
import sys
from typing import List, Optional

TRITON_MAX_TENSOR_NUMEL = sys.maxsize


def parse_xpu_arch(raw: Optional[str], default: Optional[int] = None) -> Optional[int]:
    """Parse an XPU architecture value, returning the default when invalid."""
    if raw is None or not raw.isascii() or not raw.isdigit():
        return default
    try:
        return int(raw, 10)
    except ValueError:
        return default


def get_xpu_arch(default: int = 3) -> int:
    """Return ``TRITON_XPU_ARCH`` as an integer, or ``default`` when invalid."""
    return parse_xpu_arch(os.environ.get("TRITON_XPU_ARCH"), default)


def validate_block_shape(shape: List[int]):
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d)}]")
        # XPU accepts non-power-of-two block-shape dimensions.
        numel *= d

    if numel > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"numel ({numel}) exceeds triton maximum tensor numel ({TRITON_MAX_TENSOR_NUMEL})")
    return numel


from .language import language_extend_globals


def _utils_extend_globals(globals_dict):
    # XPU's prebuilt Python modules import these helpers from ``triton._utils``.
    # Publish them only for an active XPU build.
    globals_dict["TRITON_MAX_TENSOR_NUMEL"] = TRITON_MAX_TENSOR_NUMEL
    globals_dict["get_xpu_arch"] = get_xpu_arch
    globals_dict["parse_xpu_arch"] = parse_xpu_arch


def knobs_extend_globals(globals_dict):
    # Keep the shared knobs and their scope/reset behaviour; only XPU defaults
    # and the XPU launch fast-path switch differ.
    env_bool = globals_dict["env_bool"]
    compilation_knobs = globals_dict["compilation_knobs"]
    store_binary_only = env_bool("TRITON_STORE_BINARY_ONLY", default=True)
    store_binary_only.__set_name__(compilation_knobs, "store_binary_only")
    compilation_knobs.store_binary_only = store_binary_only
    runtime_knobs = globals_dict["runtime_knobs"]
    runtime_knobs.disable_launch_fast_path = env_bool("TRITON_DISABLE_LAUNCH_FAST_PATH").get()


def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    """Benchmark a function using the active XPU runtime."""
    assert return_mode in ["min", "max", "mean", "median", "all"]

    import triton.runtime as runtime

    device_interface = runtime.driver.active.get_device_interface()
    fn()
    device_interface.synchronize()
    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    start_event = device_interface.Event(enable_timing=True)
    end_event = device_interface.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        runtime.driver.active.clear_cache(cache)
        fn()
    end_event.record()
    device_interface.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    if estimate_ms == 0:
        n_warmup = 1
        n_repeat = 1000
    else:
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))

    start_events = [device_interface.Event(enable_timing=True) for _ in range(n_repeat)]
    end_events = [device_interface.Event(enable_timing=True) for _ in range(n_repeat)]
    for _ in range(n_warmup):
        fn()
    for i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        runtime.driver.active.clear_cache(cache)
        start_events[i].record()
        fn()
        end_events[i].record()

    device_interface.synchronize()
    times = [start.elapsed_time(end) for start, end in zip(start_events, end_events)]
    return _summarize_statistics(times, quantiles, return_mode)


def _quantile(values, quantiles):
    values = sorted(values)

    def get_quantile(q):
        if not 0 <= q <= 1:
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (len(values) - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * values[lower] + t * values[upper]

    return [get_quantile(q) for q in quantiles]


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        result = _quantile(times, quantiles)
        return result[0] if len(result) == 1 else result
    if return_mode == "all":
        return times
    if return_mode == "min":
        return min(times)
    if return_mode == "max":
        return max(times)
    if return_mode == "mean":
        return statistics.mean(times)
    if return_mode == "median":
        return statistics.median(times)
