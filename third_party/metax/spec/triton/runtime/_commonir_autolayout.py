"""CommonIR layout tuning around an unmodified JITFunction."""

import hashlib
import json
import logging
import math
import os
import statistics
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton import knobs
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction, KernelInterface, compute_cache_key
from triton.runtime._distributed import DistributedRtContext, validate_node_launch_bindings

_LOGGER = logging.getLogger(__name__)
# One elimination look after 16 samples, then refine survivors to 32.
_PROTOCOL = "commonir-autolayout-v5"
_WARMUP = 5
_COARSE_SAMPLES = 16
_TOTAL_SAMPLES = 32
_ELIMINATION_ALPHA = 0.01


def _require_quantile_test():
    try:
        from scipy.stats import quantile_test
    except ImportError as error:
        raise ImportError("CommonIR autolayout tuning requires scipy>=1.12; "
                          "install it with pip install 'scipy>=1.12'.") from error
    return quantile_test


def _median_intervals(samples, candidate_count):
    quantile_test = _require_quantile_test()
    # Bonferroni over ALL candidates, at ONE predeclared look. Under i.i.d.
    # timings, simultaneous coverage is >= 99%; this is not a guarantee that
    # the final empirical winner is optimal. No repeated/adaptive CI checks.
    confidence = 1 - _ELIMINATION_ALPHA / candidate_count
    intervals = {}
    for identity, values in samples.items():
        ci = quantile_test(values, p=0.5).confidence_interval(confidence)
        lower, upper = float(ci.low), float(ci.high)
        # Too few observations for the requested confidence: never prune.
        intervals[identity] = (lower, upper) if math.isfinite(lower) and math.isfinite(upper) else None
    return intervals


def _specialization_key(bundle):
    """Key the tuning result by JIT specialization and physical device.

    Ordinary JIT specialization already captures compilation assumptions.
    Runtime shapes, strides, scalar values and launch grids do not select
    additional layouts; the first nonempty launch supplies the tuning inputs.
    """
    import torch

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload = (_PROTOCOL, "jit-specialization-v1", bundle.hash, str(getattr(device, "uuid", device.name)), device.name,
               device.multi_processor_count)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _scratch_arguments(args):
    """Copy complete storages so strides, offsets and aliases survive replay."""
    import torch

    storages = {}

    def copy(value):
        if isinstance(value, torch.Tensor):
            if not value.is_cuda or value.layout != torch.strided:
                raise ValueError("autolayout requires strided GPU tensors")
            storage = value.untyped_storage()
            identity = (str(value.device), storage.data_ptr())
            if identity not in storages:
                raw = torch.empty(0, dtype=torch.uint8, device=value.device)
                raw = raw.set_(storage, 0, (storage.nbytes(), ), (1, ))
                original = raw.clone()
                storages[identity] = (original, original.clone())
            _, scratch = storages[identity]
            result = torch.empty(0, dtype=value.dtype, device=value.device)
            return result.set_(scratch.untyped_storage(), value.storage_offset(), value.shape, value.stride())
        if isinstance(value, tuple):
            return tuple(copy(item) for item in value)
        if isinstance(value, list):
            return [copy(item) for item in value]
        if hasattr(value, "data_ptr"):
            raise TypeError("autolayout replay requires torch.Tensor pointer arguments")
        return value

    copied = tuple(copy(arg) for arg in args)

    def reset():
        for original, scratch in storages.values():
            scratch.copy_(original)

    return copied, reset


def _benchmark(kernel, grid, stream, args, reset, *, warmup=_WARMUP, repetitions=_COARSE_SAMPLES):
    import torch
    from triton.runtime.driver import driver

    launch = kernel.run  # Load binary and build the launcher before timing.
    active = driver.active
    cache = active.get_empty_cache_for_benchmark()

    def run():
        launch(*grid, stream, kernel.function, kernel.packed_metadata, None, None, None, *args)

    for _ in range(warmup):
        reset()
        active.clear_cache(cache)
        run()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in starts]
    for start, end in zip(starts, ends):
        reset()
        # Scratch restoration touches inputs; flush afterward, outside timing.
        active.clear_cache(cache)
        start.record()
        run()
        end.record()
    ends[-1].synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    if any(not math.isfinite(elapsed) or elapsed <= 0 for elapsed in samples):
        raise RuntimeError("autolayout benchmark did not produce a positive finite GPU time")
    return samples


class LayoutTuner(KernelInterface):

    def __init__(self, fn):
        if knobs.runtime.interpret:
            raise ValueError("layout_tune requires GPU compilation")
        if not isinstance(fn, JITFunction) or fn.is_gluon():
            raise TypeError("layout_tune() must directly wrap @triton.jit")
        self.fn = fn
        self.arg_names = fn.arg_names
        self._layout_lock = threading.RLock()
        self._layout_kernels = {}
        self._compile_failures = {}
        self._bundles = defaultdict(dict)
        self._warmups = defaultdict(dict)
        self._notified = set()
        self._winners = defaultdict(dict)
        self.autolayout_results = {}

    def run(self, *args, grid, warmup, **kwargs):
        from triton._common_ir import ENABLED
        from triton._C.libtriton import get_cache_invalidating_env_vars
        from triton.backends.metax.autolayout import prepare_layout_candidates
        from triton.runtime import _async_compile
        from flagtree import _flagprism

        device = driver.active.get_current_device()
        _, key_cache, target, backend, binder = self.fn.device_caches[device]
        if target.backend != "maca" or not ENABLED:
            raise ValueError("layout_tune requires the MetaX CommonIR backend")
        kwargs["debug"] = kwargs.get("debug", self.fn.debug) or knobs.runtime.debug
        kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode
        _flagprism.apply_compile_options(kwargs)
        for hook in self.fn.pre_run_hooks:
            hook(*args, **kwargs)
        bound_args, specialization, raw_options = binder(*args, **kwargs)
        key = compute_cache_key(key_cache, specialization, raw_options)
        with self._layout_lock:
            bundle = self._bundles[device].get(key)
            if bundle is None:
                options, signature, constants, attrs = self.fn._pack_args(backend, kwargs, bound_args, specialization,
                                                                          raw_options)
                if self.fn._call_hook(knobs.runtime.jit_cache_hook, key, signature, device, constants, options, [attrs],
                                      True):
                    return None
                src = self.fn.ASTSource(self.fn, signature, constants, attrs)
                bundle = prepare_layout_candidates(src, backend, options, get_cache_invalidating_env_vars())
                self._bundles[device][key] = bundle

        missing = object()
        for (name, _), (value, globals_dict) in self.fn.used_global_vals.items():
            current = globals_dict.get(name, missing)
            if current != value:
                raise RuntimeError(f"Global variable {name} has changed since we compiled this kernel, "
                                   f"from {value} to {current}")
        winners = self._winners[device]
        if warmup:
            if key in winners:
                return winners[key]
            if key not in self._warmups[device]:

                def finalize(kernel):
                    self._warmups[device][key] = kernel
                    self._finish_compile(kernel, bundle, backend, device, key)

                mode = _async_compile.active_mode.get()
                if mode is not None:
                    result = mode.submit((id(self), device, bundle.hash), lambda: self._first_compiled(bundle),
                                         finalize)
                    self._warmups[device][key] = result
                else:
                    finalize(self._first_compiled(bundle))
            return self._warmups[device][key]

        pending = self._warmups[device].get(key)
        if hasattr(pending, "result"):
            pending.result()
        assert grid is not None
        if callable(grid):
            grid = grid(bound_args)
        launch_grid = (grid[0], grid[1] if len(grid) > 1 else 1, grid[2] if len(grid) > 2 else 1)
        stream = driver.active.get_current_stream(device)
        kernel = self._select_kernel(bundle, launch_grid, stream, bound_args, kernel_cache=winners, key=key)
        self._finish_compile(kernel, bundle, backend, device, key)
        validate_node_launch_bindings(kernel, bound_args, specialization)
        launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
        ctx = DistributedRtContext()
        dist_params = [ctx.comm_ptr, ctx.mem_ptr] if ctx.is_lite_mode else []
        kernel.run(*launch_grid, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                   knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *dist_params, *bound_args.values())
        return kernel

    def _finish_compile(self, kernel, bundle, backend, device, key):
        if (device, key) not in self._notified:
            if getattr(kernel.metadata, "kernel_init_hooks", ()):
                from triton.experimental.tle.raw.cuda.runtime import run_kernel_init_hooks
                run_kernel_init_hooks(kernel)
            self.fn._call_hook(knobs.runtime.jit_post_compile_hook, key, bundle.src.signature, device,
                               bundle.src.constants, backend.parse_options(bundle.options), [bundle.src.attrs], True)
            self._notified.add((device, key))

    def _compile_variant(self, bundle, variant):
        # Bundles are per device; loaded kernel handles must not cross devices.
        identity = (id(bundle), variant["id"])
        if identity in self._layout_kernels:
            return self._layout_kernels[identity]
        if identity in self._compile_failures:
            raise RuntimeError(self._compile_failures[identity])
        try:
            cache = get_cache_manager(bundle.hash)
            path = cache.get_file(variant["file"])
            if path is None or hashlib.sha256(Path(path).read_bytes()).hexdigest() != variant["id"]:
                raise RuntimeError(f"missing or changed CommonIR candidate source: {variant['file']}")
            kernel = self.fn.compile(path, target=bundle.target, options=bundle.options, _env_vars=bundle.env_vars)
            if list(kernel.src.signature.values()) != bundle.manifest["signature"]:
                raise RuntimeError("CommonIR candidate changed the device argument signature")
            # IRSource omits constexpr host arguments. Restore the AST launch ABI.
            kernel.src = bundle.src
        except Exception as error:
            self._compile_failures[identity] = str(error)
            raise
        self._layout_kernels[identity] = kernel
        return kernel

    def _first_compiled(self, bundle, *, load=False):
        failures = {}
        for variant in bundle.manifest["variants"]:
            try:
                kernel = self._compile_variant(bundle, variant)
                if load:
                    kernel._init_handles()
                return kernel
            except Exception as error:
                failures[variant["id"]] = f"compilation/loading: {error}"
                _LOGGER.warning("CommonIR candidate %s failed compilation/loading: %s", variant["id"], error)
        self._raise_all_failed(bundle, failures)

    @staticmethod
    def _raise_all_failed(bundle, failures):
        errors = [item["error"] for item in bundle.manifest["failures"]]
        errors.extend(f"{identity}: {reason}" for identity, reason in failures.items())
        raise RuntimeError("CommonIR autolayout: all candidates failed. " + "; ".join(errors))

    def _compile_candidates(self, bundle):
        """Compile independently, then load on the caller's GPU thread in order."""
        compiled, failures = {}, {}
        variants = {v["id"]: v for v in bundle.manifest["variants"]}

        workers = int(os.getenv("TRITON_COMMONIR_AUTOLAYOUT_COMPILE_WORKERS", "4"))
        if workers < 1:
            raise ValueError("TRITON_COMMONIR_AUTOLAYOUT_COMPILE_WORKERS must be positive")
        with ThreadPoolExecutor(max_workers=min(workers, len(variants)),
                                thread_name_prefix="commonir-layout-compile") as executor:
            futures = {
                identity: executor.submit(self._compile_variant, bundle, variant)
                for identity, variant in variants.items()
            }
            # Completion order must not change benchmark order or tie-breaking.
            # Device loading stays on the caller's thread/current device.
            for identity, future in futures.items():
                try:
                    candidate = future.result()
                    candidate._init_handles()
                    compiled[identity] = candidate
                except Exception as error:
                    failures[identity] = f"compilation/loading: {error}"
                    _LOGGER.warning("CommonIR candidate %s failed compilation/loading: %s", identity, error)
        return compiled, failures

    def _benchmark_candidates(self, compiled, failures, grid, stream, args, reset):
        samples, measurements = {}, {}
        for identity, candidate in compiled.items():
            try:
                samples[identity] = _benchmark(candidate, grid, stream, args, reset)
            except Exception as error:
                failures[identity] = f"benchmark: {error}"
                _LOGGER.warning("CommonIR candidate %s failed benchmarking: %s", identity, error)
        intervals = _median_intervals(samples, len(compiled)) if samples else {}
        finite = {identity: interval for identity, interval in intervals.items() if interval is not None}
        reference = min(finite, key=lambda identity: finite[identity][1]) if finite else None
        pruned = set()
        for identity, values in samples.items():
            interval = intervals[identity]
            dominated = reference is not None and interval is not None and interval[0] > finite[reference][1]
            measurements[identity] = {
                "samples_ms": values,
                "median_ci_ms": interval,
                "status": "pruned" if dominated else "retained",
            }
            if dominated:
                pruned.add(identity)
                measurements[identity]["dominated_by"] = reference

        def refine(identities):
            for identity in identities:
                try:
                    samples[identity].extend(
                        _benchmark(compiled[identity], grid, stream, args, reset, warmup=0,
                                   repetitions=_TOTAL_SAMPLES - _COARSE_SAMPLES))
                    measurements[identity]["status"] = "complete"
                except Exception as error:
                    measurements[identity]["status"] = "failed"
                    failures[identity] = f"benchmark: {error}"
                    _LOGGER.warning("CommonIR candidate %s failed benchmarking: %s", identity, error)

        refine(identity for identity in samples if identity not in pruned)
        # If every survivor failed to launch, recover viable pruned candidates
        # instead of failing a workload that still has executable alternatives.
        if samples and not any(item["status"] == "complete" for item in measurements.values()):
            refine(identity for identity in samples if identity in pruned)
        timings = {
            identity: statistics.median(samples[identity])
            for identity, item in measurements.items()
            if item["status"] == "complete"
        }
        report = {
            "protocol": _PROTOCOL,
            "warmup": _WARMUP,
            "coarse_samples": _COARSE_SAMPLES,
            "total_samples": _TOTAL_SAMPLES,
            "elimination_alpha": _ELIMINATION_ALPHA,
            "candidate_count": len(compiled),
            "candidates": measurements,
        }
        _LOGGER.info("CommonIR autolayout benchmark: %d candidates, %d pruned, %d fully measured", len(compiled),
                     sum(item["status"] == "pruned" for item in measurements.values()), len(timings))
        return timings, report

    def _remember_winner(self, kernel_cache, key, winner):
        kernel_cache[key] = winner
        return winner

    def _select_kernel(self, bundle, grid, stream, bound_args, *, kernel_cache, key):
        if key in kernel_cache:
            return kernel_cache[key]
        manifest = bundle.manifest
        if manifest["version"] != 2:
            raise RuntimeError("unsupported CommonIR candidate manifest version")
        grid = tuple(grid) + (1, ) * (3 - len(grid))
        if any(size == 0 for size in grid):
            return self._first_compiled(bundle, load=True)
        import torch

        jit_key = key
        with self._layout_lock:
            # Another caller may have selected this specialization while we waited.
            if jit_key in kernel_cache:
                return kernel_cache[jit_key]
            if len(manifest["variants"]) == 1:
                return self._remember_winner(kernel_cache, jit_key, self._first_compiled(bundle, load=True))
            key = _specialization_key(bundle)
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("run autolayout once before capturing this specialization in a GPU graph")

            cache = get_cache_manager(key)
            path = cache.get_file("commonir-autolayout.json")
            if path:
                try:
                    record = json.loads(Path(path).read_text())
                    selected = next(v for v in manifest["variants"] if v["id"] == record["selected"])
                    if record["version"] != 1:
                        raise ValueError("stale autolayout record")
                    winner = self._compile_variant(bundle, selected)
                    winner._init_handles()
                except Exception as error:
                    _LOGGER.warning("CommonIR cached candidate could not be reused: %s", error)
                else:
                    self.autolayout_results[key] = dict(record, source="disk")
                    return self._remember_winner(kernel_cache, jit_key, winner)

            # Materialize all binaries before benchmarking, keeping compiler
            # and launcher setup out of the timed regions.
            _require_quantile_test()
            compiled, failures = self._compile_candidates(bundle)
            if not compiled:
                self._raise_all_failed(bundle, failures)
            args = tuple(bound_args.values())

            def validate_pointer(value, signature):
                if isinstance(signature, tuple):
                    for child, child_signature in zip(value, signature):
                        validate_pointer(child, child_signature)
                elif signature.startswith("*") and value is not None and not isinstance(value, torch.Tensor):
                    raise TypeError("autolayout replay requires torch.Tensor pointer arguments")

            for name, signature in bundle.src.signature.items():
                validate_pointer(bound_args[name], signature)
            # Keep PyTorch's wrapper for its current stream, including handle 0.
            replay_stream = torch.cuda.current_stream()
            if replay_stream.cuda_stream != stream:
                replay_stream = torch.cuda.ExternalStream(stream)
            with torch.no_grad(), torch.cuda.stream(replay_stream):
                replay, reset = _scratch_arguments(args)
                timings, benchmark = self._benchmark_candidates(compiled, failures, grid, stream, replay, reset)
            if not timings:
                self.autolayout_results[key] = {
                    "version": 1,
                    "selected": None,
                    "timings_ms": {},
                    "failures": failures,
                    "source": "failed",
                    "benchmark": benchmark,
                }
                self._raise_all_failed(bundle, failures)
            # Manifest order breaks ties among the successful candidates.
            selected = min(timings, key=timings.get)
            record = {
                "version": 1, "selected": selected, "timings_ms": timings, "failures": failures, "benchmark": benchmark
            }
            cache.put(json.dumps(record), "commonir-autolayout.json", binary=False)
            winner = compiled[selected]
            self.autolayout_results[key] = dict(record, source="tuned")
            _LOGGER.info("CommonIR autolayout selected %s: %.6f ms (%d candidates)", selected, timings[selected],
                         len(timings))
            return self._remember_winner(kernel_cache, jit_key, winner)
