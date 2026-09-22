"""Opt-in CommonIR layout tuning using the existing JIT launch hook."""

import hashlib
import json
import logging
import math
import statistics
import threading
from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton.runtime.jit import JITFunction

_LOGGER = logging.getLogger(__name__)
_PROTOCOL = "commonir-autolayout-v1"


def _fast_launch_key(kernel, grid, stream, args):
    """Describe the current launch without serialization or device properties."""
    import torch

    def describe(value):
        if isinstance(value, torch.Tensor):
            # Pointer plus storage_offset/dtype also determines the storage
            # base, so changed views and alias relationships cannot hit here.
            return (type(value), id(value), value.data_ptr(), tuple(value.shape), value.stride(),
                    value.storage_offset(), value.dtype, value.device)
        if isinstance(value, (tuple, list)):
            items = tuple(describe(item) for item in value)
            return None if any(item is None for item in items) else (type(value), items)
        if value is None or isinstance(value, (str, bool, int)):
            return type(value), value
        if isinstance(value, float):
            return float, value.hex()
        # Unhandled arguments still use the existing complete workload key.
        return None

    arguments = tuple(describe(arg) for arg in args)
    if any(arg is None for arg in arguments):
        return None
    return kernel, torch.cuda.current_device(), grid, stream, arguments


def _workload_key(kernel, grid, args):
    import torch

    storages = {}

    def describe(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            identity = (str(value.device), storage.data_ptr())
            alias = storages.setdefault(identity, len(storages))
            pointer = value.data_ptr()
            return ("tensor", str(value.device), str(value.dtype), tuple(value.shape), tuple(value.stride()),
                    value.storage_offset(), min(pointer & -pointer, 256), alias)
        if isinstance(value, (tuple, list)):
            return [describe(item) for item in value]
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value.hex()
        # constexpr types are already part of the compiled specialization.
        if isinstance(value, torch.dtype) or type(value).__module__.startswith("triton.language"):
            return str(value)
        raise TypeError(f"autolayout does not support argument type {type(value).__name__}")

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload = (_PROTOCOL, kernel.hash, str(getattr(device,
                                                   "uuid", device.name)), device.name, device.multi_processor_count,
               tuple(grid), [describe(arg) for arg in args])
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


def _benchmark(kernel, grid, stream, args, reset):
    import torch

    launch = kernel.run  # Load binary and build the launcher before timing.

    def run():
        launch(*grid, stream, kernel.function, kernel.packed_metadata, None, None, None, *args)

    for _ in range(5):
        reset()
        run()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(40)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in starts]
    for start, end in zip(starts, ends):
        reset()
        start.record()
        run()
        end.record()
    ends[-1].synchronize()
    elapsed = statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("autolayout benchmark did not produce a positive finite GPU time")
    return elapsed


class CommonIRAutolayoutJITFunction(JITFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layout_lock = threading.RLock()
        self._layout_kernels = {}
        self._layout_winners = {}
        self._layout_fast_winners = {}
        self.autolayout_results = {}

    def create_binder(self):
        from triton._common_ir import ENABLED

        result = super().create_binder()
        if result[2].backend != "maca" or not ENABLED:
            raise ValueError("autolayout requires the MetaX CommonIR backend")
        return result

    def run(self, *args, grid, warmup, **kwargs):
        kwargs["autolayout"] = True
        return super().run(*args, grid=grid, warmup=warmup, **kwargs)

    def _compile_variant(self, fallback, manifest, variant):
        if variant is manifest["variants"][0]:
            return fallback
        identity = (fallback.hash, variant["id"])
        if identity in self._layout_kernels:
            return self._layout_kernels[identity]
        cache = get_cache_manager(fallback.hash)
        path = cache.get_file(variant["file"])
        if path is None or hashlib.sha256(Path(path).read_bytes()).hexdigest() != variant["id"]:
            raise RuntimeError(f"missing or changed CommonIR candidate source: {variant['file']}")
        from triton.backends.metax.compiler import MACAOptions

        options = {name: getattr(fallback.metadata, name) for name in MACAOptions.__dataclass_fields__}
        options["autolayout"] = False
        kernel = self.compile(path, target=fallback.metadata.target, options=options)
        if list(kernel.src.signature.values()) != manifest["signature"]:
            raise RuntimeError("CommonIR candidate changed the device argument signature")
        # IRSource omits constexpr host arguments. Reuse the AST launch ABI
        # before the candidate's launcher is initialized.
        kernel.src = fallback.src
        self._layout_kernels[identity] = kernel
        return kernel

    def _remember_fast_winner(self, launch_key, workload_key, winner):
        if launch_key is not None:
            # Do not retain every tensor allocation identity in a long run.
            if len(self._layout_fast_winners) >= 128:
                self._layout_fast_winners.clear()
            self._layout_fast_winners[launch_key] = workload_key, winner
        return winner

    def _prepare_kernel_for_launch(self, kernel, grid, stream, bound_args):
        import torch

        manifest = getattr(kernel.metadata, "commonir_layout_candidates", None)
        if manifest is None or len(manifest["variants"]) == 1:
            return kernel
        if manifest["version"] != 1:
            raise RuntimeError("unsupported CommonIR candidate manifest version")
        grid = tuple(grid) + (1, ) * (3 - len(grid))
        if any(size == 0 for size in grid):
            return kernel
        args = tuple(bound_args.values())
        launch_key = _fast_launch_key(kernel, grid, stream, args)
        cached = self._layout_fast_winners.get(launch_key)
        if cached is not None:
            workload_key, winner = cached
            if self._layout_winners.get(workload_key) is winner:
                return winner
        key = _workload_key(kernel, grid, args)
        with self._layout_lock:
            if key in self._layout_winners:
                return self._remember_fast_winner(launch_key, key, self._layout_winners[key])
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("run autolayout once before capturing this workload in a GPU graph")

            cache = get_cache_manager(key)
            path = cache.get_file("commonir-autolayout.json")
            if path:
                try:
                    record = json.loads(Path(path).read_text())
                    selected = next(v for v in manifest["variants"] if v["id"] == record["selected"])
                    if record["version"] != 1:
                        raise ValueError("stale autolayout record")
                    winner = self._compile_variant(kernel, manifest, selected)
                except (ValueError, KeyError, TypeError, StopIteration):
                    pass
                else:
                    self._layout_winners[key] = winner
                    self.autolayout_results[key] = dict(record, source="disk")
                    return self._remember_fast_winner(launch_key, key, winner)

            # Materialize all binaries before benchmarking, keeping compiler
            # and launcher setup out of the timed regions.
            compiled, failures = {}, {}
            for index, variant in enumerate(manifest["variants"]):
                try:
                    candidate = self._compile_variant(kernel, manifest, variant)
                    candidate._init_handles()
                    compiled[variant["id"]] = candidate
                except Exception as error:
                    if index == 0:
                        raise
                    failures[variant["id"]] = str(error)
                    _LOGGER.warning("CommonIR candidate %s failed compilation/loading: %s", variant["id"], error)

            def validate_pointer(value, signature):
                if isinstance(signature, tuple):
                    for child, child_signature in zip(value, signature):
                        validate_pointer(child, child_signature)
                elif signature.startswith("*") and value is not None and not isinstance(value, torch.Tensor):
                    raise TypeError("autolayout replay requires torch.Tensor pointer arguments")

            for name, signature in kernel.src.signature.items():
                validate_pointer(bound_args[name], signature)
            # Keep PyTorch's wrapper for its current stream, including handle 0.
            replay_stream = torch.cuda.current_stream()
            if replay_stream.cuda_stream != stream:
                replay_stream = torch.cuda.ExternalStream(stream)
            with torch.no_grad(), torch.cuda.stream(replay_stream):
                replay, reset = _scratch_arguments(args)
                timings = {
                    identity: _benchmark(candidate, grid, stream, replay, reset)
                    for identity, candidate in compiled.items()
                }
            # Insertion order keeps the fallback on ties.
            selected = min(timings, key=timings.get)
            record = {"version": 1, "selected": selected, "timings_ms": timings, "failures": failures}
            cache.put(json.dumps(record), "commonir-autolayout.json", binary=False)
            winner = compiled[selected]
            self._layout_winners[key] = winner
            self.autolayout_results[key] = dict(record, source="tuned")
            _LOGGER.info("CommonIR autolayout selected %s: %.6f ms (%d candidates)", selected, timings[selected],
                         len(timings))
            return self._remember_fast_winner(launch_key, key, winner)
