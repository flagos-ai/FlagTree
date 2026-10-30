# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from functools import cached_property, partial, update_wrapper
from inspect import signature
import json
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, Callable
import warnings

if TYPE_CHECKING:
    import torch.distributed

__all__ = ["AutoTuner", "autotune", "to_hashable"]


def to_hashable(tensor):
    """Return a tensor's shape, strides and dtype as a hashable workload feature."""
    return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype


def _config_key(value):

    def encode(value):
        value_type = type(value)
        if value is None:
            return ["none"]
        if value_type in (bool, int, str):
            return [value_type.__name__, value]
        if value_type is float:
            if not isfinite(value):
                raise ValueError("Non-finite floats are unsupported in distributed configurations")
            return ["float", value.hex()]
        if value_type in (list, tuple):
            return [value_type.__name__, [encode(item) for item in value]]
        if value_type is dict:
            if any(type(key) is not str for key in value):
                raise TypeError("Distributed configuration dictionaries require string keys")
            return ["dict", [[key, encode(value[key])] for key in sorted(value)]]

        # Triton is only needed when a candidate actually contains a Config.
        try:
            from triton import Config
        except ImportError:
            Config = None
        if Config is not None and value_type is Config:
            if value.pre_hook is not None:
                raise TypeError("Distributed tuning does not support triton.Config pre_hook")
            if value.ir_override is not None or value.kwargs.get("ir_override") is not None:
                raise TypeError("Distributed tuning does not support triton.Config ir_override")
            return ["triton.Config", encode(vars(value))]
        raise TypeError(f"Unsupported distributed configuration value: {value_type.__name__}")

    return json.dumps(encode(value), indent=2, separators=(",", ":"), ensure_ascii=True)


def _get_distributed_device():
    from triton.runtime.driver import driver

    return driver.active.get_device_interface(), driver.active.get_active_torch_device()


class AutoTuner:

    def __init__(self, func, config_space, key_fn: Callable, prune_fn: Callable | None = None, *, do_bench=None):
        self.config_space = config_space
        if not self.config_space:
            raise ValueError("config_space must contain at least one configuration")
        if any(not isinstance(config, dict) for config in self.config_space):
            raise TypeError("Each configuration must be a dictionary of function keyword arguments")
        self.fn = func
        self.key_fn = key_fn
        self.prune_fn = prune_fn
        self.best_configs = {}
        self._do_bench = do_bench
        update_wrapper(self, func, updated=())

    @cached_property
    def do_bench(self):
        if self._do_bench is not None:
            return self._do_bench
        # Resolve the active backend only when benchmarking is actually needed.
        from triton.runtime.driver import driver

        return driver.active.get_benchmarker()

    def get_pruned_config(self, *args, **kwargs):
        if self.prune_fn is None:
            return self.config_space
        return [config for config in self.config_space if self.prune_fn(config, *args, **kwargs)]

    def __call__(self, *args, autotune=True, autotune_allow_arg_overwrite=False,
                 autotune_pg: torch.distributed.ProcessGroup | None = None, **kwargs):
        if autotune_pg is not None:
            return self._call_distributed(autotune_pg, autotune, *args, **kwargs)

        tuned_names = {name for name in self.config_space[0]}
        conflicts = tuned_names.intersection(kwargs)
        if conflicts:
            if not autotune_allow_arg_overwrite:
                raise ValueError(f"Arguments are also supplied by config_space: {', '.join(sorted(conflicts))}")
            else:
                warnings.warn(
                    "Duplicate keyword arguments, but autotune_allow_arg_overwrite is True, will use the argument provided by the user."
                )

        if not autotune:
            return self.fn(*args, **{**self.config_space[0], **kwargs})

        # Only workload arguments reach key_fn; tunable arguments are injected later.
        key = self.key_fn(*args, **kwargs) if self.key_fn is not None else None
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError("key_fn must return a hashable value") from exc

        best_config = self.best_configs.get(key)
        if best_config is None:
            config_space = self.get_pruned_config(*args, **kwargs)
            if not config_space:
                raise ValueError("prune_fn removed all configurations")
            timings = self.tune(config_space, *args, **kwargs)
            best_config = timings[0][1]
            self.best_configs[key] = best_config

        # Benchmark results are timings, so execute the winner to obtain the real result.
        return self.fn(*args, **{**best_config, **kwargs})

    # "group" is autotune_pg; all members align candidates even when tuning is disabled.
    def _call_distributed(self, group, autotune_enabled, *args, **kwargs):
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError("autotune_pg requires an initialized torch.distributed process group")
        rank = dist.get_rank(group=group)
        if rank < 0:
            raise ValueError("The caller must be a member of autotune_pg")

        local_configs = {}
        error = None
        tune_enabled = None
        try:
            tune_enabled = bool(autotune_enabled)
            for config in self.get_pruned_config(*args, **kwargs):
                local_configs.setdefault(_config_key(config), config)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        # Intersect original pruned candidates before applying caller overrides.
        # Every member participates even if its pruning or encoding failed.
        reports = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(reports, (error, list(local_configs), tune_enabled), group=group)
        errors = [f"group rank {i}: {report[0]}" for i, report in enumerate(reports) if report[0] is not None]
        if errors:
            raise ValueError("Distributed autotune preparation failed: " + "; ".join(errors))
        common = set(local_configs)
        for _, keys, _ in reports:
            common.intersection_update(keys)
        config_keys = sorted(common)
        if not config_keys:
            raise ValueError("No common configuration remains after distributed pruning")
        if any(report[2] != tune_enabled for report in reports):
            raise ValueError("All ranks in autotune_pg must use the same autotune setting")

        tuned_names = {name for key in config_keys for name in local_configs[key]}

        # When disabling tuning, only keep the first one.
        if not tune_enabled:
            config_keys = config_keys[:1]

        effective_keys = []
        error = None
        try:
            overrides = {name: kwargs[name] for name in tuned_names if name in kwargs}
            fn_signature = signature(self.fn)
            for config_key in config_keys:
                config = local_configs[config_key]
                # Validate only calls that survived the group intersection.
                fn_signature.bind(*args, **{**config, **kwargs})
                effective_keys.append(_config_key({**config, **overrides}))
            if tune_enabled:
                device_interface, device = _get_distributed_device()

                # Optional CPU placement for Gloo control tensors.
                # if dist.get_backend(group) == "gloo":
                #     device = "cpu"

                # Reused latency buffer and the index of the winning candidate.
                score = torch.empty(1, dtype=torch.float64, device=device)
                selected = torch.full((1, ), -1, dtype=torch.int64, device=device)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        # Overriding is allowed, but ranks must still execute the same effective configs.
        # Keep workload arguments out of the comparison: inputs may differ across ranks.
        dist.all_gather_object(reports, (error, effective_keys), group=group)
        errors = [f"group rank {i}: {report[0]}" for i, report in enumerate(reports) if report[0] is not None]
        if errors:
            raise ValueError("Distributed autotune preparation failed: " + "; ".join(errors))

        if any(report[1] != effective_keys for report in reports):
            raise ValueError(
                "Tuning parameter values differ across ranks after applying keyword argument overrides. "
                "For each candidate configuration, all ranks in autotune_pg must use the same final values.")

        if not tune_enabled:
            config = local_configs[config_keys[0]]
            return self.fn(*args, **{**config, **kwargs})

        # Distributed cache lookup and persistence are not implemented yet.
        timings = []
        for config_key in config_keys:
            config = local_configs[config_key]
            call = partial(self.fn, *args, **{**config, **kwargs})
            # Execution/communication failures propagate: continuing could desynchronize ranks.
            latency = self._bench_distributed(call, device_interface, group)
            score.fill_(latency if isfinite(latency) and latency >= 0 else float("inf"))
            dist.all_reduce(score, op=dist.ReduceOp.MAX, group=group)
            timings.append(score.item())

        if rank == 0:
            best_index = min(range(len(timings)), key=timings.__getitem__)
            if isfinite(timings[best_index]):
                selected.fill_(best_index)
        # broadcast's src is a global rank, not a rank within the supplied group.
        dist.broadcast(selected, src=dist.get_global_rank(group, 0), group=group)
        best_index = selected.item()
        if best_index < 0:
            raise RuntimeError("No configuration produced a valid distributed benchmark result")
        best_config = local_configs[config_keys[best_index]]
        # Distributed stage one deliberately neither reads nor writes best_configs.
        return self.fn(*args, **{**best_config, **kwargs})

    def _bench_distributed(self, call, device_interface, autotune_pg):
        import torch.distributed as dist

        start_event = device_interface.Event(enable_timing=True)
        end_event = device_interface.Event(enable_timing=True)

        dist.barrier(group=autotune_pg)
        for _ in range(5):
            call()
        device_interface.synchronize()
        dist.barrier(group=autotune_pg)
        # The callable must order side-stream work between these current-stream events.
        start_event.record()
        for _ in range(10):
            call()
        end_event.record()
        device_interface.synchronize()
        return start_event.elapsed_time(end_event) / 10

    def tune(self, config_space, *args, **kwargs):
        benchmark = self.do_bench
        timings = []
        first_error = None
        for config in config_space:
            bench_func = partial(self.fn, *args, **{**config, **kwargs})
            try:
                # The backend benchmarker owns warmup, repetition and synchronization.
                measurement = benchmark(bench_func, quantiles=[0.5])
                # Triton's default benchmarker unwraps a single quantile to a scalar.
                latency = float(measurement if isinstance(measurement, Real) else measurement[0])
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                latency = float("inf")
            if not isfinite(latency) or latency < 0:
                latency = float("inf")
            timings.append((latency, config))

        timings.sort(key=lambda item: item[0])
        if not timings or not isfinite(timings[0][0]):
            raise RuntimeError("No configuration produced a valid benchmark result") from first_error
        return timings


def autotune(config_space, key_fn, prune_fn=None, *, do_bench=None):
    """Tune a Python function over candidate keyword-argument dictionaries.

    ``config_space`` contains dictionaries such as ``{"config": triton.Config(...)}``.
    Values are passed unchanged to the decorated function. ``key_fn(*args, **kwargs)``
    returns a hashable workload key; use ``None`` to share one cache entry for all
    inputs. Include every input feature that affects configuration validity or
    performance in this key. ``prune_fn(config, *args, **kwargs)`` returns True to
    keep a candidate and False to discard it. In local mode it runs on a cache miss.

    ``do_bench(fn, quantiles=[0.5])`` returns a median latency in milliseconds, either
    as a scalar or a one-element sequence. It defaults to the active FlagTree
    backend's benchmarker. Tuning executes the function repeatedly; inputs must
    tolerate these repeated calls.

    In local mode, call with ``autotune=False`` to use the first unpruned candidate
    directly. Candidate names must not also be supplied by the caller unless
    ``autotune_allow_arg_overwrite=True``, in which case caller keywords take
    precedence. These control keywords and ``autotune_pg`` are reserved by the wrapper.
    ``decorated.fn`` exposes the original function, and
    ``decorated.best_configs.clear()`` clears the in-memory cache.

    Pass an initialized ``torch.distributed`` group as ``autotune_pg`` for joint
    tuning. All members must enter this call in the same order, with their active
    device selected. Candidates must preserve semantics and communication order.
    The group intersects pruned candidates by value and sorts their stable keys.
    An empty intersection is an error even when the caller overrides parameters.
    Caller keywords take precedence over candidate values without requiring
    ``autotune_allow_arg_overwrite``. ``prune_fn`` receives those keywords and
    should account for them when checking candidate validity. Effective candidate
    arguments and the ``autotune`` setting must agree across ranks; workload
    inputs may differ. With ``autotune=False``, each rank executes the first common
    candidate once, without benchmarking or resolving a benchmark device.

    With tuning enabled, the group benchmarks candidates in the common order
    with exactly 5 warmup and 10 measured calls each. The backend must support
    timing events; their elapsed milliseconds divided by 10 give the amortized
    time per call. Events are recorded on the current stream, which the function
    must preserve. Side-stream work must depend on the starting event and join
    the timing stream before the ending event; the final device synchronization
    alone does not include independent side-stream work in the event interval.
    The winner minimizes the maximum rank latency and is broadcast by group rank 0.
    Each rank then executes the winner once more and returns its local result.

    Distributed calls with tuning enabled currently always retune, ignoring
    ``key_fn``, ``best_configs`` and ``do_bench``; disk caching is not implemented.
    Supported candidate values are None, bool, int, finite float, str, list, tuple, string-keyed dict,
    and ``triton.Config`` with supported field values. Config hooks and external
    IR overrides are unsupported. Preparation failures are reported collectively;
    operator failures abort tuning and may require the group's timeout to release
    peers. No candidate recovery or distributed tuning cache is provided.

    Example::

        @autotune(
            config_space=[{"block_size": 32}, {"block_size": 64}],
            key_fn=lambda x: to_hashable(x),
            prune_fn=lambda config, x: config["block_size"] <= x.numel(),
        )
        def operation(x, block_size):
            ...  # Launch the kernel(s) using block_size.
    """

    def wrapper(func):
        return AutoTuner(func, config_space, key_fn, prune_fn, do_bench=do_bench)

    return wrapper
