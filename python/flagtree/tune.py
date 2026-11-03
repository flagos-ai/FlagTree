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

from functools import cached_property, lru_cache, partial, update_wrapper
from inspect import getsource, signature
import hashlib
import json
import logging
import marshal
from math import isfinite
from numbers import Real
import os
from pathlib import Path
import platform
import socket
import tempfile
from typing import TYPE_CHECKING, Callable
import warnings

if TYPE_CHECKING:
    import torch.distributed

__all__ = ["AutoTuner", "autotune", "to_hashable"]

_CACHE_SCHEMA = 1
_DISTRIBUTED_PROTOCOL = {"timer": "backend_event", "warmup": 5, "repeat": 10, "reduce": "max", "version": 1}
_logger = logging.getLogger(__name__)


def to_hashable(tensor):
    """Return a tensor's shape, strides and dtype as a hashable workload feature."""
    return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype


def _config_key(value, *, workload=False):

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

        # to_hashable() includes torch.dtype. Do not import torch for ordinary keys.
        if workload and value_type.__module__ == "torch":
            import torch

            if value_type in (torch.dtype, torch.device):
                return [value_type.__name__, str(value)]
            if value_type is torch.Size:
                return ["torch.Size", [encode(item) for item in value]]

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

    return json.dumps(encode(value), separators=(",", ":"), ensure_ascii=True)


def _get_distributed_device():
    from triton.runtime.driver import driver

    return driver.active.get_device_interface(), driver.active.get_active_torch_device()

# 标记，需要修改
def _get_node_id():
    # A subgroup need not include LOCAL_RANK=0. Containers may require an explicit ID.
    return os.environ.get("FLAGTREE_AUTOTUNE_NODE_ID") or socket.gethostname()


@lru_cache(maxsize=256)
def _code_identity(code):
    return {
        "source": hashlib.sha256(getsource(code).encode("utf-8")).hexdigest(),
        # Python version is part of the environment; paths/line changes may miss conservatively.
        "bytecode": hashlib.sha256(marshal.dumps(code)).hexdigest(),
    }


def _function_identity(func):
    try:
        return {
            "module": func.__module__,
            "name": func.__qualname__,
            "code": _code_identity(func.__code__),
        }
    except (AttributeError, OSError, TypeError, ValueError, RecursionError):
        return None


def _get_cache_environment():
    import torch
    import triton
    from triton.runtime.cache import triton_key
    from triton.runtime.driver import driver

    active = driver.active
    di = active.get_device_interface()
    device = active.get_active_torch_device()
    target = active.get_current_target()
    hardware = {"device": str(device), "target": vars(target), "driver": type(active).__module__}
    if hasattr(di, "get_device_name"):
        hardware["name"] = di.get_device_name(device)
    if hasattr(di, "device_count"):
        hardware["visible_devices"] = di.device_count()
    if hasattr(di, "get_device_properties"):
        properties = di.get_device_properties(device)
        for name in ("name", "major", "minor", "total_memory", "multi_processor_count", "uuid"):
            value = getattr(properties, name, None)
            if value is not None:
                hardware[name] = value if type(value) in (str, int, bool) else str(value)
    versions = {"python": platform.python_version(), "torch": str(torch.__version__),
                "triton": str(triton.__version__),
                "triton_build": hashlib.sha256(triton_key().encode("utf-8")).hexdigest()}
    for name in ("cuda", "hip", "musa", "gcu"):
        value = getattr(torch.version, name, None)
        if value is not None:
            versions[name] = str(value)
    return {"hardware": hardware, "versions": versions}


def _cache_digest(identity):
    # Workload/configuration values are already type-tagged strings. The remaining
    # metadata is ordinary JSON; re-encoding it as a configuration adds hot-path work.
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class AutoTuner:

    def __init__(self, func, config_space, key_fn: Callable, prune_fn: Callable | None = None, *,
                 cache_dir=None, cache_version: str | None = None):
        self.config_space = config_space
        if not self.config_space:
            raise ValueError("config_space must contain at least one configuration")
        if any(not isinstance(config, dict) for config in self.config_space):
            raise TypeError("Each configuration must be a dictionary of function keyword arguments")
        self.fn = func
        self.key_fn = key_fn
        self.prune_fn = prune_fn
        self.best_configs = {}
        self.best_distributed_configs = {}
        self._local_cache_keys = {}
        self._persisted_distributed = {}
        if cache_version is not None and not isinstance(cache_version, str):
            raise TypeError("cache_version must be a string or None")
        self.cache_version = cache_version
        self.cache_dir = None if cache_dir is False else Path(
            cache_dir if cache_dir is not None else os.environ.get(
                "FLAGTREE_AUTOTUNE_CACHE_DIR", str(Path.home() / ".flagtree" / "autotune"))).expanduser()
        update_wrapper(self, func, updated=())

    @cached_property
    def do_bench(self):
        # Resolve the active backend only when benchmarking is actually needed.
        from triton.runtime.driver import driver
        return driver.active.get_benchmarker()

    def _read_cache(self, mode, key, candidates):
        if self.cache_dir is None:
            return None
        path = self.cache_dir / mode / f"{key}.json"
        try:
            with path.open(encoding="utf-8") as stream:
                record = json.load(stream)
            if (not isinstance(record, dict) or type(record.get("schema")) is not int
                    or record["schema"] != _CACHE_SCHEMA or record.get("key") != key):
                return None
            best = record.get("best_config")
            return best if isinstance(best, str) and best in candidates else None
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            _logger.warning("Ignoring unreadable autotune cache %s: %s", path, exc)
            return None

    # here, key is disk_key, best_config is _config_key(best_config)
    def _write_cache(self, mode, key, best_config, identity):
        if self.cache_dir is None:
            return False
        path = self.cache_dir / mode / f"{key}.json"
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # fd: File descriptor.
            # temporary: The complete path of the temporary file.
            fd, temporary = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"schema": _CACHE_SCHEMA, "key": key, "best_config": best_config,
                           "identity": identity}, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            return True
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            _logger.warning("Could not save autotune cache %s: %s", path, exc)
            return False
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _local_cache_identity(self, key, kwargs):
        try:
            function = _function_identity(self.fn)
            prune = _function_identity(self.prune_fn) if self.prune_fn is not None else "no_pruning"
            persist = function is not None and prune is not None
            try:
                environment = _get_cache_environment()
            except Exception as exc:
                # Keep configuration/override invalidation even without a device probe.
                _logger.debug("Autotune environment unavailable: %s", exc)
                environment = None
                persist = False
            tuned_names = {name for name in self.config_space[0]}
            identity = {
                "schema": _CACHE_SCHEMA, "mode": "local", "function": function,
                "workload": _config_key(key, workload=True), "cache_version": self.cache_version,
                "configs": [_config_key(config) for config in self.config_space],
                "overrides": _config_key({name: kwargs[name] for name in tuned_names if name in kwargs}),
                "environment": environment, "prune": prune,
            }
            return identity, _cache_digest(identity), persist
        except Exception as exc:
            # Preserve memory-only caching for unsupported local keys/configurations.
            _logger.debug("Persistent local autotune caching unavailable: %s", exc)
            return None, None, False

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

        # Derive identity from workload/configuration/context metadata, cache_key from its SHA-256 digest,
        # and persist from function/prune identity and environment availability.
        identity, cache_key, persist = self._local_cache_identity(key, kwargs)
        best_config = self.best_configs.get(key)
        if best_config is not None and not any(best_config is config for config in self.config_space):
            best_config = None
        if cache_key is not None and self._local_cache_keys.get(key) != cache_key:
            best_config = None
        if best_config is None:
            config_space = self.get_pruned_config(*args, **kwargs)
            if not config_space:
                raise ValueError("prune_fn removed all configurations")
            candidates = {_config_key(config): config for config in config_space} if persist else {}
            disk_identity = {**identity, "pruned_configs": list(candidates)} if persist else None
            disk_key = _cache_digest(disk_identity) if persist else None
            cached = self._read_cache("local", disk_key, candidates) if persist else None

            if cached is not None:
                best_config = candidates[cached]
            else:
                timings = self.tune(config_space, *args, **kwargs)
                best_config = timings[0][1]
                if persist:
                    self._write_cache("local", disk_key, _config_key(best_config), disk_identity)
            self.best_configs[key] = best_config
            self._local_cache_keys[key] = cache_key

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
        # inputs may differ across ranks.
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

        # Cache preparation is collective too: a failing key_fn on one rank must
        # not leave peers waiting for a cache broadcast or entering a benchmark.
        context = None
        error = None
        try:
            key = self.key_fn(*args, **kwargs) if self.key_fn is not None else None
            try:
                hash(key)
            except TypeError as exc:
                raise TypeError("key_fn must return a hashable value") from exc
            function = _function_identity(self.fn)
            context = {
                "workload": _config_key(key, workload=True), "environment": _get_cache_environment(),
                "function": function, "node": _get_node_id(), "device": str(device),
                "cache_version": self.cache_version, "disk_cacheable": function is not None,
            }
            if not isinstance(context["node"], str) or not context["node"]:
                raise ValueError("The autotune node ID must be a nonempty string")
            _config_key(context)  # Validate before peers move to the next collective.
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        dist.all_gather_object(reports, (error, context), group=group)
        errors = [f"group rank {i}: {report[0]}" for i, report in enumerate(reports) if report[0] is not None]
        if errors:
            raise ValueError("Distributed autotune cache preparation failed: " + "; ".join(errors))
        contexts = [report[1] for report in reports]

        # Host names determine node membership and writers, not the persistent key.
        # Preserve the group-rank order and normalize equivalent node partitions.
        nodes = {}
        layout = [nodes.setdefault(item["node"], len(nodes)) for item in contexts]
        identity = {
            "schema": _CACHE_SCHEMA, "mode": "distributed", "protocol": _DISTRIBUTED_PROTOCOL,
            "backend": str(dist.get_backend(group)), "world_size": len(contexts), "node_layout": layout,
            "ranks": [{name: value for name, value in item.items() if name not in ("node", "disk_cacheable")}
                      for item in contexts],
            "configs": config_keys, "effective_configs": effective_keys,
        }
        cache_key = _cache_digest(identity)
        persist = all(item["disk_cacheable"] for item in contexts)
        writer = next(i for i, item in enumerate(contexts) if item["node"] == context["node"])

        def remember(winner, *, retuned=False):
            self.best_distributed_configs[cache_key] = winner
            if (persist and rank == writer
                    and (retuned or self._persisted_distributed.get(cache_key) != winner)):
                if self._write_cache("distributed", cache_key, winner, identity):
                    self._persisted_distributed[cache_key] = winner

        # Only group rank 0 makes the lookup decision. Other nodes' missing files
        # or empty memory caches must not cause a different collective sequence.
        proposal = [None]
        if rank == 0:
            candidate = self.best_distributed_configs.get(cache_key)
            if not isinstance(candidate, str) or candidate not in common:
                candidate = self._read_cache("distributed", cache_key, common) if persist else None
            proposal[0] = candidate
        dist.broadcast_object_list(proposal, src=dist.get_global_rank(group, 0), group=group)
        cached = proposal[0]
        valid = isinstance(cached, str) and cached in common and cached in local_configs
        dist.all_gather_object(reports, (cached, valid), group=group)
        if valid and all(item == (cached, True) for item in reports):
            remember(cached)
            return self.fn(*args, **{**local_configs[cached], **kwargs})

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
        winner = config_keys[best_index]
        best_config = local_configs[winner]
        remember(winner, retuned=True)
        return self.fn(*args, **{**best_config, **kwargs})

    def _bench_distributed(self, call, device_interface, autotune_pg):
        import torch.distributed as dist

        start_event = device_interface.Event(enable_timing=True)
        end_event = device_interface.Event(enable_timing=True)

        dist.barrier(group=autotune_pg)
        for _ in range(_DISTRIBUTED_PROTOCOL["warmup"]):
            call()
        device_interface.synchronize()
        dist.barrier(group=autotune_pg)
        # The callable must order side-stream work between these current-stream events.
        start_event.record()
        for _ in range(_DISTRIBUTED_PROTOCOL["repeat"]):
            call()
        end_event.record()
        device_interface.synchronize()
        return start_event.elapsed_time(end_event) / _DISTRIBUTED_PROTOCOL["repeat"]

    def tune(self, config_space, *args, **kwargs):
        benchmark = self.do_bench
        timings = []
        first_error = None
        for config in config_space:
            bench_func = partial(self.fn, *args, **{**config, **kwargs})
            try:
                measurement = benchmark(bench_func, quantiles=[0.5])
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


def autotune(config_space, key_fn, prune_fn=None, *, cache_dir=None, cache_version: str | None = None):
    """Tune a Python function over candidate keyword-argument dictionaries.

    ``config_space`` contains dictionaries such as ``{"config": triton.Config(...)}``.
    Values are passed unchanged to the decorated function. ``key_fn(*args, **kwargs)``
    returns a hashable workload key; use ``None`` to share one cache entry for all
    inputs. Include every input feature that affects configuration validity or
    performance in this key. ``prune_fn(config, *args, **kwargs)`` returns True to
    keep a candidate and False to discard it. In local mode it runs on a cache miss.

    Local tuning uses the active FlagTree backend's benchmarker to measure median
    latency in milliseconds. Tuning executes the function repeatedly; inputs must
    tolerate these repeated calls.

    In local mode, call with ``autotune=False`` to use the first unpruned candidate
    directly. Candidate names must not also be supplied by the caller unless
    ``autotune_allow_arg_overwrite=True``, in which case caller keywords take
    precedence. These control keywords and ``autotune_pg`` are reserved by the wrapper.
    ``decorated.fn`` exposes the original function, and
    ``decorated.best_configs.clear()`` clears the local in-memory cache.

    Both modes query memory, then disk, and only benchmark on a miss. ``cache_dir``
    defaults to ``FLAGTREE_AUTOTUNE_CACHE_DIR`` or ``~/.flagtree/autotune``;
    ``cache_dir=False`` disables disk caching. Clear the corresponding disk files
    as well as memory to force retuning. Keys include function source/code/defaults,
    candidate configurations, effective overrides, workload and runtime/device
    identity. Set ``cache_version`` to a new string when external dependencies,
    closure/global state or network topology change: these are not automatically
    fingerprinted. Unsupported local key/configuration/callable identities keep
    memory caching but disable persistence. Disk errors are treated as misses or
    failed optional saves; writes use atomic replacement.

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

    Joint tuning supports both intra-node and inter-node groups. Its separate
    ``best_distributed_configs`` cache includes all ranks' workload/environment,
    the node partition, group size/backend and timing protocol. Group rank 0
    queries memory then disk and broadcasts a candidate identity. Every member
    confirms validity before the group reuses it or retunes together. Each node's
    lowest group rank saves the same result to its local disk. Nodes are identified
    by hostname; set ``FLAGTREE_AUTOTUNE_NODE_ID`` to a shared value within a node
    and different values across nodes when containers obscure hostnames. Only the
    leader's disk participates in lookup; peer-only copies do not prevent a miss.
    Distributed memory entries can be cleared with ``best_distributed_configs.clear()``.

    Supported candidate values are None, bool, int, finite float, str, list, tuple, string-keyed dict,
    and ``triton.Config`` with supported field values. Config hooks and external
    IR overrides are unsupported in joint tuning and persistent caches. Workload
    keys also support torch dtype/device values, including ``to_hashable`` results.
    Preparation failures are reported collectively;
    operator failures abort tuning and may require the group's timeout to release
    peers. No recovery from operator/communication failure is provided.

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
        return AutoTuner(func, config_space, key_fn, prune_fn,
                         cache_dir=cache_dir, cache_version=cache_version)

    return wrapper
