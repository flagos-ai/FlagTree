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
"""Resolve Ascend NPU event and graph replay timing."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

from triton.flagtune.runtime.device import UnsupportedFlagTuneDeviceError
from triton.runtime.driver import driver


class BenchmarkMode(str, Enum):
    """Select ordinary event timing or a backend replay mechanism."""

    EVENT = "event"
    REPLAY = "replay"


def resolve_requested_mode(
    mode: BenchmarkMode | str | None = None,
    *,
    default: BenchmarkMode = BenchmarkMode.EVENT,
) -> BenchmarkMode:
    """Resolve an explicit mode or ``FLAGTUNE_BENCHMARK_MODE``.

    ``default`` lets protocol owners preserve their historical behavior while
    still honoring the explicit environment override.
    """
    requested = mode if mode is not None else os.environ.get("FLAGTUNE_BENCHMARK_MODE")
    if requested is None or requested == "":
        return BenchmarkMode(default)
    return BenchmarkMode(requested)


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Describe the exact resolved benchmark semantics used by one tuner."""

    requested_mode: BenchmarkMode
    resolved_mode: BenchmarkMode
    implementation: str
    cache_policy: str
    warmup_ms: int
    measurement_ms: int
    n_retries: int
    per_replay_ms: float | None
    fallback_reason: str | None = None

    def cache_key(self) -> tuple[Any, ...]:
        """Return a stable identity suitable for persistent benchmark caches."""
        if self.resolved_mode is BenchmarkMode.EVENT:
            # Preserve the existing FlagGems BenchmarkCache v2 event identity.
            return ("triton_do_bench", self.warmup_ms, self.measurement_ms)
        return (
            self.implementation,
            self.warmup_ms,
            self.measurement_ms,
            self.n_retries,
            self.per_replay_ms,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe audit metadata."""
        return {
            "requested_mode": self.requested_mode.value,
            "resolved_mode": self.resolved_mode.value,
            "implementation": self.implementation,
            "cache_policy": self.cache_policy,
            "warmup_ms": self.warmup_ms,
            "measurement_ms": self.measurement_ms,
            "n_retries": self.n_retries,
            "per_replay_ms": self.per_replay_ms,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class ResolvedBenchmarker:
    """Pair a two-argument Autotuner benchmark callable with its protocol."""

    protocol: BenchmarkProtocol
    benchmark: Callable[[Callable[..., Any], Sequence[float]], Sequence[float]]


def _validate_request(
    mode: BenchmarkMode | str,
    warmup_ms: int,
    measurement_ms: int,
    n_retries: int,
) -> BenchmarkMode:
    try:
        selected = BenchmarkMode(mode)
    except ValueError as exc:
        raise ValueError("benchmark mode must be 'event' or 'replay'") from exc
    if not isinstance(warmup_ms, int) or isinstance(warmup_ms, bool) or warmup_ms < 0:
        raise ValueError("benchmark warmup_ms must be a non-negative integer")
    if (not isinstance(measurement_ms, int) or isinstance(measurement_ms, bool) or measurement_ms <= 0):
        raise ValueError("benchmark measurement_ms must be a positive integer")
    if not isinstance(n_retries, int) or isinstance(n_retries, bool) or n_retries <= 0:
        raise ValueError("benchmark n_retries must be a positive integer")
    return selected


def _bench_npu_graph(device, kernel_call, rep, n_retries, quantiles):
    from triton.testing import _summarize_statistics

    caller_stream = device.current_stream()
    stream = device.Stream()
    stream.wait_stream(caller_stream)
    with device.stream(stream):
        kernel_call()
        start = device.Event(enable_timing=True)
        end = device.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            kernel_call()
        end.record()
        device.synchronize()
        estimate_ms = start.elapsed_time(end) / 5
        repeats = 1000 if estimate_ms == 0 else max(1, int(rep / estimate_ms))

        graph = device.NPUGraph()
        with device.graph(graph, stream=stream):
            for _ in range(repeats):
                kernel_call()
        device.synchronize()

        samples = []
        try:
            for _ in range(n_retries):
                start.record()
                graph.replay()
                end.record()
                device.synchronize()
                samples.append(start.elapsed_time(end) / repeats)
        finally:
            graph.reset()
    caller_stream.wait_stream(stream)
    return _summarize_statistics(samples, quantiles, "mean")


def resolve_benchmarker(
    mode: BenchmarkMode | str,
    *,
    warmup_ms: int,
    measurement_ms: int,
    n_retries: int = 10,
    allow_fallback: bool = True,
) -> ResolvedBenchmarker:
    """Resolve event timing or NPU Graph replay on the active Ascend device."""
    selected = _validate_request(mode, warmup_ms, measurement_ms, n_retries)
    active = driver.active
    backend = active.get_current_target().backend
    if backend != "npu":
        raise UnsupportedFlagTuneDeviceError(
            f"FlagTune on Triton 3.5 supports only Ascend backend 'npu', got {backend!r}")
    if selected is BenchmarkMode.REPLAY:
        get_interface = getattr(active, "get_device_interface", None)
        device = get_interface() if get_interface is not None else None
        if callable(getattr(device, "NPUGraph", None)) and callable(getattr(device, "graph", None)):
            per_replay_ms = measurement_ms / n_retries

            def replay_benchmark(kernel_call, quantiles):
                return _bench_npu_graph(device, kernel_call, per_replay_ms, n_retries, quantiles)

            return ResolvedBenchmarker(
                protocol=BenchmarkProtocol(
                    requested_mode=selected,
                    resolved_mode=BenchmarkMode.REPLAY,
                    implementation="ascend_npu_graph_replay_v1",
                    cache_policy="warm_l2",
                    # Graph timing uses a primer and a short calibration, not a timed warmup.
                    warmup_ms=0,
                    measurement_ms=measurement_ms,
                    n_retries=n_retries,
                    per_replay_ms=per_replay_ms,
                ),
                benchmark=replay_benchmark,
            )
        reason = "active Triton backend does not provide a registered replay benchmarker"
        if not allow_fallback:
            raise RuntimeError(reason)
        fallback_reason = reason
        warnings.warn(
            f"{fallback_reason}; falling back to event timing",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        fallback_reason = None

    event_benchmarker = active.get_benchmarker()

    def event_benchmark(kernel_call, quantiles):
        return event_benchmarker(
            kernel_call,
            warmup=warmup_ms,
            rep=measurement_ms,
            quantiles=quantiles,
        )

    return ResolvedBenchmarker(
        protocol=BenchmarkProtocol(
            requested_mode=selected,
            resolved_mode=BenchmarkMode.EVENT,
            implementation="triton_do_bench",
            cache_policy="cold_l2",
            warmup_ms=warmup_ms,
            measurement_ms=measurement_ms,
            n_retries=1,
            per_replay_ms=None,
            fallback_reason=fallback_reason,
        ),
        benchmark=event_benchmark,
    )
