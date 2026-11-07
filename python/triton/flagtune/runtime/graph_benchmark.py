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

import math
import time
from typing import Any, Callable, Sequence


def _timed_warmup(fn: Callable[[], Any], warmup_ms: float, device_interface: Any) -> None:
    if warmup_ms <= 0:
        return
    device_interface.synchronize()
    deadline = time.perf_counter() + warmup_ms / 1000.0
    batch = 1
    while time.perf_counter() < deadline:
        started = time.perf_counter()
        for _ in range(batch):
            fn()
        device_interface.synchronize()
        finished = time.perf_counter()
        elapsed = finished - started
        # Include asynchronous device execution before sizing the next batch.
        # Starting with one launch avoids submitting 50 very slow candidates
        # when the requested warmup is only a few milliseconds.
        if elapsed > 0:
            batch = max(1, min(50, int((deadline - finished) * batch / elapsed)))
        else:
            batch = 50


def do_bench_musa_graph(
    fn: Callable[[], Any],
    *,
    rep: float = 20,
    quantiles: Sequence[float] | None = None,
    n_retries: int = 10,
    warmup_ms: float = 0,
    device_interface: Any,
) -> Any:
    if not isinstance(n_retries, int) or isinstance(n_retries, bool) or n_retries <= 0:
        raise ValueError("n_retries must be a positive integer")

    graph_type = getattr(device_interface, "MUSAGraph", None)
    if graph_type is None or not hasattr(device_interface, "graph"):
        raise RuntimeError("the active MUSA device interface does not expose graph capture")

    with device_interface.stream(device_interface.Stream()):
        fn()
        _timed_warmup(fn, warmup_ms, device_interface)

        # Match do_bench_cudagraph: calibrate with ordinary launches before
        # capturing the one graph that is used for the actual measurement.
        start_event = device_interface.Event(enable_timing=True)
        end_event = device_interface.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            fn()
        end_event.record()
        device_interface.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        if not math.isfinite(estimate_ms) or estimate_ms <= 0:
            n_repeat = 1000
        else:
            n_repeat = max(1, min(1000, int(rep / estimate_ms)))

        graph = graph_type()
        with device_interface.graph(graph):
            for _ in range(n_repeat):
                fn()
        device_interface.synchronize()

        times = []
        for _ in range(n_retries):
            start_event = device_interface.Event(enable_timing=True)
            end_event = device_interface.Event(enable_timing=True)
            start_event.record()
            graph.replay()
            end_event.record()
            device_interface.synchronize()
            times.append(start_event.elapsed_time(end_event) / n_repeat)

    from triton.testing import _summarize_statistics

    return _summarize_statistics(times, quantiles, "mean")




def _capture_hcu_graph_duration(fn, count, api):
    """Return the graph and its median device replay duration in milliseconds."""
    graph = api.CUDAGraph()
    with api.graph(graph):
        for _ in range(count):
            fn()
    api.synchronize()
    graph.replay()
    api.synchronize()
    start, end = api.Event(enable_timing=True), api.Event(enable_timing=True)
    durations = []
    for _ in range(3):
        start.record()
        graph.replay()
        end.record()
        api.synchronize()
        durations.append(start.elapsed_time(end))
    return graph, sorted(durations)[1]


def do_bench_hcu_graph(
    fn: Callable[[], Any],
    *,
    rep: float = 20,
    quantiles: Sequence[float] | None = None,
    n_retries: int = 10,
    warmup_ms: float = 0,
    device_interface: Any = None,
) -> Any:
    """Size the measurement graph from device replays instead of host callbacks.

    LibEntry and post-hooks can make ordinary event timing mostly measure host
    submission gaps. Iteratively replay a pilot graph until its device duration
    reaches the requested budget, then measure that graph without Python calls.
    """
    if not isinstance(n_retries, int) or isinstance(n_retries, bool) or n_retries <= 0:
        raise ValueError("n_retries must be a positive integer")
    if not math.isfinite(rep) or rep <= 0:
        raise ValueError("rep must be finite and positive")
    if not math.isfinite(warmup_ms) or warmup_ms < 0:
        raise ValueError("warmup_ms must be finite and non-negative")
    if device_interface is None:
        import torch
        device_interface = torch.cuda
    api = device_interface
    caller = api.current_stream()
    stream = caller
    with api.stream(stream):
        fn()
        _timed_warmup(fn, warmup_ms, api)
        start = api.Event(enable_timing=True)
        end = api.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            fn()
        end.record()
        api.synchronize()
        ordinary_ms = start.elapsed_time(end) / 5
        count = max(1, int(rep / ordinary_ms)) if math.isfinite(ordinary_ms) and ordinary_ms > 0 else 1000
        # Short pilots include a fixed graph-launch/event cost. Recalibrate the
        # longer graph rather than extrapolating that cost per kernel forever.
        for attempt in range(3):
            graph, duration_ms = _capture_hcu_graph_duration(fn, count, api)
            if not math.isfinite(duration_ms) or duration_ms <= 0:
                next_count = 1000
            else:
                next_count = max(1, int(rep * count / duration_ms))
            if duration_ms >= 0.8 * rep or next_count == count:
                break
            count = next_count
        else:
            # Bound the calibration passes, then use the last extrapolation.
            graph = api.CUDAGraph()
            with api.graph(graph):
                for _ in range(count):
                    fn()
            api.synchronize()
        graph.replay()
        api.synchronize()
        times = []
        for _ in range(n_retries):
            start.record()
            graph.replay()
            end.record()
            api.synchronize()
            times.append(start.elapsed_time(end) / count)
    from triton.testing import _summarize_statistics
    return _summarize_statistics(times, quantiles, "mean")
