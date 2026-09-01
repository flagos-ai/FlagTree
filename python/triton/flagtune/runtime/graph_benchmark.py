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

import time
from typing import Any, Callable, Sequence


def _timed_warmup(fn: Callable[[], Any], warmup_ms: float, device_interface: Any) -> None:
    if warmup_ms <= 0:
        return
    device_interface.synchronize()
    deadline = time.perf_counter() + warmup_ms / 1000.0
    while time.perf_counter() < deadline:
        for _ in range(50):
            fn()
        device_interface.synchronize()


def _calibrate_n_repeat(
    fn: Callable[[], Any],
    rep: float,
    graph_type: Any,
    device_interface: Any,
    probe_iters: int = 256,
    max_n_repeat: int = 20000,
) -> int:
    probe = graph_type()
    with device_interface.graph(probe):
        for _ in range(probe_iters):
            fn()
    device_interface.synchronize()
    best = None
    for _ in range(3):
        start_event = device_interface.Event(enable_timing=True)
        end_event = device_interface.Event(enable_timing=True)
        start_event.record()
        probe.replay()
        end_event.record()
        device_interface.synchronize()
        sample = start_event.elapsed_time(end_event) / probe_iters
        best = sample if best is None else min(best, sample)
    if not best:
        return 1000
    # Cap the captured graph. A wedged muCtxSynchronize was observed on MTT
    # S5000 while sweeping tiny shapes, and an unbounded unroll makes both the
    # capture and every replay proportionally more exposed to it.
    return max(1, min(max_n_repeat, int(rep / best)))


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
        n_repeat = _calibrate_n_repeat(fn, rep, graph_type, device_interface)

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
