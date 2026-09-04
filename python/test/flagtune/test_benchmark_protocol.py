from contextlib import nullcontext
import inspect
from types import SimpleNamespace

import pytest
import triton.testing as triton_testing

from triton.flagtune.runtime import benchmark_protocol as benchmark_module
from triton.flagtune.runtime import graph_benchmark as graph_benchmark_module
from triton.flagtune.runtime.graph_benchmark import do_bench_musa_graph
from triton.testing import do_bench_cudagraph


class _FakeDriver:

    def __init__(self, backend="cuda"):
        self.backend = backend
        self.device_interface = object()
        self.observed = {}

    def get_current_target(self):
        return SimpleNamespace(backend=self.backend)

    def replay_benchmark(self, kernel_call, **kwargs):
        self.observed = dict(kwargs)
        kernel_call()
        return [1.0, 0.8, 1.2]

    def get_benchmarker(self):

        def benchmark(kernel_call, **kwargs):
            self.observed = dict(kwargs)
            kernel_call()
            return [2.0, 1.8, 2.2]

        return benchmark

    def get_device_interface(self):
        return self.device_interface


def _driver_for(module_name, backend):
    driver_type = type("FakeDriver", (_FakeDriver, ), {"__module__": module_name})
    return driver_type(backend)


def test_cudagraph_helper_keeps_ten_retries_as_compatible_default():
    """Expose the old hard-coded retry count as an optional final argument."""
    parameter = inspect.signature(do_bench_cudagraph).parameters["n_retries"]

    assert parameter.default == 10
    assert list(inspect.signature(do_bench_cudagraph).parameters)[-1] == "n_retries"


def test_musa_graph_helper_uses_supplied_device_interface():
    observed = {"captures": 0, "replays": 0, "synchronizes": 0}

    class _Event:

        def record(self):
            pass

        @staticmethod
        def elapsed_time(_other):
            return 5.0

    class _Graph:

        def replay(self):
            observed["replays"] += 1

    class _Interface:
        MUSAGraph = _Graph
        Stream = object

        @staticmethod
        def stream(_stream):
            return nullcontext()

        @staticmethod
        def graph(_graph):

            class _Capture:

                def __enter__(self):
                    observed["captures"] += 1

                def __exit__(self, _exc_type, _exc, _traceback):
                    return False

            return _Capture()

        @staticmethod
        def Event(enable_timing):
            assert enable_timing is True
            return _Event()

        @staticmethod
        def synchronize():
            observed["synchronizes"] += 1

    result = do_bench_musa_graph(
        lambda: None,
        rep=1,
        n_retries=2,
        device_interface=_Interface(),
    )

    # n_repeat is now calibrated from a 256-iteration probe graph instead of
    # five eager calls, so the fake 5.0ms elapsed time reads as 5.0/256 per
    # iteration and sizes the timing graph at int(rep / that) == 51.
    assert result == pytest.approx(5.0 / 51)
    # one probe capture plus the timing capture; three probe replays plus
    # n_retries timed replays; one synchronize after each capture and replay.
    assert observed == {"captures": 2, "replays": 5, "synchronizes": 7}


@pytest.mark.parametrize(
    ("module_name", "backend", "implementation"),
    [
        (
            "triton.backends.nvidia.driver",
            "cuda",
            "triton_cuda_graph_replay_v1",
        ),
        (
            "triton.backends.amd.driver",
            "hip",
            "triton_hip_graph_replay_v1",
        ),
        (
            "triton.backends.metax.driver",
            "maca",
            "triton_metax_graph_replay_v1",
        ),
        (
            "triton.backends.ppu.driver",
            "cuda",
            "triton_ppu_graph_replay_v1",
        ),
        (
            "triton.backends.hcu.driver",
            "hip",
            "triton_hcu_graph_replay_v1",
        ),
    ],
)
def test_replay_splits_total_measurement_budget(monkeypatch, module_name, backend, implementation):
    active = _driver_for(module_name, backend)
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=active))
    monkeypatch.setattr(triton_testing, "do_bench_cudagraph", active.replay_benchmark)
    launches = []

    resolved = benchmark_module.resolve_benchmarker(
        "replay",
        warmup_ms=25,
        measurement_ms=100,
        n_retries=10,
    )
    result = resolved.benchmark(lambda: launches.append(True), (0.5, 0.2, 0.8))

    assert result == [1.0, 0.8, 1.2]
    assert launches == [True]
    assert active.observed == {
        "rep": 10.0,
        "quantiles": (0.5, 0.2, 0.8),
        "n_retries": 10,
    }
    assert resolved.protocol.as_dict() == {
        "requested_mode": "replay",
        "resolved_mode": "replay",
        "implementation": implementation,
        "cache_policy": "warm_l2",
        "warmup_ms": 25,
        "measurement_ms": 100,
        "n_retries": 10,
        "per_replay_ms": 10.0,
        "fallback_reason": None,
    }
    assert resolved.protocol.cache_key() == (
        implementation,
        25,
        100,
        10,
        10.0,
    )


def test_mthreads_replay_uses_flagtune_musa_graph_helper(monkeypatch):
    active = _driver_for("triton.backends.mthreads.driver", "musa")
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=active))
    monkeypatch.setattr(
        graph_benchmark_module,
        "do_bench_musa_graph",
        active.replay_benchmark,
    )
    launches = []

    resolved = benchmark_module.resolve_benchmarker(
        "replay",
        warmup_ms=25,
        measurement_ms=100,
        n_retries=10,
    )
    result = resolved.benchmark(lambda: launches.append(True), (0.5, 0.2, 0.8))

    assert result == [1.0, 0.8, 1.2]
    assert launches == [True]
    assert active.observed == {
        "rep": 10.0,
        "quantiles": (0.5, 0.2, 0.8),
        "n_retries": 10,
        "warmup_ms": 25,
        "device_interface": active.device_interface,
    }
    assert resolved.protocol.as_dict() == {
        "requested_mode": "replay",
        "resolved_mode": "replay",
        "implementation": "triton_musa_graph_replay_v1",
        "cache_policy": "warm_l2",
        "warmup_ms": 25,
        "measurement_ms": 100,
        "n_retries": 10,
        "per_replay_ms": 10.0,
        "fallback_reason": None,
    }
    assert resolved.protocol.cache_key() == (
        "triton_musa_graph_replay_v1",
        25,
        100,
        10,
        10.0,
    )


def test_unsupported_replay_backend_warns_and_resolves_event(monkeypatch):
    active = _driver_for("triton.backends.example.driver", backend="example")
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=active))

    with pytest.warns(RuntimeWarning, match="falling back to event"):
        resolved = benchmark_module.resolve_benchmarker(
            "replay",
            warmup_ms=5,
            measurement_ms=20,
            n_retries=4,
        )
    result = resolved.benchmark(lambda: None, (0.5, 0.2, 0.8))

    assert result == [2.0, 1.8, 2.2]
    assert active.observed == {
        "warmup": 5,
        "rep": 20,
        "quantiles": (0.5, 0.2, 0.8),
    }
    assert resolved.protocol.resolved_mode is benchmark_module.BenchmarkMode.EVENT
    assert resolved.protocol.cache_key() == ("triton_do_bench", 5, 20)
    assert resolved.protocol.fallback_reason


@pytest.mark.parametrize("n_retries", [0, -1, True, 1.5])
def test_replay_rejects_invalid_retry_count(monkeypatch, n_retries):
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=_FakeDriver()))
    with pytest.raises(ValueError, match="n_retries"):
        benchmark_module.resolve_benchmarker(
            "replay",
            warmup_ms=5,
            measurement_ms=20,
            n_retries=n_retries,
        )
