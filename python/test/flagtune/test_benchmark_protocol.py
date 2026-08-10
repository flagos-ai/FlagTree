import inspect
from types import SimpleNamespace

import pytest
import triton.testing as triton_testing

from triton.flagtune.runtime import benchmark_protocol as benchmark_module
from triton.testing import do_bench_cudagraph


class _FakeDriver:

    def __init__(self, backend="cuda"):
        self.backend = backend
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


def _driver_for(module_name, backend):
    driver_type = type("FakeDriver", (_FakeDriver, ), {"__module__": module_name})
    return driver_type(backend)


def test_cudagraph_helper_keeps_ten_retries_as_compatible_default():
    """Expose the old hard-coded retry count as an optional final argument."""
    parameter = inspect.signature(do_bench_cudagraph).parameters["n_retries"]

    assert parameter.default == 10
    assert list(inspect.signature(do_bench_cudagraph).parameters)[-1] == "n_retries"


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


def test_unsupported_replay_backend_warns_and_resolves_event(monkeypatch):
    # HCU exposes a HIP target but does not use AMD's graph replay implementation.
    active = _driver_for("triton.backends.hcu.driver", backend="hip")
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
