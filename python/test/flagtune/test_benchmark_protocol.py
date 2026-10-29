import inspect
import sys
from types import SimpleNamespace

import pytest
import triton.testing as triton_testing

from triton.flagtune.runtime import benchmark_protocol as benchmark_module
from triton.testing import do_bench_cudagraph


class _FakeDriver:

    def __init__(self, backend="cuda"):
        self.backend = backend
        self.device_interface = object()
        self.observed = {}

    def get_current_target(self):
        return SimpleNamespace(backend=self.backend)

    def replay_benchmark(self, kernel_call, n_retries=10, **kwargs):
        self.observed = dict(kwargs, n_retries=n_retries)
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


@pytest.mark.parametrize("use_default_stream", [False, True])
def test_cudagraph_helper_waits_for_caller_stream(use_default_stream):
    """Preserve caller-stream work that precedes the benchmark warmup."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-compatible device")
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("requires torch.cuda._sleep to widen the ordering window")

    # Initialize lazy graph state before testing stream ordering.
    scratch = torch.zeros(1024, device="cuda")
    do_bench_cudagraph(lambda: scratch.zero_(), rep=1, n_retries=1)

    buf = torch.ones(1024, device="cuda")
    torch.cuda.synchronize()
    caller_stream = (torch.cuda.default_stream() if use_default_stream else torch.cuda.Stream())
    with torch.cuda.stream(caller_stream):
        torch.cuda._sleep(100_000_000)
        prior_read = buf.clone()
        do_bench_cudagraph(lambda: buf.fill_(2), rep=1, n_retries=1)

    torch.cuda.synchronize()
    assert prior_read.unique().tolist() == [1.0]


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


def test_mthreads_replay_initializes_torchada_and_uses_shared_helper(monkeypatch):
    active = _driver_for("triton.backends.mthreads.driver", "musa")
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=active))
    initialized = []

    def get_backend():
        initialized.append(True)
        return active.device_interface

    monkeypatch.setitem(sys.modules, "torchada", SimpleNamespace(get_backend=get_backend))

    def shared_benchmarker(kernel_call, n_retries=10, **kwargs):
        assert initialized == [True]
        return active.replay_benchmark(kernel_call, n_retries=n_retries, **kwargs)

    monkeypatch.setattr(triton_testing, "do_bench_cudagraph", shared_benchmarker)
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
    assert initialized == [True]
    assert active.observed == {
        "rep": 10.0,
        "quantiles": (0.5, 0.2, 0.8),
        "n_retries": 10,
    }
    assert resolved.protocol.as_dict() == {
        "requested_mode": "replay",
        "resolved_mode": "replay",
        "implementation": "triton_musa_torchada_graph_replay_v3",
        "cache_policy": "warm_l2",
        "warmup_ms": 0,
        "measurement_ms": 100,
        "n_retries": 10,
        "per_replay_ms": 10.0,
        "fallback_reason": None,
    }
    assert resolved.protocol.cache_key() == (
        "triton_musa_torchada_graph_replay_v3",
        0,
        100,
        10,
        10.0,
    )


@pytest.mark.parametrize("import_fails", [False, True])
def test_mthreads_replay_preserves_constexpr_helper_during_import(monkeypatch, import_fails):
    import builtins
    import triton

    active = _driver_for("triton.backends.mthreads.driver", "musa")
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=active))
    original_helper = triton.next_power_of_2
    original_import = builtins.__import__

    def patched_import(name, *args, **kwargs):
        if name == "torchada":
            monkeypatch.setattr(triton, "next_power_of_2", lambda value: value)
            if import_fails:
                raise ImportError("missing TorchADA dependency")
            return SimpleNamespace(get_backend=lambda: active.device_interface)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", patched_import)
    if import_fails:
        with pytest.raises(benchmark_module.FlagTuneDependencyError, match="requires torchada"):
            benchmark_module.resolve_benchmarker("replay", warmup_ms=25, measurement_ms=100)
    else:
        benchmark_module.resolve_benchmarker("replay", warmup_ms=25, measurement_ms=100)

    assert triton.next_power_of_2 is original_helper


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


@pytest.mark.parametrize(
    ("env_value", "explicit", "expected"),
    [
        (None, None, "event"),
        ("event", None, "event"),
        ("replay", None, "replay"),
        ("replay", "event", "event"),
    ],
)
def test_resolve_requested_mode_honors_explicit_or_environment(monkeypatch, env_value, explicit, expected):
    if env_value is None:
        monkeypatch.delenv("FLAGTUNE_BENCHMARK_MODE", raising=False)
    else:
        monkeypatch.setenv("FLAGTUNE_BENCHMARK_MODE", env_value)
    assert benchmark_module.resolve_requested_mode(explicit).value == expected


def test_resolve_requested_mode_rejects_unknown_environment_value(monkeypatch):
    monkeypatch.setenv("FLAGTUNE_BENCHMARK_MODE", "protocol")
    with pytest.raises(ValueError):
        benchmark_module.resolve_requested_mode()


def test_flagtuner_default_mode_preserves_replay(monkeypatch):
    monkeypatch.delenv("FLAGTUNE_BENCHMARK_MODE", raising=False)
    assert (benchmark_module.resolve_requested_mode(default=benchmark_module.BenchmarkMode.REPLAY)
            is benchmark_module.BenchmarkMode.REPLAY)


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
