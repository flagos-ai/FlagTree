from functools import wraps
from types import SimpleNamespace

import pytest
import torch
import triton.testing as triton_testing

from triton.flagtune.runtime import benchmark_protocol as benchmark_module


class _FakeDriver:

    def __init__(self, backend="npu"):
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


def test_flagtuner_defaults_to_npu_event_timing(monkeypatch):
    from triton.flagtune.runtime.autotuner import Flagtuner
    monkeypatch.delenv("FLAGTUNE_BENCHMARK_MODE", raising=False)
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=_FakeDriver()))
    tuner = Flagtuner(fn=lambda: None, arg_names=[], configs=[], key=[], reset_to_zero=None, restore_value=None)
    assert tuner.benchmark_protocol.requested_mode.value == "event"
    assert tuner.benchmark_protocol.resolved_mode.value == "event"


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


@pytest.mark.parametrize("mode", ["event", "replay"])
def test_ascend_uses_driver_event_benchmarker(monkeypatch, mode):
    from triton.flagtune.runtime.autotuner import Flagtuner

    active = _driver_for("triton.backends.ascend.driver", "npu")
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=active))
    monkeypatch.delenv("FLAGTUNE_BENCHMARK_MODE", raising=False)

    def no_cuda_graph(*args, **kwargs):
        pytest.fail("Ascend must not call the CUDA graph helper")

    monkeypatch.setattr(triton_testing, "do_bench_cudagraph", no_cuda_graph)

    def create_tuner():
        return Flagtuner(
            fn=lambda: None,
            arg_names=[],
            configs=[],
            key=[],
            reset_to_zero=None,
            restore_value=None,
            benchmark_mode=mode,
            warmup=5,
            rep=20,
        )

    if mode == "replay":
        with pytest.warns(RuntimeWarning, match="falling back to event"):
            tuner = create_tuner()
    else:
        tuner = create_tuner()
    calls = []
    assert tuner.do_bench(lambda: calls.append(True), (0.5, 0.2, 0.8)) == [2.0, 1.8, 2.2]
    assert calls == [True]
    assert active.observed == {"warmup": 5, "rep": 20, "quantiles": (0.5, 0.2, 0.8)}
    assert tuner.benchmark_protocol.requested_mode.value == mode
    assert tuner.benchmark_protocol.resolved_mode.value == "event"


def test_ascend_strict_replay_request_is_rejected(monkeypatch):
    active = _driver_for("triton.backends.ascend.driver", "npu")
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=active))
    with pytest.raises(RuntimeError, match="does not provide a registered replay"):
        benchmark_module.resolve_benchmarker("replay", warmup_ms=5, measurement_ms=20, allow_fallback=False)


def test_ascend_replay_uses_cudagraph_with_torchnpu_and_preserves_budget(monkeypatch):
    active = _driver_for("triton.backends.ascend.driver", "npu")
    calls = []
    active.device_interface = SimpleNamespace(
        NPUGraph=object,
        graph=lambda: None,
        Event=object,
        Stream=object,
        stream=lambda: None,
        synchronize=lambda: None,
        current_stream=lambda: SimpleNamespace(synchronize=lambda: calls.append("synchronize")),
    )
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=active))
    quantiles = (0.5, 0.2, 0.8)

    def install_compat(module):
        assert module == "torch_npu.contrib.transfer_to_npu"
        calls.append("import")
        cuda = SimpleNamespace(**vars(active.device_interface))
        cuda.CUDAGraph = cuda.NPUGraph

        @wraps(cuda.synchronize)
        def sync():
            active.device_interface.synchronize()

        cuda.synchronize = sync
        monkeypatch.setattr(torch, "cuda", cuda)

    def graph_benchmark(kernel_call, *, rep, quantiles):
        assert (rep, quantiles) == (2.0, (0.5, 0.2, 0.8))
        kernel_call()
        return [1.0, 0.8, 1.2]

    monkeypatch.setattr(benchmark_module, "import_module", install_compat)
    monkeypatch.setattr(triton_testing, "do_bench_cudagraph", graph_benchmark)
    with pytest.warns(RuntimeWarning, match="uses 10 replays, requested 4"):
        resolved = benchmark_module.resolve_benchmarker("replay", warmup_ms=5, measurement_ms=20, n_retries=4,
                                                        allow_fallback=False)
    assert resolved.benchmark(lambda: calls.append("kernel"), quantiles) == [1.0, 0.8, 1.2]
    assert calls == ["import", "synchronize", "kernel"]
    assert not active.observed
    assert resolved.protocol.as_dict() == {
        "requested_mode": "replay",
        "resolved_mode": "replay",
        "implementation": "triton_ascend_transfer_to_npu_graph_replay_v2",
        "cache_policy": "warm_l2",
        "warmup_ms": 0,
        "measurement_ms": 20,
        "n_retries": 10,
        "per_replay_ms": 2.0,
        "fallback_reason": None,
    }


def test_ascend_incomplete_cuda_aliases_fall_back_to_events(monkeypatch):
    active = _driver_for("triton.backends.ascend.driver", "npu")
    active.device_interface = SimpleNamespace(NPUGraph=object, graph=lambda: None)
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=active))
    monkeypatch.setattr(benchmark_module, "import_module", lambda _: None)
    monkeypatch.setattr(torch, "cuda", SimpleNamespace(CUDAGraph=object, graph=lambda: None))
    with pytest.warns(RuntimeWarning, match="did not redirect torch.cuda.graph"):
        resolved = benchmark_module.resolve_benchmarker("replay", warmup_ms=5, measurement_ms=20)
    assert resolved.protocol.resolved_mode is benchmark_module.BenchmarkMode.EVENT


@pytest.mark.parametrize("backend", ["cuda", "hip", "maca", "musa"])
@pytest.mark.parametrize("mode", ["event", "replay"])
def test_non_npu_benchmarkers_are_rejected(monkeypatch, backend, mode):
    from triton.flagtune.runtime.device import UnsupportedFlagTuneDeviceError
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=_FakeDriver(backend)))
    with pytest.raises(UnsupportedFlagTuneDeviceError, match="only Ascend"):
        benchmark_module.resolve_benchmarker(mode, warmup_ms=5, measurement_ms=20)
