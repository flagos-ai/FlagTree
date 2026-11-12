import contextlib
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

    def replay_benchmark(self, kernel_call, *, n_retries=10, **kwargs):
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

    # Calibration uses five ordinary launches, so the fake 5.0ms elapsed time
    # yields a 1.0ms estimate and rep=1 produces a one-launch timing graph.
    assert result == pytest.approx(5.0)
    # The implementation performs one formal capture and n_retries replays.
    # Synchronization occurs after calibration, capture, and each replay.
    assert observed == {"captures": 1, "replays": 2, "synchronizes": 4}


def test_musa_graph_helper_uses_bounded_repeat_for_invalid_calibration():
    """A zero event estimate must not divide by zero or skip graph capture."""
    observed = {"captures": 0, "replays": 0}

    class _Event:

        def record(self):
            pass

        @staticmethod
        def elapsed_time(_other):
            return 0.0

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
            pass

    result = do_bench_musa_graph(
        lambda: None,
        rep=1,
        n_retries=3,
        device_interface=_Interface(),
    )

    assert result == pytest.approx(0.0)
    assert observed == {"captures": 1, "replays": 3}


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
            "triton_hcu_graph_calibrated_currentstream_adaptivewarmup_v5",
        ),
    ],
)
def test_replay_splits_total_measurement_budget(monkeypatch, module_name, backend, implementation):
    active = _driver_for(module_name, backend)
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=active))
    monkeypatch.setattr(triton_testing, "do_bench_cudagraph", active.replay_benchmark)
    if module_name == "triton.backends.hcu.driver":
        monkeypatch.setattr(graph_benchmark_module, "do_bench_hcu_graph", active.replay_benchmark)
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
    expected_observed = {
        "rep": 10.0,
        "quantiles": (0.5, 0.2, 0.8),
        "n_retries": 10,
    }
    if module_name == "triton.backends.hcu.driver":
        expected_observed.update(warmup_ms=25, device_interface=active.device_interface)
    assert active.observed == expected_observed
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
        "implementation": "triton_musa_graph_replay_adaptivewarmup_v2",
        "cache_policy": "warm_l2",
        "warmup_ms": 25,
        "measurement_ms": 100,
        "n_retries": 10,
        "per_replay_ms": 10.0,
        "fallback_reason": None,
    }
    assert resolved.protocol.cache_key() == (
        "triton_musa_graph_replay_adaptivewarmup_v2",
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


class _HcuModelStream:
    def __init__(self, api, name):
        self.api, self.name = api, name
        self.dependencies = set()

    def wait_stream(self, other):
        self.dependencies.update({other, *other.dependencies})
        self.api.waits.append((self, other))


class _HcuTimingModel:
    def __init__(self, gpu_ms, host_ms, *, use_default_stream=True):
        self.gpu_ms, self.host_ms = gpu_ms, host_ms
        self.time, self.capture, self.calls, self.lengths = 0., None, 0, []
        self.default = _HcuModelStream(self, "default")
        self.caller = self.default if use_default_stream else _HcuModelStream(self, "existing_caller")
        self.active = self.caller
        self.stream_constructions, self.waits, self.callback_streams = 0, [], []
        self.incoming_observations = 0

    def current_stream(self):
        return self.active

    def Stream(self):
        self.stream_constructions += 1
        return _HcuModelStream(self, f"side_{self.stream_constructions}")

    @contextlib.contextmanager
    def stream(self, stream):
        assert stream.api is self
        previous, self.active = self.active, stream
        try:
            yield
        finally:
            self.active = previous

    def synchronize(self):
        pass

    def _assert_incoming_dependency(self):
        # Same-stream issue order already preserves incoming caller work. A new
        # side stream instead requires a dependency on that particular caller.
        assert self.active is self.caller or self.caller in self.active.dependencies, (
            "incoming producer stream dependency must precede callbacks")

    def fn(self):
        self._assert_incoming_dependency()
        self.incoming_observations += 1
        self.callback_streams.append(self.active)
        self.calls += 1
        if self.capture is not None:
            self.capture.n += 1
        else:
            self.time += self.gpu_ms + self.host_ms

    def Event(self, **kwargs):
        api = self
        class E:
            def record(self):
                self.time = api.time
            def elapsed_time(self, other):
                return other.time - self.time
        return E()

    def CUDAGraph(self):
        api = self
        class G:
            n = 0
            def replay(self):
                api._assert_incoming_dependency()
                api.time += self.n * api.gpu_ms + .008
        return G()

    @contextlib.contextmanager
    def graph(self, graph):
        assert self.capture is None
        self.capture = graph
        try:
            yield
        finally:
            self.lengths.append(graph.n)
            self.capture = None


@pytest.mark.parametrize("device_ms,host_ms", [(0.006, 0.3), (0.006, 0.03), (0.0072, 0.03), (2.0, 0.3)])
def test_hcu_graph_calibration_excludes_python_submission_gaps(device_ms, host_ms):
    api = _HcuTimingModel(device_ms, host_ms)
    result = graph_benchmark_module.do_bench_hcu_graph(
        api.fn, rep=1., quantiles=(0.5, 0.2, 0.8), n_retries=2, device_interface=api,
    )[0]
    assert abs(result / device_ms - 1.) < 0.025
    if device_ms < 0.01:
        assert api.lengths[-1] >= 80


def test_hcu_graph_ranking_survives_tenfold_difference_in_host_overhead():
    measured = []
    for device_ms, host_ms in [(0.006, 0.3), (0.0072, 0.03)]:
        api = _HcuTimingModel(device_ms, host_ms)
        measured.append(graph_benchmark_module.do_bench_hcu_graph(
            api.fn, rep=1., quantiles=(0.5, 0.2, 0.8), n_retries=2, device_interface=api,
        )[0])
    assert measured[0] < measured[1]


def test_hcu_graph_calibration_bounds_finite_near_zero_timings(monkeypatch):
    api = _HcuTimingModel(0.006, 0.3)
    monkeypatch.setattr(graph_benchmark_module, "_HCU_MAX_GRAPH_CALLS", 64)
    original = graph_benchmark_module._capture_hcu_graph_duration

    def near_zero_pilot(fn, count, interface):
        assert 0 < count <= 64
        graph, _ = original(fn, count, interface)
        return graph, 1e-300

    monkeypatch.setattr(graph_benchmark_module, "_capture_hcu_graph_duration", near_zero_pilot)
    result = graph_benchmark_module.do_bench_hcu_graph(
        api.fn, rep=1., n_retries=2, device_interface=api,
    )
    assert result > 0
    assert api.lengths[-1] == 64


@pytest.mark.parametrize("use_default_stream", [False, True])
def test_hcu_graph_preserves_incoming_work_on_current_caller(use_default_stream):
    api = _HcuTimingModel(0.006, 0.03, use_default_stream=use_default_stream)
    caller = api.current_stream()
    result = graph_benchmark_module.do_bench_hcu_graph(
        api.fn, rep=1., n_retries=2, device_interface=api,
    )
    assert result > 0
    assert api.incoming_observations == api.calls > 0
    assert api.current_stream() is caller


@pytest.mark.parametrize("use_default_stream", [False, True])
def test_hcu_timing_model_rejects_unordered_side_stream(use_default_stream):
    api = _HcuTimingModel(0.006, 0.03, use_default_stream=use_default_stream)
    caller, side = api.current_stream(), api.Stream()
    with api.stream(side):
        with pytest.raises(AssertionError, match="incoming producer stream dependency"):
            api.fn()
        side.wait_stream(caller)
        api.fn()
    assert api.current_stream() is caller
    assert api.calls == api.incoming_observations == 1


@pytest.mark.parametrize("use_default_stream", [False, True])
def test_hcu_graph_repeated_calls_reuse_current_stream(use_default_stream):
    # Actual runlist oversubscription motivated this resource regression. This
    # CPU model counts explicit helper Stream() requests, not HIP queue count;
    # the graph API's internal capture-stream behavior is deliberately unchanged.
    api = _HcuTimingModel(0.006, 0.03, use_default_stream=use_default_stream)
    caller, initial = api.current_stream(), api.stream_constructions
    for _ in range(6):
        graph_benchmark_module.do_bench_hcu_graph(
            api.fn, rep=1., n_retries=2, device_interface=api,
        )
    assert api.stream_constructions == initial
    assert all(stream is caller for stream in api.callback_streams)
    assert api.current_stream() is caller


@pytest.mark.parametrize("measurement_ms", [10, 50])
def test_hcu_currentstream_protocol_isolates_legacy_selection_and_timing_cache(monkeypatch, measurement_ms):
    """MM/splitK and MV must not reuse side-stream-era cache identities."""
    import hashlib
    from dataclasses import replace

    active = _driver_for("triton.backends.hcu.driver", "hip")
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=active))
    monkeypatch.setattr(graph_benchmark_module, "do_bench_hcu_graph", active.replay_benchmark)
    resolved = benchmark_module.resolve_benchmarker(
        "replay", warmup_ms=5, measurement_ms=measurement_ms, n_retries=10,
    )
    protocol = resolved.protocol
    current = protocol.cache_key()
    legacy = replace(protocol, implementation="triton_hcu_graph_calibrated_v2").cache_key()
    assert current == ("triton_hcu_graph_calibrated_currentstream_adaptivewarmup_v5", 5, measurement_ms, 10,
                       measurement_ms / 10.0)
    assert current[1:] == legacy[1:]

    # ConfigCache namespaces hash the complete protocol; BenchmarkCache keys
    # append it to the shape/dtype. Exercise both consumer key forms without
    # importing FlagGems or requiring a device or a persistent database.
    selection_suffix = lambda key: hashlib.sha256(repr(tuple(key)).encode("utf-8")).hexdigest()[:12]
    shape_dtype = (224, 256, 2048, "torch.bfloat16")
    selected_cache = {selection_suffix(legacy): "legacy-best-config"}
    timing_cache = {shape_dtype + legacy: (8.0, 7.9, 8.1)}
    assert selected_cache.get(selection_suffix(current)) is None
    assert timing_cache.get(shape_dtype + current) is None
    selected_cache[selection_suffix(current)] = "current-best-config"
    timing_cache[shape_dtype + current] = (7.8, 7.7, 7.9)
    assert selected_cache[selection_suffix(legacy)] == "legacy-best-config"
    assert timing_cache[shape_dtype + legacy] == (8.0, 7.9, 8.1)

    calls = []
    resolved.benchmark(lambda: calls.append(True), (0.5, 0.2, 0.8))
    assert calls == [True]
    assert active.observed == dict(rep=measurement_ms / 10.0, quantiles=(0.5, 0.2, 0.8),
                                   n_retries=10, warmup_ms=5, device_interface=active.device_interface)
