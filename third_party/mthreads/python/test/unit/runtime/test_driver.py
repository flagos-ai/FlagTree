import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import torch
import pytest

import triton
import triton.language as tl


def test_is_lazy():
    from importlib import reload
    reload(sys.modules["triton.runtime.driver"])
    reload(sys.modules["triton.runtime"])
    assert triton.runtime.driver._active is None
    assert triton.runtime.driver._default is None
    assert isinstance(triton.runtime.driver.active, getattr(triton.backends.driver, "DriverBase"))
    assert isinstance(triton.runtime.driver.default, getattr(triton.backends.driver, "DriverBase"))
    utils = triton.runtime.driver.active.utils  # noqa: F841


def test_kernel_in_thread(device):
    # Test calling in a new thread sets a valid device context
    buf = torch.zeros((38016 * 1024, ), dtype=torch.float32, device=device)

    @triton.jit
    def _kernel(P, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offset = pid * BLOCK + tl.arange(0, BLOCK)

        p = tl.load(P + offset)
        tl.store(P + offset, p)

    def call_triton():
        N = buf.numel()
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
        _kernel[grid](buf, BLOCK=1024)
        getattr(torch, device).synchronize()

    call_triton()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(call_triton)
        future.result()


def test_default_backend_env_selects_driver(monkeypatch):
    from importlib import import_module
    driver_mod = import_module("triton.runtime.driver")

    class FakeNvidiaDriver:

        @staticmethod
        def is_active():
            return True

    class FakeMusaDriver:

        @staticmethod
        def is_active():
            return True

    monkeypatch.setenv("TRITON_DEFAULT_BACKEND", "mthreads")
    monkeypatch.setattr(
        driver_mod,
        "backends",
        {
            "nvidia": SimpleNamespace(driver=FakeNvidiaDriver),
            "mthreads": SimpleNamespace(driver=FakeMusaDriver),
        },
    )

    selected = driver_mod._create_driver()
    assert isinstance(selected, FakeMusaDriver)


def test_do_bench_device_type_selects_requested_driver(monkeypatch):
    import triton.testing as testing

    counters = {"active_clear": 0, "musa_clear": 0, "fn": 0}

    class FakeEvent:

        def __init__(self, enable_timing=True):
            self.enable_timing = enable_timing

        def record(self):
            return None

        def elapsed_time(self, other):
            return 1.0

    class FakeDeviceInterface:
        Event = FakeEvent

        @staticmethod
        def synchronize():
            return None

    class FakeActiveDriver:

        def get_device_interface(self):
            return FakeDeviceInterface()

        def get_empty_cache_for_benchmark(self):
            return object()

        def clear_cache(self, cache):
            counters["active_clear"] += 1

    class FakeMusaDriver:

        @staticmethod
        def is_active():
            return True

        def get_device_interface(self):
            return FakeDeviceInterface()

        def get_empty_cache_for_benchmark(self):
            return object()

        def clear_cache(self, cache):
            counters["musa_clear"] += 1

    monkeypatch.setattr(testing.runtime.driver, "_active", FakeActiveDriver(), raising=False)
    monkeypatch.setattr(
        testing,
        "_available_backends",
        {"mthreads": SimpleNamespace(driver=FakeMusaDriver)},
    )
    testing._get_backend_driver.cache_clear()

    def fn():
        counters["fn"] += 1

    testing.do_bench(fn, warmup=1, rep=1, device_type="musa")

    assert counters["fn"] > 0
    assert counters["active_clear"] == 0
    assert counters["musa_clear"] > 0


def test_do_bench_device_type_unknown_backend(monkeypatch):
    import triton.testing as testing

    monkeypatch.setattr(testing, "_available_backends", {})
    testing._get_backend_driver.cache_clear()

    with pytest.raises(RuntimeError, match="Unsupported device_type/backend"):
        testing.do_bench(lambda: None, warmup=1, rep=1, device_type="unknown")
