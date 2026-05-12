import types

import pytest
import triton
from triton.compiler import compiler as triton_compiler
from triton.runtime.autotuner import Autotuner


class _DummyKernel:

    def __init__(self, error):
        self.fn = lambda: None
        self._error = error

    def run(self, *args, **kwargs):
        raise self._error


def _make_compiled_kernel(shared, num_warps=1):
    kernel = triton_compiler.CompiledKernel.__new__(triton_compiler.CompiledKernel)
    kernel.module = None
    kernel.function = None
    kernel._run = None
    kernel.src = object()
    kernel.metadata = types.SimpleNamespace(shared=shared, num_warps=num_warps, tmem_size=None)
    kernel.metadata_group = {}
    kernel.hash = "dummy_hash"
    kernel.name = "dummy_kernel"
    kernel.kernel = b"\x00"
    return kernel


def test_compiled_kernel_raises_out_of_resources(monkeypatch):

    def load_binary(*args, **kwargs):
        raise AssertionError("load_binary should not be called")

    active_driver = triton_compiler.driver.active
    monkeypatch.setattr(active_driver, "get_current_device", lambda: 0, raising=False)
    monkeypatch.setattr(active_driver, "get_current_target", lambda: types.SimpleNamespace(warp_size=32), raising=False)
    monkeypatch.setattr(active_driver, "launcher_cls", lambda src, metadata: lambda *args, **kwargs: None,
                        raising=False)
    monkeypatch.setattr(active_driver.utils, "get_device_properties", lambda device: {"max_shared_mem": 0},
                        raising=False)
    monkeypatch.setattr(active_driver.utils, "load_binary", load_binary, raising=False)
    triton_compiler.max_shared_mem.cache_clear()

    kernel = _make_compiled_kernel(shared=1)
    with pytest.raises(triton.OutOfResources):
        kernel._init_handles()


def test_compiled_kernel_loads_within_shared_limit(monkeypatch):
    calls = {}

    def load_binary(name, data, shared, device):
        calls["args"] = (name, data, shared, device)
        return ("mod", "func", 0, 0, 1024)

    active_driver = triton_compiler.driver.active
    monkeypatch.setattr(active_driver, "get_current_device", lambda: 0, raising=False)
    monkeypatch.setattr(active_driver, "get_current_target", lambda: types.SimpleNamespace(warp_size=32), raising=False)
    monkeypatch.setattr(active_driver, "launcher_cls", lambda src, metadata: lambda *args, **kwargs: None,
                        raising=False)
    monkeypatch.setattr(active_driver.utils, "get_device_properties", lambda device: {"max_shared_mem": 1024},
                        raising=False)
    monkeypatch.setattr(active_driver.utils, "load_binary", load_binary, raising=False)
    triton_compiler.max_shared_mem.cache_clear()

    kernel = _make_compiled_kernel(shared=1, num_warps=1)
    kernel._init_handles()
    assert calls["args"] == ("dummy_kernel", b"\x00", 1, 0)
    assert kernel.module == "mod"
    assert kernel.function == "func"


def test_autotuner_drops_out_of_resources():
    err = triton.OutOfResources(128, 64, "shared memory")
    fn = _DummyKernel(err)

    def fake_do_bench(kernel_call, quantiles):
        kernel_call()
        return [0.0, 0.0, 0.0]

    autotuner = Autotuner(
        fn=fn,
        arg_names=[],
        configs=[triton.Config(kwargs={})],
        key=[],
        reset_to_zero=None,
        restore_value=None,
        do_bench=fake_do_bench,
    )
    autotuner.nargs = {}
    result = autotuner._bench(config=autotuner.configs[0])
    assert result == [float("inf"), float("inf"), float("inf")]
