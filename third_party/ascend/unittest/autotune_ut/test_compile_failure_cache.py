# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import threading
from collections import defaultdict
from types import MethodType, SimpleNamespace

import pytest
import triton
import triton.language as tl
import triton.backends.ascend.runtime.autotuner as ascend_autotuner
from triton.backends.ascend.runtime.autotuner import AutoTilingTuner
from triton.runtime.autotuner import Config
from triton.runtime.errors import OutOfResources


@triton.jit
def _compile_key_kernel(
    x,
    m,
    BLOCK_SIZE: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pass


def _make_compile_key_tuner(monkeypatch, wrapped_fn):
    tuner = object.__new__(AutoTilingTuner)
    tuner.fn = wrapped_fn
    tuner.compile_parallel = False
    tuner.user_defined_pre_hook = False

    binder_calls = []

    def binder(*args, **kwargs):
        binder_calls.append(dict(kwargs))
        # Model m as a do_not_specialize runtime argument. Only constexpr
        # values injected by Config/Heuristics participate in specialization.
        specialization = [
            ("constexpr", kwargs["BLOCK_SIZE"]),
            ("constexpr", kwargs["EVEN_M"]),
        ]
        options = tuple((name, repr(kwargs.get(name))) for name in (
            "num_warps",
            "num_ctas",
            "num_stages",
            "debug",
            "instrumentation_mode",
        ))
        return {}, specialization, options

    monkeypatch.setattr(
        triton.runtime.driver,
        "_active",
        SimpleNamespace(get_current_device=lambda: 0),
    )
    monkeypatch.setattr(ascend_autotuner, "get_cache_invalidating_env_vars", lambda: {})
    monkeypatch.delenv("TRITON_ENABLE_UBTUNER", raising=False)
    monkeypatch.setattr(_compile_key_kernel, "hash", "test-jit-cache-key")
    monkeypatch.setattr(_compile_key_kernel, "debug", None)
    monkeypatch.setattr(_compile_key_kernel, "pre_run_hooks", [])
    monkeypatch.setattr(
        _compile_key_kernel,
        "device_caches",
        {0: ({}, {}, "test-target", None, binder)},
    )
    return tuner, binder_calls


def test_compile_failure_key_supports_direct_jit_function(monkeypatch):
    tuner, binder_calls = _make_compile_key_tuner(monkeypatch, _compile_key_kernel)
    config = Config({"BLOCK_SIZE": 128, "EVEN_M": True})

    compile_key = tuner._get_jit_compile_cache_key(
        object(),
        16,
        config=config,
        grid=(1, ),
    )

    assert compile_key is not None
    assert len(binder_calls) == 1
    assert binder_calls[0]["BLOCK_SIZE"] == 128
    assert binder_calls[0]["EVEN_M"] is True
    assert "grid" not in binder_calls[0]
    assert "warmup" not in binder_calls[0]


def test_compile_failure_key_replays_heuristics_before_binder(monkeypatch):
    heuristic_calls = []

    def even_m(args):
        heuristic_calls.append(dict(args))
        return args["m"] % 2 == 0

    wrapped_fn = triton.runtime.Heuristics(
        _compile_key_kernel,
        _compile_key_kernel.arg_names,
        {"EVEN_M": even_m},
    )
    tuner, binder_calls = _make_compile_key_tuner(monkeypatch, wrapped_fn)
    config = Config({"BLOCK_SIZE": 128})

    even_key = tuner._get_jit_compile_cache_key(
        object(),
        16,
        config=config,
        grid=(1, ),
    )
    odd_key = tuner._get_jit_compile_cache_key(
        object(),
        17,
        config=config,
        grid=(1, ),
    )
    second_even_key = tuner._get_jit_compile_cache_key(
        object(),
        18,
        config=config,
        grid=(1, ),
    )

    assert even_key is not None
    assert odd_key is not None
    assert second_even_key is not None
    assert even_key == second_even_key
    assert even_key != odd_key
    assert [call["EVEN_M"] for call in binder_calls] == [True, False, True]
    assert heuristic_calls[0]["BLOCK_SIZE"] == 128
    assert heuristic_calls[0]["grid"] == (1, )
    assert heuristic_calls[0]["warmup"] is False
    assert "grid" not in binder_calls[0]
    assert "warmup" not in binder_calls[0]


def test_compile_failure_key_rejects_unknown_wrapper(monkeypatch):

    class UnknownWrapper:

        def __init__(self, fn):
            self.fn = fn

    tuner, binder_calls = _make_compile_key_tuner(
        monkeypatch,
        UnknownWrapper(_compile_key_kernel),
    )
    config = Config({"BLOCK_SIZE": 128, "EVEN_M": True})

    compile_key = tuner._get_jit_compile_cache_key(
        object(),
        16,
        config=config,
        grid=(1, ),
    )

    assert compile_key is None
    assert binder_calls == []


def test_compile_failure_key_is_disabled_for_jit_pre_run_hook(monkeypatch):
    tuner, binder_calls = _make_compile_key_tuner(monkeypatch, _compile_key_kernel)
    monkeypatch.setattr(_compile_key_kernel, "pre_run_hooks", [lambda *args, **kwargs: None])
    config = Config({"BLOCK_SIZE": 128, "EVEN_M": True})

    compile_key = tuner._get_jit_compile_cache_key(
        object(),
        16,
        config=config,
        grid=(1, ),
    )

    assert compile_key is None
    assert binder_calls == []


def _make_batch_bench_tuner():
    tuner = object.__new__(AutoTilingTuner)
    tuner.compile_parallel = False
    tuner.enable_ubtuner = False
    tuner.print_autotuning = False
    tuner.user_defined_do_bench = True
    tuner._compile_failure_cache = {}
    tuner._compile_failure_cache_lock = threading.Lock()
    tuner._cached_compile_failed_configs = []

    kernel_calls = defaultdict(int)

    def get_compile_key(self, *args, config, **kwargs):
        return ("compile-key", config.kwargs["ID"])

    def make_kernel_call(self, *args, config, **kwargs):

        def kernel_call(warmup):
            kernel_calls[config] += 1
            if config.kwargs["ID"] == "bad":
                raise OutOfResources(2, 1, "UB")
            return None

        return kernel_call

    def do_bench(fn, quantiles):
        fn()
        return 1.0

    tuner._get_jit_compile_cache_key = MethodType(get_compile_key, tuner)
    tuner._make_kernel_call = MethodType(make_kernel_call, tuner)
    tuner.do_bench = do_bench
    return tuner, kernel_calls


def test_batch_bench_skips_config_cached_as_compile_failure():
    tuner, kernel_calls = _make_batch_bench_tuner()
    bad = Config({"ID": "bad"})
    good = Config({"ID": "good"})

    first_result = tuner._batch_bench(configs=[bad, good])

    assert first_result == {good: 1.0}
    assert kernel_calls[bad] == 1
    assert tuner._compile_failure_cache[("compile-key", "bad")] == {
        "exception_type": "OutOfResources",
    }
    assert tuner._cached_compile_failed_configs == []

    second_result = tuner._batch_bench(configs=[bad, good])

    assert second_result == {good: 1.0}
    assert kernel_calls[bad] == 1
    assert tuner._compile_failed_configs == []
    assert tuner._cached_compile_failed_configs == [bad]


def test_batch_bench_raises_when_all_configs_are_cached_failures():
    tuner, kernel_calls = _make_batch_bench_tuner()
    bad = Config({"ID": "bad"})
    tuner._remember_compile_failure(("compile-key", "bad"), OutOfResources(2, 1, "UB"))

    with pytest.raises(RuntimeError, match="All triton configs are cached compile failures"):
        tuner._batch_bench(configs=[bad])

    assert kernel_calls[bad] == 0
