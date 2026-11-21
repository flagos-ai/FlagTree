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

from types import MethodType, SimpleNamespace

import pytest
import triton
import triton.backends.ascend.runtime.autotuner as ascend_autotuner
from triton.runtime.autotuner import Config
from triton.backends.ascend.runtime.autotuner import AutoTilingTuner


def _make_tuner(do_bench):
    tuner = object.__new__(AutoTilingTuner)
    tuner.compile_parallel = False
    tuner.do_bench = do_bench
    tuner.user_defined_do_bench = True

    def _make_kernel_call(self, *args, config, **meta):

        def kernel_call(warmup):
            return None

        return kernel_call

    tuner._make_kernel_call = MethodType(_make_kernel_call, tuner)
    return tuner


def _make_run_tuner(configs):
    key = ("disk-cache-key", )
    tuner = object.__new__(AutoTilingTuner)

    def generate_key_and_configs(*args, **kwargs):
        tuner.nargs = {}
        return key

    tuner.arg_names = []
    tuner.cache = {}
    tuner.is_simt_mode = False
    tuner.generate_key_and_configs = generate_key_and_configs
    tuner.prune_configs = lambda kwargs: configs
    tuner.enable_ubtuner = False
    tuner.cache_results = True
    tuner.print_autotuning = False
    tuner.auto_profile_dir = None
    tuner.nargs = {}
    tuner.pre_hook = lambda kwargs, reset_only=False: None
    tuner.run_kwargs = []
    tuner.fn = SimpleNamespace(run=lambda *args, **kwargs: tuner.run_kwargs.append(kwargs) or "kernel-result")
    return tuner, key


def test_batch_bench_supports_do_bench_with_quantiles():
    record = {}

    def _do_bench(fn, quantiles):
        record["quantiles"] = quantiles
        fn()
        return (1.0, 1.0, 1.0)

    tuner = _make_tuner(_do_bench)
    cfg = Config({})

    result = tuner._batch_bench(configs=[cfg])

    assert result[cfg] == (1.0, 1.0, 1.0)
    assert record["quantiles"] == (0.5, 0.2, 0.8)


def test_batch_bench_requires_do_bench_quantiles_parameter():

    def _do_bench(fn):
        fn()
        return (2.0, 2.0, 2.0)

    tuner = _make_tuner(_do_bench)
    cfg = Config({})

    with pytest.raises(TypeError):
        tuner._batch_bench(configs=[cfg])


def test_batch_bench_npu_env_respects_user_do_bench(monkeypatch):
    calls = {"do_bench": 0}

    def _do_bench(fn, quantiles):
        calls["do_bench"] += 1
        fn()
        return (3.0, 3.0, 3.0)

    def _unexpected_do_bench_npu(*args, **kwargs):
        raise AssertionError("do_bench_npu should not be used when user do_bench is provided")

    tuner = _make_tuner(_do_bench)
    cfg0 = Config({"ID": 0})
    cfg1 = Config({"ID": 1})
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    monkeypatch.setattr("triton.backends.ascend.testing.do_bench_npu", _unexpected_do_bench_npu)

    result = tuner._batch_bench(configs=[cfg0, cfg1])

    assert calls["do_bench"] == 2
    assert result[cfg0] == (3.0, 3.0, 3.0)
    assert result[cfg1] == (3.0, 3.0, 3.0)


def test_batch_bench_npu_env_uses_do_bench_npu_without_user_do_bench(monkeypatch):

    def _do_bench(fn, quantiles):
        raise AssertionError("self.do_bench should not be used when no user do_bench is provided")

    calls = {"do_bench_npu": 0}

    def _do_bench_npu(funcs, clear_l2_cache=False, warmup=5, active=30, target_kernel_name=None, **kwargs):
        calls["do_bench_npu"] += 1
        assert len(funcs) == 2
        assert clear_l2_cache is True
        return [1.0, 2.0]

    tuner = _make_tuner(_do_bench)
    tuner.user_defined_do_bench = False
    cfg0 = Config({"ID": 0})
    cfg1 = Config({"ID": 1})
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    monkeypatch.setattr("triton.backends.ascend.testing.do_bench_npu", _do_bench_npu)

    result = tuner._batch_bench(configs=[cfg0, cfg1])

    assert calls["do_bench_npu"] == 1
    assert result[cfg0] == 1.0
    assert result[cfg1] == 2.0


def test_autotilingtuner_marks_user_defined_do_bench():
    marker = {"called": False}

    def _do_bench(fn, quantiles):
        marker["called"] = True
        return (0.0, 0.0, 0.0)

    def _dummy_kernel():
        return None

    _dummy_kernel.arg_names = []

    tuner = AutoTilingTuner(
        _dummy_kernel,
        [],
        [Config({})],
        [],
        None,
        None,
        do_bench=_do_bench,
    )

    assert tuner.user_defined_do_bench is True
    assert marker["called"] is False


def test_print_benchmark_results_formats_quantiles_and_selected_config(capsys):

    def _dummy_kernel():
        return None

    selected = Config({"BLOCK_SIZE": 128}, num_warps=4)
    other = Config({"BLOCK_SIZE": 256}, num_warps=8)
    tuner = object.__new__(AutoTilingTuner)
    tuner.print_autotuning = True
    tuner.base_fn = _dummy_kernel
    tuner.best_config = selected

    tuner._print_benchmark_results({
        selected: (1.23456, 1.0, 1.8),
        other: 2.34567,
    })

    output = capsys.readouterr().out
    assert "Triton autotuning benchmark results for function _dummy_kernel:" in output
    assert "config=BLOCK_SIZE: 128, num_warps: 4" in output
    assert "p50=1.2346 ms, p20=1.0000 ms, p80=1.8000 ms [selected]" in output
    assert "config=BLOCK_SIZE: 256, num_warps: 8" in output
    assert "mean=2.3457 ms" in output


def test_print_benchmark_results_is_disabled_without_debug_flag(capsys):
    tuner = object.__new__(AutoTilingTuner)
    tuner.print_autotuning = False

    tuner._print_benchmark_results({Config({"BLOCK_SIZE": 128}): 1.0})

    assert capsys.readouterr().out == ""


def test_run_prints_benchmark_results_after_tuning(capsys):

    def _dummy_kernel():
        return None

    selected = Config({"BLOCK_SIZE": 128}, num_warps=4)
    other = Config({"BLOCK_SIZE": 256}, num_warps=8)
    tuner, _ = _make_run_tuner([selected, other])
    tuner.cache_results = False
    tuner.print_autotuning = True
    tuner.base_fn = _dummy_kernel
    tuner.arg_names = ["cache_key", "input"]
    tuner.keys = ["cache_key"]
    tuner.nargs = {"cache_key": "disk-cache-key", "input": SimpleNamespace(dtype="torch.float32")}
    tuner.user_specified_multibuffer_mode = (2, 3)
    tuner.generate_key_and_configs = lambda *args, **kwargs: ("disk-cache-key", "torch.float32",
                                                              ("compile_mode", "simd"), ("multibuffer_mode", (2, 3)))
    tuner._batch_bench = lambda *args, configs, **kwargs: {
        selected: (1.0, 0.9, 1.1),
        other: (2.0, 1.8, 2.2),
    }

    assert tuner.run() == "kernel-result"
    output = capsys.readouterr().out
    assert "with key as " not in output
    assert ("with key fields as cache_key='disk-cache-key', input.dtype='torch.float32', "
            "compile_mode='simd', multibuffer_mode=(2, 3),") in output
    assert "Triton autotuning benchmark results for function _dummy_kernel:" in output
    assert "p50=1.0000 ms, p20=0.9000 ms, p80=1.1000 ms [selected]" in output


def test_ascend_autotune_decorator_forwards_do_bench(monkeypatch):
    import triton.backends.ascend.runtime.autotuner as ascend_autotuner

    captured = {}

    class DummyAutoTilingTuner:

        def __init__(self, *args, **kwargs):
            captured["do_bench"] = kwargs.get("do_bench")

    monkeypatch.setattr(ascend_autotuner, "AutoTilingTuner", DummyAutoTilingTuner)

    def _dummy_kernel():
        return None

    _dummy_kernel.arg_names = []
    my_do_bench = lambda kernel_call, quantiles: (0.0, 0.0, 0.0)

    ascend_autotuner.autotune(configs=[object()], key=[], do_bench=my_do_bench)(_dummy_kernel)

    assert captured["do_bench"] is my_do_bench


def test_run_skips_gc_on_autotune_disk_cache_hit(monkeypatch):
    configs = [Config({"BLOCK_SIZE": 16}), Config({"BLOCK_SIZE": 32})]
    tuner, key = _make_run_tuner(configs)
    gc_calls = []
    profile_calls = []

    def check_disk_cache(tuning_key, pruned_configs, benchmark):
        assert tuning_key == key
        tuner.cache[tuning_key] = pruned_configs[0]
        return True

    def unexpected_batch_bench(*args, **kwargs):
        raise AssertionError("benchmark must not run on a disk-cache hit")

    tuner.check_disk_cache = check_disk_cache
    tuner._batch_bench = unexpected_batch_bench
    tuner.auto_profile_dir = "profile-output"
    tuner._profile = lambda *args, config, **kwargs: profile_calls.append(config)
    monkeypatch.setattr(ascend_autotuner.gc, "collect", lambda: gc_calls.append(True))

    assert tuner.run() == "kernel-result"
    assert gc_calls == []
    assert profile_calls == []


def test_run_keeps_gc_on_autotune_disk_cache_miss(monkeypatch):
    configs = [Config({"BLOCK_SIZE": 16}), Config({"BLOCK_SIZE": 32})]
    tuner, key = _make_run_tuner(configs)
    gc_calls = []
    benchmark_calls = []
    profile_calls = []

    def check_disk_cache(tuning_key, pruned_configs, benchmark):
        assert tuning_key == key
        benchmark()
        return False

    def batch_bench(*args, configs, **kwargs):
        benchmark_calls.append(configs)
        return {configs[0]: 1.0, configs[1]: 2.0}

    tuner.check_disk_cache = check_disk_cache
    tuner._batch_bench = batch_bench
    tuner.auto_profile_dir = "profile-output"
    tuner._profile = lambda *args, config, **kwargs: profile_calls.append(config)
    monkeypatch.setattr(ascend_autotuner.gc, "collect", lambda: gc_calls.append(True))

    assert tuner.run() == "kernel-result"
    assert benchmark_calls == [configs]
    assert gc_calls == [True]
    assert profile_calls == [configs[0]]

    assert tuner.run() == "kernel-result"
    assert benchmark_calls == [configs]
    assert gc_calls == [True]
    assert profile_calls == [configs[0]]


def test_run_caches_single_config_and_skips_gc(monkeypatch):
    config = Config({"BLOCK_SIZE": 16, "compile_mode": "simt_only"})
    tuner, key = _make_run_tuner([config])
    gc_calls = []
    profile_calls = []
    prune_calls = []

    def prune_configs(kwargs):
        prune_calls.append(kwargs)
        return [config]

    def unexpected_disk_cache(*args, **kwargs):
        raise AssertionError("single-config path must not probe disk cache")

    def unexpected_batch_bench(*args, **kwargs):
        raise AssertionError("single-config path must not benchmark")

    tuner.prune_configs = prune_configs
    tuner.check_disk_cache = unexpected_disk_cache
    tuner._batch_bench = unexpected_batch_bench
    tuner.auto_profile_dir = "profile-output"
    tuner._profile = lambda *args, config, **kwargs: profile_calls.append(config)
    monkeypatch.setattr(ascend_autotuner.gc, "collect", lambda: gc_calls.append(True))

    assert tuner.run() == "kernel-result"
    assert tuner.cache[key] is config
    assert tuner.run() == "kernel-result"
    assert len(prune_calls) == 1
    assert len(tuner.run_kwargs) == 2
    assert gc_calls == []
    assert profile_calls == []


def test_run_does_not_cache_failed_single_config(monkeypatch):
    config = Config({"BLOCK_SIZE": 16})
    tuner, key = _make_run_tuner([config])
    gc_calls = []
    prune_calls = []
    run_calls = []

    def prune_configs(kwargs):
        prune_calls.append(kwargs)
        return [config]

    def run(*args, **kwargs):
        run_calls.append(kwargs)
        if len(run_calls) == 1:
            raise RuntimeError("kernel failed")
        return "kernel-result"

    def unexpected_disk_cache(*args, **kwargs):
        raise AssertionError("single-config path must not probe disk cache")

    def unexpected_batch_bench(*args, **kwargs):
        raise AssertionError("single-config path must not benchmark")

    tuner.prune_configs = prune_configs
    tuner.fn = SimpleNamespace(run=run)
    tuner.check_disk_cache = unexpected_disk_cache
    tuner._batch_bench = unexpected_batch_bench
    monkeypatch.setattr(ascend_autotuner.gc, "collect", lambda: gc_calls.append(True))

    with pytest.raises(RuntimeError, match="kernel failed"):
        tuner.run()
    assert key not in tuner.cache

    assert tuner.run() == "kernel-result"
    assert tuner.cache[key] is config
    assert tuner.run() == "kernel-result"
    assert len(prune_calls) == 2
    assert len(run_calls) == 3
    assert gc_calls == []


def test_run_caches_single_config_after_pruning(monkeypatch):
    config = Config({"BLOCK_SIZE": 16})
    tuner, key = _make_run_tuner([config, Config({"BLOCK_SIZE": 32})])
    gc_calls = []
    prune_calls = []

    def prune_configs(kwargs):
        prune_calls.append(kwargs)
        return [config]

    def unexpected_disk_cache(*args, **kwargs):
        raise AssertionError("single-config path must not probe disk cache")

    def unexpected_batch_bench(*args, **kwargs):
        raise AssertionError("single-config path must not benchmark")

    tuner.prune_configs = prune_configs
    tuner.check_disk_cache = unexpected_disk_cache
    tuner._batch_bench = unexpected_batch_bench
    monkeypatch.setattr(ascend_autotuner.gc, "collect", lambda: gc_calls.append(True))

    assert tuner.run() == "kernel-result"
    assert tuner.cache[key] is config
    assert tuner.run() == "kernel-result"
    assert len(prune_calls) == 1
    assert gc_calls == []
