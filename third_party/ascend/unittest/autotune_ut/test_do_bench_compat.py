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

from types import MethodType

import pytest
import triton
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

    def _do_bench_npu(funcs, clear_l2_cache=False):
        calls["do_bench_npu"] += 1
        assert len(funcs) == 2
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
