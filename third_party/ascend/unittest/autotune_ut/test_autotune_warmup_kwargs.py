from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import triton
from triton.backends.ascend.runtime.autotuner import AutoTilingTuner


def make_tuner(configs, parallel, monkeypatch):
    calls = []

    def warmup(*args, **kwargs):
        calls.append((args, kwargs))
        return kwargs

    tuner = object.__new__(AutoTilingTuner)
    tuner.fn = SimpleNamespace(warmup=warmup)
    tuner.nargs = None
    tuner.compile_parallel = parallel
    tuner.generate_key_and_configs = lambda *args, **kwargs: ()
    tuner.prune_configs = lambda kwargs: configs
    monkeypatch.setattr(triton, "AsyncCompileMode", lambda executor: nullcontext())
    return tuner, calls


@pytest.mark.parametrize("parallel", [False, True])
def test_warmup_caller_launch_options(parallel, monkeypatch):
    configs = [triton.Config({"BLOCK": 16}, num_warps=1), triton.Config({"BLOCK": 32}, num_warps=2)]
    original = [config.all_kwargs() for config in configs]
    tuner, calls = make_tuner(configs, parallel, monkeypatch)
    result = tuner.warmup("input", num_warps=4, num_stages=1, grid=(1, ))
    assert len(result) == 2
    assert [kwargs["BLOCK"] for _, kwargs in calls] == [16, 32]
    assert all(args == ("input", ) for args, _ in calls)
    assert all(kwargs["num_warps"] == 4 and kwargs["num_stages"] == 1 for _, kwargs in calls)
    assert all(kwargs["grid"] == (1, ) for _, kwargs in calls)
    assert [config.all_kwargs() for config in configs] == original
    assert tuner.nargs is None


@pytest.mark.parametrize("parallel", [False, True])
def test_warmup_config_defaults(parallel, monkeypatch):
    configs = [triton.Config({"BLOCK": 16}, num_warps=1), triton.Config({"BLOCK": 32}, num_warps=2)]
    tuner, calls = make_tuner(configs, parallel, monkeypatch)
    tuner.warmup("input", grid=(1, ))
    assert [kwargs["num_warps"] for _, kwargs in calls] == [1, 2]
    assert [kwargs["BLOCK"] for _, kwargs in calls] == [16, 32]


@pytest.mark.parametrize("parallel", [False, True])
def test_warmup_caller_constexpr(parallel, monkeypatch):
    tuner, calls = make_tuner([triton.Config({}, num_warps=1)], parallel, monkeypatch)
    tuner.warmup("input", BLOCK=64, num_warps=2)
    assert calls[0][1]["BLOCK"] == 64
    assert calls[0][1]["num_warps"] == 2
