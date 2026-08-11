import inspect
import importlib.util
import math
from pathlib import Path

import pytest


def _load_sparse_mla_module():
    path = (Path(__file__).resolve().parents[2] / "tutorials" / "tle" / "deepseek_v32" / "02-sparse-mla.py")
    spec = importlib.util.spec_from_file_location("sparse_mla_tutorial", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sparse_mla_autotune_configs_bind_num_warps_and_num_stages():
    module = _load_sparse_mla_module()

    expected_sparse = {
        (1, 4),
        (1, 8),
        (1, 16),
        (1, 32),
        (2, 4),
        (2, 8),
        (2, 16),
        (2, 32),
        (4, 4),
        (4, 8),
        (4, 16),
        (4, 32),
    }
    actual_sparse = {(cfg.num_stages, cfg.num_warps) for cfg in module.spar_mla_fwd_configs}
    assert actual_sparse == expected_sparse

    for cfg in module.spar_mla_fwd_configs + module.tle_spar_mla_fwd_configs:
        assert "num_warps" not in cfg.kwargs
        assert "num_stages" not in cfg.kwargs

    assert len(module.tle_spar_mla_fwd_configs) == 1
    tle_cfg = module.tle_spar_mla_fwd_configs[0]
    assert tle_cfg.num_stages == 2
    assert tle_cfg.num_warps == 8


def test_sparse_mla_bench_cases_track_flashmla_v32_prefill_and_decode():
    module = _load_sparse_mla_module()

    assert "triton" in module._BENCH_PROVIDERS
    assert "tle" in module._BENCH_PROVIDERS
    assert "tle-pipelined" not in module._BENCH_PROVIDERS
    assert len(module._BENCH_PROVIDERS) == len(module._BENCH_NAMES)
    assert len(module._BENCH_PROVIDERS) == len(module._BENCH_STYLES)

    assert module._BENCH_X_VALS == [
        (1, 4096, 8192, 128, 1, 576, 512, 2048),
        (1, 4096, 32768, 128, 1, 576, 512, 2048),
        (1, 4096, 65536, 128, 1, 576, 512, 2048),
        (1, 4096, 98304, 128, 1, 576, 512, 2048),
        (1, 4096, 131072, 128, 1, 576, 512, 2048),
    ]

    assert module._DECODE_BENCH_X_VALS == [
        (2, 2, 32768, 128, 1, 576, 512, 2048, 64),
        (64, 2, 32768, 128, 1, 576, 512, 2048, 64),
        (74, 2, 32768, 128, 1, 576, 512, 2048, 64),
        (128, 2, 32768, 128, 1, 576, 512, 2048, 64),
    ]


def test_sparse_mla_bench_seed_is_explicit_and_shared():
    module = _load_sparse_mla_module()

    assert module.BENCH_DEFAULT_SEED == 1
    assert inspect.signature(module.run_bench_table).parameters["seed"].default == module.BENCH_DEFAULT_SEED
    assert inspect.signature(module.bench_sparse_mla_fwd).parameters["seed"].default == module.BENCH_DEFAULT_SEED
    assert "seed" in inspect.signature(module.benchmark_sparse_mla_fwd.fn).parameters


def _run_forward_benchmark_callback(module, provider):
    return module.benchmark_sparse_mla_fwd.fn(
        B=1,
        S=1,
        SKV=1,
        H=1,
        HKV=1,
        DQK=1,
        DV=1,
        topk=1,
        provider=provider,
        warmup=1,
        rep=1,
        tilelang_block_I=1,
        tilelang_num_stages=1,
        tilelang_threads=32,
        input_mode="flashmla",
        seed=1,
    )


def _stub_forward_benchmark_inputs(monkeypatch, module):
    monkeypatch.setattr(module, "_get_bench_sparse_mla_inputs", lambda *args, **kwargs: (None, None, None, None))


def _benchmark_failure(message):

    def fail(*args, **kwargs):
        raise RuntimeError(message)

    return fail


@pytest.mark.parametrize("provider", ["tle-flashmla-prefill", "tilelang"])
def test_sparse_mla_benchmark_propagates_supported_provider_failure(monkeypatch, provider):
    module = _load_sparse_mla_module()
    _stub_forward_benchmark_inputs(monkeypatch, module)
    monkeypatch.setattr(module, "_HAVE_TILELANG", True)
    monkeypatch.setattr(module.triton.testing, "do_bench", _benchmark_failure("compile failed"))

    with pytest.raises(RuntimeError, match="compile failed"):
        _run_forward_benchmark_callback(module, provider)


def test_sparse_mla_benchmark_skips_unavailable_optional_provider(monkeypatch):
    module = _load_sparse_mla_module()
    _stub_forward_benchmark_inputs(monkeypatch, module)
    monkeypatch.setattr(module, "_HAVE_TILELANG", False)
    monkeypatch.setattr(module.triton.testing, "do_bench", _benchmark_failure("unavailable provider was executed"))

    result = _run_forward_benchmark_callback(module, "tilelang")

    assert all(math.isnan(value) for value in result)


def test_sparse_mla_single_benchmark_propagates_provider_failure(monkeypatch):
    module = _load_sparse_mla_module()
    _stub_forward_benchmark_inputs(monkeypatch, module)
    monkeypatch.setattr(module, "triton_sparse_mla_fwd_interface", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(module, "tle_sparse_mla_fwd_interface", _benchmark_failure("TLE compile failed"))
    monkeypatch.setattr(module, "_bench_ms", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(module, "_sparse_mla_tflops_from_topk_length", lambda *args, **kwargs: 1.0)

    with pytest.raises(RuntimeError, match="TLE compile failed"):
        module.bench_sparse_mla_fwd(B=1, S=1, SKV=1, H=1, HKV=1, DQK=1, DV=1, topk=1, check_outputs=False)


def test_sparse_mla_benchmark_rejects_non_finite_timings(monkeypatch):
    module = _load_sparse_mla_module()
    _stub_forward_benchmark_inputs(monkeypatch, module)
    monkeypatch.setattr(module.triton.testing, "do_bench", lambda *args, **kwargs: (float("nan"), 1.0, 2.0))

    with pytest.raises(RuntimeError, match="non-finite timings"):
        _run_forward_benchmark_callback(module, "triton")
