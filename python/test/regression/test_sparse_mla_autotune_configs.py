import inspect
import importlib.util
from pathlib import Path


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
