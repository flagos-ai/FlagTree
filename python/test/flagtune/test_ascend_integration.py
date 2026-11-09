"""Run FlagTune's baseline tuning path on an installed Ascend backend."""

import pytest

import triton
import triton.language as tl
from triton.flagtune import flagtune
from triton.flagtune.runtime.device import probe_flagtune_device

# Keep collection possible on machines without the optional Ascend runtime.
torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ascend_event_autotuning(dtype, monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    if triton.runtime.driver.active.get_current_target().backend != "npu":
        pytest.skip("requires Triton's Ascend driver")
    monkeypatch.setenv("FLAGTUNE_BENCHMARK_MODE", "event")

    @flagtune(
        configs=[triton.Config({"BLOCK_SIZE": block}) for block in (256, 512)],
        key=["N"],
        warmup=5,
        rep=20,
    )
    @triton.jit
    def add(X, Y, Z, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + offsets, offsets < N, other=0)
        y = tl.load(Y + offsets, offsets < N, other=0)
        tl.store(Z + offsets, x + y, offsets < N)

    descriptor = probe_flagtune_device()
    assert descriptor.vendor == "huawei"
    assert descriptor.torch_device_type == "npu"
    x = torch.randn(4097, dtype=dtype, device="npu")
    y = torch.randn_like(x)
    z = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]), )
    # First call tunes; the second exercises the selected configuration cache.
    for _ in range(2):
        add[grid](x, y, z, x.numel())
        torch.npu.synchronize()
        torch.testing.assert_close(z, x + y)
    assert add.benchmark_protocol.resolved_mode.value == "event"
    assert add.cache


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k", [1, 2])
def test_ascend_vector_add_cost_model(dtype, top_k, tmp_path, monkeypatch):
    """Train on measured NPU latencies, reload a package, and run predicted configs."""
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    if triton.runtime.driver.active.get_current_target().backend != "npu":
        pytest.skip("requires Triton's Ascend driver")
    import json
    import importlib
    import math
    from triton.flagtune.contract.archive import platform_package_name, write_platform_package
    from triton.flagtune.contract.identity import ModelIdentity, discover_gpu_metadata, make_dtype_key
    from triton.flagtune.contract.operator_schema import parse_operator_config
    from triton.flagtune.runtime import proposer
    from triton.flagtune.training.ranker import XGBoostTrainingOptions, export_ranker_model, train_xgboost_ranker

    @triton.jit
    def add(X, Y, Z, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + offsets, offsets < N, other=0)
        y = tl.load(Y + offsets, offsets < N, other=0)
        tl.store(Z + offsets, x + y, offsets < N)

    op_id = "tests/ascend_vector_add"
    blocks = [256, 512, 1024]
    variant = parse_operator_config({
        "op_id": op_id,
        "variants": {
            "vector": {
                "inputs": {"N": {}},
                "params": {"BLOCK_SIZE": {"values": blocks}},
                "features": ["N", "BLOCK_SIZE"],
            }
        },
    }).get_variant("vector")
    gpu = dict(discover_gpu_metadata())
    assert gpu["platform_key"] == "huawei-ascend910"
    dtypes = [str(dtype).removeprefix("torch.")] * 3
    identity = ModelIdentity(gpu["platform_key"], op_id, "vector", make_dtype_key(dtypes))
    records = []
    for n in [257, 4097, 65537]:
        x = torch.randn(n, dtype=dtype, device="npu")
        y = torch.randn_like(x)
        z = torch.empty_like(x)
        for block in blocks:

            def launch():
                add[(triton.cdiv(n, block), )](x, y, z, n, BLOCK_SIZE=block)

            latency = triton.testing.do_bench(launch, warmup=2, rep=5, quantiles=[0.5])
            assert math.isfinite(latency) and latency > 0
            torch.testing.assert_close(z, x + y)
            records.append({
                "schema_version": 2,
                "model_identity": {"platform_key": identity.platform_key, "dtype_key": identity.dtype_key},
                "dtypes": {"inputs": dtypes[:2], "outputs": dtypes[2:]},
                "device": {"metadata": gpu},
                "ranking_group": {
                    "operator_id": op_id, "variant": "vector", "dimensions": {"N": n}, "model_dtype_key":
                    identity.dtype_key
                },
                "inputs": {"N": n},
                "config": {"BLOCK_SIZE": block},
                "latency_ms": latency,
            })
    corpus = tmp_path / "benchmark.jsonl"
    corpus.write_text("".join(json.dumps(row) + "\n" for row in records))
    model, summary = train_xgboost_ranker(
        variant,
        corpus,
        XGBoostTrainingOptions(n_estimators=8, max_depth=2, min_train_rows=2, n_jobs=1, show_progress=False),
    )
    assert model.get_booster().get_dump()
    exported = export_ranker_model(
        model,
        variant,
        tmp_path / "child",
        summary,
        identity=identity,
        dtypes=dtypes,
        gpu=gpu,
        model_version="1.0.0",
    )
    packages = tmp_path / "packages"
    packages.mkdir()
    write_platform_package(
        packages / platform_package_name(identity.platform_key, "1.0.0"),
        platform_key=identity.platform_key,
        package_version="1.0.0",
        model_archives={identity: exported.model_path},
    )
    monkeypatch.setenv("FLAGTUNE_MODEL_DIR", str(packages))
    monkeypatch.setenv("FLAGTUNE_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FLAGTUNE_DISABLE_REMOTE", "1")
    monkeypatch.setenv("FLAGTUNE_MODEL_VERSION", "1.0.0")
    monkeypatch.setenv("FLAGTUNE_ENABLE", "1")
    monkeypatch.setenv("FLAGTUNE_TOP_K", str(top_k))
    monkeypatch.setenv("FLAGTUNE_BENCHMARK_MODE", "event")
    monkeypatch.delenv("FLAGTUNE_DISABLE_OPS", raising=False)
    public = importlib.import_module("triton.flagtune")
    monkeypatch.setattr(public, "_ENABLED", None)
    monkeypatch.setattr(proposer, "_MODEL_MANAGER", None)
    monkeypatch.setattr(proposer, "_TOP_K_CACHE", None)
    loaded = proposer.load_model_bundle(op_id, "vector", platform_key=identity.platform_key,
                                        dtype_key=identity.dtype_key)
    assert loaded.identity == identity
    predictions = []
    predict = loaded.predictor.predict

    def observed_predict(matrix):
        predictions.append(len(matrix))
        return predict(matrix)

    monkeypatch.setattr(loaded.predictor, "predict", observed_predict)
    propose = proposer.make_config_proposer(op_id, "vector", platform_key=identity.platform_key,
                                            dtype_key=identity.dtype_key)
    expected = {}
    selected = {}
    configs = [triton.Config({"BLOCK_SIZE": block}) for block in blocks]
    tuner = flagtune(configs=configs, key=["N"], op_id=op_id, variant="vector", warmup=2, rep=5)(add)
    for n in [1, 255, 1025, 131071]:
        expected[n] = propose(None, {"N": n}, [{"BLOCK_SIZE": block} for block in blocks], {})
        assert len(expected[n]) == top_k
        x = torch.randn(n, dtype=dtype, device="npu")
        y = torch.randn_like(x)
        z = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
        before = len(predictions)
        tuner[grid](x, y, z, n)
        torch.npu.synchronize()
        torch.testing.assert_close(z, x + y)
        assert len(predictions) == before + 1  # The tuner must actually call the predictor.
        assert predictions[-1] == len(blocks)
        assert tuner.best_config.kwargs in expected[n]
        selected[n] = tuner.best_config.kwargs
        before = len(predictions)
        tuner[grid](x, y, z, n)
        torch.npu.synchronize()
        torch.testing.assert_close(z, x + y)
        assert len(predictions) == before  # Cache hits avoid prediction and retuning.
    assert identity in tuner._flagtune_models
    assert len(tuner.cache) == 4
    monkeypatch.setattr(public, "_ENABLED", False)
    baseline = flagtune(configs=configs, key=["N"], op_id=op_id, variant="vector", warmup=2, rep=5)(add)
    before = len(predictions)
    baseline[grid](x, y, z, n)
    torch.npu.synchronize()
    torch.testing.assert_close(z, x + y)
    assert baseline._flagtune_models == {}
    assert len(predictions) == before

    report = {
        "dtype": str(dtype), "top_k": top_k, "training_rows": len(records), "platform_key": identity.platform_key,
        "measured_latencies_ms": [r["latency_ms"] for r in records], "selected": selected, "model_predictions":
        len(predictions)
    }
    (tmp_path / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
