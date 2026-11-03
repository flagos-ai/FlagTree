from types import SimpleNamespace

import numpy as np

from triton.flagtune.contract.operator_schema import parse_operator_config
from triton.flagtune.core.ga_search import GASearcher
from triton.flagtune.runtime import proposer as proposer_module


def test_candidate_pruner_filters_prediction_and_ga_benchmarks(monkeypatch):
    variant = parse_operator_config({
        "op_id": "vendor/mm",
        "variants": {
            "general": {
                "inputs": {"M": {}, "N": {}, "K": {}},
                "params": {
                    "BLOCK_M": {"values": [16, 32]},
                    "num_warps": {"values": [4, 8]},
                },
                "features": ["BLOCK_M"],
            },
        },
    }).get_variant("general")

    class Predictor:

        def predict(self, matrix):
            return np.arange(len(matrix), dtype=float)

    monkeypatch.setattr(
        proposer_module,
        "load_model_bundle",
        lambda *args, **kwargs: SimpleNamespace(variant=variant, predictor=Predictor()),
    )
    monkeypatch.setattr(proposer_module, "_top_k", lambda: 2)
    monkeypatch.setattr(
        GASearcher,
        "generate",
        lambda _self, _history: [
            {"config": {"BLOCK_M": 16, "num_warps": 4}},
            {"config": {"BLOCK_M": 32, "num_warps": 8}},
        ],
    )
    proposer = proposer_module.make_config_proposer("vendor/mm", "general", platform_key="thead-zw810e",
                                                    dtype_key="f32")
    measured = []

    def benchmark(config, _n_runs):
        measured.append(config)
        return [1.0]

    result = proposer(
        benchmark,
        {"M": 16, "N": 16, "K": 16},
        list(variant.iter_configs()),
        {"candidate_pruner": lambda configs: [config for config in configs if config["BLOCK_M"] == 32]},
    )
    assert result
    assert all(config["BLOCK_M"] == 32 for config in result)
    assert all(config["BLOCK_M"] == 32 for config in measured)

    other_platform_proposer = proposer_module.make_config_proposer("vendor/mm", "general", platform_key="nvidia-h800",
                                                                   dtype_key="f32")
    measured.clear()
    other_platform_proposer(
        benchmark,
        {"M": 16, "N": 16, "K": 16},
        list(variant.iter_configs()),
        {"candidate_pruner": lambda _configs: 1 / 0},
    )
    assert any(config["BLOCK_M"] == 16 for config in measured)
