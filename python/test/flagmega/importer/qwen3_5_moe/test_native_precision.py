# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.importer import import_model
from triton.flagmega.importer.numerics import VLLM_AE10_INDUCTOR_LEVEL3
from triton.flagmega.importer.qwen3_5_moe.decoder import linear
from triton.flagmega.passes.functions.lift_constant_expressions import _constant_values
from python.test.flagmega.importer.qwen3_5_moe.helpers import checkpoint


@pytest.mark.parametrize("mode,tokens", [("decode-1", 1), ("prefill", 3)])
def test_native_import_has_no_activation_casts_before_optimization(mode, tokens):
    module = import_model(checkpoint(), mode=mode, num_tokens=tokens)
    constants = _constant_values(module)
    assert not [node.id for node in module.nodes if node.op == "tensors.cast" and node.id not in constants]
    phase = "decode" if mode == "decode-1" else "prefill"
    for kind in ("linear", "attention"):
        for suffix in ("hidden", "attention_residual", "moe_output", "output"):
            assert module.node_map[f"{phase}_{kind}_{suffix}"].type.dtype == fm.DType.BFLOAT16
        router = module.node_map[f"{phase}_{kind}_router_logits"]
        assert router.op == "math.matmul" and router.attrs["output_data_type"] == "float32"
    logits = module.node_map["logits"]
    assert logits.op == "math.matmul" and logits.attrs["output_data_type"] == "float32"


def test_projection_output_dtype_does_not_round_through_bfloat16():
    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 2)))
            weight = self.input("weight", fm.tensor_type("bfloat16", (1, 2)))
            result = linear(value, weight, output_dtype="float32", name="result")
            self.function("main", (value, weight), (result,))
    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    value = torch.tensor([[1, 1 / 256]], dtype=torch.bfloat16)
    weight = torch.tensor([[1, 1]], dtype=torch.bfloat16)
    result = TorchEvaluator(DictWeightResolver({})).run(module, {"value": value, "weight": weight})[0]
    expected = value.float() @ weight.float().T
    assert not torch.equal(expected, expected.bfloat16().float())
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_explicit_compatibility_profile_retains_projection_rounding():
    module = import_model(checkpoint(), numerical_profile=VLLM_AE10_INDUCTOR_LEVEL3)
    for name in ("decode_linear_router_logits", "decode_attention_router_logits", "logits"):
        cast = module.node_map[name]
        assert cast.op == "tensors.cast" and cast.type.dtype == fm.DType.FLOAT32
        projection = module.node_map[cast.inputs[0]]
        assert projection.op == "math.matmul" and projection.type.dtype == fm.DType.BFLOAT16


@pytest.fixture
def tutorial(monkeypatch):
    path = Path(__file__).parents[4] / "tutorials/flagmega/02-qwen3.5-35b-a3b-bf16/nvidia-h800/optimize.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("native_precision_tutorial", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tutorial_import_does_not_force_wide_source_runtime_profile(tutorial):
    module = tutorial.input_module(checkpoint())
    assert module.metadata["numerical_contract"] == "nncase"
    assert module.node_map["decode_linear_hidden"].type.dtype == fm.DType.BFLOAT16
    assert module.node_map["decode_attention_hidden"].type.dtype == fm.DType.BFLOAT16


def test_tutorial_resume_preserves_declared_profile_without_reinterpretation(tutorial, tmp_path):
    source = checkpoint()
    module = tutorial.input_module(source, numerical_profile=VLLM_AE10_INDUCTOR_LEVEL3)
    path = fm.emit_module(module, tmp_path / "pinned.py")
    assert tutorial.input_module(source, path) == module
    with pytest.raises(ValueError, match="re-import"):
        tutorial.input_module(source, path, "nncase")
