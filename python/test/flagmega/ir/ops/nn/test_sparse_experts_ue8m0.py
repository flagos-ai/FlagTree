# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Routed FP8 experts carry a distinct, explicit UE8M0 scale contract."""

import hashlib
import subprocess
import sys
from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.math._block_scaled import dynamic_block_quant_dequant
from triton.flagmega.ir.ops.nn.sparse_experts import SparseExperts
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp

from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import stage_module
from python.test.flagmega.ir.ops.nn.test_sparse_experts_fp8_rounding import projection_case

MODE = "dynamic_fp8_ue8m0"


def ue8m0_projection_case(definition, *, rounded=False, input_scale=1.):
    module, values = projection_case(definition, rounded=rounded, input_scale=input_scale)
    stage = next(node for node in module.nodes if node.op == definition.op_name)
    prepared = definition.prepare(tuple(module.node_map[key] for key in stage.inputs),
                                  {**stage.attrs, "activation_quantization": MODE})
    replacement = replace(stage, attrs=prepared.attrs)
    return replace(module, nodes=tuple(replacement if node.id == stage.id else node for node in module.nodes)), values


def bf16_domain():
    maximum = torch.arange(0x7f80, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    fractions = torch.arange(128).float() / 128
    fractions[0] = 1.
    fractions[64:] *= -1
    values = (maximum[:, None] * fractions).bfloat16()
    values[:, 1], values[:, 2], values[:, 3] = maximum.bfloat16(), -maximum.bfloat16(), 0
    special = torch.zeros((4, 128), dtype=torch.bfloat16)
    special[0, 0], special[1, 0], special[2, 0] = float("nan"), float("inf"), -float("inf")
    special[3, :2] = torch.tensor((float("nan"), 1.), dtype=torch.bfloat16)
    return torch.cat((values, special))


def test_complete_bf16_domain_matches_independently_recorded_native_codes_and_scales():
    from triton.flagmega.ir.ops.math._ue8m0_quantization import ue8m0_quantize

    values = bf16_domain()
    codes, scale = ue8m0_quantize(values)
    # Official vLLM 0.31.0 CUDA per_token_group_fp8_quant, SM90, block K=128,
    # UE8M0 enabled. Both outputs were observed independently of this model.
    def digest(value):
        return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    assert digest(values) == "5e9099d921df8dc88a1f96d23ed3b163eb18ab9e5467c5c5615552a5c8d3a18d"
    assert digest(codes) == "92ceb6e7d0488ba6f44afa5417eccc3d6eb6c719eec29be7986a33f3d1d4435a"
    assert digest(scale) == "7450816f3adcb3fac2bc372705b6012c5431b69e4a7965e1b359457ef59d4529"


def test_ue8m0_rounds_the_activation_scale_up_to_a_power_of_two():
    value = torch.zeros((1, 128), dtype=torch.bfloat16)
    value[0, :2] = torch.tensor((.050537109375, .134765625))
    result = dynamic_block_quant_dequant(value, block_k=128, mode=MODE)
    torch.testing.assert_close(result[0, :2], torch.tensor((.05078125, .140625)), atol=0, rtol=0)


@pytest.mark.parametrize("definition", (SparseExpertsGateUp, SparseExpertsDown))
@pytest.mark.parametrize("rounded", (False, True))
@pytest.mark.parametrize("input_scale", (1., 1.1))
def test_stage_preserves_ue8m0_quantization_and_its_rounding_boundary(tmp_path, definition, rounded, input_scale):
    module, values = ue8m0_projection_case(definition, rounded=rounded, input_scale=input_scale)
    output, = TorchEvaluator(DictWeightResolver({})).run(module, values)
    wide = torch.tensor(128.5) * input_scale
    projection = wide.bfloat16().float() if rounded else wide
    expected = projection.square().bfloat16() if definition is SparseExpertsGateUp else projection
    torch.testing.assert_close(output, torch.full_like(output, expected.item()), atol=0, rtol=0)
    path = tmp_path / "module.py"
    path.write_text(fm.module_source(module))
    loaded = fm.load_module(path)
    assert loaded.semantic_hash == module.semantic_hash


@pytest.mark.parametrize("definition", (SparseExpertsGateUp, SparseExpertsDown, SparseExperts))
@pytest.mark.parametrize("invalid", ("mode", "activation", "weight", "scales", "block_n", "block_k"))
def test_quantization_rejects_unsupported_mode_or_storage(definition, invalid):
    attrs = {"dtype": "float32" if invalid == "activation" else "bfloat16",
             "weight_dtype": "bfloat16" if invalid == "weight" else "float8_e4m3fn",
             "block_scaled": invalid != "scales", "weight_block": 128, "hidden": 128, "intermediate": 128,
             "tokens": 1, "activation_quantization": "unknown" if invalid == "mode" else MODE}
    if invalid in {"block_n", "block_k"}:
        attrs[f"weight_{invalid}"] = 0
    with pytest.raises(IRSchemaError, match="activation_quantization|UE8M0|positive integer"):
        stage_module(definition, **attrs)


@pytest.mark.parametrize("definition", (SparseExpertsGateUp, SparseExpertsDown, SparseExperts))
def test_original_attributes_do_not_gain_a_default_quantization_mode(definition):
    module = stage_module(definition)
    assert all("activation_quantization" not in node.attrs for node in module.nodes)


@pytest.mark.parametrize("egraph", (False, True))
def test_composite_decomposition_preserves_the_quantizer(tmp_path, egraph):
    from triton.flagmega.passes.rewriter import DataflowPass, EGraphRulesPass
    from triton.flagmega.rules.neutral import decompose_sparse_experts_rule

    module = stage_module(SparseExperts, dtype="bfloat16", weight_dtype="float8_e4m3fn",
                          block_scaled=True, weight_block=128, hidden=128, intermediate=128,
                          tokens=1, activation_quantization=MODE)
    original = fm.load_module(fm.emit_module(module, tmp_path / "composite.py"))
    assert original.semantic_hash == module.semantic_hash
    if egraph:
        pass_ = EGraphRulesPass("UE8M0Decomposition", (decompose_sparse_experts_rule(),),
            cost=lambda node, _module: 100. if node.op == "nn.sparse_experts" else 1.,
            cost_model="unit-test/force-stage-extraction")
    else:
        pass_ = DataflowPass("UE8M0Decomposition", (decompose_sparse_experts_rule(),))
    rewritten = pass_.run(module)
    stages = [node for node in rewritten.nodes if node.op in
              {"nn.sparse_experts_gate_up", "nn.sparse_experts_down"}]
    assert len(stages) == 2 and all(node.attrs["activation_quantization"] == MODE for node in stages)
    values = {}
    for name in module.function_map[module.entry].parameters:
        tensor = fm.logical_type(module.node_map[name].type)
        shape = tuple(dim.fixed_value for dim in tensor.shape)
        value = torch.full(shape, 0 if name == "router_expert_ids" else .125)
        values[name] = value.to(getattr(torch, tensor.dtype.value))
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(rewritten, values)[0], evaluator.run(module, values)[0], atol=0, rtol=0)
    loaded = fm.load_module(fm.emit_module(rewritten, tmp_path / "decomposed.py"))
    assert loaded.semantic_hash == rewritten.semantic_hash


@pytest.mark.parametrize("mode", ("dynamic_fp8", MODE))
@pytest.mark.parametrize("cyclic", (False, True))
def test_reduction_shards_cannot_split_an_activation_quantization_group(mode, cyclic):
    module = stage_module(SparseExpertsDown, dtype="bfloat16", weight_dtype="float8_e4m3fn",
                          block_scaled=True, weight_block=128, hidden=256, intermediate=256,
                          tokens=1, activation_quantization=mode)
    placement = fm.Placement((4,), "x", "b")
    b = fm.SBP.broadcast()
    split = fm.SBP.split_block_cyclic((0,), 64) if cyclic else fm.SBP.split_contiguous((0,), 64)
    stage = next(node for node in module.nodes if node.op == SparseExpertsDown.op_name)
    operands = []
    for name in stage.inputs:
        kind = module.node_map[name].type
        policies = (b, b, split) if name in {"activations", "down_weight"} else (b,) * kind.rank
        operands.append(fm.Node(name, "builtin.var", (), fm.DistributedType(kind, policies, placement)))
    with pytest.raises(IRSchemaError, match="quantization group"):
        SparseExpertsDown.prepare(operands, stage.attrs)


def test_all_three_constructors_round_trip_in_a_fresh_process(tmp_path):
    arguments = []
    for definition in (SparseExperts, SparseExpertsGateUp, SparseExpertsDown):
        module = stage_module(definition, dtype="bfloat16", weight_dtype="float8_e4m3fn",
                              block_scaled=True, weight_block=128, hidden=128, intermediate=128,
                              tokens=1, activation_quantization=MODE)
        path = fm.emit_module(module, tmp_path / (definition.functional_name + ".py"))
        arguments.extend((str(path), module.semantic_hash))
    subprocess.run([sys.executable, "-c", '''
import sys
from triton.flagmega import ir as fm
for path, expected in zip(sys.argv[1::2], sys.argv[2::2]):
    module = fm.load_module(path)
    assert module.semantic_hash == expected
    assert any(node.attrs.get("activation_quantization") == "dynamic_fp8_ue8m0" for node in module.nodes)
''', *arguments], check=True)

