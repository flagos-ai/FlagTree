# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Load-time requantization changes both codes and their scaling factors."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator, materialize_constant_assets
from triton.flagmega.ir.ops.nn.requantize_block_scaled_fp8 import RequantizeBlockScaledFP8
from triton.flagmega.passes import freeze_constant_islands


class Requantized(fm.Module):
    def __init__(self, shape=(128, 128), blocks=(128, 128), *, runtime=False, project=False, casts=False,
                 scale_division="rn"):
        super().__init__(dialect="high_level", stage="imported", entry="main")
        self.shape, self.blocks, self.runtime, self.project, self.casts = shape, blocks, runtime, project, casts
        self.scale_division = scale_division

    def forward(self):
        bn, bk = self.blocks
        n, k = self.shape[-2:]
        types = {"weight": fm.tensor_type("float8_e4m3fn", self.shape),
                 "scale": fm.tensor_type("float32", (*self.shape[:-2], (n + bn - 1) // bn, (k + bk - 1) // bk))}
        values = []
        for name, value_type in types.items():
            value = (self.input(name, value_type, id=name) if self.runtime else
                     self.weight(name, value_type, source="memory", key=name, id=name))
            if self.casts and name == "scale":
                value = fm.F.tensors.cast(value, "float32", name="scale_identity")
            values.append(value)
        pair = fm.F.nn.requantize_block_scaled_fp8(*values, block_n=bn, block_k=bk,
                                                  scale_division=self.scale_division, name="requantized")
        outputs = fm.F.tensors.get_items(pair, 0, 1, name_prefix="field")
        parameters = tuple(values) if self.runtime else ()
        if self.project:
            value = self.input("value", fm.tensor_type("bfloat16", (1, k)), id="value")
            output = fm.F.math.block_scaled_matmul(value, *outputs, weight_block_n=bn, weight_block_k=bk,
                                                  name="projected")
            parameters, outputs = (value,), (output,)
        self.function("main", parameters, outputs)


def _simple_weights():
    weight = torch.zeros((128, 128))
    weight[0, 0] = 448
    return {"weight": weight.to(torch.float8_e4m3fn), "scale": torch.full((1, 1), .003)}


def test_requantization_updates_codes_and_scales_before_projection():
    weights = _simple_weights()
    evaluator = TorchEvaluator(DictWeightResolver(weights))
    codes, scales = evaluator.run(Requantized().build(), {})
    assert codes.dtype == torch.float8_e4m3fn and scales.dtype == torch.float32
    assert codes.float()[0, 0] == 352.
    assert scales.item() == 2 ** -8
    assert weights["weight"].float()[0, 0] == 448.  # source data is immutable
    assert weights["scale"].item() == torch.tensor(.003).item()
    value = torch.zeros((1, 128), dtype=torch.bfloat16)
    value[0, 0] = 1.
    output, = evaluator.run(Requantized(project=True).build(), {"value": value})
    expected = torch.zeros_like(output)
    expected[0, 0] = 1.375
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(129, 130), (2, 129, 130), (0, 128), (2, 0, 128)])
def test_tails_leading_dimensions_and_zero_blocks(shape):
    n, k = shape[-2:]
    weights = {"weight": torch.zeros(shape).to(torch.float8_e4m3fn),
               "scale": torch.full((*shape[:-2], (n + 127) // 128, (k + 127) // 128), .003)}
    if weights["weight"].numel():
        wide = weights["weight"].float()
        wide[..., -1, -1] = -448.
        weights["weight"] = wide.to(torch.float8_e4m3fn)
    result = TorchEvaluator(DictWeightResolver(weights)).run(Requantized(shape).build(), {})
    assert result[0].shape == weights["weight"].shape and result[1].shape == weights["scale"].shape
    if result[0].numel():
        assert (result[0].float()[..., -1, -1] == -352.).all()
        assert (result[1][..., -1, -1] == 2 ** -8).all()
        assert (result[1][..., 0, 0] == 2 ** -22).all()
        assert torch.count_nonzero(result[0].float()) == (2 if len(shape) == 3 else 1)


def test_noncontiguous_inputs_produce_real_contiguous_outputs():
    weights = _simple_weights()
    weights["weight"] = weights["weight"].T
    result = TorchEvaluator(DictWeightResolver(weights)).run(Requantized().build(), {})
    assert result[0].is_contiguous() and result[1].is_contiguous()
    assert result[0].float()[0, 0] == 352.


def test_block_row_chunks_keep_independent_scales_and_both_tail_axes():
    # A block row exceeds the wide-intermediate budget, forcing three chunks.
    shape = (257, 8193)
    weight = torch.zeros(shape)
    weight[::128, ::128] = 448.
    scales = torch.full((3, 65), .003)
    scales[1] = .006
    resolver = DictWeightResolver({"weight": weight.to(torch.float8_e4m3fn), "scale": scales})
    actual = TorchEvaluator(resolver).run(Requantized(shape).build(), {})
    expected = weight.clone()
    expected[::128, ::128] = 352.
    torch.testing.assert_close(actual[0].float(), expected, rtol=0, atol=0)
    expected_scales = torch.full_like(scales, 2 ** -8)
    expected_scales[1] = 2 ** -7
    torch.testing.assert_close(actual[1], expected_scales, rtol=0, atol=0)


@pytest.mark.parametrize("weight_type,scale_type", [
    (fm.tensor_type("bfloat16", (128, 128)), fm.tensor_type("float32", (1, 1))),
    (fm.tensor_type("float8_e4m3fn", (128, 128)), fm.tensor_type("bfloat16", (1, 1))),
    (fm.tensor_type(fm.vector_type("float8_e4m3fn", (8,)), (128, 16)), fm.tensor_type("float32", (1, 1))),
    (fm.tensor_type("float8_e4m3fn", (128,)), fm.tensor_type("float32", (1,))),
    (fm.tensor_type("float8_e4m3fn", (2, 128, 128)), fm.tensor_type("float32", (1, 1))),
    (fm.tensor_type("float8_e4m3fn", (128, 129)), fm.tensor_type("float32", (1, 1))),
])
def test_rejects_invalid_scalar_storage_and_scale_geometry(weight_type, scale_type):
    inputs = (SimpleNamespace(type=weight_type), SimpleNamespace(type=scale_type))
    with pytest.raises(IRSchemaError):
        RequantizeBlockScaledFP8.infer_type(inputs, {"block_n": 128, "block_k": 128})


def test_offline_cost_counts_both_tuple_fields_and_preserves_purity():
    module = Requantized().build()
    node = module.node_map["requantized"]
    cost = RequantizeBlockScaledFP8.cost(node)
    assert cost.bytes_read == cost.bytes_written == 128 * 128 + 4
    assert node.effect.is_pure
    assert RequantizeBlockScaledFP8.inplace_output_parameters == ()


def test_reciprocal_scale_arithmetic_changes_native_power_of_two_and_subnormal_boundaries():
    weight = torch.full((128, 128), 448.)
    weight[0, 0] = 2 ** -9
    weights = {"weight": weight.to(torch.float8_e4m3fn),
               "scale": torch.full((1, 1), 1.5258792700478807e-5)}
    evaluator = TorchEvaluator(DictWeightResolver(weights))
    rn = evaluator.run(Requantized().build(), {})
    reciprocal = evaluator.run(Requantized(scale_division="reciprocal_multiply").build(), {})
    assert rn[1].item() == 2 ** -16
    assert reciprocal[1].item() == 2 ** -15
    assert rn[0].float()[0, 1] == 448.
    assert reciprocal[0].float()[0, 1] == 224.
    assert rn[0].float()[0, 0] * rn[1].item() == 2 ** -25
    assert reciprocal[0].float()[0, 0] * reciprocal[1].item() == 2 ** -24


@pytest.mark.parametrize("attrs", [{"block_n": 0}, {"block_k": -1}, {"block_n": True},
                                    {"block_k": 128.5}, {"minimum_amax": 0}, {"minimum_amax": float("nan")},
                                    {"minimum_amax": float("inf")}, {"minimum_amax": True},
                                    {"minimum_amax": 1e300}, {"minimum_amax": 1e-300},
                                    {"minimum_amax": 1e-45}, {"minimum_amax": 10 ** 1000},
                                    {"scale_division": "bad"}, {"scale_division": []}])
def test_rejects_invalid_requantization_parameters(attrs):
    with pytest.raises(IRSchemaError):
        RequantizeBlockScaledFP8.normalize_attrs({"block_n": 128, "block_k": 128, **attrs})


def test_rejects_runtime_quantization_before_lowering():
    with pytest.raises(IRVerificationError, match="closed constant inputs"):
        fm.verify_module(Requantized(runtime=True).build())


def test_rejects_distributed_requantization_even_when_resumed_from_raw_ir():
    from dataclasses import replace

    from triton.flagmega.options import parse_placement

    module = Requantized().build()
    placement = parse_placement("x=2/b")

    def distributed(value_type):
        if isinstance(value_type, fm.TupleType):
            return fm.TupleType(tuple(distributed(field) for field in value_type.fields))
        return fm.DistributedType(value_type, tuple(fm.SBP.broadcast() for _ in value_type.shape), placement)

    malformed = replace(module, nodes=tuple(replace(node, type=distributed(node.type)) for node in module.nodes))
    with pytest.raises(IRVerificationError, match="offline before distribution"):
        fm.verify_module(malformed)


@pytest.mark.parametrize("mode", ["rn", "reciprocal_multiply"])
def test_freeze_materialization_and_fresh_process_round_trip(tmp_path, mode):
    module = Requantized(casts=True, scale_division=mode).build()
    attrs = module.node_map["requantized"].attrs
    assert attrs.get("scale_division", "rn") == mode
    assert ("scale_division" in attrs) == (mode != "rn")
    path = fm.emit_module(module, tmp_path / "requantized.py")
    assert fm.load_module(path).semantic_hash == module.semantic_hash
    subprocess.run([sys.executable, "-c", ("from triton.flagmega.ir import load_module; import sys; "
                    "assert load_module(sys.argv[1]).semantic_hash == sys.argv[2]"), str(path), module.semantic_hash],
                   check=True)
    frozen = freeze_constant_islands(module)
    assert all(node.op != "nn.requantize_block_scaled_fp8" for node in frozen.nodes)
    assert len(frozen.constant_recipes) == 1
    frozen_path = fm.emit_module(frozen, tmp_path / "requantized-frozen.py")
    assert fm.load_module(frozen_path).semantic_hash == frozen.semantic_hash
    weights = _simple_weights()
    resolver = DictWeightResolver(weights)
    assets = materialize_constant_assets(frozen, resolver)
    assert set(assets) == set(frozen.constant_recipes[0].outputs)
    expected = TorchEvaluator(resolver).run(module, {})
    actual = TorchEvaluator(resolver).run(frozen, {})
    torch.testing.assert_close(actual[0].float(), expected[0].float(), rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
