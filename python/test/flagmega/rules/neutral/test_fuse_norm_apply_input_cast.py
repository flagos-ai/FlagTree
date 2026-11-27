# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Input conversion belongs to NormApply without changing external stats."""

from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.target_independent import decompose_complex_ops


def graph(source_dtype, converted_dtype, *, use_mean=False, round_before_scale=False, output_dtype=None, exported_cast=False):
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type(source_dtype, (2, 32)), id="x")
            scale = self.input("scale", fm.tensor_type("bfloat16", (32,)), id="scale")
            bias = self.input("bias", fm.tensor_type("bfloat16", (32,)), id="bias")
            # Stats intentionally see the unconverted value, as in the mHC
            # collapsed numerator path. Do not recompute them from BF16 x.
            stats = fm.F.nn.norm_stats(x, axis=-1, use_mean=use_mean, name="stats")
            converted = fm.F.tensors.cast(x, converted_dtype, name="converted")
            norm = fm.F.nn.norm_apply(converted, stats, scale, bias, axis=-1, epsilon=1e-6,
                use_mean=use_mean, round_before_scale=round_before_scale, output_dtype=output_dtype, name="norm")
            self.function("main", (x, scale, bias), (norm, converted) if exported_cast else (norm,))
    return Graph(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.parametrize("source,converted", [("float32", "bfloat16"), ("bfloat16", "float32")])
@pytest.mark.parametrize("use_mean", [False, True])
@pytest.mark.parametrize("round_before_scale", [False, True])
@pytest.mark.parametrize("output_dtype", [None, "float32"])
def test_input_cast_fuses_with_independent_stats_and_all_rounding_boundaries(tmp_path, source, converted,
        use_mean, round_before_scale, output_dtype):
    original = graph(source, converted, use_mean=use_mean, round_before_scale=round_before_scale, output_dtype=output_dtype)
    original = replace(original, metadata={**original.metadata, "floating_point_contract": "strict"})
    fused = decompose_complex_ops(original)
    norm = fused.node_map["norm"]
    assert norm.op == "nn.norm_apply"
    assert norm.attrs["input_dtype"] == converted
    assert norm.inputs[0] == "x"
    assert not any(node.op == "tensors.cast" for node in fused.nodes)
    assert fused.node_map["stats"].inputs == ("x",)
    assert norm.type == original.node_map["norm"].type
    rng = torch.Generator().manual_seed(1729)
    values = {"x": (torch.randn((2, 32), generator=rng) * 3).to(getattr(torch, source)),
              "scale": torch.randn((32,), generator=rng).bfloat16(),
              "bias": torch.randn((32,), generator=rng).bfloat16()}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    old, = evaluator.run(original, values)
    new, = evaluator.run(fused, values)
    assert torch.equal(old.view(torch.uint8), new.view(torch.uint8))
    assert fm.load_module(fm.emit_module(fused, tmp_path / "norm.py")).semantic_hash == fused.semantic_hash


def test_exported_input_conversion_remains_observable():
    original = graph("float32", "bfloat16", exported_cast=True)
    fused = decompose_complex_ops(original)
    assert fused.node_map["converted"].op == "tensors.cast"
    assert sum(node.op == "nn.norm_apply" for node in fused.nodes) == 1
    assert fused.function_map["main"].outputs == ("norm", "converted")


@pytest.mark.parametrize("shared_converted_stats", [False, True])
def test_private_matmul_widening_wins_before_consumer_input_fusion(shared_converted_stats):
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (2, 32)), id="x")
            weight = self.input("weight", fm.tensor_type("bfloat16", (32, 32)), id="weight")
            scale = self.input("scale", fm.tensor_type("bfloat16", (32,)), id="scale")
            bias = self.input("bias", fm.tensor_type("bfloat16", (32,)), id="bias")
            projection = fm.F.math.matmul(x, weight, transpose_b=True, name="projection")
            converted = fm.F.tensors.cast(projection, "float32", name="converted")
            stats = (fm.F.nn.norm_stats(converted, axis=-1, use_mean=False, name="stats") if shared_converted_stats
                     else self.input("stats", fm.tensor_type("float32", (1, 2, 1)), id="stats"))
            norm = fm.F.nn.norm_apply(converted, stats, scale, bias, axis=-1, epsilon=1e-6, use_mean=False, name="norm")
            parameters = (x, weight, scale, bias) if shared_converted_stats else (x, weight, scale, bias, stats)
            self.function("main", parameters, (norm,))
    original = Graph(dialect="high_level", stage="imported", entry="main").build()
    fused = decompose_complex_ops(original)
    # Under the inferred-precision policy, the producer can directly return
    # FP32. Preserve that opportunity before recording a consumer conversion.
    assert fused.node_map["norm"].attrs.get("input_dtype") is None
    assert fused.node_map["converted"].op == "math.matmul"
    assert fused.node_map["converted"].type.dtype == fm.DType.FLOAT32
    assert not any(node.op == "tensors.cast" for node in fused.nodes)


def test_shared_rounding_keeps_the_packed_projection_epilogue(tmp_path):
    from triton.flagmega.passes.tir.fuse_packed_accumulator_cast import fuse_packed_accumulator_casts

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (2, 32)), id="x")
            weight = self.input("weight", fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (2, 4)), id="weight")
            none = fm.F.builtin.none()
            projection = fm.F.ntt.packed_matmul(x, weight, none, none, output_data_type="float32", name="projection")
            rounded_type = fm.vector_type("bfloat16", projection.type.dtype.lanes)
            converted = fm.F.tensors.cast(projection, rounded_type, name="converted")
            scale = self.input("scale", fm.tensor_type(rounded_type, projection.type.shape[1:]), id="scale")
            bias = self.input("bias", scale.type, id="bias")
            stats = fm.F.nn.norm_stats(converted, axis=-1, use_mean=False, name="stats")
            norm = fm.F.nn.norm_apply(converted, stats, scale, bias, axis=-1, epsilon=1e-6, use_mean=False, name="norm")
            self.function("main", (x, weight, scale, bias), (norm,))

    original = Graph(dialect="high_level", stage="imported", entry="main",
                     metadata={"floating_point_contract": "strict"}).build()
    decomposed = decompose_complex_ops(original)
    # Both users observe the same rounded tensor. A consumer-only bypass
    # would make the FP32 producer shared and prevent its single BF16 store.
    assert decomposed.node_map["norm"].attrs.get("input_dtype") is None
    assert decomposed.node_map["norm"].inputs[0] == "converted"
    assert decomposed.node_map["stats"].inputs == ("converted",)
    fused = fm.verify_module(fuse_packed_accumulator_casts(decomposed))
    assert fused.node_map["converted"].op == "ntt.packed_matmul"
    assert fused.node_map["converted"].attrs["output_data_type"] == "float32"
    assert fused.node_map["converted"].attrs["post_ops"]
    assert not any(node.op == "tensors.cast" for node in fused.nodes)
    assert sum(node.op == "ntt.packed_matmul" for node in fused.nodes) == 1
    assert fm.load_module(fm.emit_module(fused, tmp_path / "shared.py")).semantic_hash == fused.semantic_hash
    rng = torch.Generator().manual_seed(4951)
    values = {"x": torch.randn((2, 32), generator=rng).bfloat16(),
              "weight": torch.randn((2, 4, 8, 2, 8), generator=rng).bfloat16(),
              "scale": torch.randn((4, 2, 4), generator=rng).bfloat16(),
              "bias": torch.randn((4, 2, 4), generator=rng).bfloat16()}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    before, = evaluator.run(original, values)
    after, = evaluator.run(fused, values)
    assert torch.equal(before.view(torch.uint8), after.view(torch.uint8))
