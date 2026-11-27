# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fused statistic/residual paths retain the converted numerator contract."""

from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.target_independent import decompose_complex_ops
from triton.flagmega.passes.tir import fuse_gather_reduce_add_norm_apply, fuse_gather_reduce_norm_apply


def graph(family, use_mean, round_before_scale, output_dtype):
    mesh = fm.Placement((2, 4), "yx", "bb")
    broad = fm.SBP.broadcast()
    value_type = fm.DistributedType(fm.tensor_type(fm.vector_type("float32", (8,)), (1, 4)), (broad, broad), mesh)
    param_type = fm.DistributedType(fm.tensor_type(fm.vector_type("bfloat16", (8,)), (4,)), (broad,), mesh)
    stats_type = fm.DistributedType(fm.tensor_type("float32", (2 if use_mean else 1, 1, 1)), (broad, broad, broad), mesh)
    partial_type = replace(stats_type if family == "stats" else value_type, partial=fm.SBP.partial((0, 1)))

    class Graph(fm.Module):
        def forward(self):
            partial = self.input("partial", partial_type, id="partial")
            x = self.input("x", value_type, id="x")
            scale = self.input("scale", param_type, id="scale")
            bias = self.input("bias", param_type, id="bias")
            if family == "add":
                pair = fm.F.ntt.add_norm_stats(partial, x, axis=-1, use_mean=use_mean, name="combined")
                value, stats = fm.F.tensors.get_items(pair, 0, 1, name_prefix="combined")
                stats = fm.F.distributed.boxing(stats, stats_type, name="materialized_stats")
            else:
                value = x
                stats = fm.F.distributed.boxing(partial, stats_type, name="materialized_stats")
            converted = fm.F.tensors.cast(value, fm.vector_type("bfloat16", (8,)), name="converted")
            norm = fm.F.nn.norm_apply(converted, stats, scale, bias, axis=-1, epsilon=1e-6, use_mean=use_mean,
                round_before_scale=round_before_scale, output_dtype=output_dtype, name="norm")
            self.function("main", (partial, x, scale, bias), (value, norm) if family == "add" else (norm,))
    return Graph(dialect="ntt", stage="frozen_constants", entry="main").build()


@pytest.mark.parametrize("family", ["stats", "add"])
@pytest.mark.parametrize("use_mean,round_before_scale", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("output_dtype", [None, "float32"])
def test_collective_fusion_preserves_input_conversion_and_materialized_value(tmp_path, family, use_mean, round_before_scale, output_dtype):
    original = graph(family, use_mean, round_before_scale, output_dtype)
    converted = decompose_complex_ops(original)
    fuse = fuse_gather_reduce_norm_apply if family == "stats" else fuse_gather_reduce_add_norm_apply
    fused = fuse(converted)
    ops = [node for node in fused.nodes if node.op == f"ntt.gather_reduce_{'add_' if family == 'add' else ''}norm_apply"]
    assert len(ops) == 1
    assert ops[0].attrs["input_dtype"] == "bfloat16"
    assert not any(node.op == "tensors.cast" for node in fused.nodes)
    assert fused.function_map["main"].outputs == original.function_map["main"].outputs
    rng = torch.Generator().manual_seed(1729)
    values = {"x": torch.randn((1, 4, 8), generator=rng) * 3,
              "scale": torch.randn((4, 8), generator=rng).bfloat16(),
              "bias": torch.randn((4, 8), generator=rng).bfloat16()}
    if family == "add":
        values["partial"] = torch.randn((1, 4, 8), generator=rng)
    else:
        raw = values["x"].reshape(1, 32)
        square = raw.square().sum(-1, keepdim=True)
        values["partial"] = torch.stack((raw.sum(-1, keepdim=True), square)) if use_mean else square.unsqueeze(0)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    old, new = evaluator.run(original, values), evaluator.run(fused, values)
    for before, after in zip(old, new, strict=True):
        assert torch.equal(before.view(torch.uint8), after.view(torch.uint8))
    if family == "add":
        assert new[0].dtype == torch.float32
    assert fm.load_module(fm.emit_module(fused, tmp_path / "fused.py")).semantic_hash == fused.semantic_hash
