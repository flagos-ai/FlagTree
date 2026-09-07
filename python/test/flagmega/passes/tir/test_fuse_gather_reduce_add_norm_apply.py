# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from dataclasses import replace

import pytest
from triton.flagmega.passes.tir import fuse_gather_reduce_add_norm_apply


def _graph(*, export_stats=False, second_stats_projection=False):
    placement = fm.Placement((2, 4), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    value_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 8)),
        (broadcast, broadcast),
        placement,
    )
    partial_type = fm.DistributedType(
        value_type.tensor,
        value_type.axis_policies,
        placement,
        fm.SBP.partial((0, 1)),
    )
    parameter_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (8,)), (broadcast,), placement
    )

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="frozen_constants", entry="main")

        def forward(self):
            partial = self.input("partial", partial_type, id="partial")
            residual = self.input("residual", value_type, id="residual")
            scale = self.input("scale", parameter_type, id="scale")
            bias = self.input("bias", parameter_type, id="bias")
            combine = fm.F.ntt.matmul_norm_stats_combine(
                partial,
                residual,
                axis=-1,
                use_mean=False,
                name="combine",
            )
            value = fm.F.tensors.get_item(combine, 0, name="value")
            stats = fm.F.tensors.get_item(combine, 1, name="stats")
            if second_stats_projection:
                extra_stats = fm.F.tensors.get_item(combine, 1, name="extra_stats")
            normalized = fm.F.nn.norm_apply(
                value,
                stats,
                scale,
                bias,
                axis=-1,
                epsilon=1e-6,
                use_mean=False,
                name="normalized",
                metadata={
                    "selected_vectorization": "vectorization.norm_apply.reduction_axis",
                    "selected_vector_axes": (-1,),
                    "selected_vector_lanes": (8,),
                },
            )
            outputs = [value, normalized]
            if export_stats:
                outputs.append(stats)
            if second_stats_projection:
                outputs.append(extra_stats)
            self.function(
                "main", (partial, residual, scale, bias), tuple(outputs)
            )

    return Graph().build()


def _sharded_value_graph(
    *, export_adapted_value=False, reversible_addend_view=False
):
    """Reproduce the Qwen down-output ``S(y) -> S(y,x)`` value view."""

    placement = fm.Placement((2, 4), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    split_y = fm.SBP.split_contiguous((0,))
    split_yx = fm.SBP.split_contiguous((0, 1))
    tensor = fm.tensor_type("bfloat16", (1, 32))
    coarse_value_type = fm.DistributedType(
        tensor, (broadcast, split_y), placement
    )
    normalized_value_type = fm.DistributedType(
        tensor, (broadcast, split_yx), placement
    )
    partial_type = fm.DistributedType(
        tensor,
        coarse_value_type.axis_policies,
        placement,
        fm.SBP.partial((1,)),
    )
    parameter_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (32,)), (split_yx,), placement
    )
    materialized_stats_type = fm.DistributedType(
        fm.tensor_type("float32", (1, 1, 1)),
        (broadcast, broadcast, broadcast),
        placement,
    )

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="frozen_constants", entry="main")

        def forward(self):
            partial = self.input("partial", partial_type, id="partial")
            residual = self.input(
                "residual",
                normalized_value_type if reversible_addend_view else coarse_value_type,
                id="residual",
            )
            scale = self.input("scale", parameter_type, id="scale")
            bias = self.input("bias", parameter_type, id="bias")
            combine_addend = (
                fm.F.distributed.sharded_view(
                    residual, coarse_value_type, name="coarse_residual"
                )
                if reversible_addend_view
                else residual
            )
            combine = fm.F.ntt.matmul_norm_stats_combine(
                partial,
                combine_addend,
                axis=-1,
                use_mean=False,
                name="combine",
            )
            value = fm.F.tensors.get_item(combine, 0, name="value")
            stats = fm.F.tensors.get_item(combine, 1, name="stats")
            adapted_value = fm.F.distributed.sharded_view(
                value, normalized_value_type, name="adapted_value"
            )
            materialized_stats = fm.F.distributed.boxing(
                stats, materialized_stats_type, name="materialized_stats"
            )
            normalized = fm.F.nn.norm_apply(
                adapted_value,
                materialized_stats,
                scale,
                bias,
                axis=-1,
                epsilon=1e-6,
                use_mean=False,
                name="normalized",
            )
            outputs = [normalized]
            if export_adapted_value:
                outputs.append(adapted_value)
            self.function(
                "main", (partial, residual, scale, bias), tuple(outputs)
            )

    return Graph().build()


def test_fuses_exact_value_and_stats_projections_into_two_result_op():
    module = fuse_gather_reduce_add_norm_apply(_graph())

    fused = module.node_map["combine"]
    assert fused.op == "ntt.gather_reduce_add_norm_apply"
    assert fused.inputs == ("partial", "residual", "scale", "bias")
    assert fused.type.fields == (
        module.node_map["value"].type,
        module.node_map["normalized"].type,
    )
    assert fused.metadata["fused_norm_apply"] == "normalized"
    assert fused.metadata["private_norm_stats"] == "stats"
    assert fused.metadata["norm_vectorization"]["selected_vector_lanes"] == (8,)
    assert module.node_map["value"].inputs == ("combine",)
    assert module.node_map["normalized"].op == "builtin.get_item"
    assert module.node_map["normalized"].inputs == ("combine",)
    assert module.node_map["normalized"].attrs == {"index": 1}
    assert "stats" not in module.node_map


def test_keeps_graph_when_statistics_are_exported():
    module = fuse_gather_reduce_add_norm_apply(_graph(export_stats=True))

    assert module.node_map["combine"].op == "ntt.matmul_norm_stats_combine"
    assert module.node_map["normalized"].op == "nn.norm_apply"


def test_keeps_graph_when_combine_has_another_statistics_projection():
    module = fuse_gather_reduce_add_norm_apply(
        _graph(second_stats_projection=True)
    )

    assert module.node_map["combine"].op == "ntt.matmul_norm_stats_combine"
    assert module.node_map["extra_stats"].attrs == {"index": 1}


def test_fuses_through_single_use_sharded_value_view():
    module = fuse_gather_reduce_add_norm_apply(_sharded_value_graph())

    fused = module.node_map["combine"]
    assert fused.op == "ntt.gather_reduce_add_norm_apply"
    assert fused.inputs == ("partial", "adapted_value", "scale", "bias")
    assert fused.type.fields == (
        module.node_map["normalized"].type,
        module.node_map["normalized"].type,
    )
    assert fused.metadata["private_value_view"] == "adapted_value"
    assert fused.metadata["private_addend_view"] == "adapted_value"
    assert module.node_map["adapted_value"].inputs == ("residual",)
    assert "value" not in module.node_map
    assert "materialized_stats" not in module.node_map


def test_keeps_sharded_value_view_when_it_has_an_additional_user():
    module = fuse_gather_reduce_add_norm_apply(
        _sharded_value_graph(export_adapted_value=True)
    )

    assert module.node_map["combine"].op == "ntt.matmul_norm_stats_combine"
    assert module.node_map["adapted_value"].op == "distributed.sharded_view"


def test_rewinds_inverse_addend_view_instead_of_materializing_an_alias_cycle():
    module = fuse_gather_reduce_add_norm_apply(
        _sharded_value_graph(reversible_addend_view=True)
    )

    fused = module.node_map["combine"]
    assert fused.op == "ntt.gather_reduce_add_norm_apply"
    assert fused.inputs == ("partial", "residual", "scale", "bias")
    assert fused.type.fields == (
        module.node_map["normalized"].type,
        module.node_map["normalized"].type,
    )
    assert fused.metadata["private_addend_view"] == "coarse_residual"
    assert "coarse_residual" not in module.node_map
    assert "adapted_value" not in module.node_map


@pytest.mark.parametrize("round_before_scale", [False, True])
def test_residual_fusion_preserves_normalization_rounding(round_before_scale, tmp_path):
    source = _graph()
    source = replace(source, nodes=tuple(
        replace(node, attrs={**node.attrs, "round_before_scale": round_before_scale})
        if node.op == "nn.norm_apply" else node for node in source.nodes
    ))
    result = fuse_gather_reduce_add_norm_apply(source)
    assert result.node_map["combine"].attrs["round_before_scale"] is round_before_scale
    assert fm.load_module(fm.emit_module(result, tmp_path / "fused.py")) == result
