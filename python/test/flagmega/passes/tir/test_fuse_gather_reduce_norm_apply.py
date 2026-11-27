# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from dataclasses import replace

import pytest
from triton.flagmega.passes.tir import fuse_gather_reduce_norm_apply


def _graph(*, extra_stats_user=False, reduce_op=fm.ReduceOp.SUM, zero_bias=False):
    placement = fm.Placement((2, 4), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    split = fm.SBP.split_contiguous((0, 1))
    value_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 8)),
        (broadcast, split),
        placement,
    )
    parameter_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (8,)), (split,), placement
    )
    materialized_type = fm.DistributedType(
        fm.tensor_type("float32", (1, 1, 1)),
        (broadcast, broadcast, broadcast),
        placement,
    )
    partial_type = fm.DistributedType(
        materialized_type.tensor,
        materialized_type.axis_policies,
        placement,
        fm.SBP.partial((0, 1), reduce_op),
    )

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(
                dialect="ntt", stage="frozen_constants", entry="main"
            )

        def forward(self):
            partial = self.input("partial", partial_type, id="partial")
            value = self.input("value", value_type, id="value")
            scale = self.input("scale", parameter_type, id="scale")
            parameters = [partial, value, scale]
            if zero_bias:
                dense_zero = fm.F.builtin.splat_const(
                    parameter_type.tensor, 0.0, name="dense_zero"
                )
                bias = fm.F.distributed.sharded_view(
                    dense_zero, parameter_type, name="bias"
                )
            else:
                bias = self.input("bias", parameter_type, id="bias")
                parameters.append(bias)
            materialized = fm.F.distributed.boxing(
                partial, materialized_type, name="materialized"
            )
            output = fm.F.nn.norm_apply(
                value,
                materialized,
                scale,
                bias,
                axis=-1,
                epsilon=1e-6,
                use_mean=False,
                name="output",
                metadata={
                    "selected_vectorization": "vectorization.norm_apply.reduction_axis",
                    "selected_vector_axes": (-1,),
                    "selected_vector_lanes": (8,),
                },
            )
            outputs = (output, materialized) if extra_stats_user else (output,)
            self.function("main", tuple(parameters), outputs)

    return Graph().build()


def test_fuses_single_use_partial_boxing_and_preserves_editable_contract():
    module = fuse_gather_reduce_norm_apply(_graph())

    output = module.node_map["output"]
    assert output.op == "ntt.gather_reduce_norm_apply"
    assert output.inputs == ("partial", "value", "scale", "bias")
    assert output.attrs["materialized_stats_type"].partial is None
    assert output.attrs["has_bias"] is True
    assert output.metadata["fused_partial_stats_materialization"] == "materialized"
    assert "materialized" not in module.node_map


def test_elides_bias_only_when_splat_zero_is_proven_through_views():
    module = fuse_gather_reduce_norm_apply(_graph(zero_bias=True))

    assert module.node_map["output"].attrs["has_bias"] is False


def test_keeps_materialization_with_an_additional_exact_value_user():
    module = fuse_gather_reduce_norm_apply(_graph(extra_stats_user=True))

    assert module.node_map["output"].op == "nn.norm_apply"
    assert module.node_map["materialized"].op == "distributed.boxing"


def test_keeps_non_sum_partial_materialization():
    module = fuse_gather_reduce_norm_apply(
        _graph(reduce_op=fm.ReduceOp.MAX)
    )

    assert module.node_map["output"].op == "nn.norm_apply"
    assert module.node_map["materialized"].op == "distributed.boxing"


@pytest.mark.parametrize("round_before_scale", [False, True])
def test_fusion_preserves_normalization_rounding(round_before_scale, tmp_path):
    source = _graph()
    source = replace(source, nodes=tuple(
        replace(node, attrs={**node.attrs, "round_before_scale": round_before_scale})
        if node.op == "nn.norm_apply" else node for node in source.nodes
    ))
    result = fuse_gather_reduce_norm_apply(source)
    assert result.node_map["output"].attrs["round_before_scale"] is round_before_scale
    assert fm.load_module(fm.emit_module(result, tmp_path / "fused.py")) == result
