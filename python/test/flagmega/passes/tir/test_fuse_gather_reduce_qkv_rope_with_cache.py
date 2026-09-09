# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    PagedAttentionStateConfig,
)
from triton.flagmega.passes.tir import (
    fuse_gather_reduce_qkv_rope_with_cache,
)


def _distributed(value_type, policies, placement, partial=None):
    return fm.DistributedType(value_type, policies, placement, partial)


def _graph(*, extra_q_user=False, reduce_op=fm.ReduceOp.SUM, rotary_dim=None):
    placement = fm.Placement((2, 4), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    output_split = fm.SBP.split_block_cyclic((1,), 1)
    partial = fm.SBP.partial((0,), reduce_op)
    vector = fm.VectorType(fm.DType.BFLOAT16, (2,))
    q_packed_tensor = fm.tensor_type(vector, (1, 4))
    kv_packed_tensor = fm.tensor_type(vector, (1, 2))
    partial_qkv = fm.TupleType((
        _distributed(q_packed_tensor, (broadcast, output_split), placement, partial),
        _distributed(kv_packed_tensor, (broadcast, output_split), placement, partial),
        _distributed(kv_packed_tensor, (broadcast, output_split), placement, partial),
    ))
    materialized_qkv = fm.TupleType(tuple(
        _distributed(field.tensor, field.axis_policies, placement)
        for field in partial_qkv.fields
    ))
    q_logical = _distributed(
        fm.tensor_type(vector, (1, 2, 2)),
        (broadcast, broadcast, broadcast),
        placement,
    )
    kv_logical = _distributed(
        fm.tensor_type(vector, (1, 1, 2)),
        (broadcast, broadcast, broadcast),
        placement,
    )
    parameter = _distributed(
        fm.tensor_type(vector, (2,)), (broadcast,), placement
    )
    trig = _distributed(
        (fm.tensor_type("float32", (1, 1, rotary_dim)) if rotary_dim is not None else
         fm.tensor_type(fm.VectorType(fm.DType.FLOAT32, (2, 2)), (1, 1, 1))),
        (broadcast, broadcast, broadcast),
        placement,
    )
    state_type = PagedAttentionStateConfig(
        1, 1, 4, block_size=4, num_blocks=1, lanes=2
    ).ref_type

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="frozen_constants", entry="main")

        def forward(self):
            source = self.input("partial_qkv", partial_qkv, id="partial_qkv")
            q_scale = self.input("q_scale", parameter, id="q_scale")
            k_scale = self.input("k_scale", parameter, id="k_scale")
            bias = self.input("bias", parameter, id="bias")
            cos = self.input("cos", trig, id="cos")
            sin = self.input("sin", trig, id="sin")
            state = self.input("state", state_type, id="state")
            layer = fm.F.builtin.scalar_const(
                fm.tensor_type("int32", ()), 0, name="layer"
            )
            advance = fm.F.builtin.scalar_const(
                fm.tensor_type("bool", ()), True, name="advance"
            )
            combined = fm.F.distributed.boxing(
                source, materialized_qkv, name="combined"
            )
            packed = fm.F.tensors.get_items(
                combined, 0, 1, 2, name_prefix="packed"
            )
            broadcast_packed = tuple(
                fm.F.distributed.sharded_view(
                    value,
                    _distributed(
                        materialized_qkv.fields[index].tensor,
                        (broadcast, broadcast),
                        placement,
                    ),
                    name=f"broadcast_{index}",
                )
                for index, value in enumerate(packed)
            )
            views = tuple(
                fm.F.tensors.reshape(
                    value,
                    tuple(dimension.fixed_value for dimension in logical.tensor.shape),
                    name=f"view_{index}",
                )
                for index, (value, logical) in enumerate(
                    zip(broadcast_packed, (q_logical, kv_logical, kv_logical))
                )
            )
            qkv = fm.F.builtin.tuple(*views, name="qkv")
            fused_input = fm.F.nn.qkv_rope_with_cache(
                qkv,
                q_scale,
                k_scale,
                bias,
                bias,
                cos,
                sin,
                state,
                layer,
                advance,
                q_axis=-1,
                q_epsilon=1e-6,
                q_use_mean=False,
                k_axis=-1,
                k_epsilon=1e-6,
                k_use_mean=False,
                rotary_dim=rotary_dim,
                qkv_layout=("seq", "head", "dim"),
                attention_layout=("seq", "head", "dim"),
                name="qkv_rope",
            )
            outputs = [fused_input]
            if extra_q_user:
                outputs.append(views[0])
            self.function(
                "main",
                (source, q_scale, k_scale, bias, cos, sin, state),
                tuple(outputs),
            )

    return Graph().build()


def test_fuses_single_use_partial_materialization_and_view_chains():
    module = fuse_gather_reduce_qkv_rope_with_cache(_graph())

    fused = module.node_map["qkv_rope"]
    assert fused.op == "ntt.gather_reduce_qkv_rope_with_cache"
    assert fused.inputs[0] == "partial_qkv"
    assert fused.attrs["materialized_qkv_type"].fields[0].partial is None
    assert fused.attrs["logical_qkv_type"].fields[0].tensor.shape == (
        fm.dim(1), fm.dim(2), fm.dim(2)
    )
    assert "combined" not in module.node_map
    assert "qkv" not in module.node_map
    assert not any(node.id.startswith("packed_") for node in module.nodes)


def test_keeps_collective_when_a_q_view_has_an_additional_user():
    module = fuse_gather_reduce_qkv_rope_with_cache(_graph(extra_q_user=True))

    assert module.node_map["qkv_rope"].op == "nn.qkv_rope_with_cache"
    assert module.node_map["combined"].op == "distributed.boxing"


def test_keeps_non_sum_partial_collective():
    module = fuse_gather_reduce_qkv_rope_with_cache(
        _graph(reduce_op=fm.ReduceOp.MAX)
    )

    assert module.node_map["qkv_rope"].op == "nn.qkv_rope_with_cache"
    assert module.node_map["combined"].op == "distributed.boxing"
