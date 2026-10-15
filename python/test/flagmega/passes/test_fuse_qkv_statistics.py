# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Private QKV statistics must fuse after, and remain visible before, distribution."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig
from triton.flagmega.passes.tir.fuse_distributed_ops import fuse_distributed_ops


def qkv_statistics_graph(*, shared_stats=False, narrowing=False, use_mean=False, head_dim=64, rotary_dim=None,
                         heads_tuple=(4, 2, 2), dimension_sharded=False, hierarchy_levels="bb"):
    mesh = fm.Placement((2, 2), "yx", hierarchy_levels)
    broad = fm.SBP.broadcast()
    vector = fm.vector_type("bfloat16", (8,))
    fields = tuple(fm.DistributedType(
        fm.tensor_type(vector, (1, heads, head_dim // 8) if dimension_sharded else (1, heads * (head_dim // 8))),
        ((broad, broad, fm.SBP.split_block_cyclic((1,), 1)) if dimension_sharded
         else (broad, fm.SBP.split_block_cyclic((1,), head_dim // 8))), mesh,
    ) for heads in heads_tuple)
    materialized = fm.TupleType(fields)
    partial = fm.TupleType(tuple(replace(field, partial=fm.SBP.partial((0,))) for field in fields))
    config = PagedAttentionStateConfig(1, heads_tuple[1], head_dim, block_size=4, num_blocks=2, lanes=8)

    class Graph(fm.Module):
        def forward(self):
            source = self.input("partials", partial, id="partials")
            combined = fm.F.ntt.packed_qkv_parallel_linear_combine(source, materialized, name="combined")
            values = tuple(fm.F.tensors.reshape(
                fm.F.tensors.get_item(combined, i, name=f"field_{i}"), (1, heads, head_dim // 8), name=f"view_{i}",
            ) for i, heads in enumerate(heads_tuple))
            stats = []
            for role, value in zip(("q", "k"), values):
                wide = fm.F.ntt.vectorized_cast(value, new_type=fm.vector_type("float32", (8,)),
                                                vectorize_axes=(2,), name=f"{role}_wide")
                if narrowing:
                    wide = fm.F.ntt.vectorized_cast(wide, new_type=fm.vector_type("float16", (8,)),
                                                    vectorize_axes=(2,), name=f"{role}_narrow")
                    wide = fm.F.ntt.vectorized_cast(wide, new_type=fm.vector_type("float32", (8,)),
                                                    vectorize_axes=(2,), name=f"{role}_rewiden")
                stat = fm.F.nn.norm_stats(wide, axis=-1, use_mean=use_mean, name=f"{role}_stats")
                if stat.type.partial is not None:
                    stat = fm.F.distributed.boxing(stat, replace(stat.type, partial=None), name=f"{role}_stats_sum")
                stats.append(stat)
            def parameter(name, dtype, shape):
                tensor = fm.tensor_type(dtype, shape)
                return self.input(name, fm.DistributedType(tensor, (broad,) * len(shape), mesh))
            scale = parameter("scale", "bfloat16", (head_dim,))
            bias = parameter("bias", "bfloat16", (head_dim,))
            cos = parameter("cos", "float32", (1, 1, rotary_dim or head_dim))
            sin = parameter("sin", "float32", (1, 1, rotary_dim or head_dim))
            state = self.input("state", config.ref_type)
            layer = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), 0)
            advance = fm.F.builtin.scalar_const(fm.tensor_type("bool", ()), True)
            result = fm.F.nn.qkv_rope_with_cache(
                fm.F.builtin.tuple(*values), scale, scale, bias, bias, cos, sin, state, layer, advance, *stats,
                q_axis=-1, q_epsilon=1e-6, q_use_mean=use_mean,
                k_axis=-1, k_epsilon=1e-6, k_use_mean=use_mean,
                rotary_dim=rotary_dim,
                qkv_layout=("seq", "head", "dim"), attention_layout=("seq", "head", "dim"), name="apply",
            )
            self.function("main", (source, scale, bias, cos, sin, state),
                          (result, stats[0]) if shared_stats else (result,))

    return Graph(dialect="high_level", stage="distributed", entry="main",
                 metadata={"auto_distribution": {"placement": mesh.to_data()}}).build(), config


@pytest.mark.parametrize("use_mean", [False, True])
def test_private_qkv_statistics_and_materialization_fuse(use_mean, tmp_path):
    source, _ = qkv_statistics_graph(use_mean=use_mean)
    result = fuse_distributed_ops(source)
    fused = [node for node in result.nodes if node.op == "ntt.gather_reduce_qkv_rope_with_cache"]
    assert len(fused) == 1
    assert fused[0].type == source.node_map["apply"].type
    assert fused[0].inputs[0] == "partials"
    assert not [node for node in result.nodes if node.op in {
        "nn.norm_stats", "tensors.cast", "ntt.vectorized_cast", "ntt.packed_qkv_parallel_linear_combine",
    }]
    fm.verify_module(result)
    path = fm.emit_module(result, tmp_path / "fused.py")
    assert fm.load_module(path).semantic_hash == result.semantic_hash
    assert fuse_distributed_ops(result).semantic_hash == result.semantic_hash


@pytest.mark.parametrize("kwargs", [
    {"shared_stats": True}, {"narrowing": True}, {"dimension_sharded": True}, {"hierarchy_levels": "cb"},
])
def test_observed_statistics_and_lossy_casts_prevent_fusion(kwargs):
    source, _ = qkv_statistics_graph(**kwargs)
    result = fuse_distributed_ops(source)
    assert not [node for node in result.nodes if node.op == "ntt.gather_reduce_qkv_rope_with_cache"]
    assert len([node for node in result.nodes if node.op == "nn.norm_stats"]) == 2


@pytest.mark.parametrize("use_mean", [False, True])
@pytest.mark.parametrize("round_before_scale", [False, True])
def test_fused_statistics_preserve_query_and_all_cache_bytes(use_mean, round_before_scale):
    torch = pytest.importorskip("torch")
    from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator, create_paged_attention_state
    source, config = qkv_statistics_graph(use_mean=use_mean)
    source = replace(source, nodes=tuple(replace(node, attrs={
        **dict(node.attrs), "q_round_before_scale": round_before_scale, "k_round_before_scale": round_before_scale,
    }) if node.id == "apply" else node for node in source.nodes))
    fused = fuse_distributed_ops(source)
    generator = torch.Generator().manual_seed(764)
    partials = tuple(torch.randn((1, heads * 8, 8), generator=generator).bfloat16() for heads in (4, 2, 2))
    values = {"partials": partials}
    for node in source.nodes:
        if node.op == "builtin.var" and node.id != "partials" and isinstance(node.type, fm.DistributedType):
            dtype = node.type.tensor.dtype
            shape = tuple(dim.fixed_value for dim in node.type.tensor.shape)
            values[node.attrs["name"]] = torch.randn(shape, generator=generator).to(
                torch.bfloat16 if dtype is fm.DType.BFLOAT16 else torch.float32)
    states = [create_paged_attention_state(config) for _ in range(2)]
    evaluator = TorchEvaluator(DictWeightResolver({}))
    outputs = [evaluator.run(module, {**values, "state": state})[0][0]
               for module, state in zip((source, fused), states)]
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
    torch.testing.assert_close(states[0].kv_caches, states[1].kv_caches, atol=0, rtol=0)
    assert states[0].sequence_length == states[1].sequence_length == 1
