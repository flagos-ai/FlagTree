# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    PagedAttentionStateConfig,
)
from triton.flagmega.passes.auto_vectorize import AutoVectorizePass
from triton.flagmega.targets import NvidiaSm90Target


class _CacheUpdatingQKVRoPE(fm.Module):
    def __init__(self):
        super().__init__(dialect="nn", stage="decomposed", entry="main")
        self.config = PagedAttentionStateConfig(
            1, 1, 16, block_size=4, num_blocks=2, lanes=8
        )

    def forward(self):
        q = self.input("q", fm.tensor_type("bfloat16", (1, 2, 16)))
        k = self.input("k", fm.tensor_type("bfloat16", (1, 1, 16)))
        v = self.input("v", fm.tensor_type("bfloat16", (1, 1, 16)))
        q_scale = self.input("q_scale", fm.tensor_type("bfloat16", (16,)))
        k_scale = self.input("k_scale", fm.tensor_type("bfloat16", (16,)))
        bias = self.input("bias", fm.tensor_type("bfloat16", (16,)))
        cos = self.input("cos", fm.tensor_type("float32", (1, 1, 16)))
        sin = self.input("sin", fm.tensor_type("float32", (1, 1, 16)))
        state = self.input("state", self.config.ref_type)
        layer = self.input("layer", fm.tensor_type("int32", ()))
        advance = self.input("advance", fm.tensor_type("bool", ()))
        qkv = fm.F.builtin.tuple(q, k, v, name="qkv")
        fused = fm.F.nn.qkv_rope_with_cache(
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
            qkv_layout=("seq", "head", "dim"),
            attention_layout=("seq", "head", "dim"),
            name="fused",
        )
        self.function(
            "main",
            (
                q,
                k,
                v,
                q_scale,
                k_scale,
                bias,
                cos,
                sin,
                state,
                layer,
                advance,
            ),
            (fused,),
        )


def test_selected_effectful_vectorization_is_materialized_outside_the_egraph():
    original = _CacheUpdatingQKVRoPE().build()
    target = NvidiaSm90Target()
    proposed = AutoVectorizePass.propose(original, target)

    rewritten = AutoVectorizePass.run(proposed, target)

    fused = rewritten.node_map["fused"]
    assert fused.metadata["vectorized_from"] == "nn.qkv_rope_with_cache"
    assert fused.effect == original.node_map["fused"].effect
    qkv = rewritten.node_map[fused.inputs[0]]
    for input_id in qkv.inputs:
        tensor = rewritten.node_map[input_id].type
        assert isinstance(tensor, fm.TensorType)
        assert isinstance(tensor.dtype, fm.VectorType)
        assert tensor.dtype.lanes == (8,)
    assert tuple(
        node.id for node in rewritten.nodes if not node.effect.is_pure
    ) == ("fused",)
