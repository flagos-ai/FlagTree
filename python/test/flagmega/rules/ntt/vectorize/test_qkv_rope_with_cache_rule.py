# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import (
    DictWeightResolver,
    TorchEvaluator,
    create_paged_attention_state,
)
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    PagedAttentionStateConfig,
)
from triton.flagmega.rules import DataflowRewriter, RewriteRule
from triton.flagmega.rules.ntt.vectorize import VectorizeQKVRoPEWithCache


class _FusedModule(fm.Module):
    def __init__(self, *, head_dim: int = 16):
        super().__init__(dialect="nn", stage="decomposed", entry="main")
        self.head_dim = head_dim
        self.config = PagedAttentionStateConfig(
            1, 1, 16, block_size=4, num_blocks=2, lanes=8
        )

    def forward(self):
        q = self.input("q", fm.tensor_type("bfloat16", (1, 2, self.head_dim)))
        k = self.input("k", fm.tensor_type("bfloat16", (1, 1, self.head_dim)))
        v = self.input("v", fm.tensor_type("bfloat16", (1, 1, self.head_dim)))
        q_scale = self.input(
            "q_scale", fm.tensor_type("bfloat16", (self.head_dim,))
        )
        k_scale = self.input(
            "k_scale", fm.tensor_type("bfloat16", (self.head_dim,))
        )
        q_bias = self.input(
            "q_bias", fm.tensor_type("bfloat16", (self.head_dim,))
        )
        k_bias = self.input(
            "k_bias", fm.tensor_type("bfloat16", (self.head_dim,))
        )
        cos = self.input("cos", fm.tensor_type("float32", (1, 1, self.head_dim)))
        sin = self.input("sin", fm.tensor_type("float32", (1, 1, self.head_dim)))
        state = self.input("state", self.config.ref_type)
        layer = self.input("layer", fm.tensor_type("int32", ()))
        advance = self.input("advance", fm.tensor_type("bool", ()))
        qkv = fm.F.builtin.tuple(q, k, v, name="qkv")
        fused = fm.F.nn.qkv_rope_with_cache(
            qkv,
            q_scale,
            k_scale,
            q_bias,
            k_bias,
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
        query = fm.F.tensors.get_item(fused, 0, name="query")
        updated = fm.F.tensors.get_item(fused, 1, name="updated")
        self.function(
            "main",
            (
                q,
                k,
                v,
                q_scale,
                k_scale,
                q_bias,
                k_bias,
                cos,
                sin,
                state,
                layer,
                advance,
            ),
            (query, updated),
        )


def test_rule_packs_qkv_norm_parameters_and_double_lane_rope_tables():
    module = _FusedModule().build()
    rule = VectorizeQKVRoPEWithCache()

    candidate = rule.candidates(module.node_map["fused"], module)[0]
    result = rule.rewrite(module.node_map["fused"], module, candidate)

    replacement = result.replacement
    qkv = next(node for node in result.prefix_nodes if node.id == replacement.inputs[0])
    packed_q = next(node for node in result.prefix_nodes if node.id == qkv.inputs[0])
    packed_cos = next(
        node for node in result.prefix_nodes if node.id == replacement.inputs[5]
    )
    assert packed_q.attrs == {"lanes": (8,), "axes": (2,)}
    assert packed_cos.attrs == {"lanes": (2, 8), "axes": (2, 2)}
    assert replacement.type == module.node_map["fused"].type
    assert replacement.effect == module.node_map["fused"].effect


def test_rule_rewrite_is_evaluator_equivalent_for_query_and_cache_updates():
    module = _FusedModule().build()
    rule = VectorizeQKVRoPEWithCache()
    candidate = rule.candidates(module.node_map["fused"], module)[0]
    rewritten = DataflowRewriter(
        (
            RewriteRule(
                "VectorizeQKVRoPEWithCache:fused",
                lambda node, _: node.id == "fused"
                and "vectorized_from" not in node.metadata,
                lambda node, current: rule.rewrite(node, current, candidate),
            ),
        ),
        remove_unused=False,
    ).rewrite(module)
    torch.manual_seed(9)
    values = {
        "q": torch.randn((1, 2, 16), dtype=torch.bfloat16),
        "k": torch.randn((1, 1, 16), dtype=torch.bfloat16),
        "v": torch.randn((1, 1, 16), dtype=torch.bfloat16),
        "q_scale": torch.randn((16,), dtype=torch.bfloat16),
        "k_scale": torch.randn((16,), dtype=torch.bfloat16),
        "q_bias": torch.randn((16,), dtype=torch.bfloat16),
        "k_bias": torch.randn((16,), dtype=torch.bfloat16),
        "cos": torch.ones((1, 1, 16), dtype=torch.float32),
        "sin": torch.zeros((1, 1, 16), dtype=torch.float32),
        "layer": torch.tensor(0, dtype=torch.int32),
        "advance": torch.tensor(True),
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    original_state = create_paged_attention_state(_FusedModule().config)
    rewritten_state = original_state.clone()

    original_query, original_result_state = evaluator.run(
        module, {**values, "state": original_state}
    )
    rewritten_query, rewritten_result_state = evaluator.run(
        rewritten, {**values, "state": rewritten_state}
    )

    torch.testing.assert_close(rewritten_query, original_query)
    torch.testing.assert_close(
        rewritten_result_state.kv_caches, original_result_state.kv_caches
    )
    assert rewritten_result_state.sequence_length == original_result_state.sequence_length


def test_rule_does_not_revectorize_its_cache_native_qkv_inputs():
    module = _FusedModule().build()
    rule = VectorizeQKVRoPEWithCache()
    candidate = rule.candidates(module.node_map["fused"], module)[0]
    result = rule.rewrite(module.node_map["fused"], module, candidate)
    root_index = next(
        index for index, node in enumerate(module.nodes) if node.id == "fused"
    )
    rewritten = replace(
        module,
        nodes=(
            *module.nodes[:root_index],
            *result.prefix_nodes,
            result.replacement,
            *module.nodes[root_index + 1 :],
        ),
    )

    assert rule.candidates(rewritten.node_map["fused"], rewritten) == ()
