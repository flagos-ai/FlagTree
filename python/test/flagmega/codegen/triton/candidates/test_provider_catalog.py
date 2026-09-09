# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import inspect

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates import (
    TritonCandidateProposal,
    TritonCandidateProviderRegistry,
    default_triton_candidate_registry,
)
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.targets import NvidiaSm90Target


def test_default_catalog_has_one_explicit_provider_for_every_reviewed_op():
    registry = default_triton_candidate_registry()

    assert registry.op_names == frozenset({
        "distributed.boxing",
        "distributed.force_boxing",
        "math.add",
        "math.block_scaled_matmul",
        "math.matmul",
        "math.mul",
        "math.div",
        "math.sigmoid",
        "math.reduce_sum",
        "math.packed_block_scaled_matmul",
        "math.packed_dense_matmul",
        "math.silu",
        "math.vectorized_binary",
        "math.vectorized_unary",
        "nn.dense_matmul_glu",
        "nn.delta_rule_coefficients",
        "nn.delta_rule_log_prefix",
        "nn.delta_rule_block_update",
        "nn.delta_rule_gates",
        "nn.l2_normalization",
        "nn.embedding",
        "nn.gdn_convolution",
        "nn.gdn_recurrent_core",
        "nn.greedy_sample",
        "nn.matmul_glu",
        "nn.norm_apply",
        "nn.norm_stats",
        "nn.packed_dense_matmul_glu",
        "nn.packed_matmul_glu",
        "nn.qkv_rope_with_cache",
        "nn.rms_norm",
        "nn.rope",
        "nn.softmax",
        "nn.sparse_experts_gate_up",
        "nn.sparse_experts_down",
        "nn.rotary_embedding",
        "nn.update_paged_attention_kv_cache",
        "ntt.matmul_norm_stats",
        "ntt.matmul_norm_stats_combine",
        "ntt.gather_reduce_add_norm_apply",
        "ntt.gather_reduce_qkv_rope_with_cache",
        "ntt.gather_reduce_norm_apply",
        "ntt.packed_matmul",
        "ntt.packed_qkv_parallel_linear",
        "ntt.paged_attention_partial",
        "ntt.paged_attention_combine",
        "ntt.vectorized_cast",
        "ntt.vectorized_rope",
        "tensors.cast",
        "tensors.concat",
        "tensors.broadcast_to",
        "tensors.top_k",
        "tensors.pad",
        "tensors.pack",
        "tensors.unpack",
        "tensors.slice",
        "tensors.slice_to_shape",
    })
    assert len(registry.providers) == 25


def test_selection_orchestrator_contains_no_candidate_catalog_branches():
    source = inspect.getsource(inspect.getmodule(TritonTirSelectionPolicy))

    assert "Candidate(" not in source
    assert 'node.op ==' not in source
    assert 'node.op in' not in source


def test_injected_provider_extends_selector_without_editing_central_policy():
    class UnitProvider:
        op_names = frozenset({"unit.identity"})

        def propose(self, node, context):
            del node, context
            return TritonCandidateProposal((fm.Candidate(
                "tir.unit.identity",
                {"family": "unit", "variant": "identity"},
                {"portable_triton": True},
            ),), "tir.unit.identity")

    registry = TritonCandidateProviderRegistry()
    registry.add(UnitProvider())
    policy = TritonTirSelectionPolicy(lambda _node, values, _module, **_kwargs: values, registry)
    target = NvidiaSm90Target(tir_selection_policy=policy)
    builder = fm.IRBuilder(dialect="high_level", stage="frozen_constants")
    value_type = fm.tensor_type("float32", (16,))
    value = builder.var("value", value_type, id="value")
    result = builder.call("unit.identity", (value,), value_type, id="result")
    builder.function("main", (value,), (result,))

    proposed = target.propose_tir(builder.build(entry="main"))

    point = next(point for point in proposed.selection_points if point.id == "tir.result")
    assert point.default_candidate == "tir.unit.identity"
    assert proposed.selection_map[point.id].candidate_id == "tir.unit.identity"
