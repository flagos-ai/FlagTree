# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateProviderRegistry,
    PagedAttentionCombineCandidateProvider,
    PagedAttentionPartialCandidateProvider,
    TypeInferenceCandidateProvider,
    build_search_graph,
)
from triton.flagmega.targets import NvidiaSm90Target


def _registry(target):
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    return registry


def _silu_module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    value_type = fm.tensor_type("bfloat16", (1, 128))
    value = builder.var("value", value_type, id="value")
    output = builder.call("math.silu", (value,), value_type, id="output")
    builder.function("main", (value,), (output,))
    return builder.build(entry="main")


def test_reviewed_broadcast_provider_is_explicit_and_has_nonzero_work():
    target = NvidiaSm90Target()
    graph = build_search_graph(
        _silu_module(),
        target.distributed_placements(_silu_module())[0],
        _registry(target),
        target.distributed_reshard_realization_policy(),
    )
    bucket = graph.bucket_map["output"]
    candidate = bucket.candidates[0]

    assert bucket.executable
    assert candidate.reason == "explicit-broadcast-inference"
    assert candidate.operation_cost > 0
    assert candidate.objective_model == "flagmega.explicit-broadcast-work/v1"
    assert isinstance(candidate.return_type, fm.DistributedType)


def test_unknown_compute_op_cannot_use_silent_broadcast_fallback():
    target = NvidiaSm90Target()
    value_type = fm.tensor_type("bfloat16", (1, 16))
    value = fm.Node("value", "builtin.var", (), value_type, attrs={"name": "value"})
    # GatedDeltaNet must be decomposed before AutoDistribution and therefore
    # deliberately has no provider at this stage.
    unknown = fm.Node("output", "nn.gated_delta_net", (value.id,), value_type)
    module = fm.IRModule(
        "high_level",
        "packed",
        (value, unknown),
        (fm.Function("main", (value.id,), (unknown.id,)),),
        "main",
    )

    with pytest.raises(IRVerificationError, match="no reviewed candidate provider"):
        build_search_graph(
            module,
            target.distributed_placements(module)[0],
            _registry(target),
            target.distributed_reshard_realization_policy(),
        )


def test_all_current_semantic_families_are_explicitly_classified():
    registry = _registry(NvidiaSm90Target())
    names = set(registry.op_names)

    assert {
        "math.add",
        "math.block_scaled_matmul",
        "math.matmul",
        "math.mul",
        "math.silu",
        "nn.embedding",
        "nn.gdn_convolution",
        "nn.gdn_recurrent_core",
        "nn.greedy_sample",
        "nn.rms_norm",
        "tensors.cast",
        "tensors.bitcast",
        "tensors.concat",
        "tensors.pad",
        "tensors.permute",
        "tensors.reshape",
        "tensors.slice_to_shape",
    }.issubset(names)
    for op_name in (
        "nn.rms_norm",
        "nn.rope",
        "nn.update_paged_attention_kv_cache",
        "ntt.vectorized_cast",
        "ntt.vectorized_rope",
        "tensors.pack",
        "tensors.bitcast",
        "tensors.reshape",
        "tensors.unpack",
    ):
        assert isinstance(registry.try_get(op_name), TypeInferenceCandidateProvider)
    assert registry.try_get("nn.paged_attention") is None
    assert isinstance(
        registry.try_get("ntt.paged_attention_partial"),
        PagedAttentionPartialCandidateProvider,
    )
    assert isinstance(
        registry.try_get("ntt.paged_attention_combine"),
        PagedAttentionCombineCandidateProvider,
    )
