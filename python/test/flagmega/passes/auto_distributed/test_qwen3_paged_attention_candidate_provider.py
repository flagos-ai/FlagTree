# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateProviderRegistry,
    build_search_graph,
)
from triton.flagmega.targets import NvidiaSm90Target


@pytest.mark.parametrize(
    "op", ["nn.qwen3_paged_attention", "nn.packed_qwen3_paged_attention"]
)
def test_legacy_fused_attention_is_not_hidden_behind_a_replicated_policy(op):
    hidden = fm.tensor_type("bfloat16", (1, 2048))
    state = fm.RefType(
        "paged_attention",
        (("kv", fm.tensor_type("bfloat16", (16, 2, 8, 128))),),
    )
    value = fm.Node("value", "builtin.var", (), hidden, attrs={"name": "value"})
    cache = fm.Node("cache", "builtin.var", (), state, attrs={"name": "cache"})
    call = fm.Node(
        "output", op, (value.id, cache.id), fm.TupleType((hidden, state))
    )
    module = fm.IRModule(
        "high_level",
        "packed",
        (value, cache, call),
        (fm.Function("main", (value.id, cache.id), (call.id,)),),
        "main",
    )
    target = NvidiaSm90Target()
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)

    assert registry.try_get(op) is None
    with pytest.raises(
        IRVerificationError,
        match="no reviewed candidate provider",
    ):
        build_search_graph(
            module,
            target.distributed_placements(module)[0],
            registry,
            target.distributed_reshard_realization_policy(),
        )
