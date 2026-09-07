# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import inspect

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
from triton.flagmega.passes.auto_distributed.providers import (
    PackedQKVParallelLinearCandidateProvider,
)
from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider


class PackedQKVModule(fm.Module):
    def __init__(self):
        super().__init__(dialect="high_level", stage="packed", entry="main")

    def forward(self):
        value = self.input("value", fm.tensor_type("bfloat16", (1, 2048)))
        q_weight = self.input(
            "q_weight",
            fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)), (128, 256)),
        )
        k_weight = self.input(
            "k_weight",
            fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)), (128, 128)),
        )
        v_weight = self.input(
            "v_weight",
            fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)), (128, 128)),
        )
        none = fm.F.builtin.none(name="none")
        qkv = fm.F.ntt.packed_qkv_parallel_linear(
            value,
            q_weight,
            k_weight,
            v_weight,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            num_heads=16,
            num_kv_heads=8,
            output_data_type="bfloat16",
            name="qkv",
        )
        self.function("main", (value, q_weight, k_weight, v_weight), (qkv,))


def test_packed_qkv_provider_couples_output_and_reduction_mesh_axes():
    module = PackedQKVModule().build()
    node = module.node_map["qkv"]
    placement = fm.Placement((8, 16), "yx", "bb")
    context = DistributedCandidateContext(
        module,
        node,
        placement,
        tuple((module.node_map[input_id].type,) for input_id in node.inputs),
        PyNttDistributedSplitCandidateProvider(block_bytes=128),
    )
    candidates = PackedQKVParallelLinearCandidateProvider().get_candidates(context)

    hybrid = next(
        candidate
        for candidate in candidates
        if candidate.reason == "packed-qkv-output-K-sbp-partial"
        and all(
            field.partial == fm.SBP.partial((0,))
            and field.axis_policies[-1] == fm.SBP.split_block_cyclic((1,), 8)
            for field in candidate.return_type.fields
        )
    )
    assert hybrid.input_types[0].axis_policies[1] == fm.SBP.split_block_cyclic((0,), 64)
    for weight_type in hybrid.input_types[1:4]:
        assert weight_type.axis_policies == (
            fm.SBP.split_block_cyclic((0,), 4),
            fm.SBP.split_block_cyclic((1,), 8),
        )
    assert hybrid.input_types[4:13] == (fm.NoneType(),) * 9


def test_packed_qkv_provider_couples_heterogeneous_block_cyclic_outputs():
    module = PackedQKVModule().build()
    node = module.node_map["qkv"]
    placement = fm.Placement((8, 16), "yx", "bb")
    context = DistributedCandidateContext(
        module,
        node,
        placement,
        tuple((module.node_map[input_id].type,) for input_id in node.inputs),
        PyNttDistributedSplitCandidateProvider(block_bytes=128),
    )

    candidate = next(
        candidate
        for candidate in PackedQKVParallelLinearCandidateProvider().get_candidates(context)
        if candidate.reason == "packed-qkv-output-sbp"
        and tuple(
            field.axis_policies[-1] for field in candidate.return_type.fields
        )
        == (
            fm.SBP.split_block_cyclic((0, 1), 2),
            fm.SBP.split_block_cyclic((0, 1), 1),
            fm.SBP.split_block_cyclic((0, 1), 1),
        )
    )

    assert candidate.input_types[0].axis_policies == (
        fm.SBP.broadcast(),
        fm.SBP.broadcast(),
    )
    assert tuple(
        weight_type.axis_policies[1] for weight_type in candidate.input_types[1:4]
    ) == (
        fm.SBP.split_block_cyclic((0, 1), 2),
        fm.SBP.split_block_cyclic((0, 1), 1),
        fm.SBP.split_block_cyclic((0, 1), 1),
    )
    local_mac_work = (2048 + 1024 + 1024) * 2048 // placement.size
    assert candidate.operation_cost == local_mac_work + 6_112


def test_packed_qkv_work_counts_vector_lanes_as_scalar_outputs():
    module = PackedQKVModule().build()
    node = module.node_map["qkv"]
    placement = fm.Placement((8, 16), "yx", "bb")
    context = DistributedCandidateContext(
        module,
        node,
        placement,
        tuple((module.node_map[input_id].type,) for input_id in node.inputs),
        PyNttDistributedSplitCandidateProvider(block_bytes=128),
    )

    replicated = next(
        candidate
        for candidate in PackedQKVParallelLinearCandidateProvider().get_candidates(context)
        if candidate.reason == "broadcast-replicated"
    )

    # Q/K/V contain 2048 + 1024 + 1024 scalar outputs, each reduced over
    # K=2048.  The lower-order shape term is 0 + 1024 + 1024.
    scalar_mac_work = (2048 + 1024 + 1024) * 2048
    assert replicated.operation_cost == scalar_mac_work + 2_048


def test_packed_qkv_objective_prefers_balanced_hybrid_after_partial_combine():
    module = PackedQKVModule().build()
    node = module.node_map["qkv"]
    placement = fm.Placement((8, 16), "yx", "bb")
    context = DistributedCandidateContext(
        module,
        node,
        placement,
        tuple((module.node_map[input_id].type,) for input_id in node.inputs),
        PyNttDistributedSplitCandidateProvider(block_bytes=128),
    )
    candidates = PackedQKVParallelLinearCandidateProvider().get_candidates(context)

    direct = next(
        candidate
        for candidate in candidates
        if candidate.reason == "packed-qkv-output-sbp"
        and all(
            field.axis_policies[-1].hierarchy_axes == (0, 1)
            for field in candidate.return_type.fields
        )
    )
    hybrid = next(
        candidate
        for candidate in candidates
        if candidate.reason == "packed-qkv-output-K-sbp-partial"
        and all(
            field.partial == fm.SBP.partial((0,))
            and field.axis_policies[-1].hierarchy_axes == (1,)
            for field in candidate.return_type.fields
        )
    )

    # Both plans issue the same number of local scalar MACs.  The direct plan
    # leaves three 2048-deep reductions with only 16/8/8 scalar outputs per
    # owner, while the hybrid plan exposes 256-deep reductions and 128/64/64
    # outputs.  Its regular shape-balance advantage must survive the measured
    # 4608-byte partial materialization boundary used by the combine provider.
    assert direct.operation_cost == 65_536 + 6_112
    assert hybrid.operation_cost == 65_536 + 512
    assert hybrid.operation_cost + 4_608 < direct.operation_cost
    assert hybrid.objective_kind == "heuristic"
    assert hybrid.objective_model == "flagmega.packed-qkv-local-shape-balance/v1"
    assert "local-k-n-imbalance" in hybrid.objective_evidence


def test_packed_qkv_provider_has_no_machine_geometry_or_target_dependency():
    source = inspect.getsource(PackedQKVParallelLinearCandidateProvider)
    for spelling in (
        "sm90", "nvidia", "block_k", "num_stages", "num_warps",
        "context_mesh_size", "head_mesh_size", "qwen",
    ):
        assert spelling not in source.lower()
