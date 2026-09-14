# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import inspect
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
from triton.flagmega.passes.auto_distributed.providers import (
    PackedQKVParallelLinearCandidateProvider,
)
from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider


def _compute_cost_model():
    from triton.flagmega.passes.auto_distributed.operation_cost import DistributedOperationCostModel

    return DistributedOperationCostModel(
        block_local_read_bytes_per_cycle=1 << 40, block_local_write_bytes_per_cycle=1 << 40,
        chip_global_read_bytes_per_cycle=1 << 40, chip_global_write_bytes_per_cycle=1 << 40,
        identity="test.compute-bound/v1",
    )


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
        operation_cost_model=_compute_cost_model(),
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
    # Each of the three local projections uses the shared SIMT GEMV cost
    # contract, with a minimum 32-wide scalar output domain.
    assert candidate.operation_cost == 3 * 32 * 2048


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
        operation_cost_model=_compute_cost_model(),
    )

    replicated = next(
        candidate
        for candidate in PackedQKVParallelLinearCandidateProvider().get_candidates(context)
        if candidate.reason == "broadcast-replicated"
    )

    # Q/K/V contain 2048 + 1024 + 1024 scalar outputs, reduced over K=2048.
    scalar_mac_work = (2048 + 1024 + 1024) * 2048
    assert replicated.operation_cost == scalar_mac_work


def test_packed_qkv_cost_factors_account_for_hybrid_compute_geometry():
    module = PackedQKVModule().build()
    node = module.node_map["qkv"]
    placement = fm.Placement((8, 16), "yx", "bb")
    context = DistributedCandidateContext(
        module,
        node,
        placement,
        tuple((module.node_map[input_id].type,) for input_id in node.inputs),
        PyNttDistributedSplitCandidateProvider(block_bytes=128),
        operation_cost_model=_compute_cost_model(),
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

    # Padding is physical work; abs(K-N) is not a latency factor. Memory and
    # collective factors are tested separately with the same target model.
    assert direct.operation_cost == 3 * 32 * 2048
    assert hybrid.operation_cost == 65_536
    assert hybrid.objective_kind == "analytic"
    assert hybrid.objective_model == "test.compute-bound/v1"
    assert "op-definition-cost-factors" in hybrid.objective_evidence


def test_packed_qkv_provider_has_no_machine_geometry_or_target_dependency():
    source = inspect.getsource(PackedQKVParallelLinearCandidateProvider)
    for spelling in (
        "sm90", "nvidia", "block_k", "num_stages", "num_warps",
        "context_mesh_size", "head_mesh_size", "qwen",
    ):
        assert spelling not in source.lower()


def test_packed_qkv_cost_uses_target_throughput_instead_of_raw_macs():
    from triton.flagmega.passes.auto_distributed.operation_cost import DistributedOperationCostModel

    module = PackedQKVModule().build()
    node = module.node_map["qkv"]
    placement = fm.Placement((8, 16), "yx", "bb")
    model = DistributedOperationCostModel(
        block_local_read_bytes_per_cycle=1 << 40, block_local_write_bytes_per_cycle=1 << 40,
        chip_global_read_bytes_per_cycle=1 << 40, chip_global_write_bytes_per_cycle=1 << 40,
        identity="test.compute-bound/v1",
    )
    context = DistributedCandidateContext(
        module, node, placement, tuple((module.node_map[i].type,) for i in node.inputs),
        PyNttDistributedSplitCandidateProvider(block_bytes=128), operation_cost_model=model,
    )
    provider = PackedQKVParallelLinearCandidateProvider()
    slow = {c.id: c for c in provider.get_candidates(context)}
    fast = provider.get_candidates(replace(context, operation_cost_model=replace(model, simt_fma_per_cycle=8)))
    assert all(c.objective_kind == "analytic" for c in fast)
    assert all(c.objective_model == model.identity for c in fast)
    assert all(slow[c.id].operation_cost == c.operation_cost * 8 for c in fast)
