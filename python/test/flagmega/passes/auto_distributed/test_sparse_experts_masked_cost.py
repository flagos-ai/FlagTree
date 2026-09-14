# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
from itertools import product
from math import prod

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.passes.auto_distributed.operation_cost import DistributedOperationCostModel
from python.test.flagmega.sparse_experts.helpers import operand_types, operands


@pytest.mark.parametrize("definition", [SparseExpertsGateUp, SparseExpertsDown])
@pytest.mark.parametrize("layout", ["broadcast", "cyclic", "staged", "empty", "contiguous"])
def test_selected_bank_traffic_counts_active_slots_not_owner_capacity(definition, layout):
    b = fm.SBP.broadcast()
    routes = 4 if layout == "contiguous" else 9
    mesh = fm.Placement((8, 2, 2), "xyz", "bbb")
    r = {
        "broadcast": b,
        "cyclic": fm.SBP.split_block_cyclic((0,), 2),
        "staged": fm.SBP.split(fm.SplitStage.block_cyclic((0,), 2),
                               fm.SplitStage.block_cyclic((1,), 2)),
        "empty": fm.SBP.split_block_cyclic((0,), 4),
        "contiguous": fm.SBP.split_contiguous((1,), 4),
    }[layout]
    n = fm.SBP.split_block_cyclic((2,), 8)
    types = operand_types(tokens=3, hidden=20, intermediate=12, routes=routes, experts=16)
    policies = {"dispatched": (b, r, b), "activations": (b, r, b), "router_expert_ids": (b, r),
                "gate_weight": (b, n, b), "up_weight": (b, n, b), "down_weight": (b, n, b)}
    types = {p.name: fm.DistributedType(types[p.name], policies.get(p.name, (b, b)), mesh)
             for p in definition.input_parameters}
    inputs = operands(definition, types)
    prepared = definition.prepare(inputs, {})
    factors = definition.cost_factors(inputs, prepared.attrs, prepared.result_type)
    matrix_type = types["gate_weight" if definition is SparseExpertsGateUp else "down_weight"]
    projections = 2 if definition is SparseExpertsGateUp else 1
    expected = 0
    for owner in product(*(range(n) for n in mesh.hierarchy)):
        slots = prod(d.fixed_value for d in fm.local_shard_descriptor(types["router_expert_ids"], owner).active_shape)
        matrix = prod(d.fixed_value for d in fm.local_shard_descriptor(matrix_type, owner).active_shape[1:])
        expected += projections * slots * (matrix * matrix_type.tensor.dtype.itemsize + 8)
    memory_only = replace(factors, simt_fma_operations=0, elementwise_operations=0,
                          block_local_memory_load_bytes=0, block_local_memory_store_bytes=0)
    assert DistributedOperationCostModel().get_latency(memory_only, prepared.result_type) == expected
    # Masked arithmetic still runs over the maximum local domain.
    local_ids = fm.local_tensor_type(types["router_expert_ids"])
    local_matrix = fm.local_tensor_type(matrix_type)
    assert factors.simt_fma_operations == projections * prod(d.fixed_value for d in local_ids.shape) * prod(
        d.fixed_value for d in local_matrix.shape[1:])


def test_chip_aggregate_factors_are_not_multiplied_by_owner_count():
    model = DistributedOperationCostModel(chip_global_latency_cycles=3)
    result = fm.DistributedType(fm.tensor_type("float32", (1,)), (fm.SBP.broadcast(),),
                                fm.Placement((8, 16), "yx", "bb"))
    factors = fm.OpCostFactors(chip_global_memory_load_bytes=2, chip_global_memory_store_bytes=1,
                               chip_aggregate_memory_load_bytes=17, chip_aggregate_memory_store_bytes=5)
    assert model.get_latency(factors, result) == 3 * 128 + 17 + 5 + 3
    assert model.get_latency(factors, result.tensor) == 3 + 17 + 5 + 3


def test_down_split_k_fusion_preserves_aggregate_traffic_and_owner_reduction():
    from triton.flagmega.ir.ops.ntt.sparse_experts import SparseExpertsDownCombine
    b = fm.SBP.broadcast()
    t, r, n, k = (fm.SBP.split_block_cyclic((axis,), 2) for axis in range(4))
    mesh = fm.Placement((2, 2, 2, 2), "abcd", "bbbb")
    types = operand_types(tokens=3, routes=3, hidden=20, intermediate=12)
    policies = {"activations": (t, r, k), "router_expert_ids": (t, r),
                "router_expert_weights": (t, r), "down_weight": (b, n, k)}
    types = {p.name: fm.DistributedType(types[p.name], policies.get(p.name, (b, b)), mesh)
             for p in SparseExpertsDownCombine.input_parameters}
    inputs = operands(SparseExpertsDownCombine, types)
    down = SparseExpertsDown.prepare(inputs[:5], {})
    fused = SparseExpertsDownCombine.prepare(inputs, {})
    separate_cost = SparseExpertsDown.cost_factors(inputs[:5], down.attrs, down.result_type)
    fused_cost = SparseExpertsDownCombine.cost_factors(inputs, fused.attrs, fused.result_type)
    assert fused.result_type.partial.axes == (1, 3)
    assert fused_cost.chip_aggregate_memory_load_bytes == separate_cost.chip_aggregate_memory_load_bytes
    assert fused_cost.chip_aggregate_memory_load_bytes == 3 * 3 * 20 * 12 * 2 + 3 * 3 * 4 * 8
