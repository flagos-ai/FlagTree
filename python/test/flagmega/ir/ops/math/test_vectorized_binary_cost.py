# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Cost scalar counts are independent of packet width and logical ownership."""
from dataclasses import replace

import pytest
from triton.flagmega import ir as fm


def make(lanes, distributed=False, dynamic=False):
    n = 2048 if distributed else 16
    ty = fm.tensor_type(fm.vector_type("float32", (lanes,)), ("rows" if dynamic else 1, n // lanes))
    if distributed:
        ty = fm.DistributedType(ty, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 16 // lanes)),
                                fm.Placement((8, 16), "yx", "bb"))
    inputs = tuple(fm.Node(k, "builtin.var", (), ty, attrs={"name": k}) for k in ("lhs", "rhs"))
    definition = fm.get_definition("math.vectorized_binary")
    p = definition.prepare(inputs, {"binary_op": "add"})
    node = fm.Node("add", definition.op_name, tuple(n.id for n in inputs), p.result_type, p.effect, p.attrs)
    return definition, inputs, node


@pytest.mark.parametrize("lanes", [1, 4, 8])
@pytest.mark.parametrize("distributed", [False, True])
def test_each_scalar_counted_once(lanes, distributed):
    definition, inputs, node = make(lanes, distributed)
    cost = fm.get_cost(node)
    n = 2048 if distributed else 16
    assert (cost.flops, cost.bytes_read, cost.bytes_written) == (n, n * 8, n * 4)
    local = definition.cost_factors(inputs, node.attrs, node.type)
    assert (local.cpu_cycles, local.block_local_memory_load_bytes, local.block_local_memory_store_bytes) == (16, 128, 64)


@pytest.mark.parametrize("lanes", [1, 4, 8])
def test_unknown_outer_extent_is_unknown_cost(lanes):
    definition, inputs, node = make(lanes, True, True)
    cost = fm.get_cost(node)
    assert (cost.flops, cost.bytes_read, cost.bytes_written) == (None, None, None)
    assert definition.cost_factors(inputs, node.attrs, node.type) is None


def test_scalar_add_matches_packed_add_logical_work():
    ty = fm.tensor_type("float32", (1, 16))
    args = tuple(fm.Node(k, "builtin.var", (), ty, attrs={"name": k}) for k in ("lhs", "rhs"))
    op = fm.get_definition("math.add")
    p = op.prepare(args, {})
    scalar = op.cost(fm.Node("scalar", op.op_name, tuple(n.id for n in args), p.result_type, p.effect, p.attrs))
    assert scalar.flops == 16
    for lanes in (1, 4, 8):
        _, _, packed = make(lanes)
        assert fm.get_cost(packed).flops == scalar.flops
