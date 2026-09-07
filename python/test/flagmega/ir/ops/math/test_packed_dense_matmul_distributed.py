# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.math.packed_dense_matmul import PackedDenseMatMul


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def _types(*, reduction_axes, output_axes=()):
    placement = fm.Placement((8, 16), "yx", "bb")
    lhs = fm.DistributedType(
        fm.tensor_type("bfloat16", [1, 6144]),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous(reduction_axes)),
        placement,
    )
    weight_policies = [fm.SBP.broadcast() for _ in range(4)]
    weight_policies[0] = fm.SBP.split_contiguous(reduction_axes)
    if output_axes:
        weight_policies[1] = fm.SBP.split_contiguous(output_axes)
    weight = fm.DistributedType(
        fm.tensor_type("bfloat16", [384, 256, 2, 64]),
        tuple(weight_policies),
        placement,
    )
    return lhs, weight


def test_packed_k_major_aligned_k_splits_produce_sum_partial():
    lhs, weight = _types(reduction_axes=(0, 1))

    result = PackedDenseMatMul.infer_type(
        (_node("lhs", lhs), _node("weight", weight)),
        {"packed_layout": "k_major_n8_k16", "logical_n": None},
    )

    assert result.tensor == fm.tensor_type("bfloat16", [1, 2048])
    assert result.axis_policies == (fm.SBP.broadcast(), fm.SBP.broadcast())
    assert result.partial == fm.SBP.partial((0, 1))


def test_packed_k_major_supports_disjoint_output_and_reduction_mesh_axes():
    lhs, weight = _types(reduction_axes=(0,), output_axes=(1,))

    result = PackedDenseMatMul.infer_type(
        (_node("lhs", lhs), _node("weight", weight)),
        {"packed_layout": "k_major_n8_k16", "logical_n": None},
    )

    assert result.axis_policies == (
        fm.SBP.broadcast(), fm.SBP.split_contiguous((1,)))
    assert result.partial == fm.SBP.partial((0,))


def test_distributed_type_rejects_one_mesh_axis_assigned_to_k_and_n():
    with pytest.raises(IRSchemaError, match="not distributable"):
        _types(reduction_axes=(0,), output_axes=(0,))


def test_packed_k_major_scales_physical_k_and_n_split_units():
    placement = fm.Placement((8, 16), "yx", "bb")
    lhs = fm.DistributedType(
        fm.tensor_type("bfloat16", [1, 6144]),
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 64)),
        placement,
    )
    weight = fm.DistributedType(
        fm.tensor_type("bfloat16", [384, 256, 2, 64]),
        (
            fm.SBP.split_block_cyclic((0,), 4),
            fm.SBP.split_block_cyclic((1,), 8),
            fm.SBP.broadcast(),
            fm.SBP.broadcast(),
        ),
        placement,
    )

    result = PackedDenseMatMul.infer_type(
        (_node("lhs", lhs), _node("weight", weight)),
        {"packed_layout": "k_major_n8_k16", "logical_n": None},
    )

    assert result.axis_policies == (
        fm.SBP.broadcast(),
        fm.SBP.split_block_cyclic((1,), 64),
    )
    assert result.partial == fm.SBP.partial((0,))
