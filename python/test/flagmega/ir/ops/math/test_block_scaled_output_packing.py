# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Activation packing is independent of FP8 weight packing and SBP units."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError


def _call(packed, *, lanes=(), dtype="bfloat16", policies=None):
    kinds = [fm.tensor_type(dtype, (1, 1024)),
             fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), (1024, 32)) if packed
             else fm.tensor_type("float8_e4m3fn", (1024, 1024)),
             fm.tensor_type("float32", (8, 8))]
    if policies:
        placement = fm.Placement((8,), "x", "b")
        kinds = [fm.DistributedType(kind, policy, placement) for kind, policy in zip(kinds, policies)]
    inputs = tuple(fm.Node(str(index), "builtin.var", (), kind, attrs={"name": str(index)})
                   for index, kind in enumerate(kinds))
    attrs = {"weight_block_n": 128, "weight_block_k": 128, "output_lanes": lanes}
    if packed:
        attrs.update(k_pack=2, k_vector=16)
    definition = fm.get_definition("math.packed_block_scaled_matmul" if packed else "math.block_scaled_matmul")
    return definition.prepare(inputs, attrs)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("lanes", [(), (8,), (2, 4), (1,)])
def test_fp8_output_type_and_old_default_are_preserved(packed, dtype, lanes):
    result = _call(packed, lanes=lanes, dtype=dtype)
    expected = fm.tensor_type(fm.vector_type(dtype, lanes) if lanes else dtype, (1, 1024 // (8 if lanes in {(8,), (2, 4)} else 1)))
    assert result.result_type == expected
    assert result.attrs.get("output_lanes", ()) == lanes
    if not lanes:
        assert "output_lanes" not in result.attrs


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("lanes", [None, 8, (0,), (True,), (2.5,)])
def test_fp8_output_lanes_are_validated(packed, lanes):
    with pytest.raises(IRSchemaError, match="output_lanes.*positive integers"):
        _call(packed, lanes=lanes)


@pytest.mark.parametrize("packed", [False, True])
def test_fp8_output_n_must_divide_lane_product(packed):
    with pytest.raises(IRSchemaError, match="N must be divisible"):
        _call(packed, lanes=(3,))


@pytest.mark.parametrize("packed", [False, True])
def test_fp8_output_split_units_are_scaled_from_weight_rows_to_vector_groups(packed):
    broadcast = fm.SBP.broadcast()
    scalar_split = fm.SBP.split_block_cyclic((0,), 64)
    result = _call(packed, lanes=(2, 4), policies=[(broadcast, broadcast),
                                                 (scalar_split, broadcast), (broadcast, broadcast)])
    assert result.result_type.axis_policies == (broadcast, fm.SBP.split_block_cyclic((0,), 8))
    assert result.result_type.tensor.dtype == fm.vector_type("bfloat16", (2, 4))


@pytest.mark.parametrize("packed", [False, True])
def test_fp8_output_split_rejects_a_boundary_inside_a_vector(packed):
    broadcast = fm.SBP.broadcast()
    scalar_split = fm.SBP.split_block_cyclic((0,), 4)
    with pytest.raises(IRSchemaError, match="N split must align"):
        _call(packed, lanes=(8,), policies=[(broadcast, broadcast),
                                          (scalar_split, broadcast), (broadcast, broadcast)])


@pytest.mark.parametrize("packed", [False, True])
def test_fp8_packed_output_keeps_k_partial_reduction(packed):
    broadcast = fm.SBP.broadcast()
    value_split = fm.SBP.split_contiguous((0,))
    result = _call(packed, lanes=(8,), policies=[(broadcast, value_split), (broadcast, value_split),
                                              (broadcast, broadcast)])
    assert result.result_type.partial == fm.SBP.partial((0,))
    assert result.result_type.tensor == fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 128))
