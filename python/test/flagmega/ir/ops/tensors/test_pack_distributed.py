# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.tensors.pack import Pack
from triton.flagmega.ir.ops.tensors.unpack import Unpack


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def test_pack_and_unpack_scale_block_cyclic_units_exactly():
    placement = fm.Placement((8, 16), "yx", "bb")
    logical = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 2048)),
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 64)),
        placement,
    )

    packed = Pack.infer_type(
        (_node("logical", logical),),
        {"lanes": (2, 8), "axes": (1, 1)},
    )
    assert packed.tensor == fm.tensor_type(
        fm.VectorType(fm.DType.BFLOAT16, (2, 8)), (1, 128)
    )
    assert packed.axis_policies == (
        fm.SBP.broadcast(),
        fm.SBP.split_block_cyclic((0,), 4),
    )

    unpacked = Unpack.infer_type(
        (_node("packed", packed),),
        {"axes": (1, 1)},
    )
    assert unpacked == logical


def test_pack_rejects_split_boundary_that_cuts_vector_lane_group():
    placement = fm.Placement((8,), "x", "b")
    logical = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 2048)),
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 4)),
        placement,
    )

    with pytest.raises(IRSchemaError, match="cannot scale axis 1 split policy"):
        Pack.infer_type(
            (_node("logical", logical),),
            {"lanes": (8,), "axes": (1,)},
        )
