# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError


PLACEMENT = fm.Placement((4, 8), "yx", "bb")


def _reshape(source_type: fm.IRType, shape: tuple[int, ...]) -> fm.IRType:
    source = fm.Node("source", "builtin.var", (), source_type, attrs={"name": "source"})
    return fm.get_definition("tensors.reshape").infer_type((source,), {"shape": shape})


def test_split_flat_axis_maps_to_the_first_non_unit_expanded_axis():
    source = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 2048)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 128)),
        PLACEMENT,
    )

    result = _reshape(source, (1, 16, 128))

    assert result == fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 16, 128)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 1), fm.SBP.broadcast()),
        PLACEMENT,
    )


def test_unrelated_split_and_inserted_unit_axes_are_preserved():
    source = fm.DistributedType(
        fm.tensor_type("float32", (1, 48, 1024)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 12), fm.SBP.broadcast()),
        PLACEMENT,
    )

    result = _reshape(source, (1, 48, 1, 64, 16))

    assert result == fm.DistributedType(
        fm.tensor_type("float32", (1, 48, 1, 64, 16)),
        (
            fm.SBP.broadcast(),
            fm.SBP.split_contiguous((0,), 12),
            fm.SBP.broadcast(),
            fm.SBP.broadcast(),
            fm.SBP.broadcast(),
        ),
        PLACEMENT,
    )


def test_split_unit_that_cuts_an_expanded_axis_is_rejected():
    source = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 2048)),
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 64)),
        PLACEMENT,
    )

    with pytest.raises(IRSchemaError, match="cannot preserve distributed layout"):
        _reshape(source, (1, 16, 128))


def test_first_split_axis_can_be_preserved_when_axes_are_flattened():
    source = fm.DistributedType(
        fm.tensor_type("float32", (1, 48, 64, 16)),
        (
            fm.SBP.broadcast(),
            fm.SBP.broadcast(),
            fm.SBP.split_contiguous((0,), 8),
            fm.SBP.broadcast(),
        ),
        PLACEMENT,
    )

    result = _reshape(source, (1, 48, 1024))

    assert result == fm.DistributedType(
        fm.tensor_type("float32", (1, 48, 1024)),
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 128)),
        PLACEMENT,
    )


def test_trailing_split_axis_cannot_be_flattened_after_a_non_unit_axis():
    source = fm.DistributedType(
        fm.tensor_type("float32", (1, 48, 64, 16)),
        (
            fm.SBP.broadcast(),
            fm.SBP.broadcast(),
            fm.SBP.broadcast(),
            fm.SBP.split_contiguous((0,), 2),
        ),
        PLACEMENT,
    )

    with pytest.raises(IRSchemaError, match="cannot preserve distributed layout"):
        _reshape(source, (1, 48, 1024))


def test_ordered_block_cyclic_splits_are_composed_when_axes_are_flattened():
    source = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (16, 16, 1)),
        (
            fm.SBP.split_block_cyclic((1,), 2),
            fm.SBP.split_block_cyclic((0,), 4),
            fm.SBP.broadcast(),
        ),
        PLACEMENT,
    )

    result = _reshape(source, (1, 256))

    assert result == fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 256)),
        (
            fm.SBP.broadcast(),
            fm.SBP.split(
                fm.SplitStage.block_cyclic((1,), 32),
                fm.SplitStage.block_cyclic((0,), 4),
            ),
        ),
        PLACEMENT,
    )


def test_unmappable_reshape_rejects_a_non_broadcast_layout():
    source = fm.DistributedType(
        fm.tensor_type("float32", (2, 30)),
        (fm.SBP.split_contiguous((0,), 1), fm.SBP.broadcast()),
        fm.Placement((2,), "b", "b"),
    )

    with pytest.raises(IRSchemaError, match="cannot preserve distributed layout"):
        _reshape(source, (3, 20))


def test_unmappable_reshape_still_accepts_a_fully_broadcast_value():
    placement = fm.Placement((2,), "b", "b")
    source = fm.DistributedType(
        fm.tensor_type("float32", (2, 30)),
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )

    result = _reshape(source, (3, 20))

    assert result == fm.DistributedType(
        fm.tensor_type("float32", (3, 20)),
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
