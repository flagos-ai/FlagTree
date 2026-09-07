# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _distributed_unique_writer_active,
    _partial_group_owner_expression,
    _partial_group_owner_expression_for_axes,
)
from triton.flagmega.codegen.triton.physical_access import (
    emit_local_scalar_offset,
)
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.codegen.triton.tir_package import _local_mesh, _mesh_context
from triton.flagmega.errors import CodegenError


def _distributed_abi(
    *,
    hierarchy=(2, 3, 5),
    partial_axes=None,
    split_axes=(),
):
    return {
        "storage": "workspace",
        "storage_kind": "canonical_global",
        "scalar_storage_strides": (1,),
        "coordinate_space": "canonical_global",
        "logical_coordinate_expressions": (
            "local_coord_0 + shard_coord_2 * 6",
        ),
        "active_shape_expressions": ("6",),
        "distributed_type": {
            "placement": {"hierarchy": hierarchy},
            "axis_policies": ({
                "kind": "split",
                "stages": ({"hierarchy_axes": tuple(split_axes)},),
            },) if split_axes else ({"kind": "broadcast"},),
            "partial": (
                None
                if partial_axes is None
                else {"axes": tuple(partial_axes), "reduce_op": "sum"}
            ),
        },
    }


def _elementwise_call():
    return {
        "symbol": "_flagmega_rank_generic",
        "signature": "value, result",
        "execution_kind": "local_shard",
        "internal_grid_barriers": 0,
        "family": "elementwise",
        "variant": "silu",
        "local_capacity": 1,
        "tile": 1,
        "active": "True",
        "lhs": "value",
        "lhs_offset": "0",
        "result": "result",
        "result_offset": "0",
        "output_type": "tl.float32",
    }


def test_rank_three_mesh_uses_row_major_coordinates_without_2d_aliases():
    mesh = _mesh_context({
        "hierarchy": (2, 3, 5),
        "name": "abc",
        "hierarchy_levels": "bbb",
    })

    source = TritonTemplateRegistry().render(
        "kernels/elementwise/silu.py.jinja",
        {
            **mesh,
            "distributed_entry": True,
            "render_calls": (_elementwise_call(),),
        },
    )

    assert mesh["mesh_rank"] == 3
    assert mesh["mesh_size"] == 30
    assert mesh["mesh_axes_repr"] == "[('block_a', 2), ('block_b', 3), ('block_c', 5)]"
    assert "shard_coord0 =" in source
    assert "shard_coord1 =" in source
    assert "shard_coord2 =" in source
    assert "((shard_coord0 * 3 + shard_coord1) * 5 + shard_coord2)" in source
    assert "shard_y =" not in source
    assert "shard_x =" not in source


def test_unit_non_block_axes_are_zero_and_excluded_from_physical_mesh():
    mesh = _mesh_context({
        "hierarchy": (1, 1, 4),
        "name": "cdb",
        "hierarchy_levels": "cdb",
    })
    source = TritonTemplateRegistry().render(
        "kernels/elementwise/silu.py.jinja",
        {
            **mesh,
            "distributed_entry": True,
            "render_calls": (_elementwise_call(),),
        },
    )

    assert mesh["mesh_axes_repr"] == "[('block_b', 4)]"
    assert "shard_coord0 = tl.full((), 0, tl.int64)" in source
    assert "shard_coord1 = tl.full((), 0, tl.int64)" in source
    assert "tle.shard_id(FLAGMEGA_GRID_MESH, 'block_b')" in source


def test_ordinary_entry_retains_logical_mesh_coordinates_as_zero():
    mesh = _local_mesh({
        "hierarchy": (8, 16),
        "name": "yx",
        "hierarchy_levels": "bb",
    })
    source = TritonTemplateRegistry().render(
        "kernels/elementwise/silu.py.jinja",
        {
            **mesh,
            "distributed_entry": False,
            "render_calls": (_elementwise_call(),),
        },
    )

    assert mesh["grid_mesh"] is None
    assert mesh["mesh_size"] == 1
    assert "shard_coord0 = tl.full((), 0, tl.int64)" in source
    assert "shard_coord1 = tl.full((), 0, tl.int64)" in source
    assert "shard_y = shard_coord0" in source
    assert "shard_x = shard_coord1" in source


@pytest.mark.parametrize(
    "placement",
    (
        {"hierarchy": (2, 4), "name": "ab", "hierarchy_levels": "cb"},
        {"hierarchy": (1, 1), "name": "ab", "hierarchy_levels": "cd"},
    ),
)
def test_unmaterializable_hierarchical_mesh_is_rejected(placement):
    with pytest.raises(CodegenError):
        _mesh_context(placement)


def test_rank_three_physical_access_binds_every_placement_coordinate():
    offset = emit_local_scalar_offset(
        _distributed_abi(),
        ("lane",),
    )

    assert "shard_coord2" in offset
    assert "shard_y" not in offset
    assert "shard_x" not in offset
    assert "local_coord_" not in offset


def test_rank_three_partial_owner_linearization_handles_non_adjacent_axes():
    abi = _distributed_abi(partial_axes=(0, 2))
    expression = _partial_group_owner_expression_for_axes(
        abi, (0, 2), "member"
    )

    for member in range(10):
        for shard_coord1 in range(3):
            owner = eval(
                expression,
                {"__builtins__": {}},
                {"member": member, "shard_coord1": shard_coord1},
            )
            assert owner == (member // 5) * 15 + shard_coord1 * 5 + member % 5


def test_rank_three_single_partial_axis_preserves_other_owner_coordinates():
    abi = _distributed_abi(partial_axes=(2,))
    expression = _partial_group_owner_expression(abi, "member")

    owner = eval(
        expression,
        {"__builtins__": {}},
        {"member": 4, "shard_coord0": 1, "shard_coord1": 2},
    )
    assert owner == 29


def test_rank_three_unique_writer_keeps_each_distinct_split_shard():
    abi = _distributed_abi(split_axes=(1,))

    assert _distributed_unique_writer_active(abi) == (
        "(shard_coord0 == 0) & (shard_coord2 == 0)"
    )


def test_rank_three_unique_writer_ordinal_uses_redundant_axes_row_major():
    abi = _distributed_abi(split_axes=(1,))

    assert _distributed_unique_writer_active(abi, writer_ordinal=7) == (
        "(shard_coord0 == 1) & (shard_coord2 == 2)"
    )


def test_rank_three_unique_writer_ordinal_wraps_redundant_owner_count():
    abi = _distributed_abi(split_axes=(1,))

    assert _distributed_unique_writer_active(abi, writer_ordinal=10) == (
        "(shard_coord0 == 0) & (shard_coord2 == 0)"
    )
