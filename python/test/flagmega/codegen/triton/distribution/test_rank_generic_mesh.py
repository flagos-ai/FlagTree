# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _canonical_writer_active,
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
        "fusion_lines": (),
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


def test_chip_axis_lowers_to_device_mesh_and_runtime_rank():
    mesh = _mesh_context({
        "hierarchy": (4, 8, 16),
        "name": "cyx",
        "hierarchy_levels": "cbb",
    })
    source = TritonTemplateRegistry().render(
        "kernels/elementwise/silu.py.jinja",
        {
            **mesh,
            "distributed_entry": True,
            "render_calls": (_elementwise_call(),),
        },
    )
    assert mesh["mesh_topology_repr"] == "{'device': 4, 'block': [('block_y', 8), ('block_x', 16)]}"
    assert mesh["mesh_requires_device_dptr"] is True
    assert 'tle.shard_id(FLAGMEGA_GRID_MESH, "device", device_dptr=device_dptr)' in source
    assert "shard_coord0 = ((_flagmega_device_rank // 1) % 4)" in source


def test_chip_node_mesh_records_complete_runtime_owner_extent():
    mesh = _mesh_context({
        "hierarchy": (2, 3, 8),
        "name": "cdx",
        "hierarchy_levels": "cdb",
    })

    assert mesh["mesh_device_extent"] == 2
    assert mesh["mesh_node_extent"] == 3
    assert mesh["mesh_size"] == 48
    assert mesh["mesh_requires_device_dptr"] is True


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


def test_mesh_without_block_axis_is_rejected():
    with pytest.raises(CodegenError):
        _mesh_context({"hierarchy": (2, 1), "name": "cd", "hierarchy_levels": "cd"})


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


def _owner_coordinates(expression, hierarchy):
    """Evaluate a rendered writer predicate over an entire mesh.

    Returns the coordinate tuples that satisfy it, i.e. the elected owners.
    Election asserts *how many* writers each logical shard gets, not which
    physical owner is chosen, so these helpers let the tests pin the contract
    instead of one rendering of it.
    """
    import itertools

    names = (("shard_y", "shard_x") if len(hierarchy) == 2
             else tuple(f"shard_coord{axis}" for axis in range(len(hierarchy))))
    owners = []
    for coordinate in itertools.product(*(range(n) for n in hierarchy)):
        if eval(  # noqa: S307 - rendered predicate over fixed local names
            expression, {"__builtins__": {}},
            dict(zip(names, coordinate)),
        ):
            owners.append(coordinate)
    return owners


def _writers_per_shard(expression, hierarchy, split_axes):
    from collections import Counter

    owners = _owner_coordinates(expression, hierarchy)
    return Counter(tuple(c[axis] for axis in split_axes) for c in owners)


def test_rank_three_unique_writer_keeps_each_distinct_split_shard():
    abi = _distributed_abi(split_axes=(1,))

    writers = _writers_per_shard(
        _distributed_unique_writer_active(abi), (2, 3, 5), (1,)
    )

    assert len(writers) == 3 and set(writers.values()) == {1}


def test_rank_three_unique_writer_ordinal_uses_redundant_axes_row_major():
    abi = _distributed_abi(split_axes=(1,))

    first = _owner_coordinates(
        _distributed_unique_writer_active(abi, writer_ordinal=7), (2, 3, 5)
    )
    second = _owner_coordinates(
        _distributed_unique_writer_active(abi, writer_ordinal=8), (2, 3, 5)
    )

    # Consecutive ordinals select different redundant owners of the same shards.
    assert len(first) == len(second) == 3
    assert not set(first) & set(second)


def test_rank_three_unique_writer_ordinal_wraps_redundant_owner_count():
    abi = _distributed_abi(split_axes=(1,))
    # 2 * 5 redundant owners, so ordinal 10 wraps onto ordinal 0.
    assert (
        _distributed_unique_writer_active(abi, writer_ordinal=10)
        == _distributed_unique_writer_active(abi, writer_ordinal=0)
    )


def test_canonical_split_writer_preserves_each_block_shard_on_every_chip():
    abi = _distributed_abi(hierarchy=(4, 8, 16), split_axes=(1,))
    abi["distributed_type"]["placement"]["hierarchy_levels"] = "cbb"

    owners = _owner_coordinates(_canonical_writer_active(abi), (4, 8, 16))

    # Every chip keeps one writer for each of its 8 block shards: the chip
    # axis is never elected away, because each chip owns a private instance.
    assert len({coordinate[0] for coordinate in owners}) == 4
    assert len(owners) == 4 * 8
    writers = _writers_per_shard(
        _canonical_writer_active(abi), (4, 8, 16), (0, 1)
    )
    assert set(writers.values()) == {1}


def test_canonical_broadcast_writer_reduces_block_axes_but_replicates_chip_axis():
    abi = _distributed_abi(hierarchy=(4, 8, 16))
    abi["distributed_type"]["placement"]["hierarchy_levels"] = "cbb"
    assert _canonical_writer_active(abi) == (
        "(shard_coord1 == 0) & (shard_coord2 == 0)"
    )
