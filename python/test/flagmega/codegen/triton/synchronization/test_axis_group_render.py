# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.tir_package import (
    _attach_grid_barrier_axis_groups,
    render_tir_package,
)
from triton.flagmega.errors import CodegenError


def _mesh():
    return {
        "grid_mesh": {
            "hierarchy": [8, 16],
            "name": "yx",
            "hierarchy_levels": "bb",
        },
        "mesh_hierarchy": (8, 16),
        "mesh_axis_names": ("block_y", "block_x"),
        "mesh_axes": (
            {"placement_axis": 0, "name": "block_y", "size": 8, "level": "b"},
            {"placement_axis": 1, "name": "block_x", "size": 16, "level": "b"},
        ),
        "mesh_axes_repr": "[('block_y', 8), ('block_x', 16)]",
    }


def test_axis_group_descriptor_and_barrier_are_rendered_from_tir_event():
    events = [{
        "kind": "tir.kernel_call",
        "call": "consumer",
        "family": "unit",
        "variant": "unit",
        "execution_kind": "local_shard",
        "barrier_before": True,
        "barrier_scope": "grid",
        "barrier_axis_group_axes": (0,),
        "symbol": "_unit",
        "arguments": "",
    }]
    groups = _attach_grid_barrier_axis_groups(events, None, _mesh())
    descriptor = {
        **_mesh(),
        "symbol": "entry",
        "signature": "",
        "signature_arguments": (),
        "kernel_templates": (),
        "entry_events": events,
        "pipeline_schedule": None,
        "use_tle": True,
        "distributed_entry": True,
        "grid_barrier_axis_groups": groups,
        "shared_silu": False,
    }

    source = render_tir_package(descriptor, "unit")

    assert groups == ({
        "key": "0x8",
        "axis_names_repr": "('block_y',)",
        "shape_repr": "(8,)",
        "axes": (0,),
        "shape": (8,),
    },)
    assert "FLAGMEGA_GRID_AXIS_GROUP_0x8 = tl.constexpr(" in source
    assert ".axis_group(\n        ('block_y',),\n        group_shape=(8,)," in source
    assert "tle.distributed_barrier(FLAGMEGA_GRID_AXIS_GROUP_0x8)" in source


def test_axis_group_rejects_non_block_placement_axes():
    events = [{
        "kind": "barrier",
        "scope": "grid",
        "axis_group_axes": (2,),
    }]

    with pytest.raises(CodegenError, match="not physical block axes"):
        _attach_grid_barrier_axis_groups(events, None, _mesh())


def test_all_block_axes_use_canonical_full_mesh_barrier():
    events = [{
        "kind": "barrier",
        "scope": "grid",
        "axis_group_axes": (0, 1),
    }]

    groups = _attach_grid_barrier_axis_groups(events, None, _mesh())

    assert groups == ()
    assert events[0]["axis_group_axes"] == ()
