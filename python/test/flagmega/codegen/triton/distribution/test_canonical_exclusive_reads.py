# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A global E writer does not restrict readers of its canonical allocation."""

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import _exclusive_participation_active


def _binding(*, canonical=True, levels="bb", axes=(0, 1), owners=None):
    return {"buffers": [{"abi": {
        "storage_kind": "canonical_global" if canonical else "compact_local",
        "coordinate_space": "canonical_global" if canonical else "local",
        "distributed_type": {
            "placement": {"hierarchy": (4,) * len(levels), "hierarchy_levels": levels},
            "exclusive": {"axes": axes, "owner_coordinates": owners},
        },
    }}]}


@pytest.mark.parametrize("barriers", [0, 1])
def test_canonical_input_retains_every_block_consumer(barriers):
    # A broadcast global source feeds a split vocabulary result. Its writer
    # is CTA zero, but every CTA must produce its own output shard.
    raw = {"inputs": [_binding()], "outputs": [],
           "facts": {"internal_grid_barriers": barriers}}
    assert _exclusive_participation_active(raw) == "True"


@pytest.mark.parametrize("section", ["outputs", "workspaces"])
def test_canonical_writes_still_restrict_participation(section):
    raw = {"inputs": [_binding()], section: [_binding(owners=(1, 2))]}
    assert _exclusive_participation_active(raw) == "(shard_y == 1) & (shard_x == 2)"


def test_private_input_still_requires_its_owner():
    assert _exclusive_participation_active({"inputs": [_binding(canonical=False)]}) == (
        "(shard_y == 0) & (shard_x == 0)")


@pytest.mark.parametrize("levels,expected", [
    ("cbb", "(shard_coord0 == 2)"),
    ("bdb", "(shard_coord1 == 1)"),
])
def test_canonical_input_preserves_chip_and_node_ownership(levels, expected):
    raw = {"inputs": [_binding(levels=levels, axes=(0, 1, 2), owners=(2, 1, 3))]}
    assert _exclusive_participation_active(raw) == expected
