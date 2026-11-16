# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Owner election on redundant placement axes must spread, not collapse.

``_distributed_unique_writer_active`` elects one physical writer per logical
shard by fixing a coordinate on every axis the value does not split.  When the
elected coordinate is a *constant*, two values that split **different** axes of
the same mesh receive owner sets that both contain the mesh's low-coordinate
corner, so their owners intersect.  Every such intersection concentrates
several roles' work onto one block while other blocks of the same launch own
nothing, which lengthens that block's arrival at the next rendezvous.

The election is still one writer per shard in both cases -- this is a work
placement contract, not a store-ownership contract.  The tests below pin the
placement property (disjointness across values, and a stable owner count).
"""

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _distributed_unique_writer_active,
)


HIERARCHY = (8, 16)


def _abi(split_axis, *, hierarchy=HIERARCHY):
    """One canonical value split on ``split_axis`` of a 2-D block mesh."""
    return {
        "storage": "workspace",
        "storage_kind": "canonical_global",
        "memory_sharing_scope": "chip",
        "distributed_type": {
            "placement": {
                "hierarchy": hierarchy,
                "hierarchy_levels": "b" * len(hierarchy),
            },
            "axis_policies": (
                {"kind": "split", "stages": ({"hierarchy_axes": (split_axis,)},)},
            ),
        },
    }


def _owners(expression, hierarchy=HIERARCHY):
    """Evaluate a rendered predicate over the whole mesh.

    Returns the set of flat block indices whose coordinates satisfy it, which
    is exactly the set of blocks that execute the elected role.
    """
    rows, columns = hierarchy
    owners = set()
    for shard_y in range(rows):
        for shard_x in range(columns):
            if eval(  # noqa: S307 - rendered predicate, fixed local names
                expression,
                {"__builtins__": {}},
                {"shard_y": shard_y, "shard_x": shard_x,
                 "shard_index": shard_y * columns + shard_x},
            ):
                owners.add(shard_y * columns + shard_x)
    return owners


def test_split_x_value_keeps_one_owner_per_shard():
    """One writer per logical shard: 16 owners for a 16-wide split."""
    owners = _owners(_distributed_unique_writer_active(_abi(1)))

    assert len(owners) == HIERARCHY[1]


def test_split_y_value_keeps_one_owner_per_shard():
    owners = _owners(_distributed_unique_writer_active(_abi(0)))

    assert len(owners) == HIERARCHY[0]


def test_values_splitting_different_axes_get_disjoint_owners():
    """The defect: a constant redundant coordinate collapses owner sets.

    A value split on ``block_x`` is free on ``block_y`` and a value split on
    ``block_y`` is free on ``block_x``.  Electing a constant on the free axis
    puts the first on row 0 and the second on column 0, which intersect at the
    corner block.  Electing from the value's own split coordinate does not.
    """
    split_x_owners = _owners(_distributed_unique_writer_active(_abi(1)))
    split_y_owners = _owners(_distributed_unique_writer_active(_abi(0)))

    assert not (split_x_owners & split_y_owners), (
        "values split on different axes share "
        f"{sorted(split_x_owners & split_y_owners)}"
    )


def test_distinct_writer_ordinals_stay_disjoint_on_the_same_split_axis():
    """Two values on the same split axis must not collide either.

    ``writer_ordinal`` is how one call separates two roles that split the same
    axis (the K and V cache writers).  Both roles' owner sets must remain
    disjoint from each other after any election change.
    """
    first = _owners(_distributed_unique_writer_active(_abi(0), writer_ordinal=0))
    second = _owners(_distributed_unique_writer_active(_abi(0), writer_ordinal=1))

    assert len(first) == len(second) == HIERARCHY[0]
    assert not (first & second)


def test_three_roles_of_one_call_cover_distinct_blocks():
    """The measured qkv_rope shape: Q on block_x, K and V on block_y.

    33 role units today land on 30 blocks.  Each role keeps one owner per
    shard, so a correct election places 16 + 8 + 8 = 32 units on 32 distinct
    blocks of the 128-block mesh.
    """
    q = _owners(_distributed_unique_writer_active(_abi(1)))
    k = _owners(_distributed_unique_writer_active(_abi(0), writer_ordinal=0))
    v = _owners(_distributed_unique_writer_active(_abi(0), writer_ordinal=1))

    assert len(q) + len(k) + len(v) == 32
    assert len(q | k | v) == 32, (
        "roles overlap: "
        f"Q&K={sorted(q & k)} Q&V={sorted(q & v)} K&V={sorted(k & v)}"
    )


def test_fully_split_value_elects_every_block_unchanged():
    """A value that splits every axis has no redundant axis to elect on."""
    abi = _abi(0)
    abi["distributed_type"]["axis_policies"] = (
        {"kind": "split", "stages": ({"hierarchy_axes": (0, 1)},)},
    )

    assert _distributed_unique_writer_active(abi) == "True"


def test_fully_replicated_value_elects_exactly_one_block():
    """A value that splits nothing keeps a single owner for the whole mesh."""
    abi = _abi(0)
    abi["distributed_type"]["axis_policies"] = ({"kind": "broadcast"},)

    assert len(_owners(_distributed_unique_writer_active(abi))) == 1


def test_rejects_negative_writer_ordinal():
    from triton.flagmega.errors import CodegenError

    with pytest.raises(CodegenError):
        _distributed_unique_writer_active(_abi(0), writer_ordinal=-1)
