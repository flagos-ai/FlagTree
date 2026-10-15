# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A partial Boxing whose reduction group is chip-local but whose source is not.

Two independent properties of a partial source decide how its leaf is lowered:
which placement axes the reduction group covers, and which axes distinguish the
source's elements.  When a chip axis carries the group, the group's members live
in peer instances and the reduction loops over peers.  When a chip axis instead
*splits* the source, every destination element's whole group lives inside one
chip - but not necessarily this one, so the leaf has to iterate its destination
and select that chip per element.

The renderer used to treat the second case as the first one missing a
precondition, and rejected it.
"""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _cross_chip_partial_source,
)
from triton.flagmega.errors import CodegenError

from .test_partial_reduce_local_abi import _abi, _prepare


_PLACEMENT = fm.Placement((4, 8, 16), "cyx", "cbb")
_TENSOR = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 8))


def _distributed_data(policies, *, partial_axes=None):
    """Serialize a real DistributedType so the ABI matches what codegen sees."""

    value = fm.DistributedType(_TENSOR, policies, _PLACEMENT)
    data = value.to_data()
    if partial_axes is not None:
        data["partial"] = {
            "kind": "partial", "axes": tuple(partial_axes), "reduce_op": "sum",
        }
    return data


def _chip_split_source():
    """A ``cbb`` source split on the chip axis, reduced over a block axis.

    This is the ABI ``decode_mla_attention_kv_a.vectorized.compute`` presents in
    the GLM-5.3-Flash ``cyx=4x8x16/cbb`` model: the head axis is split across
    chips and then block-cyclically across ``y``, while ``x`` carries the partial
    sum of the matmul's K split.
    """

    abi = _abi(
        (1, 64), local_shape=(1, 2),
        storage_kind="compact_per_owner", coordinate_space="local",
        coordinates=(
            "local_coord_0",
            "(shard_coord_1 + (local_coord_1 * 8) + (shard_coord_0 * 16))",
        ),
        partial_axes=(2,), owner_stride=16, lane_count=8,
        hierarchy=(4, 8, 16), hierarchy_levels="cbb",
    )
    abi["memory_space"] = "workspace"
    abi["storage"] = "workspace"
    abi["distributed_type"] = _distributed_data(
        (
            fm.SBP.broadcast(),
            fm.SBP.split(
                fm.SplitStage.contiguous((0,)),
                fm.SplitStage.block_cyclic((1,), 1),
            ),
        ),
        partial_axes=(2,),
    )
    return abi


def _canonical_result():
    abi = _abi(
        (1, 64), local_shape=(1, 8),
        coordinates=("local_coord_0", "(local_coord_1 + (shard_coord_1 * 8))"),
        lane_count=8, hierarchy=(4, 8, 16), hierarchy_levels="cbb",
    )
    abi["memory_space"] = "workspace"
    abi["distributed_type"] = _distributed_data(
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 8)),
    )
    return abi


def test_chip_split_source_selects_its_peer_per_element():
    leaf = _prepare(
        (_chip_split_source(),), (_canonical_result(),), ("gather_reduce_scatter",)
    )["leaves"][0]

    # The whole reduction group is inside the selected chip, so the peer loop
    # does not consume any of it.
    assert leaf["partial_owner_count"] == 16
    # Every chip is visited once and a per-element predicate keeps its elements.
    assert leaf["remote_peers"] == 4
    assert "remote_owner" in leaf
    assert "remote_source" in leaf
    assert leaf["remote_window_pool"] == "workspace"
    # The destination owns the iteration domain: this chip's own source region
    # covers only a quarter of it.
    assert leaf["capacity"] == 64


def test_chip_split_source_reduces_the_selected_chips_own_components():
    leaf = _prepare(
        (_chip_split_source(),), (_canonical_result(),), ("gather_reduce_scatter",)
    )["leaves"][0]

    # A remote read addresses the peer's instance at this pool's own offsets, so
    # the component index is the same decomposition a chip-local leaf uses: the
    # element's owner outside the group plus the group member's ordinal.
    assert "boxing_partial_member" in leaf["partial_owner"]
    assert str(leaf["owner_stride"]) in leaf["source_offset"]


def test_group_spanning_chips_still_loops_peers_over_the_group():
    """The pre-existing lowering is unchanged: here the group is the chip axis."""

    source = _chip_split_source()
    source["distributed_type"] = _distributed_data(
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 1)),
        partial_axes=(0,),
    )
    result = _canonical_result()

    leaf = _prepare((source,), (result,), ("gather_reduce_scatter",))["leaves"][0]

    # The chip axis is the reduction group, so the peer loop consumes it.
    assert leaf["remote_peers"] == 4
    assert leaf["partial_owner_count"] == 1
    # Peers carry the group, so no per-element peer predicate is emitted.
    assert "remote_owner" not in leaf


def test_a_crossing_axis_that_neither_reduces_nor_splits_is_rejected():
    """Peer selection needs the axis to be in the group or to split the source.

    ``boxing_crossing_chip_axes`` reports an axis as crossing when it is reduced,
    when it is exclusively owned, or when it splits a region the destination does
    not keep chip-local.  A partial source cannot also be exclusive, so the
    middle case is unreachable through a module; the guard is asserted directly
    so a future crossing reason cannot silently reach the peer-selection path.
    """

    source = _chip_split_source()
    # Reduce over a block axis and split over its sibling: nothing then relates
    # the chip axis to a peer.
    source["distributed_type"] = _distributed_data(
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 1)),
        partial_axes=(2,),
    )

    with pytest.raises(CodegenError, match="neither"):
        _cross_chip_partial_source(
            source, (0,), axes=(2,), routed=True,
            logical_coordinates=("0", "0"),
            member_variable="boxing_partial_member",
        )


def test_element_selected_peers_require_the_destination_domain():
    """Iterating this chip's source region would skip the peers' elements."""

    with pytest.raises(CodegenError, match="iterate its destination domain"):
        _cross_chip_partial_source(
            _chip_split_source(), (0,), axes=(2,), routed=False,
            logical_coordinates=("0", "0"),
            member_variable="boxing_partial_member",
        )
