# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A reshard of a function parameter cannot read the parameter's peer copies.

A chip-crossing Boxing addresses its source on every peer chip inside the
FlagCX symmetric window, so the source bytes must live in a pool the runtime
registers as that window.  A function parameter is an opaque pointer supplied
by the caller: the peer chip's copy of it is at an unrelated address, so no
peer-side address for it exists.

The realization policy used to answer BOXING for those types anyway.  The
boundary-layout planner trusted that answer, promoted a callee parameter to a
chip-split ABI, and emitted a chip-crossing restore Boxing from it - which only
failed much later in codegen, at ``emit_remote_storage_pointer``, with
``Remote reads require pooled storage registered as the device window, got
storage 'parameter'``.
"""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    DistributedReshardRealization,
    DistributedReshardRealizationContext,
    DistributedReshardSourceKind,
    DistributedReshardUsageKind,
    NttDistributedReshardRealizationPolicy,
    PyNttDistributedReshardRealizationPolicy,
)

# ``cbb``: axis 0 is a chip axis, axes 1 and 2 are block axes of one chip.
_PLACEMENT = fm.Placement((4, 2, 4), "cyx", "cbb")
_TENSOR = fm.tensor_type("float32", (1, 64))


def _types():
    broadcast = fm.DistributedType(
        _TENSOR, (fm.SBP.broadcast(), fm.SBP.broadcast()), _PLACEMENT)
    chip_split = fm.DistributedType(
        _TENSOR,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 16)),
        _PLACEMENT,
    )
    block_split = fm.DistributedType(
        _TENSOR,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 32)),
        _PLACEMENT,
    )
    return broadcast, chip_split, block_split


def _policies():
    return (
        PyNttDistributedReshardRealizationPolicy(),
        NttDistributedReshardRealizationPolicy(),
    )


@pytest.mark.parametrize("policy", _policies(), ids=lambda value: type(value).__name__)
def test_a_chip_crossing_reshard_of_a_parameter_is_not_realizable(policy):
    broadcast, chip_split, _ = _types()
    # The restore an all-gather would need: the source distinguishes chips, the
    # target does not, so every chip has to read all the others' components.
    assert fm.boxing_crossing_chip_axes(chip_split, broadcast) == (0,)
    assert policy.classify(DistributedReshardRealizationContext(
        chip_split,
        broadcast,
        DistributedReshardSourceKind.FUNCTION_PARAMETER,
        DistributedReshardUsageKind.INTERNAL,
    )) is DistributedReshardRealization.UNSUPPORTED


@pytest.mark.parametrize("policy", _policies(), ids=lambda value: type(value).__name__)
@pytest.mark.parametrize(
    "source_kind",
    (
        DistributedReshardSourceKind.INTERNAL,
        DistributedReshardSourceKind.CONSTANT,
    ),
)
def test_a_chip_crossing_reshard_of_addressable_storage_stays_realizable(
    policy, source_kind
):
    """Only a parameter is opaque.

    An internal value lives in the workspace pool and a constant in rdata, both
    of which the runtime can register as the device window, so the same types
    remain realizable for them.  Without this the fix would be a blanket ban on
    all-gathers rather than a storage-addressability constraint.
    """

    broadcast, chip_split, _ = _types()
    assert policy.classify(DistributedReshardRealizationContext(
        chip_split,
        broadcast,
        source_kind,
        DistributedReshardUsageKind.INTERNAL,
    )) is not DistributedReshardRealization.UNSUPPORTED


@pytest.mark.parametrize("policy", _policies(), ids=lambda value: type(value).__name__)
def test_a_block_only_reshard_of_a_parameter_stays_realizable(policy):
    """Block owners share one chip instance, so no peer is read."""

    broadcast, _, block_split = _types()
    assert fm.boxing_crossing_chip_axes(block_split, broadcast) == ()
    assert policy.classify(DistributedReshardRealizationContext(
        block_split,
        broadcast,
        DistributedReshardSourceKind.FUNCTION_PARAMETER,
        DistributedReshardUsageKind.INTERNAL,
    )) is not DistributedReshardRealization.UNSUPPORTED


def test_a_parameter_view_of_the_same_chip_crossing_types_is_still_a_view():
    """The constraint applies to the answer, not the request.

    A ``sharded_view`` reads nothing: it only renames the region a reader
    addresses.  Rejecting the types outright would also forbid views that are
    perfectly addressable, so an exclusive transition - which PyNTT realizes as
    a view - must survive even from a parameter.
    """

    policy = PyNttDistributedReshardRealizationPolicy()
    exclusive = fm.DistributedType(
        _TENSOR,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        _PLACEMENT,
        exclusive=fm.SBP.exclusive((0,)),
    )
    _, chip_split, _ = _types()
    assert fm.boxing_crossing_chip_axes(chip_split, exclusive) == (0,)
    assert policy.classify(DistributedReshardRealizationContext(
        chip_split,
        exclusive,
        DistributedReshardSourceKind.FUNCTION_PARAMETER,
        DistributedReshardUsageKind.INTERNAL,
    )) is DistributedReshardRealization.SHARDED_VIEW
