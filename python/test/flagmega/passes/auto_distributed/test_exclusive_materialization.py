# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Exclusive participation never makes missing peer-chip bytes resident."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    DistributedReshardRealization as Realization,
)
from triton.flagmega.passes.auto_distributed import (
    DistributedReshardRealizationContext as Context,
)
from triton.flagmega.passes.auto_distributed import (
    DistributedReshardSourceKind as Source,
)
from triton.flagmega.passes.auto_distributed import (
    DistributedReshardUsageKind as Usage,
)
from triton.flagmega.passes.auto_distributed import (
    NttDistributedReshardRealizationPolicy,
    PyNttDistributedReshardRealizationPolicy,
)

MESH = fm.Placement((4, 8, 16), "cyx", "cbb")
POLICIES = (NttDistributedReshardRealizationPolicy, PyNttDistributedReshardRealizationPolicy)


def qa_types(lanes=8):
    tensor = fm.tensor_type(fm.vector_type("bfloat16", (lanes,)) if lanes > 1 else "bfloat16",
                            (1, 1536 // lanes))
    split = fm.SBPSplit(stages=(*fm.SBP.split_contiguous((0,)).stages,
                               *fm.SBP.split_block_cyclic((1, 2), 1).stages))
    source = fm.DistributedType(tensor, (fm.SBP.broadcast(), split), MESH)
    target = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.broadcast()), MESH,
                                exclusive=fm.SBP.exclusive((1, 2)))
    return source, target


@pytest.mark.parametrize("policy_type", POLICIES)
@pytest.mark.parametrize("kind", (Source.INTERNAL, Source.FUNCTION_PARAMETER))
@pytest.mark.parametrize("lanes", (1, 8))
def test_chip_gather_with_block_exclusive_target_requires_materialization(policy_type, kind, lanes):
    source, target = qa_types(lanes)
    assert fm.boxing_crossing_chip_axes(source, target) == (0,)
    expected = Realization.UNSUPPORTED if kind is Source.FUNCTION_PARAMETER else Realization.BOXING
    assert policy_type().classify(Context(source, target, kind, Usage.INTERNAL)) is expected


@pytest.mark.parametrize("policy_type", POLICIES)
def test_publishing_chip_exclusive_value_requires_materialization(policy_type):
    _, target = qa_types()
    source = replace(target, exclusive=fm.SBP.exclusive((0,)))
    target = replace(target, exclusive=None)
    assert policy_type().classify(Context(source, target, Source.INTERNAL, Usage.INTERNAL)) is Realization.BOXING


@pytest.mark.parametrize("policy_type", POLICIES)
@pytest.mark.parametrize("publish", (False, True))
def test_block_exclusive_transition_of_a_complete_chip_value_remains_view(policy_type, publish):
    _, target = qa_types()
    source = replace(target, exclusive=None)
    if publish:
        source, target = target, source
    assert fm.boxing_crossing_chip_axes(source, target) == ()
    assert policy_type().classify(Context(source, target, Source.INTERNAL, Usage.INTERNAL)) is Realization.SHARDED_VIEW


def test_block_shard_coarsening_with_exclusive_owner_remains_chip_local_view():
    _, target = qa_types()
    source = replace(target, exclusive=None,
                     axis_policies=(fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1, 2), 1)))
    assert PyNttDistributedReshardRealizationPolicy().classify(
        Context(source, target, Source.INTERNAL, Usage.INTERNAL)) is Realization.SHARDED_VIEW


@pytest.mark.parametrize("kind", (Source.INTERNAL, Source.FUNCTION_PARAMETER))
def test_changing_chip_granularity_is_not_proven_local_by_axis_owner_alone(kind):
    source, _ = qa_types()
    source = replace(source, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 16)))
    target = replace(source, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 8)))
    assert fm.boxing_crossing_chip_axes(source, target) == (0,)
    expected = Realization.UNSUPPORTED if kind is Source.FUNCTION_PARAMETER else Realization.BOXING
    assert PyNttDistributedReshardRealizationPolicy().classify(
        Context(source, target, kind, Usage.INTERNAL)) is expected
