# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    CanonicalStorageReshardRealizationPolicy,
    DistributedReshardRealization,
    DistributedReshardRealizationContext,
    DistributedReshardSourceKind,
    DistributedReshardUsageKind,
    NttDistributedReshardRealizationPolicy,
    PyNttDistributedReshardRealizationPolicy,
)


def test_pyntt_policy_distinguishes_alias_copy_and_unsupported():
    policy = PyNttDistributedReshardRealizationPolicy()
    tensor = fm.tensor_type("bfloat16", [1, 128])
    other = fm.tensor_type("bfloat16", [1, 64])
    placement = fm.Placement((8,), "b", "b")
    split = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 16)),
        placement,
    )
    broadcast = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )

    assert policy.uses_sharded_views_for_constants()
    assert policy.classify(DistributedReshardRealizationContext(
        tensor,
        split,
        DistributedReshardSourceKind.CONSTANT,
        DistributedReshardUsageKind.INTERNAL,
    )) == DistributedReshardRealization.SHARDED_VIEW
    assert policy.classify(DistributedReshardRealizationContext(
        broadcast,
        split,
        DistributedReshardSourceKind.CONSTANT,
        DistributedReshardUsageKind.FUNCTION_BOUNDARY,
    )) == DistributedReshardRealization.SHARDED_VIEW
    assert policy.classify(DistributedReshardRealizationContext(
        broadcast,
        split,
        DistributedReshardSourceKind.INTERNAL,
        DistributedReshardUsageKind.INTERNAL,
    )) == DistributedReshardRealization.SHARDED_VIEW
    assert policy.classify(DistributedReshardRealizationContext(
        split,
        broadcast,
        DistributedReshardSourceKind.INTERNAL,
        DistributedReshardUsageKind.INTERNAL,
    )) == DistributedReshardRealization.SHARDED_VIEW
    assert policy.classify(DistributedReshardRealizationContext(
        split,
        tensor,
        DistributedReshardSourceKind.INTERNAL,
        DistributedReshardUsageKind.PROGRAM_OUTPUT,
    )) == DistributedReshardRealization.BOXING
    assert policy.classify(DistributedReshardRealizationContext(
        other,
        split,
        DistributedReshardSourceKind.INTERNAL,
        DistributedReshardUsageKind.INTERNAL,
    )) == DistributedReshardRealization.UNSUPPORTED


def test_ntt_base_policy_only_aliases_constants_with_unified_storage():
    tensor = fm.tensor_type("bfloat16", [4, 128])
    placement = fm.Placement((8,), "b", "b")
    split = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 16)),
        placement,
    )
    context = DistributedReshardRealizationContext(
        tensor,
        split,
        DistributedReshardSourceKind.CONSTANT,
        DistributedReshardUsageKind.INTERNAL,
    )

    assert (
        NttDistributedReshardRealizationPolicy().classify(context)
        == DistributedReshardRealization.BOXING
    )
    assert (
        NttDistributedReshardRealizationPolicy(
            unified_shared_storage=True
        ).classify(context)
        == DistributedReshardRealization.SHARDED_VIEW
    )


def test_pyntt_canonical_chip_view_can_reassign_physical_block_axes():
    tensor = fm.tensor_type("bfloat16", [128, 128])
    placement = fm.Placement((8, 16), "yx", "bb")
    source = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 1)),
        placement,
    )
    target = fm.DistributedType(
        tensor,
        (
            fm.SBP.split_block_cyclic((0,), 1),
            fm.SBP.split_block_cyclic((1,), 1),
        ),
        placement,
    )

    result = PyNttDistributedReshardRealizationPolicy().classify(
        DistributedReshardRealizationContext(
            source,
            target,
            DistributedReshardSourceKind.INTERNAL,
            DistributedReshardUsageKind.INTERNAL,
        )
    )

    assert result == DistributedReshardRealization.SHARDED_VIEW


def test_pyntt_canonical_chip_view_preserves_non_block_axis_owner():
    tensor = fm.tensor_type("bfloat16", [128, 128])
    placement = fm.Placement((8, 16), "dc", "bd")
    source = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 1)),
        placement,
    )
    target = fm.DistributedType(
        tensor,
        (
            fm.SBP.split_block_cyclic((1,), 1),
            fm.SBP.split_block_cyclic((0,), 1),
        ),
        placement,
    )

    result = PyNttDistributedReshardRealizationPolicy().classify(
        DistributedReshardRealizationContext(
            source,
            target,
            DistributedReshardSourceKind.INTERNAL,
            DistributedReshardUsageKind.INTERNAL,
        )
    )

    assert result == DistributedReshardRealization.BOXING


def test_pyntt_does_not_promote_function_parameter_to_canonical_chip_storage():
    tensor = fm.tensor_type("bfloat16", [128, 128])
    placement = fm.Placement((8, 16), "yx", "bb")
    source = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 1)),
        placement,
    )
    target = fm.DistributedType(
        tensor,
        (
            fm.SBP.split_block_cyclic((0,), 1),
            fm.SBP.split_block_cyclic((1,), 1),
        ),
        placement,
    )

    result = PyNttDistributedReshardRealizationPolicy().classify(
        DistributedReshardRealizationContext(
            source,
            target,
            DistributedReshardSourceKind.FUNCTION_PARAMETER,
            DistributedReshardUsageKind.INTERNAL,
        )
    )

    assert result == DistributedReshardRealization.BOXING


def test_legacy_policy_name_remains_a_pyntt_compatibility_alias():
    assert isinstance(
        CanonicalStorageReshardRealizationPolicy(),
        PyNttDistributedReshardRealizationPolicy,
    )
