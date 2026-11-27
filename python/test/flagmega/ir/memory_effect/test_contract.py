# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError


def test_memory_effect_round_trip_preserves_every_orthogonal_contract():
    effect = fm.MemoryEffect.REDUCTION_READ_WRITE.in_fixed_block(3)
    effect = effect.partitioned_by_argument(5).across_partial_owners()

    assert fm.MemoryEffect.from_data(effect.to_data()) == effect
    assert effect.mode is fm.MemoryAccessMode.READ_WRITE
    assert effect.physical_mode is fm.MemoryAccessMode.WRITE
    assert effect.scope is fm.MemoryAccessScope.INFERRED
    assert effect.kind is fm.MemoryEffectKind.REDUCTION_ACCUMULATOR
    assert effect.access_domain == fm.MemoryAccessDomain.fixed_block(3)
    assert effect.access_partition == fm.MemoryAccessPartition.by_argument(5)
    assert effect.owner_access is fm.MemoryOwnerAccess.PARTIAL_GROUP


def test_direct_read_write_retains_both_physical_accesses():
    assert fm.MemoryEffect.READ_WRITE.physical_mode == (
        fm.MemoryAccessMode.READ | fm.MemoryAccessMode.WRITE
    )
    assert fm.MemoryEffect.CHIP_READ.scope is fm.MemoryAccessScope.CHIP


@pytest.mark.parametrize(
    "constructor, message",
    (
        (lambda: fm.MemoryAccessDomain.fixed_block(-1), "non-negative"),
        (lambda: fm.MemoryAccessPartition.by_argument(-1), "non-negative"),
        (
            lambda: fm.MemoryAccessDomain(
                fm.MemoryAccessDomainKind.ALL_BLOCKS, 0
            ),
            "cannot name a block",
        ),
    ),
)
def test_invalid_memory_effect_refinements_fail_at_construction(
    constructor, message
):
    with pytest.raises(IRSchemaError, match=message):
        constructor()
