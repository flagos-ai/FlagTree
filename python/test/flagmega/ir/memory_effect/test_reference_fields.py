# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.memory_effect import expand_memory_effect


def test_nested_reference_effects_round_trip_and_follow_field_order():
    tensor = fm.tensor_type("float32", (16,))
    nested = fm.RefType("inner", (("state", tensor), ("counter", tensor)))
    bundle = fm.RefType("outer", (("unchanged", tensor), ("inner", nested)))
    effect = fm.MemoryEffect.for_fields(inner=fm.MemoryEffect.for_fields(
        counter=fm.MemoryEffect.READ, state=fm.MemoryEffect.CHIP_READ_WRITE))
    assert effect.physical_mode == fm.MemoryAccessMode.READ_WRITE
    assert fm.MemoryEffect.from_data(effect.to_data()) == effect
    assert expand_memory_effect(bundle, effect) == (fm.MemoryEffect.NONE, fm.MemoryEffect.CHIP_READ_WRITE,
                                                   fm.MemoryEffect.READ)
    refined = effect.in_fixed_block(3).partitioned_by_argument(2).across_partial_owners()
    assert expand_memory_effect(bundle, refined)[1] == (
        fm.MemoryEffect.CHIP_READ_WRITE.in_fixed_block(3).partitioned_by_argument(2).across_partial_owners())


def test_whole_reference_effect_remains_whole():
    tensor = fm.tensor_type("float32", (16,))
    bundle = fm.RefType("state", (("a", tensor), ("b", tensor)))
    assert expand_memory_effect(bundle, fm.MemoryEffect.READ_WRITE) == (fm.MemoryEffect.READ_WRITE,) * 2
    assert "field_effects" not in fm.MemoryEffect.READ_WRITE.to_data()


@pytest.mark.parametrize("violation", ("missing_field", "tensor", "aggregate_mode", "parent_scope"))
def test_invalid_field_contract_is_not_silently_ignored(violation):
    tensor = fm.tensor_type("float32", (16,))
    bundle = fm.RefType("state", (("a", tensor), ("b", tensor)))
    effect = fm.MemoryEffect.for_fields(a=fm.MemoryEffect.READ)
    with pytest.raises(IRSchemaError, match="[Ff]ield"):
        if violation == "missing_field":
            expand_memory_effect(bundle, fm.MemoryEffect.for_fields(typo=fm.MemoryEffect.READ))
        elif violation == "tensor":
            expand_memory_effect(tensor, effect)
        elif violation == "aggregate_mode":
            replace(effect, mode=fm.MemoryAccessMode.WRITE)
        else:
            replace(effect, scope=fm.MemoryAccessScope.CHIP)
