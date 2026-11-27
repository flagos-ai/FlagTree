# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Partition redundant owners without changing logical sharding or aliasing."""

from copy import deepcopy

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import _gather_reduce_add_norm_apply_call
from triton.flagmega.errors import CodegenError
from test_sum import _abi, _raw


@pytest.mark.parametrize("values", [1, 3, 17, 129, 2049])
def test_redundant_owners_cover_each_value_once_including_empty_owners(values):
    partial = _abi(
        (1, values), storage_kind="compact_per_owner", coordinate_space="local",
        partial_axes=(0, 1), owner_stride=values,
    )
    call = _gather_reduce_add_norm_apply_call(_raw(
        partial=partial, value=_abi((1, values)), parameter=_abi((values,)),
    ))
    count = call["work_partition_count"]
    assert count == 128
    assert call["stats_owner_count"] == count
    assert call["tile"] <= (1 << (((values + count - 1) // count) - 1).bit_length())
    visits = [0] * values
    for owner in range(count):
        for start in range(owner * call["tile"], values, count * call["tile"]):
            for index in range(start, min(start + call["tile"], values)):
                visits[index] += 1
    assert visits == [1] * values


def test_private_output_copies_violate_collective_publication_contract():
    raw = deepcopy(_raw())
    for output in raw["outputs"]:
        output["buffers"][0]["abi"].update(
            storage_kind="compact_local", coordinate_space="local",
        )
    with pytest.raises(CodegenError, match="CHIP_WRITE effect contract"):
        _gather_reduce_add_norm_apply_call(raw)
