# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import DType, IRBuilder, make_buffer_plan, tensor_type


def _grouped_module(indices=(0, 1, 2), *, count=3):
    builder = IRBuilder(dialect="high_level", stage="selected_tir")
    values = []
    for index in indices:
        values.append(builder.weight(
            f"layers.{index}.projection.weight",
            tensor_type(DType.BFLOAT16, [16, 16]),
            source="metadata.safetensors",
            key=f"layers.{index}.projection.weight",
            id=f"weight_{index}",
            metadata={
                "rdata_group": {
                    "name": "decoder.projection.weight",
                    "index": index,
                    "count": count,
                }
            },
        ))
    builder.function("main", (), (values[-1],))
    return builder.build(entry="main")


def test_rdata_group_members_are_contiguous_in_logical_index_order():
    # Graph order is deliberately different from physical repeated-region order.
    plan = make_buffer_plan(_grouped_module((2, 0, 1)))
    grouped = sorted(
        (value for value in plan.buffers if value.rdata_group),
        key=lambda value: value.group_index,
    )

    assert [value.id for value in grouped] == ["weight_0", "weight_1", "weight_2"]
    assert [value.offset for value in grouped] == [0, 512, 1024]
    assert all(value.group_count == 3 for value in grouped)


def test_rdata_group_rejects_a_missing_repeated_parameter():
    with pytest.raises(IRVerificationError, match="requires indices 0..2"):
        make_buffer_plan(_grouped_module((0, 2)))
