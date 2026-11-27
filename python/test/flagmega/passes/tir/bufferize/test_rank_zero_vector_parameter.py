# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Rank-zero vector tensors retain their element storage at function entry."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.bufferize import plan_buffers


@pytest.mark.parametrize("lanes", ((), (4,), (2, 2)))
def test_rank_zero_vector_parameter_is_a_buffer_and_scalar_stays_by_value(lanes):
    dtype = fm.vector_type("float32", lanes) if lanes else fm.DType.FLOAT32
    value = fm.Node("value", "builtin.var", (), fm.tensor_type(dtype, ()))
    module = fm.IRModule(
        dialect="semantic_tir", stage="packaged_tir", nodes=(value,),
        functions=(fm.Function("main", (value.id,), (value.id,)),), entry="main",
    )
    descriptor = next(buffer for buffer in plan_buffers(module).buffers if buffer.id == value.id)
    assert descriptor.storage == ("input" if lanes else "scalar")
    assert descriptor.role == ("parameter" if lanes else "scalar_parameter")
    assert descriptor.mem_span.nbytes == dtype.itemsize
