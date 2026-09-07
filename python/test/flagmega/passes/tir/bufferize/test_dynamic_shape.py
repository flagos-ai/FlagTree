# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import DType, IRBuilder, dim, make_buffer_plan, tensor_type


def test_bufferization_uses_a_symbolic_dimension_upper_bound():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    value_type = tensor_type(DType.FLOAT32, [dim("tokens", 1, 32), 64])
    source = builder.var("source", value_type, id="source")
    output = builder.call("test.unary", [source], value_type, id="output")
    builder.function("main", [source], [output])

    plan = make_buffer_plan(builder.build(entry="main"))
    assert plan.buffer_map["source"].shape == (32, 64)
    assert plan.buffer_map["source"].nbytes == 32 * 64 * 4


def test_bufferization_rejects_only_an_unbounded_dynamic_dimension():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    value_type = tensor_type(DType.FLOAT32, [dim("tokens", 1, None), 64])
    source = builder.var("source", value_type, id="source")
    builder.function("main", [source], [source])

    with pytest.raises(IRVerificationError, match="finite upper bound"):
        make_buffer_plan(builder.build(entry="main"))
