# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.kernels.distributed_boxing.reduction import partial_reduction_context
from triton.flagmega.errors import CodegenError


@pytest.mark.parametrize("dtype,accumulator,lower,upper", [
    ("bfloat16", "tl.float32", "float('-inf')", "float('inf')"),
    ("int8", "tl.int32", "-128", "127"),
    ("uint16", "tl.uint32", "0", "65535"),
    ("int32", "tl.int32", "-2147483648", "2147483647"),
    ("int64", "tl.int64", "-9223372036854775808", "9223372036854775807"),
    ("uint64", "tl.uint64", "0", "18446744073709551615"),
])
@pytest.mark.parametrize("reduce_op", ["sum", "min", "max", "prod"])
def test_identity_and_accumulator_preserve_scalar_reduction_semantics(dtype, accumulator, lower, upper, reduce_op):
    context = partial_reduction_context(reduce_op, dtype)
    assert context["mode"] == f"partial_{reduce_op}"
    assert context["accumulator_type"] == accumulator
    assert context["neutral"] == {"sum": "0", "prod": "1", "min": upper, "max": lower}[reduce_op]


@pytest.mark.parametrize("reduce_op,dtype", [("mean", "float32"), ("sum", "bool"), ("sum", "complex64")])
def test_unreviewed_operator_or_dtype_is_not_silently_computed_as_fp32_sum(reduce_op, dtype):
    with pytest.raises(CodegenError):
        partial_reduction_context(reduce_op, dtype)
