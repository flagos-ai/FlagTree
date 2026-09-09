# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.python_ir import _python_value_expr


@pytest.mark.parametrize("dtype", [
    fm.vector_type("bfloat16", (2, 4)),
    fm.PointerType(fm.DType.FLOAT32),
    fm.MaskVectorType(fm.MaskVectorStyle.SLIM, 8, 4)
])
def test_python_call_dtype_values_use_executable_type_constructors(dtype):
    source = _python_value_expr(dtype, {})
    assert eval(source, {"fm": fm}) == dtype
    assert "fm." in source
