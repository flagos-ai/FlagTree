# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Runtime tensor boundaries expose vector lanes as trailing scalar axes."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.runtime.module import _allocate_tensor, _validate_ir_tensor


@pytest.mark.parametrize("lanes", [(), (8,), (2, 4)])
def test_runtime_allocation_and_validation_use_physical_vector_shape(lanes):
    torch = pytest.importorskip("torch")
    dtype = fm.vector_type("float32", lanes) if lanes else fm.DType.FLOAT32
    value_type = fm.tensor_type(dtype, (2, 17))
    value = _allocate_tensor(torch, value_type, "cpu")
    assert tuple(value.shape) == (2, 17, *lanes)
    _validate_ir_tensor("input", value, value_type, "cpu")
    if lanes:
        with pytest.raises(RuntimeContractError, match="must have shape"):
            _validate_ir_tensor("input", torch.empty((2, 17)), value_type, "cpu")
