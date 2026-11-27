# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.core import tensor_elements, tensor_nbytes


def test_vector_tensor_elements_and_bytes_count_lane_payload_exactly_once():
    value_type = fm.tensor_type(fm.vector_type("bfloat16", (2, 8)), (3, 5))

    assert tensor_elements(value_type) == 3 * 5 * 2 * 8
    assert tensor_nbytes(value_type) == 3 * 5 * 2 * 8 * 2


def test_scalar_tensor_byte_count_is_unchanged():
    value_type = fm.tensor_type("float32", (3, 5))

    assert tensor_elements(value_type) == 15
    assert tensor_nbytes(value_type) == 60
