# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed.policy import _peel_vector_operand


def boundary(dtype):

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("float32", (2, 16)), id="value")
            result = fm.F.tensors.bitcast(
                value, dtype, name="boundary", metadata={
                    "vectorization_internal": True, "vectorization_role": "pack", "vectorization_root": "compute"
                })
            self.function("main", (value, ), (result, ))

    return Graph(dialect="ntt", stage="vectorized", entry="main").build()


def test_recover_operand_through_canonical_lane_view():
    module = boundary(fm.vector_type("float32", (4, )))
    assert _peel_vector_operand("boundary", module, compute_roots={}, seen=frozenset()) == "value"


def test_numeric_reinterpretation_is_not_a_vectorization_boundary():
    module = boundary(fm.vector_type("int32", (4, )))
    with pytest.raises(ValueError, match="element type"):
        _peel_vector_operand("boundary", module, compute_roots={}, seen=frozenset())
