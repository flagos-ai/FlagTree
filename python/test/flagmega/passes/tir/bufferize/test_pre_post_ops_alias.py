# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler


@pytest.mark.parametrize("target", ["float32", "bfloat16"])
def test_fused_add_alias_requires_compatible_external_storage(target):

    @fm.fusion(fm.tensor_type("float32", (4, 32)))
    def convert(x):
        return fm.F.tensors.cast(x, target)

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", convert.input_type)
            y = self.input("y", convert.input_type)
            result = fm.F.with_ops(fm.F.math.add, x, y, post_ops=(convert, ))
            self.function("main", (x, y), (result, ))

    source = Graph(dialect="high_level", stage="frozen_constants", entry="main").build()
    result = Compiler().compile(source).module
    dispatch, = (value.dispatch for value in result.kernel_definitions)
    assert bool(dispatch.inplace_alias_candidates) is (target == "float32")
