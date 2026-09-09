# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm


@pytest.mark.parametrize("lanes", [(2, ), (2, 4)])
def test_vector_cast_pattern_compares_canonical_nested_dtype_attributes(lanes):

    class Module(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type(fm.vector_type("float32", lanes), (4, )))
            result = fm.F.tensors.cast(value, fm.vector_type("bfloat16", lanes), name="cast")
            self.function("main", (value, ), (result, ))

    module = Module(dialect="ntt", stage="packed", entry="main").build()
    pattern = pm.F.tensors.is_cast(dtype=fm.vector_type("bfloat16", lanes))
    assert pm.try_match_root(module.node_map["cast"], pattern, module) is not None
    wrong = pm.F.tensors.is_cast(dtype=fm.vector_type("float32", lanes))
    assert pm.try_match_root(module.node_map["cast"], wrong, module) is None


def test_op_pattern_snapshots_mutable_nested_attribute_constraints():
    attrs = {"axes": [0, 1], "contract": {"lanes": [2, 4]}}
    pattern = pm.is_op("test.op", attributes=attrs)
    node = fm.Node("test", "test.op", (), fm.tensor_type("float32", (1, )), attrs=attrs)
    attrs["contract"]["lanes"].append(8)
    assert pattern.match_leaf(node)
    assert not pattern.match_leaf(fm.Node("other", "test.op", (), node.type, attrs=attrs))
