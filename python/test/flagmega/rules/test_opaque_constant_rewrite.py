# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm, pattern_match as pm
from triton.flagmega.errors import IRVerificationError, StageError
from triton.flagmega.passes.constants import FreezeConstantIslandsPass
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules import RewriteResult, RewriteRule


def frozen_module():

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", fm.tensor_type("float32", (8, )))
            constant = fm.F.builtin.splat_const(x.type, 1.0, name="weight")
            output = fm.F.math.add(x, constant, name="output")
            self.function("main", (x, ), (output, ))

    return FreezeConstantIslandsPass().run(Graph(dialect="ntt", stage="canonical_constants", entry="main").build())


def test_post_freeze_rules_treat_recipes_as_opaque_leaves():
    module = frozen_module()
    rule = RewriteRule("add-to-mul", pm.F.math.is_add(), lambda r, m: replace(r.root, op="math.mul"))
    with pytest.raises(StageError, match="constants_open"):
        DataflowPass("default", (rule, )).run(module)
    result = DataflowPass("late-fusion", (rule, ), rewrite_constants=False).run(module)
    assert result.node_map["output"].op == "math.mul"
    assert result.constant_recipes == module.constant_recipes
    assert result.node_map["weight"] == module.node_map["weight"]


def test_region_cannot_modify_a_frozen_asset_through_an_extra_replacement():
    module = frozen_module()
    rule = RewriteRule(
        "illegal", pm.F.math.is_add(), lambda r, m: RewriteResult(
            r.root, extra_replacements=(replace(m.node_map["weight"], metadata={"changed": True}), )))
    with pytest.raises(IRVerificationError, match="opaque constant"):
        DataflowPass("late-fusion", (rule, ), rewrite_constants=False).run(module)
