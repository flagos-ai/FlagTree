# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import StageError


def test_unknown_stop_boundary_is_rejected_before_any_pass_runs(monkeypatch):

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 16)))
            self.function("main", (value, ), (value, ))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()

    def unexpected_pass(*args, **kwargs):
        pytest.fail("Unknown stop boundary must not enter a compiler pass")

    monkeypatch.setattr("triton.flagmega.compiler.PassManager.run", unexpected_pass)
    with pytest.raises(StageError, match="Unknown.*stop"):
        Compiler().compile(module, stop_after="TargetIndependet")


def test_pipeline_group_name_is_an_explicit_resume_boundary():

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 16)))
            self.function("main", (value, ), (fm.F.math.silu(value), ))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    result = Compiler().compile(module, stop_after="TargetIndependentPass")
    assert result.module.stage == "call_invariants_hoisted"
    resumed = Compiler().compile(result.module, stop_after="TargetIndependentPass")
    assert resumed.module == result.module and resumed.reports == ()
