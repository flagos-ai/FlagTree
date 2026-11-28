# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.stages import next_stage
from triton.flagmega.passes.functions import function_nodes
from .helpers import module


@pytest.mark.parametrize("nested", [False, True])
def test_freeze_compiles_unannotated_weight_expressions_offline(nested):
    source = module(nested=nested)
    result = Compiler().compile(source, stop_after="freeze-constants").module
    assert not any(n.op == "tensors.cast" for n in function_nodes(result, result.function_map["worker"]))
    assert result.node_map["output"].op == "math.add"
    assert len([f for f in result.functions if f.name == "worker"]) == 1
    assert all(result.node_map[result.node_map[f"call{i}"].inputs[-1]].op == "builtin.const_asset" for i in range(2))


def test_dynamic_call_cannot_be_silently_specialized_as_a_constant():
    result = Compiler().compile(module(dynamic=True), stop_after="freeze-constants").module
    assert result.node_map["cast"].inputs == ("p", )
    assert result.function_map["worker"].parameters == ("p", "x")


@pytest.mark.parametrize("boundary", ["lift-constant-parameters", "constant_parameters_lifted"])
def test_resume_stops_at_new_boundary_and_does_not_freeze_or_reapply(tmp_path, boundary):
    compiler = Compiler()
    result = compiler.compile(module(), stop_after=boundary).module
    assert result.stage == "constant_parameters_lifted"
    assert not result.constant_recipes
    assert result.metadata.get("constant_phase", "open") == "open"
    restored = fm.load_module(fm.emit_module(result, tmp_path / "lifted.py"))
    assert compiler.compile(restored, stop_after=boundary).module == result
    assert next_stage(result.stage).name == "freeze-constants"
    frozen = compiler.compile(restored, stop_after="freeze-constants").module
    assert frozen.metadata["constant_phase"] == "frozen"
