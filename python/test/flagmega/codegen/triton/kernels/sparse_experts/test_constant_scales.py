# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Immutable uniform scales are constants, while runtime scales stay loads."""

from dataclasses import replace
import re
import struct

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.passes.constants import freeze_constant_islands
from python.test.flagmega.codegen.triton.kernels.sparse_experts import helpers


class ConstantOnlyCheckpoint:
    def load_tensor(self, key, *, device="cpu"):
        raise AssertionError(f"Splat scales must not request checkpoint tensor {key!r}.")


def constant_scales_module(definition, value=1.0):
    module = helpers.stage_module(definition, tokens=1)
    scales = {node.id for node in module.nodes if node.id.endswith("_scale")}
    nodes = tuple(replace(node, op="builtin.splat_const", attrs={"value": value}) if node.id in scales else node
                  for node in module.nodes)
    functions = tuple(replace(function, parameters=tuple(name for name in function.parameters if name not in scales))
                      for function in module.functions)
    result = replace(module, nodes=nodes, functions=functions)
    fm.verify_module(result)
    return freeze_constant_islands(result)


@pytest.mark.parametrize("definition", [SparseExpertsGateUp, SparseExpertsDown])
@pytest.mark.parametrize("constant", [1.0, 0.5, 1.00000001])
def test_uniform_scales_render_typed_literals(tmp_path, definition, constant):
    compiled = Compiler().compile(constant_scales_module(definition, constant)).module
    render_triton_package(compiled, tmp_path / "generated")
    source = (tmp_path / "generated" / "generated_kernels.py").read_text()
    expected = struct.unpack("f", struct.pack("f", constant))[0]
    names = ["_fm_gate_input_scale", "_fm_gate_scale", "_fm_up_input_scale", "_fm_up_scale"] \
        if definition is SparseExpertsGateUp else ["_fm_input_scale", "_fm_scale"]
    for name in names:
        assignments = re.findall(rf"^\s*{name} = (.+)$", source, re.MULTILINE)
        assert assignments == [f"tl.full((), {expected!r}, tl.float32)"]


@pytest.mark.parametrize("definition", [SparseExpertsGateUp, SparseExpertsDown])
def test_runtime_scales_are_not_assumed_to_be_one(tmp_path, definition):
    compiled = Compiler().compile(helpers.stage_module(definition, tokens=1)).module
    render_triton_package(compiled, tmp_path / "generated")
    source = (tmp_path / "generated" / "generated_kernels.py").read_text()
    assert re.search(r"_fm_(?:gate_)?input_scale = tl.load\(", source)


@pytest.mark.parametrize("definition", [SparseExpertsGateUp, SparseExpertsDown])
@pytest.mark.parametrize("constant", [1.0, 0.5])
def test_constant_scales_device_matches_evaluator(tmp_path, definition, constant):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    actual, expected, _ = helpers.execute_and_reference(constant_scales_module(definition, constant), tmp_path, torch,
                                                       checkpoint=ConstantOnlyCheckpoint())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
