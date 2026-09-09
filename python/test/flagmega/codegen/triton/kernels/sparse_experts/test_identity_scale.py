# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Unit readonly scales need no correctly-rounded runtime division by one."""

import ast
from dataclasses import replace

import pytest

from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from python.test.flagmega.codegen.triton.kernels.sparse_experts import helpers
from python.test.flagmega.codegen.triton.kernels.sparse_experts.test_constant_scales import constant_scales_module


@pytest.mark.parametrize("definition", [SparseExpertsGateUp, SparseExpertsDown])
@pytest.mark.parametrize("constant", [1.0, 1.00000001, 0.5, -1.0, 0.0, None])
def test_only_proven_float32_one_eliminates_scale_division(tmp_path, definition, constant):
    module = helpers.stage_module(definition, tokens=1) if constant is None else constant_scales_module(
        definition, constant)
    compiled = Compiler().compile(module).module
    render_triton_package(compiled, tmp_path / "generated")
    tree = ast.parse((tmp_path / "generated" / "generated_kernels.py").read_text())
    divisions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "div_rn"
    ]
    assert len(divisions) == (0 if constant in (1.0, 1.00000001) else 2 if definition is SparseExpertsGateUp else 1)


@pytest.mark.parametrize("scale_name,divided_values", [
    ("gate_input_scale", ["_fm_input"]),
    ("up_input_scale", ["_fm_input"]),
    ("gate_proj_scale", ["_fm_input", "_fm_input"]),
])
def test_each_input_scale_requires_its_own_identity_proof(tmp_path, scale_name, divided_values):
    module = helpers.stage_module(SparseExpertsGateUp, tokens=1)
    nodes = tuple(
        replace(node, op="builtin.splat_const", attrs={"value": 1.0}) if node.id == scale_name else node
        for node in module.nodes)
    functions = tuple(
        replace(function, parameters=tuple(name
                                           for name in function.parameters
                                           if name != scale_name))
        for function in module.functions)
    compiled = Compiler().compile(freeze_constant_islands(replace(module, nodes=nodes, functions=functions))).module
    render_triton_package(compiled, tmp_path / "generated")
    tree = ast.parse((tmp_path / "generated" / "generated_kernels.py").read_text())
    divisions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "div_rn"
    ]
    assert [node.args[0].id for node in divisions] == divided_values
    expected_scales = {"_fm_up_input_scale"} if scale_name == "gate_input_scale" else \
        {"_fm_gate_input_scale"} if scale_name == "up_input_scale" else \
        {"_fm_gate_input_scale", "_fm_up_input_scale"}
    assert {node.args[1].id for node in divisions} == expected_scales


def test_float32_division_by_one_preserves_bits_at_numeric_boundaries():
    """Check the rewrite's number contract separately from expert GEMV tests."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    import triton
    import triton.language as tl

    @triton.jit
    def divide(source, scale, result, N: tl.constexpr):
        offsets = tl.arange(0, 32)
        values = tl.load(source + offsets, offsets < N, other=0)
        # The denominator is a runtime load, so this tests actual div_rn.
        denominator = tl.load(scale)
        tl.store(result + offsets, tl.div_rn(values, denominator), offsets < N)

    # Positive/negative zero, subnormal, normal boundary, maximal finite and
    # infinities. NaN payload/signaling are not an expert tensor-value contract.
    positive = [0, 1, 0x007fffff, 0x00800000, 0x00800001, 0x3f800000, 0x7f7fffff, 0x7f800000]
    bits = torch.tensor([*positive, *(value | 0x80000000 for value in positive)], dtype=torch.int64,
                        device="cuda").int()
    source = bits.view(torch.float32)
    scale = torch.ones((), device="cuda")
    result = torch.empty_like(source)
    divide[(1, )](source, scale, result, source.numel())
    assert torch.equal(result.view(torch.int32), bits)
