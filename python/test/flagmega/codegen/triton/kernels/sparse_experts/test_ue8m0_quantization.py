# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Actual routed-expert quantization, packed lanes and scale groups on SM90."""

import importlib.util
from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.math._ue8m0_quantization import ue8m0_quantize
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp

from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import execute_and_reference, stage_module
from python.test.flagmega.codegen.triton.kernels.sparse_experts.test_sharded_block_scales import _module
from python.test.flagmega.ir.ops.nn.test_sparse_experts_ue8m0 import (
    MODE,
    bf16_domain,
    ue8m0_projection_case,
)
from python.test.flagmega.devices import is_nvidia_sm90


def require_sm90():
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA is required")


@pytest.mark.parametrize("definition", (SparseExpertsGateUp, SparseExpertsDown))
@pytest.mark.parametrize("rounded", (False, True))
@pytest.mark.parametrize("input_scale", (1., 1.1))
def test_actual_expert_preserves_ue8m0_and_projection_rounding(tmp_path, definition, rounded, input_scale):
    require_sm90()
    module, values = ue8m0_projection_case(definition, rounded=rounded, input_scale=input_scale)
    actual, expected, runtime = execute_and_reference(module, tmp_path, torch, overrides=values)
    try:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("definition", (SparseExpertsGateUp, SparseExpertsDown))
@pytest.mark.parametrize("block", (16, 128))
@pytest.mark.parametrize("packed", (False, True))
def test_packed_and_ragged_grouped_experts_preserve_ue8m0(tmp_path, definition, block, packed):
    require_sm90()
    module = stage_module(definition, dtype="bfloat16", weight_dtype="float8_e4m3fn",
                          block_scaled=True, weight_block=block, hidden=144, intermediate=136,
                          tokens=1, packed=packed, activation_quantization=MODE)
    actual, expected, runtime = execute_and_reference(module, tmp_path, torch)
    try:
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=.005)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("definition", (SparseExpertsGateUp, SparseExpertsDown))
def test_readonly_weights_keep_the_quantization_contract(tmp_path, definition):
    from triton.flagmega.importer import MemoryCheckpoint, TensorInfo

    require_sm90()
    base, values = ue8m0_projection_case(definition, input_scale=1.1)
    stage = next(node for node in base.nodes if node.op == definition.op_name)
    names = {name for name in stage.inputs if name.endswith(("_weight", "_proj_scale"))}

    class Graph(fm.Module):
        def forward(self):
            inputs, operands = [], []
            for name in stage.inputs:
                kind = base.node_map[name].type
                if name in names:
                    operand = self.weight(name, kind, source="memory", key=name, id=name)
                else:
                    operand = self.input(name, kind, id=name)
                    inputs.append(operand)
                operands.append(operand)
            result = definition.construct(*operands, **stage.attrs, name=stage.id)
            self.function("main", inputs, (result,))

    module = Graph(dialect="nn", stage="imported", entry="main").build()
    weights = {name: values[name] for name in names}
    infos = {name: TensorInfo(name, base.node_map[name].type.dtype, tuple(value.shape), "memory")
             for name, value in weights.items()}
    checkpoint = MemoryCheckpoint({}, infos, weights)
    actual, expected, runtime = execute_and_reference(module, tmp_path, torch, checkpoint=checkpoint,
        overrides={name: value for name, value in values.items() if name not in names})
    try:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert runtime.resource_report["spill_bytes"] == 0
        final = fm.load_module(tmp_path / "artifact/ir/final.py")
        assert not any(node.op == "tensors.pack" for node in final.nodes)
    finally:
        runtime.close()


@pytest.mark.parametrize("definition,split_k", ((SparseExpertsGateUp, False),
                                               (SparseExpertsDown, False), (SparseExpertsDown, True)))
@pytest.mark.parametrize("block,cyclic", ((16, False), (128, True)))
def test_ue8m0_preserves_sharded_weight_scale_coordinates(tmp_path, definition, split_k, block, cyclic):
    require_sm90()
    module = _module(definition, block, split_k, cyclic)
    nodes = []
    for node in module.nodes:
        if node.op == definition.op_name:
            prepared = definition.prepare(tuple(module.node_map[key] for key in node.inputs),
                                          {**node.attrs, "activation_quantization": MODE})
            node = replace(node, attrs=prepared.attrs)
        nodes.append(node)
    module = replace(module, nodes=tuple(nodes))
    proposed = Compiler().compile(module, stop_after="propose-tir").module
    assert not any(node.op == "tensors.pack" for node in proposed.nodes)
    actual, expected, runtime = execute_and_reference(module, tmp_path, torch)
    try:
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=.005)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("definition", (SparseExpertsGateUp, SparseExpertsDown))
@pytest.mark.parametrize("mode,implementation", (("dynamic_fp8", "simt"), (MODE, "simt"),
                                                ("dynamic_fp8", "pipeline")))
def test_quantization_group_can_span_multiple_compute_tiles(tmp_path, monkeypatch, definition, mode, implementation):
    require_sm90()
    if implementation == "pipeline":
        from python.test.flagmega.codegen.triton.kernels.sparse_experts import helpers
        from python.test.flagmega.codegen.triton.kernels.sparse_experts.test_pipeline import pipeline_target

        compiler = Compiler()
        compiler.target = pipeline_target()
        monkeypatch.setattr(helpers, "Compiler", lambda: compiler)
    module = stage_module(definition, dtype="bfloat16", weight_dtype="bfloat16" if implementation == "pipeline" else "float8_e4m3fn",
                          block_scaled=True, weight_block=256, hidden=264, intermediate=264,
                          tokens=1, activation_quantization=mode)
    activation = definition.input_parameters[0].name
    values = {}
    for name in module.function_map[module.entry].parameters:
        tensor = fm.logical_type(module.node_map[name].type)
        shape = tuple(dim.fixed_value for dim in tensor.shape)
        if name.endswith("_scale"):
            value = torch.ones(shape)
        else:
            value = torch.zeros(shape)
            if name == activation:
                value[..., 0], value[..., 1], value[..., 128] = 2.**-10, 1., 256.
            elif name.endswith("_weight"):
                value[..., 0] = 448.
        values[name] = value.to(getattr(torch, tensor.dtype.value))
    actual, expected, runtime = execute_and_reference(module, tmp_path, torch, overrides=values)
    try:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert runtime.resource_report["spill_bytes"] == 0
        if implementation == "pipeline":
            assert "simt_tma_pipeline" in (tmp_path / "artifact/generated_kernels.py").read_text()
    finally:
        runtime.close()


@pytest.mark.parametrize("groups", (1, 8))
def test_complete_bf16_amax_domain_matches_actual_gpu_instructions(tmp_path, groups):
    require_sm90()
    source = "import triton\nimport triton.language as tl\n" + TritonTemplateRegistry().render(
        "kernels/_ue8m0_quantization.py.jinja", {}) + '''
@triton.jit
def quantize(X, Q, S, ROWS: tl.constexpr, GROUPS: tl.constexpr):
    row = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    lane = tl.arange(0, 128)
    value = tl.load(X + row[:, None] * 128 + lane[None, :], row[:, None] < ROWS, other=0.).to(tl.float32)
    codes, scale = _flagmega_quantize_ue8m0(value, 1)
    tl.store(Q + row[:, None] * 128 + lane[None, :], codes, row[:, None] < ROWS)
    tl.store(S + row[:, None], scale, row[:, None] < ROWS)
'''
    path = tmp_path / "ue8m0_domain.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("ue8m0_domain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = bf16_domain()
    expected_codes, expected_scale = ue8m0_quantize(values)
    codes = torch.empty(values.shape, device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.empty((len(values),), device="cuda")
    kernel = module.quantize[((len(values) + groups - 1) // groups,)](
        values.cuda(), codes, scale, len(values), groups, num_warps=4)
    torch.testing.assert_close(codes.cpu().view(torch.uint8), expected_codes.view(torch.uint8), atol=0, rtol=0)
    torch.testing.assert_close(scale.cpu().view(torch.int32), expected_scale.flatten().view(torch.int32), atol=0, rtol=0)
    assert kernel.n_spills == 0
