# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega.compiler import Compiler
from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.targets import NvidiaSm90Target

from python.test.flagmega.codegen.triton.kernels.sparse_experts import helpers
from python.test.flagmega.sparse_experts.helpers import operand_types


def pipeline_target(*, block_k=None, stages=None):
    target = NvidiaSm90Target()
    model = target.triton_implementation_model
    if block_k is not None or stages is not None:
        implementations = []
        for implementation in model.implementations:
            if implementation.variant == "simt_tma_pipeline":
                k = block_k or implementation.parameters["block_k"]
                count = stages or implementation.parameters["num_stages"]
                implementation = replace(
                    implementation, parameters={**implementation.parameters, "block_k": k, "num_stages": count},
                    shared_workspaces=tuple(replace(workspace, type=fm.tensor_type(
                        workspace.type.dtype, (count, 1, implementation.parameters["block_n"], k)))
                        for workspace in implementation.shared_workspaces),
                    transfer_pipeline=replace(implementation.transfer_pipeline, capacity=count))
            implementations.append(implementation)
        model = replace(model, implementations=tuple(implementations))
    preferences = dict(model.preferences)
    for family in ("sparse_experts_gate_up", "sparse_experts_down"):
        preferences[family] = tuple(i.id for i in model.implementations
                                    if i.family == family and i.variant == "simt_tma_pipeline") + preferences.get(family, (f"tir.{family}.simt",))
    target.triton_implementation_model = replace(model, preferences=preferences)
    return target


@pytest.mark.parametrize("definition", (SparseExpertsGateUp, SparseExpertsDown))
@pytest.mark.parametrize("dtype", ("bfloat16", "float32"))
@pytest.mark.parametrize("packed", (False, True))
def test_pipeline_ragged_tiles_and_numerical_boundaries(definition, dtype, packed, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires SM90")
    compiler = Compiler()
    compiler.target = pipeline_target()
    monkeypatch.setattr(helpers, "Compiler", lambda: compiler)
    graph = helpers.stage_module(definition, dtype=dtype, packed=packed)
    actual, expected, runtime = helpers.execute_and_reference(graph, tmp_path, torch)
    torch.testing.assert_close(actual, expected, rtol=0.016 if dtype == "bfloat16" else 1e-5, atol=1e-5)
    source = (tmp_path / "artifact/generated_kernels.py").read_text()
    assert "simt_tma_pipeline" in source
    assert "def _flagmega_main_call_0_expert_stage__producer(" in source
    assert runtime._prepared.compiled_kernel.n_spills == 0


def distributed_stage(definition, dtype, partition):
    mesh = fm.Placement((2, 4), "yx", "bb")
    b = fm.SBP.broadcast()
    is_down = definition is SparseExpertsDown
    route = fm.SBP.split_block_cyclic((1,), 1) if partition == "route" else b
    feature = (fm.SBP.split_contiguous((0, 1), 16) if partition == "output"
               else fm.SBP.split_contiguous((0,), 48 if is_down else 24))
    reduction = fm.SBP.split_contiguous((1,), 16) if partition == "split_k" else b
    types = operand_types(dtype=dtype, tokens=2, hidden=72, intermediate=40, routes=3, experts=5)
    policies = {"dispatched": (b, route, b), "activations": (b, route, reduction),
                "router_expert_ids": (b, route), "gate_weight": (b, feature, b),
                "up_weight": (b, feature, b), "down_weight": (b, feature, reduction)}

    class Graph(fm.Module):
        def forward(self):
            inputs = tuple(self.input(parameter.name,
                                      fm.DistributedType(types[parameter.name], policies.get(parameter.name, (b, b)), mesh),
                                      id=parameter.name) for parameter in definition.input_parameters)
            value = definition.construct(*inputs, name="expert_stage")
            result = fm.F.distributed.boxing(value, value.type.tensor)
            self.function("main", inputs, (result,))
    return Graph(dialect="nn", stage="frozen_constants", entry="main",
                 metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()


@pytest.mark.parametrize("definition,partition", ((SparseExpertsGateUp, "route"),
                                                  (SparseExpertsGateUp, "output"),
                                                  (SparseExpertsDown, "route"),
                                                  (SparseExpertsDown, "output"),
                                                  (SparseExpertsDown, "split_k")))
@pytest.mark.parametrize("dtype", ("bfloat16", "float32"))
def test_pipeline_ragged_owners_and_explicit_partial(definition, partition, dtype, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires SM90")
    compiler = Compiler()
    compiler.target = pipeline_target()
    monkeypatch.setattr(helpers, "Compiler", lambda: compiler)
    actual, expected, runtime = helpers.execute_and_reference(distributed_stage(definition, dtype, partition), tmp_path, torch)
    torch.testing.assert_close(actual, expected, rtol=0.016 if dtype == "bfloat16" else 1e-5, atol=1e-5)
    assert "simt_tma_pipeline" in (tmp_path / "artifact/generated_kernels.py").read_text()
    assert runtime._prepared.compiled_kernel.n_spills == 0


@pytest.mark.parametrize("definition", (SparseExpertsGateUp, SparseExpertsDown))
@pytest.mark.parametrize("dtype", ("bfloat16", "float32"))
def test_pipeline_large_shared_tile_uses_multiple_tma_boxes(definition, dtype, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires SM90")
    compiler = Compiler()
    compiler.target = pipeline_target(block_k=128, stages=2)
    monkeypatch.setattr(helpers, "Compiler", lambda: compiler)
    actual, expected, _ = helpers.execute_and_reference(distributed_stage(definition, dtype, "route"), tmp_path, torch)
    torch.testing.assert_close(actual, expected, rtol=0.016 if dtype == "bfloat16" else 1e-5, atol=1e-5)


@pytest.mark.parametrize("definition", (SparseExpertsGateUp, SparseExpertsDown))
@pytest.mark.parametrize("view_kind", ("materialized", "alias"))
def test_pipeline_reads_router_ids_written_in_the_same_invocation(definition, view_kind, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires SM90")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
    from triton.flagmega.runtime import load

    mesh = fm.Placement((2, 4), "yx", "bb")
    b = fm.SBP.broadcast()
    route = fm.SBP.split_block_cyclic((1,), 1)
    feature = fm.SBP.split_contiguous((0,), 24)
    types = operand_types(tokens=2, hidden=48, intermediate=40, routes=3, experts=5)
    policies = {"dispatched": (b, route, b), "activations": (b, route, b),
                "gate_weight": (b, feature, b), "up_weight": (b, feature, b),
                "down_weight": (b, feature, b)}

    class Graph(fm.Module):
        def forward(self):
            logits = self.input("logits", fm.DistributedType(fm.tensor_type("float32", (2, 5)), (b, b), mesh),
                                id="logits")
            top = fm.F.tensors.top_k(logits, k=3, name="router")
            _, indices = fm.F.tensors.get_items(top, 0, 1, name_prefix="router")
            reshard = fm.F.distributed.sharded_view if view_kind == "alias" else fm.F.distributed.boxing
            ids = reshard(indices, fm.DistributedType(indices.type.tensor, (b, route), mesh))
            inputs = []
            operands = []
            for parameter in definition.input_parameters:
                if parameter.name == "router_expert_ids":
                    operands.append(ids)
                else:
                    value = self.input(parameter.name, fm.DistributedType(
                        types[parameter.name], policies.get(parameter.name, (b, b)), mesh), id=parameter.name)
                    inputs.append(value)
                    operands.append(value)
            result = definition.construct(*operands, name="expert_stage")
            self.function("main", (logits, *inputs), (fm.F.distributed.boxing(result, result.type.tensor),))

    graph = Graph(dialect="nn", stage="frozen_constants", entry="main",
                  metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()
    compiler = Compiler()
    compiler.target = pipeline_target()
    compiled = compiler.compile(graph).module
    from triton.flagmega.ir.tir.visitor import TIRVisitor

    class CheckProducerReads(TIRVisitor):
        def __init__(self):
            self.found = 0

        def visit_kernel_invoke(self, invocation):
            if invocation.call_id == "expert_stage":
                ids_binding = next(binding for binding in invocation.arguments if binding.formal == "router_expert_ids")
                assert ids_binding.actual in invocation.transfer_sources
                self.found += 1

    check = CheckProducerReads()
    check.visit(compiled.execution_function_map["main"])
    assert check.found == 2
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    source = (artifact / "generated_kernels.py").read_text()
    assert "simt_tma_pipeline" in source
    assert "handoff" in source
    runtime = load(artifact, device="cuda:0")
    generator = torch.Generator().manual_seed(1027)
    values = {}
    parameters = graph.function_map[graph.entry].parameters
    for name in parameters:
        tensor = fm.logical_type(graph.node_map[name].type)
        shape = tuple(extent.fixed_value for extent in tensor.shape)
        values[name] = ((torch.ones(shape) if name.endswith("_scale")
                         else torch.randn(shape, generator=generator) / 8).to(getattr(torch, tensor.dtype.value)))
    arguments = tuple(values[name].cuda() for name in parameters)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    output = runtime.create_outputs()
    runtime.prepare(*arguments, output=output)
    # Reuse allocations with different IDs to expose a stale producer read.
    for iteration in range(20):
        values["logits"] = torch.roll(torch.arange(5).float().repeat(2, 1), iteration, dims=1)
        arguments[0].copy_(values["logits"])
        expected = evaluator.run(graph, values)[0]
        runtime.run_into(output, *arguments)
        torch.cuda.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0.016, atol=0.000244140625)


@pytest.mark.parametrize("tokens", (1, 3))
@pytest.mark.parametrize("shared_width", (16, 24))
def test_pipeline_imported_moe_with_independent_shared_width(tokens, shared_width, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires SM90")
    from python.test.flagmega.codegen.triton.kernels.sparse_experts.test_shared_groups import imported_moe

    graph, checkpoint = imported_moe(torch, tokens, shared_width)
    compiler = Compiler()
    compiler.target = pipeline_target()
    monkeypatch.setattr(helpers, "Compiler", lambda: compiler)
    output, expected, _ = helpers.execute_and_reference(graph, tmp_path, torch, checkpoint=checkpoint)
    torch.testing.assert_close(output, expected, rtol=0.016, atol=0.03125)
    source = (tmp_path / "artifact/generated_kernels.py").read_text()
    assert "sparse_experts_gate_up/simt_tma_pipeline" in source
    assert "sparse_experts_down/simt_tma_pipeline" in source
