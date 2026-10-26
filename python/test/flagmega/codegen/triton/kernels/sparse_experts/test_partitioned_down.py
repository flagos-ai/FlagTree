# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Actual partitioned projections through the normal compiler and runtime."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.runtime import load

from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import execute_and_reference
from python.test.flagmega.ir.ops.nn.test_partitioned_sparse_experts_down import PartitionedProjection, operands
from python.test.flagmega.devices import is_nvidia_sm90


@pytest.mark.parametrize("packed,lanes", ((False, ()), (True, (2, 2))))
@pytest.mark.parametrize("rounded", (False, True))
def test_actual_partitioned_projection_keeps_each_result(tmp_path, packed, lanes, rounded):
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA is required")
    graph = PartitionedProjection(packed=packed, lanes=lanes, rounded=rounded)
    actual, expected, runtime = execute_and_reference(graph.build(), tmp_path, torch, overrides=operands(graph))
    try:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert runtime.resource_report["spill_bytes"] == 0
        final = fm.load_module(tmp_path / "artifact/ir/final.py")
        assert not any(node.op == "tensors.pack" for node in final.nodes)
    finally:
        runtime.close()


@pytest.mark.parametrize("lanes", ((), (2, 2)))
def test_local_rounding_then_global_partition_sum(tmp_path, lanes):
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA is required")
    graph = PartitionedProjection(lanes=lanes, combine=True)
    module, values = graph.build(), operands(graph)
    expected = TorchEvaluator(DictWeightResolver({})).run(module, values)
    compiled = Compiler().compile(module).module
    assert not any(node.op == "tensors.pack" for node in compiled.nodes)
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    try:
        inputs = {name: value.cuda() for name, value in values.items()}
        result_specs = [spec for spec in runtime.external_arguments if spec["role"] == "result"]
        outputs = {spec["value"]: torch.empty_like(value, device="cuda")
                   for spec, value in zip(result_specs, expected, strict=True)}
        arguments = tuple(outputs[spec["value"]] if spec["role"] == "result" else inputs[spec["value"]]
                          for spec in runtime.external_arguments)
        runtime.prepare(*arguments)
        runtime.run_into(*arguments)
        for spec, target in zip(result_specs, expected, strict=True):
            got = outputs[spec["value"]]
            torch.testing.assert_close(got.cpu(), target, atol=0, rtol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("mode", ("dynamic_fp8", "dynamic_fp8_ue8m0"))
def test_partitioned_ragged_axes_and_multiple_quantization_groups(tmp_path, packed, mode):
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA is required")
    graph = PartitionedProjection(packed=packed, lanes=(2, 2), mode=mode, partitions=3, tokens=2,
                                  hidden=20, reduction=136, block=16)
    values = operands(graph)
    values["router_expert_ids"] = torch.tensor(((0, 1), (2, 0)), dtype=torch.int32)
    values["down_proj_scale"] *= torch.arange(1, 4).float()[:, None, None, None]
    values["down_proj_scale"] *= torch.tensor((1., 2.))[None, None, :, None]
    actual, expected, runtime = execute_and_reference(graph.build(), tmp_path, torch, overrides=values)
    try:
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=.005)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("packed", (False, True))
def test_partitioned_readonly_fp8_weights_keep_their_representation(tmp_path, packed):
    from triton.flagmega.importer import MemoryCheckpoint, TensorInfo

    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA is required")
    graph = PartitionedProjection(packed=packed, lanes=(2, 2), constants=True)
    module, values = graph.build(), operands(graph)
    names = {"down_weight", "down_proj_scale", "down_input_scale"}
    weights = {name: values[name] for name in names}
    infos = {name: TensorInfo(name, module.node_map[name].type.dtype, tuple(value.shape), "memory")
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


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("cyclic", (False, True))
def test_partition_and_hidden_owners_index_global_scale_tables(tmp_path, packed, cyclic):
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA is required")
    graph = PartitionedProjection(packed=packed, lanes=(2, 2), partitions=3 if cyclic else 4, tokens=2,
                                  hidden=64, reduction=128, block=16)
    base = graph.build()
    placement = fm.Placement((2, 2), "yx", "bb")
    b = fm.SBP.broadcast()
    p = fm.SBP.split_block_cyclic((0,), 1) if cyclic else fm.SBP.split_contiguous((0,))
    h = fm.SBP.split_block_cyclic((1,), 8) if cyclic else fm.SBP.split_contiguous((1,), 32)
    policies = {"activations": (b, b, p, b), "down_weight": (b, p, h, b)}

    class DistributedProjection(fm.Module):
        def forward(self):
            inputs = {}
            for name in base.function_map[base.entry].parameters:
                tensor = base.node_map[name].type
                kind = fm.DistributedType(tensor, policies.get(name, (b,) * tensor.rank), placement)
                inputs[name] = self.input(name, kind, id=name)
            result = fm.F.nn.partitioned_sparse_experts_down(**inputs, weight_block_n=16, weight_block_k=16,
                activation_quantization="dynamic_fp8_ue8m0", output_lanes=(2, 2))
            result = fm.F.distributed.boxing(result, result.type.tensor)
            self.function("main", tuple(inputs.values()), (result,))

    module = DistributedProjection(dialect="nn", stage="frozen_constants", entry="main",
        metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    values = operands(graph)
    values["router_expert_ids"] = torch.tensor(((0, 1), (2, 0)), dtype=torch.int32)
    values["down_proj_scale"] *= torch.arange(1, 5).float()[None, None, :, None]
    actual, expected, runtime = execute_and_reference(module, tmp_path, torch, overrides=values)
    try:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
