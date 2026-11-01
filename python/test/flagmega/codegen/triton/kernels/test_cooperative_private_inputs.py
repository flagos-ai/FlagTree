# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Full broadcast activations can be read from each CTA's own replica."""

from dataclasses import replace
from math import prod

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.codegen.triton.cooperative_access import require_owner_input
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError
from triton.flagmega.options import CompileOptions

from python.test.flagmega.codegen.triton.kernels.block_fp8.test_block_fp8_cooperative_wgmma import (
    TensorCheckpoint,
    projection_graph,
    projection_target,
    projection_values,
)
from python.test.flagmega.codegen.triton.kernels.matmul_glu.test_matmul_glu_cooperative_wgmma import (
    glu_graph,
    glu_target,
    glu_values,
)


def private_input_stage(family, *, private=True):
    mesh = fm.Placement((8, 16), "yx", "bb")
    factory, graph_factory = (projection_target, projection_graph) if family == "block_fp8" else (glu_target, glu_graph)
    kwargs = {"lanes": (8,)} if family == "block_fp8" else {}
    graph = graph_factory(n=1024, k=256, placement=mesh, readonly=True,
                          private_input=True, private_result=True, **kwargs)
    target = factory()
    if private:
        model = target.triton_implementation_model
        original = model.implementation(f"tir.{family}.cooperative_wgmma")
        alternative = replace(original, id=original.id + ".owner_local_value",
                              facts={**original.facts, "source_memory_scopes": {"value": "inferred"}})
        target.triton_implementation_model = replace(model, implementations=(*model.implementations, alternative),
            preferences={**model.preferences, family: (alternative.id, *model.preferences[family])})
    compiler = Compiler(CompileOptions(placement=mesh))
    compiler.target = target
    module = compiler.compile(graph).module
    plan = fm.verify_buffer_plan(module)
    call, = (node for node in module.nodes if node.op == "tir.call"
             and (dispatch := fm.kernel_dispatch_for_call(module, node)) is not None
             and dispatch.microkernel.family == family)
    assert plan.buffer_map[call.inputs[0]].storage == ("block_local_data" if private else "workspace")
    return graph, module


@pytest.mark.parametrize("family", ["block_fp8", "matmul_glu"])
def test_normal_codegen_accepts_the_named_owner_local_activation(tmp_path, family):
    _, module = private_input_stage(family)
    report = render_triton_package(module, tmp_path / "codegen")
    assert any(pool["storage"] == "block_local_data" for pool in report["runtime_binding"]["pools"])


@pytest.mark.parametrize("cause", ["backing", "view", "index", "stride", "split", "partial"])
def test_private_activation_rejects_unproved_or_different_owner_components(cause):
    mesh = fm.Placement((8, 16), "yx", "bb")
    kind = fm.DistributedType(fm.tensor_type("bfloat16", (1, 256)), (fm.SBP.broadcast(),) * 2, mesh)
    abi = {"memory_sharing_scope": "block", "coordinate_space": "local", "storage_kind": "compact_local",
           "distributed_type": kind.to_data(), "distributed_backing_type": None,
           "pool_scope_count": 128, "pool_scope_stride_bytes": 512, "pool_scope_index": "block_index"}
    if cause == "backing":
        abi["distributed_backing_type"] = kind.to_data()
    elif cause == "view":
        abi["view_byte_offset_expression"] = "16"
    elif cause == "index":
        abi["pool_scope_index"] = "shard_index"
    elif cause == "stride":
        abi["pool_scope_stride_bytes"] = 0
    elif cause == "split":
        kind = replace(kind, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 32)))
        abi["distributed_type"] = kind.to_data()
    elif cause == "partial":
        abi["distributed_type"] = replace(kind, partial=fm.SBP.partial((1,))).to_data()
    with pytest.raises(CodegenError, match="complete"):
        require_owner_input({"abi": abi}, "value", owner="test")


@pytest.mark.parametrize("family", ["block_fp8", "matmul_glu"])
def test_private_activation_matches_shared_bits_guards_and_successor(tmp_path, family):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime

    graph, private = private_input_stage(family)
    _, shared = private_input_stage(family, private=False)
    values = (projection_values if family == "block_fp8" else glu_values)(torch, graph)
    device_values = {key: value.cuda() for key, value in values.items() if not key.endswith("weight")}
    cases, runtimes = [], []
    try:
        for label, module in (("shared", shared), ("private", private)):
            path = write_artifact(module, tmp_path / label, target="nvidia-sm90", emit_executable=True,
                                  checkpoint=TensorCheckpoint(values))
            runtime = load(path, device="cuda:0")
            runtimes.append(runtime)
            outputs, guards = {}, []
            for argument in runtime.external_arguments:
                if argument["role"] != "result":
                    continue
                descriptor = runtime.buffer_plan.buffer_map[argument["buffer"]]
                kind = descriptor.dtype
                lanes = kind.lanes if isinstance(kind, fm.VectorType) else ()
                scalar = kind.elem_type if lanes else kind
                shape = (*descriptor.component_shape, *lanes)
                storage = torch.full((64 + prod(shape) + 64,), -123., dtype=getattr(torch, scalar.value), device="cuda")
                outputs[argument["value"]] = storage[64:-64].reshape(shape)
                guards.extend((storage[:64], storage[-64:]))
            arguments = tuple({**device_values, **outputs}[spec["value"]] for spec in runtime.external_arguments)
            cases.append((arguments, tuple(outputs.values()), guards))
        for mutation in range(4):
            if mutation:
                device_values["value"].mul_(1.03125).add_(1 / 512)
                for key, value in device_values.items():
                    if key.endswith("scale"):
                        value.mul_(1.03125)
                device_values["sentinel"].add_(1 / 8)
            for runtime, (arguments, outputs, guards) in zip(runtimes, cases, strict=True):
                for output in outputs:
                    output.fill_(float("nan"))
                Runtime.prepare(runtime, *arguments)
                Runtime.run_into(runtime, *arguments)
                torch.cuda.synchronize()
                assert runtime.resource_report["grid"] == [128, 1, 1]
                assert runtime.resource_report["spill_bytes"] == 0
                assert all(bool((guard == -123.).all()) for guard in guards)
                torch.testing.assert_close(outputs[1], device_values["sentinel"] * 2, rtol=0, atol=0)
            for left, right in zip(cases[0][1], cases[1][1], strict=True):
                assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    finally:
        for runtime in runtimes:
            runtime.close()
