# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Ordinary compilation, tensor-map ABI, chip publication and GPU results."""

from math import prod

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.core import construction_scope
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

from python.test.flagmega.ir.ops.nn.test_kda_input_projection import graph


def test_injected_catalog_preserves_nvmma_alignment():
    from triton.flagmega.targets import NvidiaSm90Target
    from triton.flagmega.targets.portable_triton_implementations import portable_triton_implementation_model

    NvidiaSm90Target(triton_implementation_model=portable_triton_implementation_model())


@pytest.mark.parametrize("k,variant", [(512, "wgmma_local"), (80, "mma")])
def test_normal_codegen_declares_storage_and_descriptor_abi(tmp_path, k, variant):
    compiled = Compiler().compile(graph(k=k)).module
    dispatch, = (function.dispatch for function in compiled.kernel_definitions
                 if function.dispatch.semantic_op == "nn.kda_input_projection")
    assert dispatch.microkernel.variant == variant
    assert compiled.metadata["launch_contract"]["cooperative_grid"]
    assert tuple(compiled.metadata["codegen_package_plan"]["launch"]["grid"]) == (128, 1, 1)
    assert all(effect.scope == fm.MemoryAccessScope.CHIP for name, effect in dispatch.memory_effect_map.items())
    path = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    source = (path / "generated_kernels.py").read_text()
    assert f"# flagmega-kernel: kda_input_projection/{variant}" in source
    if variant == "wgmma_local":
        assert len(dispatch.microkernel.shared_workspaces) == 2
        assert dispatch.microkernel.transfer_pipeline is None
        assert "warp_specialize" not in dispatch.microkernel.requires
        assert "reinterpret_tensor_map" in source and "tle.gpu.wgmma_wait" in source
    restored = fm.load_module(fm.emit_module(compiled, tmp_path / "compiled.py"))
    assert restored.semantic_hash == compiled.semantic_hash


@pytest.mark.parametrize("k,heads,width,rank_f,rank_g,packed,resident", [
    (80, 8, 16, 64, 128, False, False),
    (512, 8, 16, 64, 128, False, False),
    (512, 8, 16, 64, 128, True, False),
    (4096, 16, 128, 128, 128, True, False),
    (4096, 16, 128, 128, 128, True, True),
    (512, 32, 128, 64, 128, True, True),
])
def test_real_artifact_six_results_and_guards(tmp_path, k, heads, width, rank_f, rank_g, packed, resident):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    lanes = ((2, 4), (4,), (), (), (8,), (2, 4)) if packed else ((),) * 6
    module = Compiler().compile(graph(k=k, heads=heads, width=width, rank_f=rank_f, rank_g=rank_g,
                                      lanes=lanes, input_lanes=(2, 4) if packed else (),
                                      placement=fm.Placement((8, 16), "yx", "bb") if resident else None)).module
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    if resident:
        assert "distributed_boxing/tensor_load" not in (artifact / "generated_kernels.py").read_text()
    runtime = load(artifact, device="cuda:0")
    generator = torch.Generator().manual_seed(7931)
    inputs = {node.id: torch.randint(-8, 8, tuple(d.fixed_value for d in node.type.shape) +
                                    tuple(getattr(node.type.dtype, "lanes", ())), generator=generator).float()
                                    .div_(128).bfloat16().cuda()
              for node in module.nodes if node.op == "builtin.var"}
    specs = tuple(spec for spec in runtime.external_arguments if spec["role"] == "result")
    storage, outputs = [], {}
    for spec in specs:
        kind = fm.logical_type(module.node_map[spec["value"]].type)
        shape = tuple(d.fixed_value for d in kind.shape) + tuple(getattr(kind.dtype, "lanes", ()))
        allocation = torch.full((prod(shape) + 32,), 123., dtype=torch.bfloat16, device="cuda")
        storage.append(allocation)
        outputs[spec["value"]] = allocation[16:-16].reshape(shape)
    try:
        for mutation in range(3):
            inputs["x"].add_(1 / 128 if mutation else 0)
            arguments = tuple(outputs[spec["value"]] if spec["role"] == "result" else inputs[spec["value"]]
                              for spec in runtime.external_arguments)
            GeneratedTirCallGraphModule.prepare(runtime, *arguments)
            GeneratedTirCallGraphModule.run_into(runtime, *arguments)
            torch.cuda.synchronize()
            for spec, name, allocation in zip(specs, ("q", "k", "v", "b", "f", "g"), storage, strict=True):
                expected = (inputs["x"].cpu().reshape(1, k).float()
                            @ inputs[name].cpu().reshape(-1, k).float().T).bfloat16()
                torch.testing.assert_close(outputs[spec["value"]].reshape(expected.shape).cpu(), expected,
                                           rtol=0, atol=0)
                assert bool(torch.all(allocation[:16] == 123.)) and bool(torch.all(allocation[-16:] == 123.))
            assert runtime.resource_report["spill_bytes"] == 0
            assert runtime.resource_report["num_warps"] <= 8
    finally:
        runtime.close()


def test_reusable_projection_rebinds_six_weights_and_intermediate_activation(tmp_path):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    placement = fm.Placement((8, 16), "yx", "bb")
    builder = fm.IRBuilder(dialect="distributed", stage="frozen_constants",
                           metadata={"auto_distribution": {"placement": placement.to_data()}})
    types = (fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 64)),
             *(fm.tensor_type("bfloat16", (4, 16, 512)),) * 3,
             fm.tensor_type("bfloat16", (4, 512)), fm.tensor_type("bfloat16", (512, 512)),
             fm.tensor_type("bfloat16", (64, 512)))
    distributed = tuple(fm.DistributedType(kind, (fm.SBP.broadcast(),) * kind.rank, placement) for kind in types)
    formal = tuple(builder.var(f"arg{i}", kind, id=f"arg{i}") for i, kind in enumerate(distributed))
    with construction_scope(builder):
        projection = fm.F.nn.kda_input_projection(*formal, output_lanes=((), (), (), (), (8,), ()), name="projection")
    builder.function("project", formal, (projection,), attrs={"reusable": True})
    x = builder.var("x", distributed[0], id="x")
    banks = tuple(tuple(builder.var(f"{label}{name}", kind, id=f"{label}{name}")
                        for name, kind in zip(("q", "k", "v", "b", "f", "g"), distributed[1:], strict=True))
                  for label in ("a", "b"))
    with construction_scope(builder):
        first = fm.F.builtin.call(x, *banks[0], callee="project", result_type=projection.type, name="first")
        intermediate = fm.F.tensors.get_item(first, 4)
        second = fm.F.builtin.call(intermediate, *banks[1], callee="project", result_type=projection.type, name="second")
        outputs = (*fm.F.tensors.get_items(first, *range(6)), *fm.F.tensors.get_items(second, *range(6)))
    builder.function("main", (x, *banks[0], *banks[1]), outputs)
    compiled = Compiler().compile(fm.verify_module(builder.build(entry="main"))).module
    restored = fm.load_module(fm.emit_module(compiled, tmp_path / "reusable.py"))
    assert restored.semantic_hash == compiled.semantic_hash
    runtime = load(write_artifact(restored, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    generator = torch.Generator().manual_seed(1314)
    values, allocations, ordered_outputs = {}, [], []
    for spec in runtime.external_arguments:
        kind = fm.logical_type(compiled.node_map[spec["value"]].type)
        shape = tuple(d.fixed_value for d in kind.shape) + tuple(getattr(kind.dtype, "lanes", ()))
        if spec["role"] == "result":
            storage = torch.full((prod(shape) + 32,), 123., dtype=torch.bfloat16, device="cuda")
            values[spec["value"]] = storage[16:-16].reshape(shape)
            allocations.append(storage)
            ordered_outputs.append(values[spec["value"]])
        else:
            values[spec["value"]] = (torch.randint(-4, 4, shape, generator=generator).float() / 128).bfloat16().cuda()
    try:
        arguments = tuple(values[spec["value"]] for spec in runtime.external_arguments)
        GeneratedTirCallGraphModule.prepare(runtime, *arguments)
        GeneratedTirCallGraphModule.run_into(runtime, *arguments)
        torch.cuda.synchronize()
        source = values["x"].cpu().reshape(1, 512).float()
        expected = []
        for label in ("a", "b"):
            fields = tuple((source @ values[f"{label}{name}"].cpu().reshape(-1, 512).float().T).bfloat16()
                           for name in ("q", "k", "v", "b", "f", "g"))
            expected.extend(fields)
            source = fields[4].float()
        for actual, reference, storage in zip(ordered_outputs, expected, allocations, strict=True):
            torch.testing.assert_close(actual.cpu().reshape(reference.shape), reference, rtol=0, atol=0)
            assert bool(torch.all(storage[:16] == 123.)) and bool(torch.all(storage[-16:] == 123.))
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
