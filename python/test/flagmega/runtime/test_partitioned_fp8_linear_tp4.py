# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Run with torchrun on four idle SM90 GPUs; P ownership is observable."""

import json
import os
from dataclasses import replace
from math import prod

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime

from python.test.flagmega.codegen.triton.kernels.block_fp8.test_block_fp8_cooperative_wgmma import projection_target
from python.test.flagmega.ir.ops.nn.test_partitioned_block_scaled_linear import Graph as Reference


@pytest.fixture(scope="module")
def device_group():
    import torch
    import torch.distributed as dist

    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    dist.init_process_group("gloo")
    yield device
    dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("WORLD_SIZE") != "4", reason="requires torchrun with four SM90 GPUs")
@pytest.mark.parametrize("ownership", ("partition", "rows"))
@pytest.mark.parametrize("lanes", ((), (2, 4)))
@pytest.mark.parametrize("readonly_block", (False, True))
def test_partition_scale_coordinates_and_rounded_local_add(tmp_path, ownership, lanes, readonly_block, device_group):
    import torch
    import torch.distributed as dist

    placement, b = fm.Placement((4, 8, 16), "cyx", "cbb"), fm.SBP.broadcast()
    p = fm.SBP.split_contiguous((0,)) if ownership == "partition" else b
    n = fm.SBP.split_contiguous((0,)) if ownership == "rows" else b
    tokens, hidden, reduction, parts = 2, 4096 if readonly_block else 512, 2048, 4
    scalar = lambda name: fm.vector_type(name, lanes) if lanes else name
    roots = (fm.tensor_type(scalar("bfloat16"), (tokens, reduction // prod(lanes))),
             fm.tensor_type("float8_e4m3fn", (hidden, reduction)),
             fm.tensor_type("float32", (hidden // 128, 16)),
             fm.tensor_type(scalar("bfloat16"), (parts, tokens, hidden // prod(lanes))))
    kinds = (fm.DistributedType(roots[0], (b, p), placement), fm.DistributedType(roots[1], (n, p), placement),
             fm.DistributedType(roots[2], (b, b), placement), fm.DistributedType(roots[3], (p, b, n), placement))
    if readonly_block:
        axes = (1, 2) if ownership == "partition" else (0, 1, 2)
        kind = fm.DistributedType(roots[3], (p, b, fm.SBP.split_contiguous(axes, hidden // (128 if ownership == "partition" else 512))), placement)
        kinds = (*kinds[:3], kind)

    class Graph(fm.Module):
        def forward(self):
            inputs, operands = [], []
            for i, (name, root, kind) in enumerate(zip(("x", "weight", "scales", "routed"), roots, kinds)):
                if readonly_block and i in (1, 2):
                    value = self.weight(name, kind, source="memory", key=name, id=name)
                else:
                    value = self.input(name, root, id=name)
                    inputs.append(value)
                    value = fm.F.distributed.sharded_view(value, kind)
                operands.append(value)
            x, w, scales, routed = operands
            projected = fm.F.nn.partitioned_block_scaled_linear(x, w, scales, partitions=parts,
                activation_quantization="dynamic_fp8_sm90", output_lanes=lanes, name="projection")
            material = fm.F.distributed.force_boxing(projected, roots[3], name="material_parts")
            owned = fm.F.distributed.force_boxing(projected, kinds[3]) if readonly_block else projected
            combined = fm.F.math.add(owned, routed, name="rounded_local_add")
            wide = fm.F.tensors.cast(combined, scalar("float32"))
            if lanes:
                wide = fm.F.tensors.unpack(wide, axes=(-1,) * len(lanes))
            total = fm.F.math.reduce_sum(wide, axes=(0,), keep_dims=False, output_lanes=lanes)
            published = fm.F.distributed.force_boxing(total, fm.logical_type(total.type))
            result = fm.F.tensors.cast(published, scalar("bfloat16"), name="result")
            self.function("main", inputs, (material, result))

    device = device_group
    runtime = reference_runtime = None
    try:
        module = Compiler().compile(Graph(dialect="distributed", stage="frozen_constants", entry="main",
            metadata={"auto_distribution": {"placement": placement.to_data()}}).build()).module
        x_cpu = torch.empty((tokens, reduction), dtype=torch.bfloat16)
        row, column = torch.arange(hidden)[:, None], torch.arange(reduction)[None, :]
        weight = (((row * 5 + column * 13) % 9 - 4).float() / 8).to(torch.float8_e4m3fn)
        scales = torch.exp2(-4 - torch.arange(hidden // 128)[:, None] - torch.arange(16)[None, :] // 4).float()
        routed = ((torch.arange(parts * tokens * hidden).reshape(parts, tokens, hidden) % 11 - 5).float() / 8).bfloat16()
        class Checkpoint:
            def load_tensor(self, key, *, device="cpu"):
                return {"weight": weight, "scales": scales}[key].to(device)
        artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True,
                                  checkpoint=Checkpoint() if readonly_block else None)
        runtime = load(artifact, device=device)
        values = {"x": x_cpu.reshape(tokens, reduction // prod(lanes), *lanes).to(device),
                  "weight": weight.to(device), "scales": scales.to(device),
                  "routed": routed.reshape(parts, tokens, hidden // prod(lanes), *lanes).to(device)}
        guard_parts = torch.full((parts * tokens * hidden + 32,), 123, dtype=torch.bfloat16, device=device)
        guard_sum = torch.full((tokens * hidden + 32,), 123, dtype=torch.bfloat16, device=device)
        outputs = {"material_parts": guard_parts[16:-16].reshape(parts, tokens, hidden // prod(lanes), *lanes),
                   "result": guard_sum[16:-16].reshape(tokens, hidden // prod(lanes), *lanes)}
        arguments = tuple(outputs[spec["value"]] if spec["role"] == "result" else values[spec["value"]]
                          for spec in runtime.external_arguments)
        Runtime.prepare(runtime, *arguments)
        reference = Reference(tokens=tokens, n=hidden, k=reduction, input_lanes=lanes, output_lanes=lanes).build()
        evaluator = TorchEvaluator(DictWeightResolver({}))
        # Compare distributed ownership against the pre-existing GPU projection
        # schedule. Its FP8 WGMMA arithmetic is not bitwise identical to a CPU
        # FP32 dot for general nonuniform K inputs.
        reference_graph = Reference(tokens=tokens, n=hidden, k=reduction, explicit=True,
            input_lanes=lanes, output_lanes=lanes,
            kinds=tuple(fm.DistributedType(root, (b,) * root.rank, placement) for root in roots[:3])).build()
        reference_graph = replace(reference_graph, dialect="distributed", stage="frozen_constants",
                                  metadata={"auto_distribution": {"placement": placement.to_data()}})
        reference_compiler = Compiler(CompileOptions(placement=placement))
        reference_compiler.target = projection_target(cooperative=False, warps=8)
        reference_module = reference_compiler.compile(reference_graph).module
        reference_artifact = write_artifact(reference_module, tmp_path / "reference", target="nvidia-sm90",
                                            emit_executable=True)
        reference_runtime = load(reference_artifact, device=device)
        reference_output = torch.empty_like(outputs["material_parts"])
        reference_arguments = tuple(reference_output if spec["role"] == "result" else values[spec["value"]]
                                    for spec in reference_runtime.external_arguments)
        Runtime.prepare(reference_runtime, *reference_arguments)
        cpu_difference_counts = []
        for mutation in range(3):
            for part in range(parts):
                for token in range(tokens):
                    amplitude = (0 if mutation == 1 else
                                 (1 + part / 8 + token / 2) * (-1 if mutation == 2 else 1))
                    x_cpu[token, part * 512:(part + 1) * 512] = (
                        amplitude * (1 - 2 * (torch.arange(512) % 3 == 0).float()))
            values["x"].copy_(x_cpu.reshape(values["x"].shape))
            cpu_projected, = evaluator.run(reference, {"x": x_cpu.reshape(tokens, reduction // prod(lanes), *lanes),
                                                       "weight": weight, "scales": scales})
            Runtime.run_into(reference_runtime, *reference_arguments)
            torch.cuda.synchronize()
            projected = reference_output.cpu()
            cpu_difference_counts.append(int((projected != cpu_projected).sum()))
            expected = (projected.reshape(parts, tokens, hidden) + routed).float().sum(0).bfloat16()
            for output in outputs.values():
                output.fill_(float("nan"))
            dist.barrier()
            Runtime.run_into(runtime, *arguments)
            torch.cuda.synchronize()
            torch.testing.assert_close(outputs["material_parts"].cpu(), projected, rtol=0, atol=0)
            torch.testing.assert_close(outputs["result"].cpu().reshape(tokens, hidden), expected, rtol=0, atol=0)
            for guard in (guard_parts, guard_sum):
                assert bool((guard[:16] == 123).all() & (guard[-16:] == 123).all())
        assert runtime.resource_report["spill_bytes"] == 0
        assert reference_runtime.resource_report["spill_bytes"] == 0
        (tmp_path / "result.json").write_text(json.dumps({"passed": True, "rank": dist.get_rank(),
            "ownership": ownership, "lanes": lanes, "readonly_block": readonly_block, "mutations": 3, "guards": True,
            "reference": "pre-existing expanded GPU FP8 MMA projections", "cpu_difference_counts": cpu_difference_counts,
            "resources": runtime.resource_report, "source_sha256": runtime.codegen["source_sha256"]}, indent=2) + "\n")
    finally:
        if reference_runtime is not None:
            reference_runtime.close()
        if runtime is not None:
            runtime.close()
        dist.barrier()
