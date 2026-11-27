# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Run with torchrun on two/four idle SM90 GPUs to check real chip ownership."""

import json
import os
from math import prod

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime

from python.test.flagmega.codegen.triton.kernels.sparse_experts.test_complete_k_wgmma import compile_complete
from python.test.flagmega.codegen.triton.kernels.sparse_experts.test_cooperative_wgmma import (
    TensorCheckpoint,
    compile_stage,
    stage_graph,
    stage_values,
)


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


@pytest.mark.skipif(os.environ.get("WORLD_SIZE") not in ("2", "4"), reason="requires torchrun on two/four SM90 GPUs")
@pytest.mark.parametrize("task_n,packed,chip_rows,reduction", [(32, False, 512, 4096), (32, True, 512, 4096), (64, True, 640, 1536)])
def test_complete_k_chip_publication_matches_ordered_reference(tmp_path, task_n, packed, chip_rows, reduction, device_group):
    import torch
    import torch.distributed as dist

    size, device = dist.get_world_size(), device_group
    mesh = fm.Placement((size, 8, 16), "cyx", "cbb")
    rows = size * chip_rows
    graph = stage_graph(placement=mesh, packed=packed, readonly=True, publish_result=True,
                        rows=rows, reduction=reduction, experts=17)
    values = stage_values(torch, graph)
    shapes = [(1, 8, rows // 8, 8) if packed else (1, 8, rows), (128,)]
    expected = []
    for label in ("reference", "complete"):
        module = compile_stage(graph, mesh, True) if label == "reference" else compile_complete(graph, mesh, task_n, stages=3)
        artifact = write_artifact(module, tmp_path / label, target="nvidia-sm90", emit_executable=True,
                                  checkpoint=TensorCheckpoint(values))
        runtime = load(artifact, device=device)
        inputs = {name: value.to(device) for name, value in values.items() if not name.endswith("_weight")}
        storages, outputs = [], {}
        for spec in runtime.external_arguments:
            if spec["role"] != "result":
                continue
            shape = shapes[len(outputs)]
            dtype = torch.bfloat16 if not outputs else torch.float32
            storage = torch.full((prod(shape) + 64,), -123., device=device, dtype=dtype)
            storages.append(storage)
            outputs[spec["value"]] = storage[32:-32].reshape(shape)
        arguments = tuple({**inputs, **outputs}[spec["value"]] for spec in runtime.external_arguments)
        try:
            Runtime.prepare(runtime, *arguments)
            assert runtime.resource_report["spill_bytes"] == 0
            for mutation in range(3):
                x = (((torch.arange(8 * reduction, device=device) * 5 + mutation * 7) % 67 - 33).float() / 16)
                inputs["dispatched"].copy_(x.reshape(inputs["dispatched"].shape).bfloat16())
                if mutation == 1:
                    inputs["dispatched"].zero_()
                inputs["router_expert_ids"].copy_(((torch.arange(8, device=device) * 7 + mutation * 3) % 17).reshape(1, 8).int())
                inputs["gate_proj_scale"].copy_(values["gate_proj_scale"].to(device) * (1 + mutation / 32))
                inputs["up_input_scale"].copy_(values["up_input_scale"].to(device) + mutation / 32)
                for value in outputs.values():
                    value.fill_(float("nan"))
                dist.barrier()
                Runtime.run_into(runtime, *arguments)
                torch.cuda.synchronize()
                actual = tuple(value.cpu().clone() for value in outputs.values())
                if label == "reference":
                    expected.append(actual)
                else:
                    for old, new in zip(expected[mutation], actual, strict=True):
                        assert torch.isfinite(new).all()
                        assert torch.equal(old.view(torch.uint8), new.view(torch.uint8))
                assert all(bool((value[:32] == -123).all() & (value[-32:] == -123).all()) for value in storages)
                torch.testing.assert_close(actual[1], values["sentinel"] * 2, rtol=0, atol=0)
            (tmp_path / f"{label}.json").write_text(json.dumps({
                "passed": True, "rank": dist.get_rank(), "world_size": size, "task_n": task_n,
                "packed": packed, "rows_per_chip": chip_rows, "reduction": reduction,
                "mutations": 3, "bitwise": True, "guards": True, "publication": "all chip shards to every rank",
                "source_sha256": runtime.codegen["source_sha256"], "resources": runtime.resource_report}, indent=2) + "\n")
        finally:
            runtime.close()
            dist.barrier()
