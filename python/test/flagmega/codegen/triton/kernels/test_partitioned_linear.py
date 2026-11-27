# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Ordinary compilation and executing packed, guarded partition projections."""

from math import prod

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

from python.test.flagmega.ir.ops.nn.test_partitioned_linear import graph


@pytest.mark.parametrize("k,variant", [(320, "mma"), (2048, "wgmma_local")])
def test_normal_compiler_and_emitted_checkpoint(tmp_path, k, variant):
    module = Compiler().compile(graph(k=k)).module
    dispatch, = (function.dispatch for function in module.kernel_definitions
                 if function.dispatch.semantic_op == "nn.partitioned_linear")
    assert dispatch.microkernel.variant == variant
    assert all(effect.scope == fm.MemoryAccessScope.CHIP for _, effect in dispatch.memory_effect_map.items())
    assert module.metadata["launch_contract"]["cooperative_grid"]
    path = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    assert f"# flagmega-kernel: partitioned_linear/{variant}" in (path / "generated_kernels.py").read_text()
    restored = fm.load_module(fm.emit_module(module, tmp_path / "compiled.py"))
    assert restored.semantic_hash == module.semantic_hash


@pytest.mark.parametrize("k,n,parts,packed", [
    (320, 83, 4, False), (2048, 83, 4, False), (4096, 80, 4, True), (2048, 4096, 1, True),
])
def test_artifact_gpu_preserves_rounded_parts_and_guards(tmp_path, k, n, parts, packed):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    module = Compiler().compile(graph(k=k, n=n, parts=parts, lanes=(2, 4) if packed else (),
                                      input_lanes=(2, 4) if packed else (),
                                      placement=fm.Placement((8, 16), "yx", "bb"))).module
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True), device="cuda:0")
    generator = torch.Generator().manual_seed(3168)
    values, guards = {}, []
    for spec in runtime.external_arguments:
        kind = fm.logical_type(module.node_map[spec["value"]].type)
        shape = tuple(d.fixed_value for d in kind.shape) + tuple(getattr(kind.dtype, "lanes", ()))
        if spec["role"] == "result":
            storage = torch.full((prod(shape) + 32,), 123., dtype=torch.bfloat16, device="cuda")
            values[spec["value"]] = storage[16:-16].reshape(shape)
            guards.append(storage)
        else:
            values[spec["value"]] = (torch.randint(-7, 8, shape, generator=generator).float() / 128).bfloat16().cuda()
    try:
        for mutation in range(3):
            if mutation:
                values["x"].add_(1 / 128)
            arguments = tuple(values[spec["value"]] for spec in runtime.external_arguments)
            GeneratedTirCallGraphModule.prepare(runtime, *arguments)
            GeneratedTirCallGraphModule.run_into(runtime, *arguments)
            torch.cuda.synchronize()
            x, w = values["x"].cpu().reshape(1, k).float(), values["w"].cpu().float()
            span = k // parts
            expected = sum((x[:, i:i + span] @ w[:, i:i + span].T).bfloat16().float()
                           for i in range(0, k, span)).bfloat16()
            result, = (values[spec["value"]] for spec in runtime.external_arguments if spec["role"] == "result")
            torch.testing.assert_close(result.cpu().reshape(expected.shape), expected, rtol=0, atol=0)
            assert all(bool(torch.all(storage[:16] == 123.)) and bool(torch.all(storage[-16:] == 123.)) for storage in guards)
            assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
