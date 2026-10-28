# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The normal artifact path must publish transformed FP8 codes and scales."""

import pytest
import torch
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from triton.flagmega.selection import apply_plan, override_plan

from python.test.flagmega.ir.ops.nn.test_requantize_block_scaled_fp8 import Requantized, _simple_weights
from python.test.flagmega.devices import is_nvidia_sm90


@pytest.mark.parametrize("variant", ["tir.block_fp8.simt", "tir.block_fp8.mma"])
@pytest.mark.parametrize("mode", ["rn", "reciprocal_multiply"])
def test_offline_requantized_weights_reach_real_gpu_consumer(tmp_path, variant, mode):
    if not is_nvidia_sm90():
        pytest.skip("SM90 CUDA is required")
    weights = _simple_weights()
    infos = {name: TensorInfo(name, DType.FLOAT8_E4M3FN if name == "weight" else DType.FLOAT32,
                               tuple(value.shape), "memory") for name, value in weights.items()}
    checkpoint = MemoryCheckpoint({}, infos, weights)
    compiler = Compiler()
    proposed = compiler.compile(Requantized(project=True, scale_division=mode).build(), stop_after="propose-tir").module
    assert not any(node.op in {"nn.requantize_block_scaled_fp8", "tensors.pack"} for node in proposed.nodes)
    assert any(node.op == "nn.requantize_block_scaled_fp8"
               for recipe in proposed.constant_recipes for node in recipe.nodes)
    choices = [(point.id, variant) for point in proposed.selection_points
               if any(candidate.id == variant for candidate in point.candidates)]
    assert choices
    selected = apply_plan(proposed, override_plan(proposed, choices))
    lowered = compiler.compile(selected).module
    path = write_artifact(lowered, tmp_path / "requantized", target="nvidia-sm90",
                          checkpoint=checkpoint, emit_executable=True)
    runtime = load(path, device="cuda:0")
    try:
        value = torch.zeros((1, 128), dtype=torch.bfloat16, device="cuda")
        value[0, 0] = 1.
        output = torch.empty_like(value)
        args = tuple(output if argument["role"] == "result" else value for argument in runtime.external_arguments)
        assert len(args) == 2  # weights are readonly assets, not runtime parameters
        GeneratedTirCallGraphModule.prepare(runtime, *args)
        GeneratedTirCallGraphModule.run_into(runtime, *args)
        torch.cuda.synchronize()
        expected = torch.zeros_like(output)
        expected[0, 0] = 1.375
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
