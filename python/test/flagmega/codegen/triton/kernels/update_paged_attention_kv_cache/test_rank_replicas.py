# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A paged-cache RefType is an independent allocation on each device rank."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
from python.test.flagmega.devices import has_nvidia_sm90_devices


def _worker(artifact):
    import torch
    from triton.flagmega import ir as fm
    from triton.flagmega.artifacts import load_artifact
    from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
    from triton.flagmega.runtime import load

    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    _, module = load_artifact(artifact)
    runtime = load(artifact, device=device)
    try:
        config = PagedAttentionStateConfig(3, 4, 32, block_size=16, num_blocks=4)
        state = create_paged_attention_state(config, device=device)
        state.kv_caches.fill_(-9)
        state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], device=device, dtype=torch.int32))
        semantic = torch.arange(3 * 128, device=device).reshape(3, 4, 32).bfloat16()
        slots = semantic.reshape(3, 4, 4, 8)
        state_buffers = (state.kv_caches, state.query_start_loc, state.seq_lens, state.slot_mapping, state.block_table)
        binding = runtime.buffer_plan.function_map[module.entry]
        buffers = {}
        for value, names in binding.parameters:
            if isinstance(module.node_map[value].type, fm.RefType):
                buffers.update(zip(names, state_buffers, strict=True))
            else:
                buffers[names[0]] = slots
        for _, names in binding.outputs:
            buffers.update(zip(names, state_buffers, strict=True))
        arguments = [buffers[str(argument['buffer'])] for argument in runtime.external_arguments]
        runtime.prepare(*arguments)
        expected = state.kv_caches.clone()
        for position in (13, 31):
            state.seq_lens.fill_(position)
            runtime.run_into(*arguments)
            for row in range(3):
                page = (2, 0, 3, 1)[(position + row) // 16]
                expected[page, 1, 1, (position + row) % 16].copy_(semantic[row].reshape(4, 4, 8))
            torch.testing.assert_close(state.kv_caches, expected, rtol=0, atol=0)
            assert state.seq_lens.item() == position + 3
            assert state.slot_mapping.item() == position
            assert state.query_start_loc.tolist() == [0, 3]
    finally:
        runtime.close()


@pytest.mark.parametrize("hierarchy,levels", [((4, 2, 2), "cbb"), ((2, 4, 2), "bcb")])
def test_paged_cache_updates_each_rank_replica(tmp_path, hierarchy, levels):
    pytest.importorskip("torch")
    if not has_nvidia_sm90_devices(4):
        pytest.skip("Four SM90 CUDA devices are required")
    from triton.flagmega import ir as fm
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.evaluator import PagedAttentionStateConfig
    from python.test.flagmega.codegen.triton.kernels.update_paged_attention_kv_cache.test_packed_slots_runtime import (
        cache_update_module,
    )
    source = cache_update_module(
        PagedAttentionStateConfig(3, 4, 32, block_size=16, num_blocks=4),
        True, ("seq", "head", "dim"), True, True, tokens=3,
        placement=fm.Placement(hierarchy, "abc", levels),
    )
    module = Compiler().compile(source).module
    artifact = write_artifact(module, tmp_path / "cache", target="nvidia-sm90", emit_executable=True)
    result = subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
        str(Path(__file__).resolve()), str(artifact),
    ], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _worker(sys.argv[1])
