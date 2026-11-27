# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Real compiled chunk writes, including page tails and repeated calls."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.runtime import load

from python.test.flagmega.codegen.triton.kernels.update_paged_attention_kv_cache.test_packed_slots_runtime import (
    cache_update_module, )


@pytest.mark.parametrize("tokens", (3, 9))
@pytest.mark.parametrize("packed,layout,split_heads,advance", (
    (False, ("seq", "head", "dim"), False, False),
    (True, ("seq", "head", "dim"), True, True),
    (True, ("head", "dim", "seq"), True, False),
))
def test_cache_chunk_writes_every_row(tmp_path, tokens, packed, layout, split_heads, advance):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    config = PagedAttentionStateConfig(3, 4, 32, block_size=16, num_blocks=4)
    source = cache_update_module(config, packed, layout, split_heads, advance, tokens=tokens)
    fm.emit_module(source, tmp_path / "input.py")
    module = Compiler().compile(fm.load_module(tmp_path / "input.py")).module
    artifact = write_artifact(module, tmp_path / "cache", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    state = create_paged_attention_state(config, device="cuda")
    state.kv_caches.fill_(-9)
    state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], device="cuda", dtype=torch.int32))
    semantic = torch.arange(tokens * 128, device="cuda").reshape(tokens, 4, 32).bfloat16()
    canonical = semantic.reshape(tokens, 4, 4, 8) if packed else semantic
    permutation = tuple(("seq", "head", "dim").index(axis) for axis in layout)
    slots = canonical.permute(*permutation, 3).contiguous() if packed else canonical.permute(*permutation).contiguous()
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
    arguments = [buffers[str(argument["buffer"])] for argument in runtime.external_arguments]
    runtime.prepare(*arguments)
    expected = state.kv_caches.clone()
    for position in (13, 31):
        state.seq_lens.fill_(position)
        runtime.run_into(*arguments)
        for row in range(tokens):
            page = (2, 0, 3, 1)[(position + row) // config.block_size]
            expected[page, 1, 1, (position + row) % config.block_size].copy_(semantic[row].reshape(4, 4, 8))
        torch.testing.assert_close(state.kv_caches, expected, rtol=0, atol=0)
        assert state.slot_mapping.item() == position
        assert state.seq_lens.item() == position + (tokens if advance else 0)
        assert state.query_start_loc.tolist() == [0, tokens]
