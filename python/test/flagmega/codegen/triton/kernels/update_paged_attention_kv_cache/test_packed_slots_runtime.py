# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Cache update alone must write every scalar lane and no neighboring slot."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.runtime import load


def cache_update_module(config, packed, layout, split_heads, advance):
    placement = fm.Placement((2, 2), "yx", "bb")

    class Update(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="distributed", entry="main",
                             metadata={"auto_distribution": {"placement": placement.to_data()}})

        def forward(self):
            extents = {"seq": 1, "head": config.num_kv_heads,
                       "dim": config.head_dim // (config.lanes if packed else 1)}
            dtype = fm.vector_type("bfloat16", (config.lanes,)) if packed else fm.DType.BFLOAT16
            tensor = fm.tensor_type(dtype, tuple(extents[axis] for axis in layout))
            policies = [fm.SBP.broadcast()] * 3
            if split_heads:
                policies[layout.index("head")] = fm.SBP.split(fm.SplitStage((0,), fm.ContiguousSplit()))
            value = self.input("slots", tensor)
            slots = fm.F.distributed.boxing(value, fm.DistributedType(tensor, tuple(policies), placement))
            state = self.input("state", config.ref_type)
            layer = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), 1)
            increment = fm.F.builtin.scalar_const(fm.tensor_type("bool", ()), advance)
            updated = fm.F.nn.update_paged_attention_kv_cache(
                slots, state, layer, increment, cache_kind="value", layout=layout)
            # Aggregate output intentionally selects the general flattened ABI.
            result = fm.F.builtin.tuple(updated)
            self.function("main", (value, state), (result,))

    return Update().build()


@pytest.mark.parametrize("packed,layout,split_heads,advance", [
    (False, ("seq", "head", "dim"), False, False),
    (True, ("seq", "head", "dim"), False, False),
    (True, ("seq", "head", "dim"), True, True),
    (True, ("head", "dim", "seq"), True, False),
])
def test_cache_update_preserves_complete_paged_storage(tmp_path, packed, layout, split_heads, advance):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    config = PagedAttentionStateConfig(3, 4, 32, num_blocks=4)
    module = Compiler().compile(cache_update_module(config, packed, layout, split_heads, advance)).module
    artifact = write_artifact(module, tmp_path / "cache", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    state = create_paged_attention_state(config, device="cuda")
    state.kv_caches.fill_(-9)
    state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], device="cuda", dtype=torch.int32))
    semantic = torch.arange(128, device="cuda", dtype=torch.float32).reshape(1, 4, 32).bfloat16()
    canonical = semantic.reshape(1, 4, 4, 8) if packed else semantic
    permutation = tuple(("seq", "head", "dim").index(axis) for axis in layout)
    slots = canonical.permute(*permutation, 3).contiguous() if packed else canonical.permute(*permutation).contiguous()
    binding = runtime.buffer_plan.function_map[module.entry]
    buffers = {}
    state_buffers = (state.kv_caches, state.query_start_loc, state.seq_lens, state.slot_mapping, state.block_table)
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
    for position in (31, 255, 256, 300):
        state.seq_lens.fill_(position)
        runtime.run_into(*arguments)
        page = (2, 0, 3, 1)[position // config.block_size]
        expected[page, 1, 1, position % config.block_size].copy_(semantic.reshape(4, 4, 8))
        torch.testing.assert_close(state.kv_caches, expected, rtol=0, atol=0)
        assert state.slot_mapping.item() == position
        assert state.seq_lens.item() == position + advance
