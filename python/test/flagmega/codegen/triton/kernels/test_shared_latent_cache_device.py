# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
import pytest, torch
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Run
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig, create_paged_attention_state
from python.test.flagmega.ir.ops.nn.test_update_paged_attention_kv_cache_pair import state_arguments
from python.test.flagmega.ir.ops.nn._shared_latent_cache_fixtures import SharedModule, physical


@pytest.mark.parametrize(
    "lanes,layout,split_axis,cyclic,heads,advance,reusable",
    [
        ((), ("seq", "head", "dim"), None, False, 4, False, False),
        ((1,), ("seq", "head", "dim"), None, False, 4, True, False),
        ((4,), ("seq", "head", "dim"), "head", False, 4, True, False),
        ((2, 4), ("head", "dim", "seq"), "head", True, 7, False, False),
        ((16,), ("dim", "seq", "head"), "dim", False, 4, True, False),
        ((8,), ("head", "seq", "dim"), "seq", True, 4, True, False),
        ((8,), ("seq", "head", "dim"), "head", False, 4, True, True),
    ],
)
def test_shared_field_writes_preserve_all_cache_bytes_and_epoch(
    tmp_path, lanes, layout, split_axis, cyclic, heads, advance, reusable
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    config = PagedAttentionStateConfig(3, heads, 32, block_size=16, num_blocks=4)
    mesh = fm.Placement((2, 2), "yx", "bb")
    runtimes = {}
    states = {}
    args = {}
    semantic, slots = physical(config, 3, lanes, layout, device="cuda")
    # Keep source guards in scalar storage even when the semantic dim is a
    # packed, non-last axis. The physical view retains its actual lane order.
    storage = torch.full((slots.numel() + 32,), 123, dtype=slots.dtype, device="cuda")
    guarded = storage[16:-16].reshape(slots.shape)
    guarded.copy_(slots)
    slots = guarded
    expected = None
    try:
        for shared in (None, True):
            source = SharedModule(
                config,
                shared_kv=shared,
                lanes=lanes,
                layout=layout,
                advance=advance,
                placement=mesh,
                split_axis=split_axis,
                cyclic=cyclic,
                reusable=reusable,
            ).build()
            c = Compiler(CompileOptions(placement=mesh))
            m = c.compile(source).module
            path = write_artifact(m, tmp_path / str(shared), target="nvidia-sm90", emit_executable=True)
            text = (path / "generated_kernels.py").read_text()
            if shared:
                assert "flagmega-kernel: concat/" not in text and "flagmega-kernel: pack/" not in text
            r = load(path, device="cuda:0")
            runtimes[shared] = r
            state = create_paged_attention_state(config, device="cuda")
            states[shared] = state
            state.kv_caches.fill_(-9)
            state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device="cuda"))
            binding = r.buffer_plan.function_map[m.entry]
            buffers = {}
            for value, names in binding.parameters:
                if isinstance(m.node_map[value].type, fm.RefType):
                    buffers.update(zip(names, state_arguments(state), strict=True))
                else:
                    buffers[names[0]] = slots
            for _, names in binding.outputs:
                buffers.update(zip(names, state_arguments(state), strict=True))
            args[shared] = tuple(buffers[str(spec["buffer"])] for spec in r.external_arguments)
            Run.prepare(r, *args[shared])
            assert r.resource_report["spill_bytes"] == 0
        expected = states[True].kv_caches.clone()
        for mutation, position in enumerate((13, 15, 16, 31, 34)):
            slots.copy_(guarded + (0.125 * mutation))
            # Unpack via the independently constructed scalar fixture, not
            # via the implementation's physical offset expressions.
            if mutation:
                semantic.add_(0.125 * mutation)
            for shared, r in runtimes.items():
                states[shared].seq_lens.fill_(position)
                Run.run_into(r, *args[shared])
            torch.cuda.synchronize()
            for row in range(3):
                page = (2, 0, 3, 1)[(position + row) // 16]
                for kind in range(2):
                    expected[page, 1, kind, (position + row) % 16].copy_(semantic[row].reshape(heads, 4, 8))
            for state in states.values():
                assert torch.equal(state.kv_caches.view(torch.uint8), expected.view(torch.uint8))
                assert (
                    state.seq_lens.item() == position + 3 * advance
                    and state.slot_mapping.item() == position
                    and state.query_start_loc.tolist() == [0, 3]
                )
            for field in ["kv_caches", "seq_lens", "slot_mapping", "query_start_loc", "block_table"]:
                assert torch.equal(getattr(states[None], field), getattr(states[True], field))
            assert bool((storage[:16] == 123).all()) and bool((storage[-16:] == 123).all())
    finally:
        for r in runtimes.values():
            r.close()
