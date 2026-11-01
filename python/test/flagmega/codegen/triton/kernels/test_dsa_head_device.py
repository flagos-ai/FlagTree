# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from math import prod
import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Run
from python.test.flagmega.ir.ops.nn._dsa_head_fixtures import compile_attention
from python.test.flagmega.ir.ops.nn._mla_head_fixtures import policies


def _arguments(runtime, module, state, query, topk, output):
    buffers = {}
    binding = runtime.buffer_plan.function_map[module.entry]
    for value, ids in binding.parameters:
        if isinstance(module.node_map[value].type, fm.RefType):
            buffers.update(
                zip(
                    ids,
                    (state.kv_caches, state.query_start_loc, state.seq_lens, state.slot_mapping, state.block_table),
                    strict=True,
                )
            )
        else:
            buffers[ids[0]] = {"query": query, "topk": topk}[module.node_map[value].attrs["name"]]
    for _, ids in binding.outputs:
        buffers[ids[0]] = output
    return tuple(buffers[spec["buffer"]] for spec in runtime.external_arguments)


@pytest.mark.parametrize(
    "schedule,traversal,cyclic,kv_heads,lanes",
    [
        ("bf16_mma", "compact", False, 1, (2, 4)),
        ("bf16_mma", "slots", True, 1, (4,)),
        ("simt", "slots", True, 1, (2, 4)),
        ("simt", "slots", False, 2, ()),
    ],
)
def test_attention_head_shards_keep_order_cache_mapping_and_guards(
    tmp_path, schedule, traversal, cyclic, kv_heads, lanes
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    mesh = fm.Placement((2, 2), "yx", "bb")
    heads = 7 if cyclic else 8
    hp, _ = policies(mesh, heads=heads, cyclic=cyclic)
    config = PagedAttentionStateConfig(1, kv_heads, 32, num_blocks=8)
    state = create_paged_attention_state(config, device="cuda")
    state.block_table.copy_(torch.tensor([[3, 0, 6, 2, 7, 1, 5, 4]], dtype=torch.int32, device="cuda"))
    state.seq_lens.fill_(67)
    state.slot_mapping.fill_(66)
    query_storage = torch.full((heads * 32 + 32,), 123, dtype=torch.bfloat16, device="cuda")
    query = query_storage[16:-16].reshape((1, heads, 32 // prod(lanes)) + lanes)
    topk_storage = torch.full((162,), -7, dtype=torch.int32, device="cuda")
    topk = topk_storage[16:-16].reshape(1, 130)
    outputs = {}
    storages = {}
    args = {}
    runtimes = {}
    try:
        for split in (False, True):
            m, p = compile_attention(
                tmp_path / str(split),
                mesh=mesh,
                heads=heads,
                head_policy=hp if split else None,
                schedule=schedule,
                traversal=traversal,
                lanes=lanes,
                config=config,
            )
            r = load(p, device="cuda:0")
            runtimes[split] = r
            storage = torch.full((heads * 32 + 32,), 123, dtype=torch.bfloat16, device="cuda")
            storages[split] = storage
            out = storage[16:-16].reshape(1, heads, 4, 8)
            outputs[split] = out
            args[split] = _arguments(r, m, state, query, topk, out)
            Run.prepare(r, *args[split])
            assert r.resource_report["spill_bytes"] == 0
        gen = torch.Generator().manual_seed(92211)
        for mutation in range(6):
            query.zero_() if mutation < 3 else query.copy_(torch.randn(query.shape, generator=gen).bfloat16())
            state.kv_caches.zero_()
            for token in range(67):
                physical = (3, 0, 6, 2, 7, 1, 5, 4)[token // 16]
                slot = token % 16
                for kv in range(kv_heads):
                    state.kv_caches[physical, 0, 0, slot, kv].copy_(torch.randn((4, 8), generator=gen).bfloat16())
                    value = (kv * 16 + token % 7 + 1) if mutation < 3 else None
                    if value is not None:
                        state.kv_caches[physical, 0, 1, slot, kv].fill_(value)
                    else:
                        state.kv_caches[physical, 0, 1, slot, kv].copy_(torch.randn((4, 8), generator=gen).bfloat16())
            topk.fill_(-1)
            if mutation == 0:
                slots = [0, 1, 2]
                tokens = [0, 1, 2]
            elif mutation == 1:
                slots = [0, 64, 129]
                tokens = [2, 99, 2]
            elif mutation == 2:
                slots = []
                tokens = []
            elif mutation == 3:
                slots = list(range(65))
                tokens = list(range(65))
            elif mutation == 4:
                slots = list(range(4)) + list(range(64, 125))
                tokens = list(range(65))
            else:
                slots = [0, 1, 63, 64, 65, 128, 129]
                tokens = [-3, 66, 12, 99, 0, 12, 66]
            if slots:
                topk[0, slots] = torch.tensor(tokens, dtype=torch.int32, device="cuda")
            cache_before = [
                v.clone()
                for v in (state.kv_caches, state.query_start_loc, state.seq_lens, state.slot_mapping, state.block_table)
            ]
            for split, r in runtimes.items():
                outputs[split].fill_(-17)
                Run.run_into(r, *args[split])
            torch.cuda.synchronize()
            assert torch.equal(outputs[True], outputs[False])
            if mutation < 3:
                for h in range(heads):
                    expected = 0 if mutation == 2 else (2 if mutation == 0 else 3) + (h // (heads // kv_heads)) * 16
                    assert bool((outputs[True][0, h] == expected).all())
            for original, current in zip(
                cache_before,
                (state.kv_caches, state.query_start_loc, state.seq_lens, state.slot_mapping, state.block_table),
            ):
                assert torch.equal(original, current)
            for storage in storages.values():
                assert bool((storage[:16] == 123).all()) and bool((storage[-16:] == 123).all())
            assert bool((query_storage[:16] == 123).all()) and bool((query_storage[-16:] == 123).all())
            assert bool((topk_storage[:16] == -7).all()) and bool((topk_storage[-16:] == -7).all())
    finally:
        for r in runtimes.values():
            r.close()
