# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""BF16 MMA score accumulation preserves the selected online schedule."""

from dataclasses import replace

import pytest
import torch
import triton
import triton.language as tl
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.runtime import load
from triton.flagmega.runtime.prepared import ResourceContract, _spill_resource_report
from triton.flagmega.targets import NvidiaSm90Target
from python.test.flagmega.devices import is_nvidia_sm90


def _graph(config, width, *, probability="online_bf16", heads=None, index_traversal="slots"):
    heads = config.num_kv_heads if heads is None else heads

    class Graph(fm.Module):
        def forward(self):
            query = self.input(
                "query", fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, heads, config.head_dim // 8)), id="query"
            )
            topk = self.input("topk", fm.tensor_type("int32", (1, width)), id="topk")
            state = self.input("state", config.ref_type, id="state")
            layer = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), 0)
            result = fm.F.nn.dsa_paged_attention(
                query,
                topk,
                state,
                layer,
                scale=1.0,
                layout=("seq", "head", "dim"),
                hidden_size=heads * config.head_dim,
                probability_rounding=probability,
                index_traversal=index_traversal,
            )
            self.function("main", (query, topk, state), (result,))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _compile(module, path, schedule):
    compiler = Compiler()
    base = compiler.target.triton_implementation_model
    catalog = replace(
        base,
        implementations=tuple(
            replace(value, parameters={**value.parameters, "decode_schedule": schedule, "compute_num_warps": 8})
            if value.id == "tir.dsa_paged_attention.decode"
            else value
            for value in base.implementations
        ),
    )
    compiler.target = NvidiaSm90Target(triton_implementation_model=catalog)
    compiled = compiler.compile(module).module
    restored = fm.load_module(fm.emit_module(compiled, path / "selected.py"))
    assert restored.semantic_hash == compiled.semantic_hash
    assert any(
        d.dispatch.microkernel.parameters.get("decode_schedule") == schedule for d in restored.kernel_definitions
    )
    artifact = write_artifact(compiled, path / "artifact", target="nvidia-sm90", emit_executable=True)
    return compiled, artifact


def _arguments(runtime, module, state, query, topk, output):
    buffers = {}
    for value, ids in runtime.buffer_plan.function_map[module.entry].parameters:
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
    for _, ids in runtime.buffer_plan.function_map[module.entry].outputs:
        buffers[ids[0]] = output
    return tuple(buffers[spec["buffer"]] for spec in runtime.external_arguments)


@pytest.mark.parametrize("schedule", ["simt", "bf16_mma"])
@pytest.mark.parametrize("slot", [0, 64])
def test_two_key_cancellation_uses_selected_dot_accumulation(tmp_path, schedule, slot):
    if not is_nvidia_sm90():
        pytest.skip("requires SM90")
    config = PagedAttentionStateConfig(1, 1, 32, num_blocks=2)
    module, artifact = _compile(_graph(config, 130), tmp_path, schedule)
    state = create_paged_attention_state(config, device="cuda")
    state.seq_lens.fill_(2)
    state.slot_mapping.fill_(1)
    query = torch.zeros((1, 1, 4, 8), dtype=torch.bfloat16, device="cuda")
    query.flatten()[0], query.flatten()[8], query.flatten()[16] = 4096, 1, -4096
    key = state.kv_caches[0, 0, 0, 0].flatten()
    key[0], key[8], key[16] = 4096, 1, 4096
    state.kv_caches[0, 0, 1, 0].fill_(1)
    topk = torch.full((1, 130), -1, dtype=torch.int32, device="cuda")
    topk[0, slot : slot + 2] = torch.tensor([0, 1], device="cuda")
    output = torch.empty_like(query)
    runtime = load(artifact, device="cuda:0")
    args = _arguments(runtime, module, state, query, topk, output)
    try:
        runtime.prepare(*args)
        runtime.run_into(*args)
        # BF16 MMA rounds 2**24 + 1 in its first K16 instruction before the
        # second instruction cancels 2**24. Tree reduction cancels first.
        expected = 0.5 if schedule == "bf16_mma" else torch.sigmoid(torch.tensor(1.0)).bfloat16().item()
        torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("slots", [(0, 1, 2), (64, 65, 66), (0, 65, 129), ()])
def test_mma_online_empty_tiles_and_partial_head_group(tmp_path, slots):
    if not is_nvidia_sm90():
        pytest.skip("requires SM90")
    config = PagedAttentionStateConfig(1, 1, 48, num_blocks=2)
    module, artifact = _compile(_graph(config, 130, heads=17), tmp_path, "bf16_mma")
    state = create_paged_attention_state(config, device="cuda")
    state.seq_lens.fill_(3)
    state.slot_mapping.fill_(2)
    query = torch.zeros((1, 17, 6, 8), dtype=torch.bfloat16, device="cuda")
    topk = torch.full((1, 130), -1, dtype=torch.int32, device="cuda")
    for logical, slot in enumerate(slots):
        topk[0, slot] = logical
    for logical, value in enumerate((1, 2, 4)):
        state.kv_caches[0, 0, 1, logical].fill_(value)
    output = torch.full_like(query, float("nan"))
    runtime = load(artifact, device="cuda:0")
    try:
        args = _arguments(runtime, module, state, query, topk, output)
        runtime.prepare(*args)
        runtime.run_into(*args)
        expected = torch.tensor(7 / 3 if slots else 0, dtype=torch.bfloat16).item()
        torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("count", [5, 65])
def test_compact_traversal_preserves_results_when_padding_moves_tokens(tmp_path, count):
    if not is_nvidia_sm90():
        pytest.skip("requires SM90")
    config = PagedAttentionStateConfig(1, 1, 32, num_blocks=2)
    module, artifact = _compile(_graph(config, 2175, heads=16, index_traversal="compact"), tmp_path, "bf16_mma")
    state = create_paged_attention_state(config, device="cuda")
    state.seq_lens.fill_(count)
    state.slot_mapping.fill_(count - 1)
    generator = torch.Generator().manual_seed(31)
    query = torch.randn((1, 16, 4, 8), generator=generator).bfloat16().cuda()
    keys = torch.randn((count, 4, 8), generator=generator).bfloat16().cuda()
    values = torch.randn((count, 4, 8), generator=generator).bfloat16().cuda()
    for position in range(count):
        block, offset = divmod(position, config.block_size)
        state.kv_caches[block, 0, 0, offset, 0].copy_(keys[position])
        state.kv_caches[block, 0, 1, offset, 0].copy_(values[position])
    topk = torch.full((1, 2175), -1, dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)
    runtime = load(artifact, device="cuda:0")
    try:
        args = _arguments(runtime, module, state, query, topk, output)
        runtime.prepare(*args)
        topk[0, :count] = torch.arange(count, device="cuda")
        runtime.run_into(*args)
        compact = output.clone()
        topk.fill_(-1)
        topk[0, :4] = torch.arange(4, device="cuda")
        topk[0, 2048 : 2048 + count - 4] = torch.arange(4, count, device="cuda")
        runtime.run_into(*args)
        torch.testing.assert_close(output, compact, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("count", [2, 3, 5, 6, 7, 65])
def test_compact_traversal_reads_each_valid_cache_entry(tmp_path, count):
    if not is_nvidia_sm90():
        pytest.skip("requires SM90")
    config = PagedAttentionStateConfig(1, 1, 32, num_blocks=2)
    module, artifact = _compile(_graph(config, 2175, heads=16, index_traversal="compact"), tmp_path, "bf16_mma")
    state = create_paged_attention_state(config, device="cuda")
    state.seq_lens.fill_(count)
    state.slot_mapping.fill_(count - 1)
    query = torch.zeros((1, 16, 4, 8), dtype=torch.bfloat16, device="cuda")
    for position in range(count):
        block, offset = divmod(position, config.block_size)
        state.kv_caches[block, 0, 1, offset, 0].fill_(position + 1)
    topk = torch.full((1, 2175), -1, dtype=torch.int32, device="cuda")
    pooled = count // 4 * 4
    topk[0, :pooled] = torch.arange(pooled, device="cuda")
    topk[0, 2048 : 2048 + count - pooled] = torch.arange(pooled, count, device="cuda")
    output = torch.empty_like(query)
    runtime = load(artifact, device="cuda:0")
    try:
        args = _arguments(runtime, module, state, query, topk, output)
        runtime.prepare(*args)
        runtime.run_into(*args)
        # Zero logits give equal probabilities, so every cache value must
        # contribute exactly once. Padding invariance alone cannot prove this.
        torch.testing.assert_close(output, torch.full_like(output, (count + 1) / 2), rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


def test_compact_traversal_preserves_duplicates_and_filters_future_indices(tmp_path):
    if not is_nvidia_sm90():
        pytest.skip("requires SM90")
    config = PagedAttentionStateConfig(1, 1, 32, num_blocks=2)
    module, artifact = _compile(_graph(config, 2175, heads=16, index_traversal="compact"), tmp_path, "bf16_mma")
    state = create_paged_attention_state(config, device="cuda")
    state.seq_lens.fill_(3)
    state.slot_mapping.fill_(2)
    for position in range(3):
        state.kv_caches[0, 0, 1, position, 0].fill_(position + 1)
    query = torch.zeros((1, 16, 4, 8), dtype=torch.bfloat16, device="cuda")
    topk = torch.full((1, 2175), -1, dtype=torch.int32, device="cuda")
    topk[0, [0, 64, 2048, 2049]] = torch.tensor([2, 99, 0, 2], dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)
    runtime = load(artifact, device="cuda:0")
    try:
        args = _arguments(runtime, module, state, query, topk, output)
        runtime.prepare(*args)
        runtime.run_into(*args)
        torch.testing.assert_close(output, torch.full_like(output, 7 / 3), rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


def test_compact_traversal_rejects_an_unsupported_physical_schedule(tmp_path):
    config = PagedAttentionStateConfig(1, 1, 32, num_blocks=2)
    with pytest.raises(CodegenError, match="compact index_traversal requires the bf16_mma schedule"):
        _compile(_graph(config, 2175, index_traversal="compact"), tmp_path, "simt")


@pytest.mark.parametrize(
    "schedule,probability,kv_heads",
    [
        ("unknown", "online_bf16", 1),
        ("bf16_mma", "normalized", 1),
        ("bf16_mma", "online_bf16", 2),
    ],
)
def test_unsupported_selected_schedule_rejected(tmp_path, schedule, probability, kv_heads):
    config = PagedAttentionStateConfig(1, kv_heads, 32, num_blocks=2)
    with pytest.raises(CodegenError, match="DSA.*(decode_schedule|bf16_mma)"):
        _compile(_graph(config, 130, probability=probability), tmp_path, schedule)


def test_resumed_mma_schedule_requires_sm90(tmp_path, monkeypatch):
    import triton.flagmega.codegen.triton.tir_package as package

    monkeypatch.setattr(package, "module_template_target", lambda module: ("iluvatar", "bi-v150"))
    config = PagedAttentionStateConfig(1, 1, 32, num_blocks=2)
    with pytest.raises(CodegenError, match="bf16_mma requires the NVIDIA SM90 target"):
        _compile(_graph(config, 130), tmp_path, "bf16_mma")


def test_mma_full_head_domain_stays_within_register_contract(tmp_path):
    if not is_nvidia_sm90():
        pytest.skip("requires SM90")
    config = PagedAttentionStateConfig(1, 1, 512, num_blocks=2)
    module, artifact = _compile(_graph(config, 2175, heads=64), tmp_path, "bf16_mma")
    state = create_paged_attention_state(config, device="cuda")
    state.seq_lens.fill_(2)
    state.slot_mapping.fill_(1)
    state.kv_caches[0, 0, 1, 0].fill_(1)
    state.kv_caches[0, 0, 1, 1].fill_(2)
    query = torch.zeros((1, 64, 64, 8), dtype=torch.bfloat16, device="cuda")
    topk = torch.full((1, 2175), -1, dtype=torch.int32, device="cuda")
    topk[0, :2] = torch.tensor([0, 1], device="cuda")
    output = torch.empty_like(query)
    runtime = load(artifact, device="cuda:0")
    try:
        args = _arguments(runtime, module, state, query, topk, output)
        runtime.prepare(*args)
        runtime.run_into(*args)
        torch.testing.assert_close(output, torch.full_like(output, 1.5), rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@triton.jit
def _attention_with_live_caller(saved, query, topk, s0, s1, s2, s3, s4, result, pool, FN: tl.constexpr):
    offsets = tl.arange(0, 32768)
    # The callee writes this same buffer. Its previous values must survive the
    # call, as live caller state does in a complete model's shared function.
    previous = tl.load(result + offsets)
    FN(query, topk, s0, s1, s2, s3, s4, result, pool)
    tl.store(saved + offsets, previous.to(tl.float32) + 1.0)


@pytest.mark.parametrize("index_traversal", ["slots", "compact"])
def test_mma_register_budget_includes_live_caller(tmp_path, index_traversal):
    if not is_nvidia_sm90():
        pytest.skip("requires SM90")
    config = PagedAttentionStateConfig(1, 1, 512, num_blocks=2)
    module, artifact = _compile(_graph(config, 2175, heads=64, index_traversal=index_traversal), tmp_path, "bf16_mma")
    state = create_paged_attention_state(config, device="cuda")
    state.seq_lens.fill_(2)
    state.slot_mapping.fill_(1)
    state.kv_caches[0, 0, 1, 0].fill_(1)
    state.kv_caches[0, 0, 1, 1].fill_(2)
    query = torch.zeros((1, 64, 64, 8), dtype=torch.bfloat16, device="cuda")
    topk = torch.full((1, 2175), -1, dtype=torch.int32, device="cuda")
    topk[0, :2] = torch.tensor([0, 1], device="cuda")
    output = torch.arange(32768, dtype=torch.float32, device="cuda").remainder(29).bfloat16().reshape_as(query)
    expected_saved = output.flatten().float() + 1
    saved = torch.empty_like(expected_saved)
    runtime = load(artifact, device="cuda:0")
    try:
        args = _arguments(runtime, module, state, query, topk, output)
        assert runtime.codegen["signature_arguments"] == [
            *(spec["name"] for spec in runtime.external_arguments),
            "block_local_data",
        ]
        pools = runtime.codegen["runtime_binding"]["pools"]
        assert len(pools) == 1 and pools[0]["name"] == "block_local_data"
        pool = torch.empty(pools[0]["nbytes"], dtype=torch.uint8, device="cuda")
        compiled = _attention_with_live_caller[(1,)](
            saved, *args, pool, runtime.kernel, num_warps=8, ptx_options="--minnctapersm=1"
        )
        torch.testing.assert_close(output, torch.full_like(output, 1.5), rtol=0, atol=0)
        torch.testing.assert_close(saved, expected_saved, rtol=0, atol=0)
        resources = _spill_resource_report(compiled, ResourceContract(compute_num_warps=8, resident_blocks_per_sm=1))
        assert resources["spill_bytes"] == 0, resources
    finally:
        runtime.close()
