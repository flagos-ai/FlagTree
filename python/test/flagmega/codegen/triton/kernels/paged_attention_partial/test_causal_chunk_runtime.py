# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Causal query chunks through formal partial/combine and their physical ABIs."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.runtime import load


def chunk_attention(config, tokens, layout, token_split, local_input, *, query_heads=4, token_policy=None):
    placement = fm.Placement((2, 2, 2), "zyx", "bbb") if token_split else fm.Placement((2, 2), "yx", "bb")
    extents = {"seq": tokens, "head": query_heads, "dim": config.head_dim // 8}
    tensor = fm.tensor_type(fm.vector_type("bfloat16", (8, )), tuple(extents[axis] for axis in layout))
    by_kind = {
        "seq": (token_policy or fm.SBP.split_block_cyclic((1, ), 2)) if token_split else fm.SBP.broadcast(), "head":
        fm.SBP.split(fm.SplitStage((placement.rank - 1, ), fm.ContiguousSplit())), "dim": fm.SBP.broadcast()
    }
    distributed = fm.DistributedType(tensor, tuple(by_kind[axis] for axis in layout), placement)

    class Attention(fm.Module):

        def forward(self):
            value = self.input("query", tensor if local_input else distributed)
            query = fm.F.distributed.force_boxing(value, distributed) if local_input else value
            state = self.input("state", config.ref_type)
            layer = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), 1)
            partial = fm.F.ntt.paged_attention_partial(query, state, layer, scale=config.head_dim**-.5, layout=layout,
                                                       hidden_size=query_heads * config.head_dim,
                                                       split_hierarchy_axis=0, split_count=2)
            output = fm.F.ntt.paged_attention_combine(*(fm.F.tensors.get_item(partial, index) for index in range(3)),
                                                      layout=layout, hidden_size=query_heads * config.head_dim,
                                                      output_data_type=tensor.dtype, output_type=query.type,
                                                      split_hierarchy_axis=0, split_count=2)
            result = fm.F.builtin.tuple(fm.F.distributed.force_boxing(output, tensor))
            self.function("main", (value, state), (result, ))

    return Attention(dialect="distributed", stage="frozen_constants", entry="main",
                     metadata={"auto_distribution": {"placement": placement.to_data()}}).build()


@pytest.mark.parametrize("tokens,past,dimension,query_heads,block_size", (
    (3, 0, 32, 4, 16),
    (9, 13, 32, 4, 16),
    (2, 33, 128, 2, 256),
))
@pytest.mark.parametrize("layout,token_split,local_input", (
    (("seq", "head", "dim"), False, False),
    (("head", "dim", "seq"), False, True),
    (("seq", "head", "dim"), True, True),
))
def test_attention_chunk_uses_per_row_causal_length(tmp_path, tokens, past, dimension, query_heads, block_size, layout,
                                                    token_split, local_input):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    config = PagedAttentionStateConfig(3, 2, dimension, block_size=block_size, num_blocks=4)
    token_policy = fm.SBP.split(fm.SplitStage((1, ), fm.ContiguousSplit())) if tokens == 2 else None
    module = chunk_attention(config, tokens, layout, token_split, local_input, query_heads=query_heads,
                             token_policy=token_policy)
    fm.emit_module(module, tmp_path / "input.py")
    module = Compiler().compile(fm.load_module(tmp_path / "input.py")).module
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    variant = "mma_tma_smem_pipeline" if tokens == 2 and token_split else "decode"
    assert "paged_attention_partial/" + variant in (artifact / "generated_kernels.py").read_text()
    runtime = load(artifact, device="cuda:0")
    generator = torch.Generator().manual_seed(313)
    semantic_query = torch.randn((tokens, query_heads, dimension), generator=generator).bfloat16()
    keys = torch.randn((past + tokens, 2, dimension), generator=generator).bfloat16()
    values = torch.randn((past + tokens, 2, dimension), generator=generator).bfloat16()
    permutation = tuple(("seq", "head", "dim").index(axis) for axis in layout)
    query = semantic_query.reshape(tokens, query_heads, dimension // 8, 8).permute(*permutation, 3).contiguous().cuda()
    state = create_paged_attention_state(config, device="cuda")
    state.kv_caches.fill_(-9)
    state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device="cuda"))
    state.seq_lens.fill_(past + tokens)
    state.slot_mapping.fill_(past)
    state.query_start_loc.copy_(torch.tensor([0, tokens], dtype=torch.int32, device="cuda"))
    for position in range(past + tokens):
        page = (2, 0, 3, 1)[position // config.block_size]
        offset = position % config.block_size
        state.kv_caches[page, 1, 0, offset].copy_(keys[position].reshape(2, dimension // 8, 8))
        state.kv_caches[page, 1, 1, offset].copy_(values[position].reshape(2, dimension // 8, 8))
    saved_state = state.clone()
    output = torch.full_like(query, float("nan"))
    binding = runtime.buffer_plan.function_map[module.entry]
    state_buffers = (state.kv_caches, state.query_start_loc, state.seq_lens, state.slot_mapping, state.block_table)
    buffers = {}
    for value, names in binding.parameters:
        if isinstance(module.node_map[value].type, fm.RefType):
            buffers.update(zip(names, state_buffers, strict=True))
        else:
            buffers[names[0]] = query
    for _, names in binding.outputs:
        assert len(names) == 1
        buffers[names[0]] = output
    arguments = [buffers[str(arg["buffer"])] for arg in runtime.external_arguments]
    runtime.prepare(*arguments)
    for _ in range(2):
        output.fill_(float("nan"))
        runtime.run_into(*arguments)
        expected = []
        for row in range(tokens):
            length = past + row + 1
            k = keys[:length].repeat_interleave(query_heads // 2, 1).permute(1, 0, 2).float()
            v = values[:length].repeat_interleave(query_heads // 2, 1).permute(1, 0, 2).float()
            score = torch.einsum("hd,hsd->hs", semantic_query[row].float(), k) * (dimension**-.5)
            expected.append(torch.einsum("hs,hsd->hd", score.softmax(-1), v).bfloat16())
        expected = torch.stack(expected).reshape(tokens, query_heads, dimension // 8, 8).permute(*permutation, 3)
        # CPU FP32 softmax reference; pinned native/model acceptance stays exact.
        torch.testing.assert_close(output.cpu(), expected, rtol=.01, atol=.005)
    for field in ("kv_caches", "query_start_loc", "seq_lens", "slot_mapping", "block_table"):
        torch.testing.assert_close(getattr(state, field), getattr(saved_state, field), rtol=0, atol=0)
