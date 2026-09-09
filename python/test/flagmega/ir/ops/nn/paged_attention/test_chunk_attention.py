# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Each query row sees its own causal prefix, even after sequence advance."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, PagedAttentionStateConfig, TorchEvaluator
from triton.flagmega.evaluator import create_paged_attention_state
from triton.flagmega.ir.ops.ntt._paged_attention_split import pack_dim_from_scalar, unpack_dim_to_scalar


@pytest.mark.parametrize("tokens,past", ((1, 3), (3, 0), (5, 3)))
@pytest.mark.parametrize("packed,layout", ((False, ("seq", "head", "dim")), (True, ("head", "dim", "seq"))))
def test_chunk_attention_is_causal_and_layout_independent(tmp_path, tokens, past, packed, layout):
    config = PagedAttentionStateConfig(2, 2, 8, block_size=4, num_blocks=4)
    dtype = fm.vector_type("bfloat16", (2, 2)) if packed else fm.DType.BFLOAT16
    extents = {"seq": tokens, "head": 4, "dim": 2 if packed else 8}
    query_type = fm.tensor_type(dtype, tuple(extents[axis] for axis in layout))

    class Attention(fm.Module):

        def forward(self):
            q = self.input("query", query_type)
            state = self.input("state", config.ref_type)
            layer = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), 1)
            result = fm.F.nn.paged_attention(q, state, layer, scale=8**-.5, layout=layout, hidden_size=32)
            self.function("main", (q, state), (result, ))

    source = Attention(dialect="high_level", stage="imported", entry="main").build()
    fm.emit_module(source, tmp_path / "attention.py")
    module = fm.load_module(tmp_path / "attention.py")
    generator = torch.Generator().manual_seed(129)
    query = torch.randn((tokens, 4, 8), generator=generator).bfloat16()
    keys = torch.randn((past + tokens, 2, 8), generator=generator).bfloat16()
    values = torch.randn((past + tokens, 2, 8), generator=generator).bfloat16()
    state = create_paged_attention_state(config)
    state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], dtype=torch.int32))
    # Seed independently; do not use the cache update under test as an oracle.
    for index in range(past + tokens):
        block, offset = divmod(index, config.block_size)
        page = int(state.block_table[0, block])
        state.kv_caches[page, 1, 0, offset].copy_(keys[index].reshape(2, 1, 8))
        state.kv_caches[page, 1, 1, offset].copy_(values[index].reshape(2, 1, 8))
    state.slot_mapping.fill_(past)
    state.seq_lens.fill_(past + tokens)
    state.query_start_loc.copy_(torch.tensor([0, tokens], dtype=torch.int32))
    before = state.clone()
    expected = []
    heads = torch.tensor([0, 0, 1, 1])
    for row in range(tokens):
        length = past + row + 1
        k = keys[:length, heads].permute(1, 0, 2)
        v = values[:length, heads].permute(1, 0, 2)
        scores = torch.einsum("hd,hsd->hs", query[row].float(), k.float()) * (8**-.5)
        probs = torch.softmax(scores, dim=-1).bfloat16()
        expected.append(torch.einsum("hs,hsd->hd", probs, v))
    permutation = tuple(("seq", "head", "dim").index(axis) for axis in layout)
    physical_query = pack_dim_from_scalar(query.permute(permutation), dtype, layout.index("dim"))
    result = TorchEvaluator(DictWeightResolver({})).run(module, {"query": physical_query, "state": state})[0]
    scalar = unpack_dim_to_scalar(result, query_type, layout.index("dim"))
    torch.testing.assert_close(scalar, torch.stack(expected).permute(permutation), rtol=0, atol=0)
    for field in ("kv_caches", "query_start_loc", "seq_lens", "slot_mapping", "block_table"):
        torch.testing.assert_close(getattr(state, field), getattr(before, field), rtol=0, atol=0)
