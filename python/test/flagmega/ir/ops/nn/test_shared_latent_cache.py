# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from dataclasses import replace
from itertools import permutations
import pytest, torch
from triton.flagmega import ir as fm, pattern_match as pm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig, create_paged_attention_state
from python.test.flagmega.ir.ops.nn._shared_latent_cache_fixtures import SharedModule, physical


@pytest.mark.parametrize("layout", tuple(permutations(("seq", "head", "dim"))))
@pytest.mark.parametrize("lanes", [(), (2, 4), (16,)])
@pytest.mark.parametrize("advance", [False, True])
def test_shared_latent_preserves_the_whole_state_without_concat(layout, lanes, advance):
    config = PagedAttentionStateConfig(3, 4, 32, block_size=4, num_blocks=4)
    source = SharedModule(config, shared_kv=None, lanes=lanes, layout=layout, advance=advance).build()
    fused = SharedModule(config, lanes=lanes, layout=layout, advance=advance).build()
    assert any(n.op == "tensors.concat" for n in source.nodes)
    assert not any(
        n.op in {"tensors.concat", "tensors.slice", "tensors.pack", "tensors.unpack", "tensors.cast"}
        for n in fused.nodes
    )
    semantic, value = physical(config, 3, lanes, layout)
    states = [create_paged_attention_state(config) for _ in range(2)]
    for graph, state in zip((source, fused), states, strict=True):
        state.kv_caches.fill_(-9)
        state.seq_lens.fill_(3)
        state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], dtype=torch.int32))
        result = TorchEvaluator(DictWeightResolver({})).run(graph, {"slots": value, "state": state})[0][0]
        assert result is state
        expected = torch.full_like(state.kv_caches, -9)
        for row in range(3):
            page = (2, 0, 3, 1)[(3 + row) // 4]
            for kind in range(2):
                expected[page, 1, kind, (3 + row) % 4].copy_(semantic[row].reshape(4, 4, 8))
        torch.testing.assert_close(state.kv_caches, expected, rtol=0, atol=0)
        assert (
            state.seq_lens.item() == 3 + 3 * advance
            and state.slot_mapping.item() == 3
            and state.query_start_loc.tolist() == [0, 3]
        )
    for field in ["kv_caches", "seq_lens", "slot_mapping", "query_start_loc", "block_table"]:
        assert torch.equal(getattr(states[0], field), getattr(states[1], field))


def test_shared_mode_roundtrip_patterns_cost_and_old_default(tmp_path):
    config = PagedAttentionStateConfig(3, 4, 32)
    original = SharedModule(config, shared_kv=None, lanes=(2, 4)).build()
    explicit = SharedModule(config, shared_kv=False, lanes=(2, 4)).build()
    assert original.semantic_hash == explicit.semantic_hash
    for shared in (False, True):
        graph = SharedModule(config, shared_kv=shared, lanes=(2, 4)).build()
        restored = fm.load_module(fm.emit_module(graph, tmp_path / f"{shared}.py"))
        assert restored.semantic_hash == graph.semantic_hash
        n = restored.node_map["update"]
        for requested, expected in [(None, True), (False, not shared), (True, shared)]:
            assert (
                bool(
                    pm.try_match_root(n, pm.F.nn.is_update_paged_attention_kv_cache_pair(shared_kv=requested), restored)
                )
                == expected
            )
        definition = fm.get_definition(n.op)
        factors = definition.cost_factors([restored.node_map[i] for i in n.inputs], n.attrs, n.type)
        assert factors.block_local_memory_load_bytes == 3 * 4 * 32 * 2 * (1 if shared else 2)
        assert factors.block_local_memory_store_bytes == 3 * 4 * 32 * 2 * 2
        assert not n.effect.is_pure and n.type == config.ref_type


@pytest.mark.parametrize("shared", [0, 1, "yes"])
def test_shared_mode_rejects_non_bool_values(shared):
    with pytest.raises(IRSchemaError, match="shared_kv"):
        SharedModule(PagedAttentionStateConfig(3, 4, 32), shared_kv=shared).build()


@pytest.mark.parametrize("kind", ["chip", "partial", "width", "dtype"])
def test_shared_mode_keeps_input_legality(kind):
    config = PagedAttentionStateConfig(3, 4, 32)
    graph = SharedModule(config).build()
    n = graph.node_map["update"]
    inputs = [graph.node_map[i] for i in n.inputs]
    t = inputs[0].type
    if kind in {"chip", "partial"}:
        b = fm.SBP.broadcast()
        mesh = fm.Placement((2, 2), "cb", "cb")
        t = fm.DistributedType(
            t,
            (b, b, fm.SBP.split_contiguous((0,))) if kind == "chip" else (b, b, b),
            mesh,
            partial=fm.SBP.partial((1,)) if kind == "partial" else None,
        )
    elif kind == "width":
        t = fm.tensor_type("bfloat16", (3, 4, 64))
    else:
        t = fm.tensor_type("float32", (3, 4, 32))
    inputs[0] = replace(inputs[0], type=t)
    with pytest.raises(IRSchemaError):
        fm.get_definition(n.op).infer_type(inputs, n.attrs)
