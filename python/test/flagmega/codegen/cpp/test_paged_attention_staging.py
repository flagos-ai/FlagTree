# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Parallel attention requires uniform queries, GQA pairs and bounded shared storage."""
import ctypes
from copy import deepcopy
from types import SimpleNamespace
import pytest
from triton.flagmega.codegen.cpp import renderer
from triton.flagmega.codegen.cpp import toolchain_available
from triton.flagmega.codegen.cpp.build import build_cpp_package
from triton.flagmega.codegen.cpp.templates import CppTemplateRegistry


def _call():
    return {"family": "paged_attention_partial", "variant": "decode_t16", "token_tile": 16,
            "head_dim": 128, "num_query_heads": 16, "num_kv_heads": 8,
            "query_lanes": 8, "query_dim_stride": 8, "local_capacity": 4,
            "head": "(attention_state_offsets + shard_x * 4)",
            "query_token": "attention_state_offsets // 4",
            "semantic_attrs": {"layout": ["seq", "head", "dim"]},
            "inputs": [{"formal": "q", "buffers": [{"abi": {
                "scalar_dtype": "bfloat16", "logical_shape": (1, 16, 16)}}]},
                {"formal": "state", "buffers": [{"abi": {
                    "scalar_dtype": "bfloat16", "logical_shape": (16, 2, 2, 256, 8, 16),
                    "scalar_lane_count": 8,
                    "scalar_storage_strides": (1048576, 524288, 262144, 1024, 128, 8)}}]}],
            "outputs": [{"formal": "result_0", "buffers": [{"abi": {
                "local_capacity_shape": (1, 4, 1),
                "logical_coordinate_expressions": ("local_coord_0", "local_coord_1 + shard_coord_1 * 4", "local_coord_2")}}]}]}


def _select(call, budget=131072, warp=64):
    descriptor = {"render_calls": [call], "block_threads": 1024, "warp_size": warp}
    renderer._select_attention_staging(descriptor, SimpleNamespace(max_shared_memory_bytes=budget))
    return call


def test_decode_attention_selects_parallel_staging():
    call = _select(_call())
    assert call["cpp_attention_staging"]["heads"] == 4
    assert call["cpp_attention_staging"]["token_tile"] == 64
    assert call["cpp_attention_shared_bytes"] <= 131072


@pytest.mark.parametrize("change", [
    {"head_dim": 64}, {"num_query_heads": 8}, {"query_dim_stride": 16},
    {"local_capacity": 3}, {"variant": "mma_tma_smem_pipeline"},
    {"variant": "decode_t32", "token_tile": 32},
])
def test_unproven_geometry_keeps_generic_mapping(change):
    call = _call(); call.update(change)
    assert not _select(call).get("cpp_attention_staging")


def test_queries_cannot_diverge_at_cta_barriers():
    call = _call(); call["inputs"][0]["buffers"][0]["abi"]["logical_shape"] = (2, 16, 16)
    assert not _select(call).get("cpp_attention_staging")


def test_gqa_pair_cannot_cross_owner_origin():
    call = _call()
    call["outputs"][0]["buffers"][0]["abi"]["logical_coordinate_expressions"] = (
        "local_coord_0", "local_coord_1 + shard_coord_1 * 3", "local_coord_2")
    assert not _select(call).get("cpp_attention_staging")


def test_cache_storage_must_match_contiguous_scalar_addresses():
    call = _call(); call["inputs"][1]["buffers"][0]["abi"]["scalar_storage_strides"] = (1, 2, 3, 4, 5, 6)
    assert not _select(call).get("cpp_attention_staging")


def test_shared_budget_and_warp_are_selection_constraints():
    assert not _select(_call(), budget=65536).get("cpp_attention_staging")
    assert not _select(_call(), warp=32).get("cpp_attention_staging")


@pytest.fixture(scope="module")
def attention_runner(tmp_path_factory):
    if not toolchain_available("iluvatar-bi-v150"):
        pytest.skip("The COREX device compiler is required")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    call = _call()
    call.update(arguments=["q", "kv", "slot", "table", "stats", "acc"],
                token_tile=16, context_shard="shard_y", context_shards=4,
                layer_id="1", query_token="0", active="True", block_size=256,
                num_blocks=16, num_layers=2, scale=128 ** -0.5,
                query="q + attention_head * 128", kv_cache="kv", slot_mapping="slot",
                block_table="table", accumulator_dim_stride=1,
                partial_max="stats + shard_index * 4 + attention_state_offsets",
                partial_sum="stats + 64 + shard_index * 4 + attention_state_offsets",
                partial_accumulator="acc + (shard_index * 4 + attention_state_offsets) * 128")
    context = dict(block_threads=1024, warp_size=64, shared_arena_reuse=True,
                   distributed_entry=True, mesh_hierarchy=[4, 4], mesh_rank=2,
                   mesh_levels="bb", paged_attention_partial_variant="decode_t16",
                   argument_types={name: {"type": dtype, "pointer": True} for name, dtype in
                       [("q", "ftt::bfloat16_t"), ("kv", "ftt::bfloat16_t"),
                        ("slot", "long long"), ("table", "int"),
                        ("stats", "float"), ("acc", "float")]})
    registry = CppTemplateRegistry()
    code = "#include <ftt/ftt.h>\n#include <cuda_runtime.h>\n"
    code += "__shared__ ftt::bfloat16_t _ftt_sme_arena[65536];\n"
    for staged in (False, True):
        leaf = deepcopy(call)
        leaf["symbol"] = "staged" if staged else "original"
        if staged:
            _select(leaf)
        code += registry.render("kernels/paged_attention_partial/_function.cpp.jinja",
                                dict(context, render_calls=[leaf]))
    code += '''
extern "C" __global__ void attention_test(ftt::bfloat16_t *q,ftt::bfloat16_t *kv,
    long long *slot,int *table,float *stats,float *acc) {
    original(q,kv,slot,table,stats,acc);
    ftt::arch::cta_sync();
    staged(q,kv,slot,table,stats+128,acc+8192);
}
extern "C" int run(void *q,void *kv,void *slot,void *table,void *stats,void *acc) {
    attention_test<<<16,1024>>>((ftt::bfloat16_t*)q,(ftt::bfloat16_t*)kv,
        (long long*)slot,(int*)table,(float*)stats,(float*)acc);
    if (cudaGetLastError()!=cudaSuccess) return -1;
    return (int)cudaDeviceSynchronize();
}
'''
    directory = tmp_path_factory.mktemp("attention-staging")
    source = directory / "attention.cpp"
    source.write_text(code)
    library = build_cpp_package(source, directory / "attention.so", target="iluvatar-bi-v150")
    runner = ctypes.CDLL(str(library)).run
    runner.argtypes = [ctypes.c_void_p] * 6
    runner.restype = ctypes.c_int
    return runner


@pytest.mark.parametrize("context", [1, 17, 31, 32, 33, 127, 255, 256, 511, 1024, 2048])
@pytest.mark.parametrize("unaligned", [False, True])
def test_staged_attention_paged_tails_and_unaligned_cache_on_gpu(attention_runner, context, unaligned):
    torch = pytest.importorskip("torch")
    torch.manual_seed(41)
    q = torch.randn((16, 128), device="cuda", dtype=torch.bfloat16)
    storage = torch.randn(16 * 2 * 2 * 256 * 8 * 128 + int(unaligned),
                          device="cuda", dtype=torch.bfloat16)
    kv = storage[int(unaligned):].view(16, 2, 2, 256, 8, 128)
    assert (kv.data_ptr() % 16 != 0) == unaligned
    # Noncontiguous physical pages, including the legal page clamp domain.
    table = torch.tensor([3, 5, 7, 9, 11, 13, -1, 99, 2, 4, 6, 8, 10, 12, 14, 0],
                         device="cuda", dtype=torch.int32)
    slot = torch.tensor([context - 1], device="cuda", dtype=torch.int64)
    stats = torch.full((2, 2, 4, 16), float("nan"), device="cuda")
    acc = torch.full((2, 4, 16, 128), float("nan"), device="cuda")
    torch.cuda.synchronize()
    assert attention_runner(*(t.data_ptr() for t in (q, kv, slot, table, stats, acc))) == 0
    assert torch.equal(stats[1], stats[0])
    assert torch.equal(acc[1], acc[0])
    # An independent softmax oracle also checks head/page ownership and empty shards.
    for shard in range(4):
        tokens = torch.arange(context * shard // 4, context * (shard + 1) // 4, device="cuda")
        if not tokens.numel():
            assert torch.isneginf(stats[1, 0, shard]).all()
            assert (stats[1, 1, shard] == 0).all()
            assert (acc[1, shard] == 0).all()
            continue
        pages = table[tokens // 256].long().clamp(0, 15)
        keys = kv[pages, 1, 0, tokens % 256].float().repeat_interleave(2, dim=1)
        values = kv[pages, 1, 1, tokens % 256].float().repeat_interleave(2, dim=1)
        scores = (keys * q.float()).sum(-1) * (128 ** -0.5)
        probabilities = torch.softmax(scores, dim=0)
        expected = (probabilities[..., None] * values).sum(0)
        result = acc[1, shard] / stats[1, 1, shard, :, None]
        torch.testing.assert_close(result, expected, rtol=3e-5, atol=3e-5)
