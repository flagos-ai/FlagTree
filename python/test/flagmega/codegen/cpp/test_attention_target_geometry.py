# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Generated attention on a CPU CTA simulator; CUDA compilation never launches."""

import ctypes
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from triton.flagmega.codegen.cpp.build import FTT_INCLUDE_DIR, build_cpp_package, toolchain_available
from triton.flagmega.codegen.cpp.templates import CppTemplateRegistry


def _attention_source(head_dim, threads=128, warp=32, *, symbol="attention", platform="nvidia", dtype="float"):
    call = dict(family="paged_attention_partial", variant="decode_t16", symbol=symbol,
                arguments=["q", "kv", "slot", "table", "stats", "acc"], token_tile=16,
                head_dim=head_dim, num_query_heads=4, num_kv_heads=2, local_capacity=2,
                head="attention_state_offsets + shard_x * 2", active="True",
                context_shard="shard_y", context_shards=2, layer_id="1", query_token="0",
                block_size=4, num_blocks=3, num_layers=2, scale=head_dim ** -0.5,
                query_lanes=1, query_dim_stride=2,
                query=f"q + attention_head * {head_dim * 2}", kv_cache="kv",
                slot_mapping="slot", block_table="table", accumulator_dim_stride=3,
                partial_max="stats + shard_y * 4 + attention_head",
                partial_sum="stats + 8 + shard_y * 4 + attention_head",
                partial_accumulator=f"acc + (shard_y * 4 + attention_head) * {head_dim * 3}")
    context = dict(render_calls=[call], block_threads=threads, warp_size=warp,
                   platform=platform, distributed_entry=True, mesh_hierarchy=[2, 2],
                   mesh_rank=2, mesh_levels="bb", paged_attention_partial_variant="decode_t16",
                   argument_types={name: {"type": dtype, "pointer": True} for name, dtype in
                                   [("q", dtype), ("kv", dtype), ("slot", "long long"),
                                    ("table", "int"), ("stats", "float"), ("acc", "float")]})
    return CppTemplateRegistry().render("kernels/paged_attention_partial/_function.cpp.jinja", context)


def _host_runner(directory, head_dim, threads=128, warp=32):
    if shutil.which("c++") is None:
        pytest.skip("A C++20 host compiler is required")
    body = _attention_source(head_dim, threads, warp, platform="corex" if warp == 64 else "nvidia")
    group = int(re.search(r"lane_group_reduce_sum<(\d+)>", body)[1])
    body = re.sub(r"__shared__ float (\w+)\[(\d+)\];", r"static bounded_array<\2> \1;", body)
    body = body.replace("ftt::arch::thread_id()", "test_thread")
    body = body.replace("ftt::arch::block_id()", "test_shard[0] * 2 + test_shard[1]")
    body = re.sub(r"ftt::arch::mesh_shard_id<(\d+)>\([^)]*\)", r"test_shard[\1]", body)
    body = body.replace("ftt::arch::cta_sync()", "test_cta.arrive_and_wait()")
    body = body.replace("ftt::arch::lane_group_reduce_sum<", "test_group_sum<")
    # NVIDIA receives per-runtime counters; this local attention function has
    # no grid barrier and the CPU runner supplies an unused nullptr.
    counter = "" if warp == 64 else ", nullptr"
    source = f'''#include <atomic>
#include <barrier>
#include <memory>
#include <thread>
#include <vector>
#include <cmath>
struct float4 {{ float x, y, z, w; }};
#include <ftt/ftt.h>
static constexpr int THREADS = {threads};
static thread_local int test_thread;
static int test_shard[2];
static std::barrier<> test_cta(THREADS);
static std::unique_ptr<std::barrier<>> test_groups[THREADS / {group}];
static float test_partial[THREADS];
static std::atomic<bool> test_bounds_failure;
template <int Count> struct bounded_array {{
    float value[Count]{{}};
    float &operator[](long long index) {{
        if (index < 0 || index >= Count) {{
            test_bounds_failure.store(true);
            static thread_local float dummy;
            return dummy;
        }}
        return value[index];
    }}
}};
template <int Width> float test_group_sum(float value) {{
    const int base = test_thread / Width * Width;
    auto &sync = *test_groups[test_thread / Width];
    for (int offset = Width / 2; offset; offset >>= 1) {{
        test_partial[test_thread] = value;
        sync.arrive_and_wait();
        value += test_partial[base + ((test_thread % Width) ^ offset)];
        sync.arrive_and_wait();
    }}
    return value;
}}
{body}
extern "C" int run(float *q, float *kv, long long *slot, int *table, float *stats, float *acc) {{
    test_bounds_failure.store(false);
    for (auto &group : test_groups) group = std::make_unique<std::barrier<>>({group});
    for (test_shard[0] = 0; test_shard[0] < 2; ++test_shard[0]) {{
        for (test_shard[1] = 0; test_shard[1] < 2; ++test_shard[1]) {{
            std::vector<std::thread> workers;
            for (int thread = 0; thread < THREADS; ++thread)
                workers.emplace_back([=] {{
                    test_thread = thread;
                    attention(q, kv, slot, table, stats, acc{counter});
                }});
            for (auto &worker : workers) worker.join();
        }}
    }}
    return test_bounds_failure.load() ? 1 : 0;
}}
'''
    path = directory / "attention.cpp"
    path.write_text(source)
    library = directory / "attention.so"
    subprocess.run(["c++", "-std=c++20", "-O1", "-shared", "-fPIC", "-pthread", "-DFTT_ARCH_ILUVATAR",
                    f"-I{FTT_INCLUDE_DIR}", str(path), "-o", str(library)], check=True, capture_output=True)
    function = ctypes.CDLL(str(library)).run
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_void_p] * 6
    return function


@pytest.mark.parametrize("head_dim,threads,warp", [(31, 128, 32), (65, 128, 32), (128, 128, 32),
                                                   (256, 128, 32), (257, 128, 32), (128, 256, 64)])
def test_generated_attention_covers_all_dimensions_and_owner_states(tmp_path, head_dim, threads, warp):
    runner = _host_runner(tmp_path, head_dim, threads, warp)
    rng = np.random.default_rng(31)
    queries = rng.standard_normal((4, head_dim)).astype(np.float32)
    q = np.full((4, head_dim * 2), -987.0, np.float32)
    q[:, ::2] = queries
    kv = rng.standard_normal((3, 2, 2, 4, 2, head_dim)).astype(np.float32)
    table = np.asarray([2, 0, 1, 0, -1, 99, 1, 2] * 3, dtype=np.int32)
    sentinel = np.float32(-321.5)
    for context in (0, 1, 17, 65):
        slot = np.asarray([context - 1], dtype=np.int64)
        stats_storage = np.full(16 + 16 + 16, sentinel, np.float32)
        acc_storage = np.full(16 + 2 * 4 * head_dim * 3 + 16, sentinel, np.float32)
        stats = stats_storage[16:-16].reshape(2, 2, 4)
        acc = acc_storage[16:-16].reshape(2, 4, head_dim, 3)
        status = runner(*(array.ctypes.data for array in (q, kv, slot, table, stats, acc)))
        assert status == 0, "generated shared partial indexing exceeded its declared array"
        assert np.all(stats_storage[:16] == sentinel) and np.all(stats_storage[-16:] == sentinel)
        assert np.all(acc_storage[:16] == sentinel) and np.all(acc_storage[-16:] == sentinel)
        assert np.all(acc[..., 1:] == sentinel), "partial accumulator stride was ignored"
        for context_shard in range(2):
            tokens = np.arange(context * context_shard // 2, context * (context_shard + 1) // 2)
            if not len(tokens):
                assert np.isneginf(stats[0, context_shard]).all()
                assert (stats[1, context_shard] == 0).all()
                assert (acc[context_shard, ..., 0] == 0).all()
                continue
            pages = np.clip(table[tokens // 4], 0, 2)
            keys = kv[pages, 1, 0, tokens % 4].repeat(2, axis=1)
            values = kv[pages, 1, 1, tokens % 4].repeat(2, axis=1)
            scores = (keys * queries[None]).sum(-1, dtype=np.float32) * np.float32(head_dim ** -0.5)
            maxima = scores.max(0)
            probabilities = np.exp(scores - maxima)
            sums = probabilities.sum(0, dtype=np.float32)
            expected = (probabilities[..., None] * values).sum(0, dtype=np.float32)
            np.testing.assert_allclose(stats[0, context_shard], maxima, rtol=3e-5, atol=3e-5)
            np.testing.assert_allclose(stats[1, context_shard], sums, rtol=3e-5, atol=3e-5)
            np.testing.assert_allclose(acc[context_shard, ..., 0] / sums[:, None],
                                       expected / sums[:, None], rtol=3e-5, atol=3e-5)


def test_nvidia_attention_source_uses_warp32_and_uniform_dimension_tiles():
    source = _attention_source(257)
    assert "lane_group_reduce_sum<32>" in source
    assert "lane_group_reduce_sum<64>" not in source
    assert "attention_dimension_base" in source
    assert "attention_dimension_active" in source
    assert "_ftt_sme_arena" not in source


def test_iluvatar_attention_retains_eight_token_groups_at_its_original_geometry():
    source = _attention_source(128, 1024, 64, platform="corex")
    assert "lane_group_reduce_sum<64>" in source
    assert "attention_token_group < 8LL" in source
    assert "token_lane += 8LL" in source
    assert "ftt_attention_value_partial[1024]" in source


def test_nvidia_attention_compiles_with_real_nvcc_without_launch(tmp_path):
    if not toolchain_available("nvidia-sm120"):
        pytest.skip("NVCC is required")
    source = "#include <ftt/ftt.h>\n"
    for dim, dtype in [(dim, "float") for dim in (31, 65, 128, 256, 257)] + [
            (dim, "ftt::bfloat16_t") for dim in (256, 257)]:
        suffix = f"{dim}_{'bf16' if dtype != 'float' else 'f32'}"
        source += _attention_source(dim, symbol=f"attention_{suffix}", dtype=dtype)
        source += f'''FTT_GLOBAL void probe_{suffix}({dtype} *q,{dtype} *kv,long long *slot,int *table,
            float *stats,float *acc,unsigned int *counters) {{
            attention_{suffix}(q,kv,slot,table,stats,acc,counters);
        }}\n'''
    path = tmp_path / "attention.cu"
    path.write_text(source)
    library = build_cpp_package(path, tmp_path / "attention.so", target="nvidia-sm120")
    assert library.stat().st_size > 0


@pytest.mark.parametrize("head_dim", [31, 128, 256, 257])
def test_bf16_attention_dimension_coverage_and_paged_tails_on_gpu(tmp_path, head_dim):
    """Device gate for the orchestrator/Validator; CPU assignment deselects it."""
    torch = pytest.importorskip("torch")
    if (not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0)
            or not toolchain_available("nvidia-sm120")):
        pytest.skip("An SM120 device and NVCC are required")
    source = "#include <ftt/ftt.h>\n"
    source += _attention_source(head_dim, dtype="ftt::bfloat16_t")
    source += '''FTT_GLOBAL void probe(ftt::bfloat16_t *q,ftt::bfloat16_t *kv,long long *slot,
        int *table,float *stats,float *acc) { attention(q,kv,slot,table,stats,acc,nullptr); }
extern "C" int run(void *q,void *kv,void *slot,void *table,void *stats,void *acc) {
    probe<<<4,128>>>((ftt::bfloat16_t*)q,(ftt::bfloat16_t*)kv,(long long*)slot,
        (int*)table,(float*)stats,(float*)acc);
    return (int)cudaGetLastError();
}
'''
    path = tmp_path / "attention.cu"
    path.write_text(source)
    library = build_cpp_package(path, tmp_path / "attention.so", target="nvidia-sm120")
    run = ctypes.CDLL(str(library)).run
    run.argtypes = [ctypes.c_void_p] * 6
    run.restype = ctypes.c_int
    generator = torch.Generator().manual_seed(31)
    q = torch.randn((4, head_dim * 2), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((3, 2, 2, 4, 2, head_dim), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    table = torch.tensor([2, 0, 1, 0, -1, 99, 1, 2] * 3, dtype=torch.int32, device="cuda")
    for context in (0, 1, 17, 65):
        slot = torch.tensor([context - 1], dtype=torch.int64, device="cuda")
        stats = torch.full((2, 2, 4), float("nan"), device="cuda")
        acc = torch.full((2, 4, head_dim, 3), -321.5, device="cuda")
        assert run(*(tensor.data_ptr() for tensor in (q, kv, slot, table, stats, acc))) == 0
        torch.cuda.synchronize()
        assert (acc[..., 1:] == -321.5).all()
        for shard in range(2):
            tokens = torch.arange(context * shard // 2, context * (shard + 1) // 2, device="cuda")
            if not tokens.numel():
                assert torch.isneginf(stats[0, shard]).all()
                assert (stats[1, shard] == 0).all() and (acc[shard, ..., 0] == 0).all()
                continue
            pages = table[tokens // 4].long().clamp(0, 2)
            keys = kv[pages, 1, 0, tokens % 4].float().repeat_interleave(2, dim=1)
            values = kv[pages, 1, 1, tokens % 4].float().repeat_interleave(2, dim=1)
            scores = (keys * q[:, ::2].float()).sum(-1) * (head_dim ** -0.5)
            expected = (torch.softmax(scores, dim=0)[..., None] * values).sum(0)
            torch.testing.assert_close(acc[shard, ..., 0] / stats[1, shard, :, None],
                                       expected, rtol=3e-5, atol=3e-5)
