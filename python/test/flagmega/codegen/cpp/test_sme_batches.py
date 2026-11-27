# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SME FIFO waits preserve pending work and never read an unfinished tile."""
import ctypes

import pytest

from triton.flagmega.codegen.cpp import renderer, toolchain_available
from triton.flagmega.codegen.cpp.build import build_cpp_package
from triton.flagmega.codegen.cpp.templates import CppTemplateRegistry


@pytest.mark.parametrize("expression,expected", [
    ("((0) * 2048) + qkv_local_k_offsets * 1", (0, 1)),
    ("16 + qkv_local_k_offsets", (16, 1)),
    ("qkv_local_k_offsets * 2", (0, 2)),
    ("qkv_local_k_offsets // 8", None),
    ("qkv_local_k_offsets * qkv_local_k_offsets", None),
    ("4 + qkv_local_k_offsets", None),
])
def test_sme_source_walk_proves_all_addresses(expression, expected):
    assert renderer._affine_source_walk(expression, "qkv_local_k_offsets") == expected


@pytest.mark.parametrize("every_caller_has_barrier", [False, True])
def test_shared_exit_requires_every_enclosing_call_barrier(every_caller_has_barrier):
    call = {"symbol": "projection", "family": "dense_matmul",
            "variant": "packed_k_major_gemv_norm_stats"}
    first = {"kind": "function_call", "callee": "layer"}
    descriptor = {"render_calls": [call],
        "device_functions": [{"function": "layer", "schedule": {
            "consumer_events": [{"kind": "tir.kernel_call", "symbol": "projection"}]}}],
        "entry_events": [first, {"kind": "barrier"}, first.copy()]}
    if every_caller_has_barrier:
        descriptor["entry_events"].append({"kind": "barrier"})
    renderer._mark_shared_exit_barriers(descriptor)
    assert bool(call.get("cpp_shared_exit_barrier_in_schedule")) == every_caller_has_barrier


def _render_pipeline(batch, concurrency, projections, groups=128):
    return CppTemplateRegistry().render("kernels/_sme_batches.cpp.jinja", {
        "sme_issuer_lane": "lane", "sme_buffer": "buf", "sme_warp": "warp",
        "sme_goff": "goff", "sme_groups": groups,
        "sme_concurrency": concurrency, "sme_lane": 16, "sme_stride": "512LL",
        "sme_act": "activation", "sme_run": "run", "sme_projections": projections,
        "sme_sources": [f"source{i}" for i in range(projections)],
        "sme_accumulators": [f"acc{i}" for i in range(projections)],
        "sme_batch": batch, "chunk": 8, "run_load": "load_run",
    })


@pytest.mark.parametrize("batch,concurrency,projections", [
    (1, 1, 2), (2, 1, 2), (2, 4, 1), (2, 8, 1), (3, 4, 1),
])
def test_sme_wait_retains_one_pending_batch(batch, concurrency, projections):
    source = _render_pipeline(batch, concurrency, projections)
    prologue, loop = source.split("for (ftt::dim_t ftt_sme_group", 1)
    # A full drain after every consumed tile loses issue depth. Complete only
    # the oldest batch, retaining the known count of newer FIFO transactions.
    assert prologue.count("sme_copy_block_from") == 2 * batch * projections
    assert f"sme_wait_pending<{batch * projections}>()" in prologue
    assert f"sme_wait_pending<{batch * projections}>()" in loop
    assert "sme_wait_pending<0>()" in loop
    assert "sme_drain()" not in source
    assert "cta_sync" not in source


def _render_payload(capacity, symbol="projection", sme=True, packed_capacity=None):
    packed_capacity = packed_capacity or (capacity + 7) // 8 * 8
    call = {"family": "dense_matmul", "variant": "packed_k_major_gemv",
        "symbol": symbol, "arguments": ["weights", "acts", "output"],
        "source": "acts", "weight": "weights", "result": "output",
        "local_m_capacity": 1, "local_n_capacity": capacity, "local_k_capacity": 2048,
        "weight_k_lane": 16, "weight_block_n": 8, "weight_block_k": 16,
        "weight_group_stride": packed_capacity * 16, "k_streams": 4,
        "global_n": "dense_local_n_offsets", "global_k": "dense_local_k_offsets",
        "weight_offset": f"(dense_global_k // 16) * {packed_capacity * 16} + dense_global_n * 16 + dense_global_k % 16",
        "source_offset": "dense_local_k_offsets", "result_offset": "dense_local_n_offsets",
        "result_active": "True", "source_active": "True", "k_predicate_redundant": True}
    context = {"render_calls": [call], "argument_types": {
        name: {"type": dtype, "pointer": True} for name, dtype in
        [("weights", "ftt::bfloat16_t"), ("acts", "ftt::bfloat16_t"), ("output", "float")]},
        "mesh_hierarchy": [1], "distributed_entry": False,
        "block_threads": 1024, "warp_size": 64, "warps": 16,
        "sme": sme, "sme_slots": 4, "sme_pad": 0,
        "run_load": "load_run", "tiled_depth": 4}
    return CppTemplateRegistry().render("kernels/dense_matmul/packed_k_major_gemv.cpp.jinja", context)


@pytest.mark.parametrize("capacity,overlaps", [(549, True), (792, True),
    (1023, False), (1024, False), (1032, False), (9496, True)])
def test_scalar_tail_only_uses_complete_idle_warps(capacity, overlaps):
    source = _render_payload(capacity)
    assert ("else if (dense_n_start" in source) == overlaps
    # The scalar tail always keeps its four K streams and 16-lane fold.
    assert "float dense_partial[4]" in source
    assert "lane_group_reduce_sum<16>" in source


def test_sme_payload_requires_aligned_group_byte_offsets():
    source = _render_payload(549, packed_capacity=549)
    assert "sme_copy_block_from" not in source


@pytest.mark.skipif(not toolchain_available("iluvatar-bi-v150"),
                    reason="The COREX device compiler is required")
@pytest.mark.parametrize("capacity,padded", [(549, True), (792, True),
    (1023, True), (1032, True), (549, False)])
def test_scalar_tail_overlap_matches_canonical_arithmetic_on_gpu(tmp_path, capacity, padded):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    packed_capacity = (capacity + 7) // 8 * 8 if padded else capacity
    code = """
#include <ftt/ftt.h>
#include <cuda_runtime.h>
__shared__ ftt::bfloat16_t _ftt_sme_arena[65536];
""" + _render_payload(capacity, "scalar", False, packed_capacity) + _render_payload(capacity, "staged", True, packed_capacity) + f"""
FTT_GLOBAL void launch(ftt::bfloat16_t *w,ftt::bfloat16_t *a,float *o) {{
    scalar(w,a,o);
    ftt::arch::cta_sync();
    staged(w,a,o+{capacity});
}}
extern "C" int run(void *w,void *a,void *o) {{
    launch<<<1,1024>>>((ftt::bfloat16_t*)w,(ftt::bfloat16_t*)a,(float*)o);
    if (cudaGetLastError()!=cudaSuccess) return -1;
    return (int)cudaDeviceSynchronize();
}}
"""
    source = tmp_path / "tail.cpp"
    source.write_text(code)
    library = build_cpp_package(source, tmp_path / "tail.so", target="iluvatar-bi-v150")
    runner = ctypes.CDLL(str(library)).run
    runner.argtypes = [ctypes.c_void_p] * 3
    runner.restype = ctypes.c_int
    torch.manual_seed(401)
    weights = torch.randint(-2, 3, (128, packed_capacity, 16), device="cuda").to(torch.bfloat16)
    acts = torch.randint(-2, 3, (128, 16), device="cuda").to(torch.bfloat16)
    expected = (weights[:, :capacity].float() * acts.float()[:, None, :]).sum(dim=(0, 2))
    output = torch.empty((2, capacity), device="cuda")
    for _ in range(3):
        output.fill_(float("nan"))
        torch.cuda.synchronize()
        assert runner(weights.data_ptr(), acts.data_ptr(), output.data_ptr()) == 0
        assert torch.equal(output[0], expected)
        assert torch.equal(output[1], output[0])
    # Nonintegral BF16 operands expose a changed summation order in the tail.
    # Full SME tiles have their own packed-dot arithmetic; compare the scalar
    # tail directly against its original four streams and 16-lane fold.
    weights.copy_(torch.randn_like(weights))
    acts.copy_(torch.randn_like(acts))
    output.fill_(float("nan"))
    torch.cuda.synchronize()
    assert runner(weights.data_ptr(), acts.data_ptr(), output.data_ptr()) == 0
    tail_start = capacity // 32 * 32
    assert torch.equal(output[1, tail_start:], output[0, tail_start:])


@pytest.mark.parametrize("capacity", [37, 1024])
def test_sme_large_payload_arena(monkeypatch, capacity):
    monkeypatch.setattr(renderer, "CPP_USE_SME", True)
    monkeypatch.setattr(renderer, "CPP_SME_SLOTS", 4)
    monkeypatch.setattr(renderer, "CPP_SME_PAD", 0)
    descriptor = {"render_calls": [{"family": "dense_matmul",
        "local_n_capacity": capacity, "local_k_capacity": 2048,
        "weight_k_lane": 16, "weight_group_stride": 32768}]}
    assert renderer._sme_arena_elems(descriptor, 16) >= 16 * 4 * 512


def test_sme_qkv_only_arena(monkeypatch):
    monkeypatch.setattr(renderer, "CPP_USE_SME", True)
    monkeypatch.setattr(renderer, "CPP_SME_SLOTS", 4)
    monkeypatch.setattr(renderer, "CPP_SME_PAD", 0)
    descriptor = {"render_calls": [{"family": "qkv_parallel_linear",
        "k_lane": 16, "local_k_capacity": 2048,
        "outputs": [{"local_n_capacity": 128}, {"local_n_capacity": 64}]}]}
    assert renderer._sme_arena_elems(descriptor, 16) >= 16 * 4 * 512


@pytest.mark.skipif(not toolchain_available("iluvatar-bi-v150"),
                    reason="The COREX device compiler is required")
def test_argmax_shared_arena_keeps_values_and_int64_indices_disjoint(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    call = {"family": "greedy_sample", "variant": "distributed_argmax",
        "symbol": "sample", "arguments": ["logits", "partial_max", "partial_index", "result"],
        "logits": "logits", "partial_max": "partial_max", "partial_index": "partial_index",
        "result": "result", "batch_capacity": 1, "result_capacity": 1,
        "local_capacity": 2061, "vocab_size": 2061, "active": "True",
        "logits_offset": "vocab_offsets", "logical_index": "vocab_offsets",
        "partial_batch_logical": "0", "result_logical_batch": "0",
        "result_active": "True", "result_writer": "True", "result_offset": "0"}
    context = {"render_calls": [call], "argument_types": {
        name: {"type": dtype, "pointer": True} for name, dtype in
        [("logits", "float"), ("partial_max", "float"),
         ("partial_index", "ftt::dim_t"), ("result", "int")]},
        "mesh_hierarchy": [1], "mesh_size": 1, "distributed_entry": False,
        "block_threads": 1024, "shared_arena_reuse": True}
    code = """
#include <ftt/ftt.h>
#include <cuda_runtime.h>
__shared__ ftt::bfloat16_t _ftt_sme_arena[6144];
""" + CppTemplateRegistry().render("kernels/greedy_sample/distributed_argmax.cpp.jinja", context) + """
FTT_GLOBAL void launch(float *l,float *p,ftt::dim_t *i,int *o) {sample(l,p,i,o);}
extern "C" int run(void *l,void *p,void *i,void *o) {
    launch<<<1,1024>>>((float*)l,(float*)p,(ftt::dim_t*)i,(int*)o);
    if (cudaGetLastError()!=cudaSuccess) return -1;
    return (int)cudaDeviceSynchronize();
}
"""
    source = tmp_path / "sample.cpp"
    source.write_text(code)
    library = build_cpp_package(source, tmp_path / "sample.so", target="iluvatar-bi-v150")
    runner = ctypes.CDLL(str(library)).run
    runner.argtypes = [ctypes.c_void_p] * 4
    runner.restype = ctypes.c_int
    logits = torch.zeros(2061, device="cuda")
    partial = torch.empty(1, device="cuda")
    index = torch.empty(1, device="cuda", dtype=torch.int64)
    output = torch.empty(1, device="cuda", dtype=torch.int32)
    for locations, value in [((2060,), 3.0), ((71, 1031), 5.0), ((9, 2055), float("nan"))]:
        logits.zero_()
        for location in locations:
            logits[location] = value
        torch.cuda.synchronize()
        assert runner(logits.data_ptr(), partial.data_ptr(), index.data_ptr(), output.data_ptr()) == 0
        assert output.item() == logits.argmax().item() == index.item()


@pytest.mark.skipif(not toolchain_available("iluvatar-bi-v150"),
                    reason="The COREX device compiler is required")
@pytest.mark.parametrize("batch,concurrency,projections,groups", [
    (1, 1, 2, 4), (2, 1, 2, 4), (2, 1, 2, 128),
    (2, 4, 1, 128), (2, 8, 1, 128),
    (3, 4, 1, 384),
])
def test_sme_fifo_payload_on_gpu(tmp_path, batch, concurrency, projections, groups):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    declarations = "\n".join(
        f"const auto source{i}=ftt::arch::sme_source_for("
        f"weights+({i}*16LL+warp)*{groups}LL*512LL,64);\nfloat acc{i}=0.0f;"
        for i in range(projections)
    )
    stores = "\n".join(f"output[{i}*1024+threadIdx.x]=acc{i};" for i in range(projections))
    shape = f"[2][512]" if projections == 2 else "[512]"
    code = f"""
#include <ftt/ftt.h>
#include <cuda_runtime.h>
FTT_GLOBAL void fifo(const ftt::bfloat16_t *weights,
                     const ftt::bfloat16_t *acts,float *output) {{
    __shared__ ftt::bfloat16_t buf[16][{2 * batch}]{shape};
    const int warp=threadIdx.x/64,lane=threadIdx.x%64;
    const int goff=warp%{concurrency},run=lane*8;
    const auto *activation=acts+goff*16+run%16;
    {declarations}
{_render_pipeline(batch, concurrency, projections, groups)}
    {stores}
}}
extern "C" int run(const void *weights,const void *acts,void *output) {{
    fifo<<<1,1024>>>((const ftt::bfloat16_t*)weights,
                     (const ftt::bfloat16_t*)acts,(float*)output);
    if (cudaGetLastError() != cudaSuccess) return -1;
    return (int)cudaDeviceSynchronize();
}}
"""
    source = tmp_path / "fifo.cpp"
    source.write_text(code)
    library = build_cpp_package(source, tmp_path / "fifo.so", target="iluvatar-bi-v150")
    runner = ctypes.CDLL(str(library)).run
    runner.argtypes = [ctypes.c_void_p] * 3
    runner.restype = ctypes.c_int
    # Small integer operands make every product and FP32 sum exact. Distinct
    # values expose stale slots and tail reads without a reduction tolerance.
    torch.manual_seed(173)
    weights = torch.randint(-2, 3, (projections, 16, groups, 64, 8),
                            device="cuda").to(torch.bfloat16)
    acts = torch.randint(-2, 3, (groups, 16), device="cuda").to(torch.bfloat16)
    output = torch.empty((projections, 16, 64), device="cuda", dtype=torch.float32)
    expected = torch.empty_like(output)
    for warp in range(16):
        for lane in range(64):
            expected[:, warp, lane] = (
                weights[:, warp, warp % concurrency::concurrency, lane].float()
                * acts[warp % concurrency::concurrency, (lane % 2) * 8:(lane % 2 + 1) * 8].float()
            ).sum(dim=(-2, -1))
    for _ in range(3):
        output.fill_(float("nan"))
        # The C shim uses the legacy stream; complete PyTorch's stream writes
        # before it reads the operands or replaces the sentinel output.
        torch.cuda.synchronize()
        assert runner(weights.data_ptr(), acts.data_ptr(), output.data_ptr()) == 0
        assert torch.equal(output, expected)


@pytest.mark.skipif(not toolchain_available("iluvatar-bi-v150"),
                    reason="The COREX device compiler is required")
@pytest.mark.parametrize("fused", [False, True])
def test_qkv_arena_reuse_waits_for_output_readers(tmp_path, fused):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    call = {
        "family": "qkv_parallel_linear", "variant": "packed_fused_gemv",
        "symbol": "projection", "arguments": ["weights", "acts", "output"],
        "source": "acts", "weight": "weights", "k_lane": 16, "n_lane": 4 if fused else 8,
        "cpp_fused_sme": fused,
        "local_k_capacity": 2048, "k_streams": 1, "k_predicate_redundant": True,
        "weight_owner_stride": 0, "weight_n_group_stride": 64 if fused else 128,
        "weight_k_group_stride": 4096, "source_offset": "qkv_local_k_offsets",
        "outputs": [{"result": f"output+{start}", "local_n_capacity": size,
                     "weight_group_start": start // (4 if fused else 8), "active": "True",
                     "result_offset": "qkv_local_n_offsets"}
                    for start, size in [(0, 128), (128, 64), (192, 64)]],
    }
    context = {
        "render_calls": [call], "argument_types": {
            name: {"type": typ, "pointer": True} for name, typ in
            [("weights", "ftt::bfloat16_t"), ("acts", "ftt::bfloat16_t"),
             ("output", "float")]},
        "mesh_hierarchy": [1], "distributed_entry": False,
        "block_threads": 1024, "warp_size": 64, "warps": 16,
        "sme": True, "sme_slots": 4, "sme_pad": 0,
        "qkv_tiled": False, "qkv_hoist": False, "qkv_pair": False,
        "run_load": "load_run",
    }
    projection = CppTemplateRegistry().render(
        "kernels/qkv_parallel_linear/packed_fused_gemv.cpp.jinja", context)
    # Make reducers lag behind non-reducers. The next output must not overwrite
    # their shared partials even when unrelated warps finish much earlier.
    projection = projection.replace("float qkv_accumulator = 0.0f;", """
        float qkv_accumulator = 0.0f;
#if defined(FTT_DEVICE_PASS)
        for (int delay=0; delay<1000; ++delay) asm volatile("sl_nop 1");
#endif
""")
    if fused:
        projection = projection.replace("float acc=0;", """
        float acc=0;
#if defined(FTT_DEVICE_PASS)
        for (int delay=0; delay<1000; ++delay) asm volatile("sl_nop 1");
#endif
""")
    code = """
#include <ftt/ftt.h>
#include <cuda_runtime.h>
__shared__ ftt::bfloat16_t _ftt_sme_arena[65536];
""" + projection + """
FTT_GLOBAL void launch(ftt::bfloat16_t *w,ftt::bfloat16_t *a,float *o) {
    projection(w,a,o);
""" + ("""
    // A following shared-memory consumer may finish before delayed reducers.
    if (threadIdx.x >= 128) {
        reinterpret_cast<float *>(_ftt_sme_arena)[threadIdx.x - 128] = -999.0f;
    }
""" if fused else "") + """
}
extern "C" int run(void *w,void *a,void *o) {
    launch<<<1,1024>>>((ftt::bfloat16_t*)w,(ftt::bfloat16_t*)a,(float*)o);
    if (cudaGetLastError()!=cudaSuccess) return -1;
    return (int)cudaDeviceSynchronize();
}
"""
    source = tmp_path / "qkv.cpp"
    source.write_text(code)
    library = build_cpp_package(source, tmp_path / "qkv.so", target="iluvatar-bi-v150")
    runner = ctypes.CDLL(str(library)).run
    runner.argtypes = [ctypes.c_void_p] * 3
    runner.restype = ctypes.c_int
    torch.manual_seed(315)
    weights = torch.randint(-2, 3, (128, 256, 16), device="cuda").to(torch.bfloat16)
    acts = torch.randint(-2, 3, (128, 16), device="cuda").to(torch.bfloat16)
    output = torch.empty(256, device="cuda")
    expected = (weights.float() * acts.float()[:, None, :]).sum((0, 2))
    for _ in range(3):
        output.fill_(float("nan"))
        torch.cuda.synchronize()
        assert runner(weights.data_ptr(), acts.data_ptr(), output.data_ptr()) == 0
        assert torch.equal(output, expected)
