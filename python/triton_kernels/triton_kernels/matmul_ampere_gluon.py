import argparse

import pytest
import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp, mma_v2

def is_ampere_or_newer():
    try:
        target = triton.runtime.driver.active.get_current_target()
    except RuntimeError:
        return False
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 8


_WARP = 32  # NVIDIA warp size (Ampere / H20)


@gluon.constexpr_function
def _mma_acc_layout(num_warps: gl.constexpr, element_bitwidth: gl.constexpr) -> gl.constexpr:
    # NVMMADistributedLayout(version=[2,0], warps_per_cta=[num_warps,1], instr_shape=[16,8])
    # warps 全沿 M 铺；instr_shape/version 由 m16n8k16 v2.0 钉死。
    return gl.NVMMADistributedLayout([2, 0], [num_warps, 1], [16, 8])

@gluon.constexpr_function
def _mma_smem_layouts(element_bitwidth: gl.constexpr):
    # A: transposed=False ; B: transposed=True (ldmatrix.trans)
    # swizzle_byte_width=128（最优），rank=2（2D GEMM，钉死）
    a = gl.NVMMASharedLayout(128, element_bitwidth, 2, False)
    b = gl.NVMMASharedLayout(128, element_bitwidth, 2, True)
    return a, b


@gluon.constexpr_function
def _default_blocked_layout(shape: gl.constexpr, num_warps: gl.constexpr) -> gl.constexpr:
    """A plain register blocked layout (mirrors the Triton gluon translator default)."""
    rank = len(shape)
    size_per_thread = [1 for _ in range(rank)]
    threads_per_warp = [1 for _ in range(rank)]
    threads_per_warp[rank - 1] = _WARP
    warps_per_cta = [1 for _ in range(rank)]
    warps_per_cta[0] = num_warps
    order = [i for i in range(rank - 1, -1, -1)]
    return gl.BlockedLayout(size_per_thread=size_per_thread, threads_per_warp=threads_per_warp,
                            warps_per_cta=warps_per_cta, order=order)


@gluon.constexpr_function
def _ptr_blocked_layout(block0: gl.constexpr, block1: gl.constexpr,
                        num_warps: gl.constexpr,
                        contig_dim: gl.constexpr,
                        cp_async_elem: gl.constexpr = 8) -> gl.constexpr:
    """2D cp.async pointer BlockedLayout for a tile of shape [block0, block1].

    `cp_async_elem` is the number of elements per 16-byte cp.async vector
    (8 for fp16, 16 for int8). `contig_dim` (0 or 1) selects the memory-contiguous
    dim: it carries the cp.async vectors and gets the fastest `order` index; the
    other dim is "slow". Per-dim block == size_per_thread * threads_per_warp *
    warps_per_cta, with threads_per_warp prod == 32 and warps_per_cta prod ==
    num_warps, so the tile is covered exactly once. For fp16 (block0, block1) ==
    (128, 128) and num_warps=4 this yields [16,8]/[4,8]/[2,2]/[1,0] (contig_dim=1)
    and [8,16]/[8,4]/[2,2]/[0,1] (contig_dim=0).
    """
    spt_fast = cp_async_elem
    if contig_dim == 1:
        block_slow, block_fast = block0, block1
    else:
        block_slow, block_fast = block1, block0
    assert block_fast % spt_fast == 0, \
        "contiguous block dim must be a multiple of cp_async_elem (16-byte cp.async)"
    fast_units = block_fast // spt_fast              # == tpw_fast * wpc_fast
    total_threads = _WARP * num_warps
    tile_area = block_slow * block_fast
    assert tile_area % total_threads == 0, \
        "tile area must be a multiple of 32 * num_warps"
    spt_area = tile_area // total_threads
    assert spt_area % spt_fast == 0, \
        "tile too small to vectorize the non-contiguous dim at 16-byte cp.async"
    spt_slow = spt_area // spt_fast
    # Prefer 8 threads on the contiguous dim (one 128-byte cache line per warp row);
    # fall back to smaller powers of two when divisibility does not allow it.
    candidates = [c for c in (8, 4, 2, 1)
                  if fast_units % c == 0 and num_warps % (fast_units // c) == 0]
    assert candidates, \
        f"no valid pointer-layout factorization for slow={block_slow}, fast={block_fast}, num_warps={num_warps}"
    tpw_fast = candidates[0]
    wpc_fast = fast_units // tpw_fast
    tpw_slow = _WARP // tpw_fast
    wpc_slow = num_warps // wpc_fast
    assert block_slow == spt_slow * tpw_slow * wpc_slow, "internal: slow-dim block mismatch"
    if contig_dim == 1:                               # [slow, fast], fast is dim1
        spt = [spt_slow, spt_fast]
        tpw = [tpw_slow, tpw_fast]
        wpc = [wpc_slow, wpc_fast]
        order = [1, 0]
    else:                                             # [fast, slow], fast is dim0
        spt = [spt_fast, spt_slow]
        tpw = [tpw_fast, tpw_slow]
        wpc = [wpc_fast, wpc_slow]
        order = [0, 1]
    return gl.BlockedLayout(size_per_thread=spt, threads_per_warp=tpw,
                            warps_per_cta=wpc, order=order)


@gluon.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                  GROUP_M: gl.constexpr, NUM_BUFFERS: gl.constexpr):
    """fp16 TN matmul kernel (Gluon/Ampere). See module docstring for semantics & launch."""
    # ---- CTA / tile selection (grouped grid, identical to the MACA reference) ----
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- Layouts (all constexpr) ----
    # fp16 MMA m16n8k16 -> fp32 accumulator; instr_shape is the instruction's [M,N] (K=16 implicit).
    EBW: gl.constexpr = 16                                   # fp16
    acc_layout: gl.constexpr = _mma_acc_layout(gl.num_warps(), EBW)          # warps_per_cta=[num_warps,1]
    KW: gl.constexpr = 32 // EBW                             # k_width = 2
    a_op: gl.constexpr = gl.DotOperandLayout(0, acc_layout, KW)
    b_op: gl.constexpr = gl.DotOperandLayout(1, acc_layout, KW)
    a_smem_layout: gl.constexpr = _mma_smem_layouts(EBW)[0]  # transposed=False
    b_smem_layout: gl.constexpr = _mma_smem_layouts(EBW)[1]  # transposed=True
    # Pointer tiles: block == tile, 16-byte (8 x fp16) cp.async vectors on the K dim.
    # `order` is fastest-first and must match memory contiguity, else cp.async falls back
    # to <16 bytes. Layouts derived from BLOCK_M/N/K (num_warps fixed at 4).
    a_ptr_layout: gl.constexpr = _ptr_blocked_layout(BLOCK_M, BLOCK_K, gl.num_warps(), 1, 8)  # [M,K], K=dim1 contig
    b_ptr_layout: gl.constexpr = _ptr_blocked_layout(BLOCK_K, BLOCK_N, gl.num_warps(), 0, 8)  # [K,N], K=dim0 contig
    out_layout: gl.constexpr = _default_blocked_layout([BLOCK_M, BLOCK_N], gl.num_warps())

    # ---- Index tensors (each 1D arange gets a SliceLayout derived from its 2D parent) ----
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, a_ptr_layout))
    offs_k_a = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, a_ptr_layout))
    offs_k_b = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, b_ptr_layout))
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, b_ptr_layout))

    num_k_tiles = gl.cdiv(K, BLOCK_K)

    # ---- Multi-buffer shared memory for the software pipeline ----
    a_smem = gl.allocate_shared_memory(gl.float16, [NUM_BUFFERS, BLOCK_M, BLOCK_K], a_smem_layout)
    b_smem = gl.allocate_shared_memory(gl.float16, [NUM_BUFFERS, BLOCK_K, BLOCK_N], b_smem_layout)

    # ---- Prologue: fill the pipeline (issue the first NUM_BUFFERS K-tiles) ----
    # Mask each issue by (k < num_k_tiles) so K not divisible by BLOCK_K / short-K cases
    # do not read past the K end (matches the reference's has_first/has_next guards).
    for k in gl.static_range(NUM_BUFFERS):
        k_guard = k < num_k_tiles
        a_mask = gl.full((BLOCK_M, BLOCK_K), k_guard, gl.int1)
        b_mask = gl.full((BLOCK_K, BLOCK_N), k_guard, gl.int1)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k * BLOCK_K + offs_k_a)[None, :] * stride_ak
        b_ptrs = b_ptr + (k * BLOCK_K + offs_k_b)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        cp.async_copy_global_to_shared(a_smem.index(k % NUM_BUFFERS), a_ptrs, mask=a_mask)
        cp.async_copy_global_to_shared(b_smem.index(k % NUM_BUFFERS), b_ptrs, mask=b_mask)
        cp.commit_group()

    acc = gl.full((BLOCK_M, BLOCK_N), 0.0, gl.float32, layout=acc_layout)

    # ---- Steady state + epilogue: overlap next load with current compute ----
    for k in range(num_k_tiles):
        cp.wait_group(NUM_BUFFERS - 1)                     # oldest stage ready
        a_frag = a_smem.index(k % NUM_BUFFERS).load(a_op)  # shared -> register (DotOperandLayout)
        b_frag = b_smem.index(k % NUM_BUFFERS).load(b_op)
        acc = mma_v2(a_frag, b_frag, acc)                  # fp16 x fp16 -> fp32
        nk = k + NUM_BUFFERS
        
        if nk < num_k_tiles:                               # issue the stage NUM_BUFFERS ahead
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (nk * BLOCK_K + offs_k_a)[None, :] * stride_ak
            b_ptrs = b_ptr + (nk * BLOCK_K + offs_k_b)[:, None] * stride_bk + offs_n[None, :] * stride_bn
            cp.async_copy_global_to_shared(a_smem.index(nk % NUM_BUFFERS), a_ptrs)
            cp.async_copy_global_to_shared(b_smem.index(nk % NUM_BUFFERS), b_ptrs)
            cp.commit_group()

    # ---- Epilogue store: C[M, N] = fp16(acc) ----
    acc_out = gl.convert_layout(acc, out_layout)           # MMA layout -> plain blocked (fp32)
    c = gl.cast(acc_out, gl.float16)
    m_out: gl.constexpr = gl.SliceLayout(1, out_layout)    # M-axis (len BLOCK_M)
    n_out: gl.constexpr = gl.SliceLayout(0, out_layout)    # N-axis (len BLOCK_N)
    offs_m_o = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=m_out)
    offs_n_o = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=n_out)
    c_ptrs = c_ptr + offs_m_o[:, None] * stride_cm + offs_n_o[None, :] * stride_cn
    c_mask = (offs_m_o[:, None] < M) & (offs_n_o[None, :] < N)
    gl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, c, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, NUM_BUFFERS=2):
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                        c.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, NUM_BUFFERS, num_warps=4)
    return c


@pytest.mark.skipif(not is_ampere_or_newer(), reason="Requires NVIDIA Ampere-or-newer CUDA target")
@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (256, 256, 256)])
def test_ampere_matmul(M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).transpose(0, 1)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    matmul(a, b, c)
    torch_output = torch.matmul(a, b).to(torch.float16)
    torch.testing.assert_close(c, torch_output, atol=1e-2, rtol=1e-2)


def _tflops(M, N, K, ms):
    return 2.0 * M * N * K * 1e-12 / (ms * 1e-3)


def _shape_name(M, N, K):
    return f"{M}x{N}x{K}"


def _assert_benchmark_shapes():
    assert BENCHMARK_SIZES == [128 * i for i in range(2, 33)]
    for M, N, K in BENCHMARK_SHAPES:
        assert M == N == K
        assert M % 128 == 0 and N % 128 == 0 and K % 128 == 0


def _make_inputs(M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).transpose(0, 1)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    return a, b, c


def _measure_accuracy(a, b, c):
    matmul(a, b, c)
    torch_output = torch.matmul(a, b).to(torch.float16)
    diff = (c - torch_output).abs()
    rel = diff / torch_output.abs().clamp_min(1e-6)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": rel.max().item(),
        "allclose": torch.allclose(c, torch_output, atol=1e-2, rtol=1e-2),
    }


def _print_markdown_table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(str(item) for item in row) + " |")


def run_accuracy_cases(shapes=((128, 128, 128), (256, 256, 256))):
    rows = []
    for M, N, K in shapes:
        a, b, c = _make_inputs(M, N, K)
        result = _measure_accuracy(a, b, c)
        rows.append([
            _shape_name(M, N, K),
            f"{result['max_abs']:.6g}",
            f"{result['mean_abs']:.6g}",
            f"{result['max_rel']:.6g}",
            result["allclose"],
        ])
    _print_markdown_table(["shape", "max_abs", "mean_abs", "max_rel", "allclose"], rows)


def run_benchmark(warmup=25, rep=100, check_correctness=True):
    _assert_benchmark_shapes()
    rows = []
    for M, N, K in BENCHMARK_SHAPES:
        a, b, c = _make_inputs(M, N, K)
        accuracy = _measure_accuracy(a, b, c)
        if check_correctness and not accuracy["allclose"]:
            torch_output = torch.matmul(a, b).to(torch.float16)
            torch.testing.assert_close(c, torch_output, atol=1e-2, rtol=1e-2)

        torch_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=warmup, rep=rep)
        gluon_ms = triton.testing.do_bench(lambda: matmul(a, b, c), warmup=warmup, rep=rep)
        torch_tflops = _tflops(M, N, K, torch_ms)
        gluon_tflops = _tflops(M, N, K, gluon_ms)
        rows.append([
            _shape_name(M, N, K),
            f"{torch_ms:.4f}",
            f"{gluon_ms:.4f}",
            f"{torch_tflops:.2f}",
            f"{gluon_tflops:.2f}",
            f"{torch_ms / gluon_ms:.3f}",
            f"{accuracy['max_abs']:.6g}",
            accuracy["allclose"],
        ])
    _print_markdown_table(
        ["shape", "torch_ms", "gluon_ms", "torch_tflops", "gluon_tflops", "speedup", "max_abs", "allclose"],
        rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="benchmark square matmul shapes from 256 to 4096")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--no-check", action="store_true", help="skip correctness check during benchmark")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(warmup=args.warmup, rep=args.rep, check_correctness=not args.no_check)
    else:
        run_accuracy_cases()
