from __future__ import annotations

import argparse
import ctypes
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.experimental.tle.language as tle
import triton.experimental.tle.language.raw as tle_raw
import triton.language as tl
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.nvshmem.utils import (
    init_torch_distributed,
    tensor_from_pointer,
)

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

POISON_U32 = 0x80000000
PACK_BYTES = 16
ALLREDUCE_WARPS = 16
CLUSTER_MAX = 8

PDL_CHAIN = True

HERE = Path(__file__).resolve().parent
CU_FILE = HERE / "fuse-ar-rmsnorm-device.cu"

# ---------------------------------------------------------------------------
# TLE-Raw: wrap the .cu device functions into launchable Triton kernels
# ---------------------------------------------------------------------------


def _device_dialect(function_name):

    @dialect(name="cuda", compiler="clang", file=CU_FILE, extern_func_name=function_name)
    def _fn(*args, **kwargs):
        ...

    return _fn


_ALLREDUCE_FUNC = _device_dialect("fused_ar_rmsnorm_allreduce_bf16")


@triton.jit
def fused_ar_rmsnorm_allreduce_kernel(input_ptr, peer_scatter_ptrs, multicast_broadcast, num_tokens, packs_per_token,
                                      rank: tl.constexpr, world_size: tl.constexpr, CLUSTER_SIZE: tl.constexpr):
    tle_raw.call(_ALLREDUCE_FUNC, [
        input_ptr,
        peer_scatter_ptrs,
        multicast_broadcast,
        tl.full((), num_tokens, tl.int32),
        tl.full((), packs_per_token, tl.int32),
        tl.full((), rank, tl.int32),
        tl.full((), world_size, tl.int32),
    ])


@triton.jit
def _float_to_bf16_bits(x):
    bits = x.to(tl.uint32, bitcast=True)
    bits = bits + 0x7FFF + ((bits >> 16) & 1)
    return (bits >> 16).to(tl.uint16)


@triton.jit
def _pack_words_to_bf16(words):
    lo = (words & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    hi = ((words >> 16) & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    return tl.interleave(lo, hi)


@triton.jit
def _poison_broadcast_pack(bc_ptr, token, packs_per_token, pack_idx, m, BLOCK_PACKS: tl.constexpr):
    wptr = bc_ptr + ((token * packs_per_token + pack_idx)[:, None] * 4 + tl.arange(0, 4)[None, :])
    tl.store(wptr, tl.full([BLOCK_PACKS, 4], 0x80000000, tl.uint32), mask=m[:, None])


@triton.jit
def _spin_broadcast_pack(bc_ptr, token, packs_per_token, pack_idx, m, BLOCK_PACKS: tl.constexpr):
    wptr = bc_ptr + ((token * packs_per_token + pack_idx)[:, None] * 4 + tl.arange(0, 4)[None, :])
    words = tle.load(wptr, mask=m[:, None], other=0, volatile=True)
    ready = tl.sum(tl.sum(tl.where(m[:, None], words != 0x80000000, True).to(tl.int32), axis=1),
                   axis=0) == BLOCK_PACKS * 4
    while ready == 0:
        words = tle.load(wptr, mask=m[:, None], other=0, volatile=True)
        ready = tl.sum(tl.sum(tl.where(m[:, None], words != 0x80000000, True).to(tl.int32), axis=1),
                       axis=0) == BLOCK_PACKS * 4
    return words


@triton.jit
def fused_ar_rmsnorm_norm_bf16(
    norm_out_ptr,
    prenorm_out_ptr,
    residual_ptr,
    gamma_ptr,
    local_broadcast_ptr,
    cluster_scratch_ptr,
    num_tokens,
    packs_per_token,
    hidden,
    eps,
    CLUSTER_SIZE: tl.constexpr,
    BLOCK_PACKS: tl.constexpr,
    PREFETCH: tl.constexpr,
    PDL_CHAIN: tl.constexpr = False,
):
    token = tl.program_id(0)
    if token >= num_tokens:
        return
    block_rank = tl.program_id(1)

    # This CTA handles a contiguous chunk of the token's hidden row.
    chunk_packs = (packs_per_token + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    pack_begin = block_rank * chunk_packs
    pack_end = tl.minimum(pack_begin + chunk_packs, packs_per_token)
    n_iters = (pack_end - pack_begin + BLOCK_PACKS - 1) // BLOCK_PACKS

    # PDL: let dependent work start early; never wait for the allreduce grid,
    # the broadcast sentinels below are the synchronisation.
    tl.extra.cuda.gdc_launch_dependents()

    pack_lane = tl.arange(0, BLOCK_PACKS)
    elem = tl.arange(0, 8)
    row = token * packs_per_token

    prefetch_slots = tl.arange(0, PREFETCH)
    prefetch_packs = pack_begin + prefetch_slots[:, None, None] * BLOCK_PACKS + pack_lane[None, :, None]
    prefetch_mask = prefetch_packs < pack_end
    prefetch_elems = elem[None, None, :]
    prefetch_residual = tl.load(residual_ptr + (row + prefetch_packs) * 8 + prefetch_elems, mask=prefetch_mask,
                                other=0.0)
    prefetch_gamma = tl.load(gamma_ptr + prefetch_packs * 8 + prefetch_elems, mask=prefetch_mask, other=0.0)

    # ---- pass 1: consume broadcast, write pre_norm, accumulate the square sum
    acc = tl.zeros([BLOCK_PACKS, 8], dtype=tl.float32)
    for slot in tl.static_range(PREFETCH):
        pack_idx = pack_begin + slot * BLOCK_PACKS + pack_lane
        m = pack_idx < pack_end
        if slot < n_iters:
            words = _spin_broadcast_pack(local_broadcast_ptr, token, packs_per_token, pack_idx, m, BLOCK_PACKS)
            x = _pack_words_to_bf16(words).to(tl.float32)

            r = tl.reshape(tle.extract_tile(prefetch_residual, index=[slot, 0, 0], tile_shape=(1, BLOCK_PACKS, 8)),
                           (BLOCK_PACKS, 8)).to(tl.float32)
            y = x + r
            prenorm_bits = _float_to_bf16_bits(y)
            tl.store(prenorm_out_ptr + (row + pack_idx)[:, None] * 8 + elem[None, :],
                     prenorm_bits.to(tl.bfloat16, bitcast=True), mask=m[:, None])
            if not PDL_CHAIN:
                _poison_broadcast_pack(local_broadcast_ptr, token, packs_per_token, pack_idx, m, BLOCK_PACKS)
            acc += tl.where(m[:, None], y * y, 0.0)

    for it in range(PREFETCH, n_iters):
        pack_idx = pack_begin + it * BLOCK_PACKS + pack_lane
        m = pack_idx < pack_end
        r = tl.load(residual_ptr + (row + pack_idx)[:, None] * 8 + elem[None, :], mask=m[:, None], other=0.0)
        words = _spin_broadcast_pack(local_broadcast_ptr, token, packs_per_token, pack_idx, m, BLOCK_PACKS)
        x = _pack_words_to_bf16(words).to(tl.float32)
        y = x + r.to(tl.float32)
        prenorm_bits = _float_to_bf16_bits(y)
        tl.store(prenorm_out_ptr + (row + pack_idx)[:, None] * 8 + elem[None, :],
                 prenorm_bits.to(tl.bfloat16, bitcast=True), mask=m[:, None])
        if not PDL_CHAIN:
            _poison_broadcast_pack(local_broadcast_ptr, token, packs_per_token, pack_idx, m, BLOCK_PACKS)
        acc += tl.where(m[:, None], y * y, 0.0)

    # ---- block reduction of the square sum
    full_sum = tl.sum(tl.sum(acc, axis=1), axis=0)

    # ---- cross-CTA reduce through the CGA cluster shared-memory mapping
    if CLUSTER_SIZE > 1:
        partial_smem = tle.gpu.alloc([CLUSTER_SIZE], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                                     nv_mma_shared_layout=False)
        tle.distributed_barrier()
        for dst in tl.static_range(CLUSTER_SIZE):
            remote_smem = tle.remote(partial_smem, dst)
            rptr = tle.gpu.local_ptr(remote_smem, (block_rank, ))
            tl.store(rptr, full_sum)
        tle.distributed_barrier()
        full_sum = tl.zeros((), dtype=tl.float32)
        for i in tl.static_range(CLUSTER_SIZE):
            full_sum += tl.load(tle.gpu.local_ptr(partial_smem, (i, )))

    rcp_rms = tl.rsqrt(full_sum / hidden.to(tl.float32) + eps)

    # ---- pass 2: normalize from the global pre_norm buffer and gamma
    for slot in tl.static_range(PREFETCH):
        pack_idx = pack_begin + slot * BLOCK_PACKS + pack_lane
        m = pack_idx < pack_end
        if slot < n_iters:
            pre = tl.load(prenorm_out_ptr + (row + pack_idx)[:, None] * 8 + elem[None, :], mask=m[:, None],
                          other=0.0).to(tl.float32)
            g = tl.reshape(tle.extract_tile(prefetch_gamma, index=[slot, 0, 0], tile_shape=(1, BLOCK_PACKS, 8)),
                           (BLOCK_PACKS, 8)).to(tl.float32)
            z = pre * g * rcp_rms
            out_bits = _float_to_bf16_bits(z)
            tl.store(norm_out_ptr + (row + pack_idx)[:, None] * 8 + elem[None, :],
                     out_bits.to(tl.bfloat16, bitcast=True), mask=m[:, None])

    for it in range(PREFETCH, n_iters):
        pack_idx = pack_begin + it * BLOCK_PACKS + pack_lane
        m = pack_idx < pack_end
        pre = tl.load(prenorm_out_ptr + (row + pack_idx)[:, None] * 8 + elem[None, :], mask=m[:, None],
                      other=0.0).to(tl.float32)
        g = tl.load(gamma_ptr + pack_idx[:, None] * 8 + elem[None, :], mask=m[:, None], other=0.0).to(tl.float32)
        z = pre * g * rcp_rms
        out_bits = _float_to_bf16_bits(z)
        tl.store(norm_out_ptr + (row + pack_idx)[:, None] * 8 + elem[None, :], out_bits.to(tl.bfloat16, bitcast=True),
                 mask=m[:, None])

    if PDL_CHAIN:
        tl.extra.cuda.gdc_wait()
        for it in range(n_iters):
            pack_idx = pack_begin + it * BLOCK_PACKS + pack_lane
            m = pack_idx < pack_end
            _poison_broadcast_pack(local_broadcast_ptr, token, packs_per_token, pack_idx, m, BLOCK_PACKS)


# ---------------------------------------------------------------------------
# symmetric memory workspace
# ---------------------------------------------------------------------------


class FusedArRmsNormWorkspace:
    # Symmetric scratch: slot 0 = scatter, slot 1 = broadcast.
    def __init__(self, world_size, rank, max_token_num, hidden_dim, dtype, group):
        self.world_size = world_size
        self.rank = rank
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.group = group

        padded_tokens = triton.cdiv(max_token_num, world_size) * world_size
        packs_per_token = hidden_dim * dtype.itemsize // PACK_BYTES
        shape = (1, 2, padded_tokens, packs_per_token, 4)

        self.local_slots = symm_mem.empty(shape, dtype=torch.uint32, device="cuda")
        self.handle = symm_mem.rendezvous(self.local_slots, group)
        if not self.handle.multicast_ptr:
            raise RuntimeError("no multicast mapping available")
        self.multicast_slots = tensor_from_pointer(ctypes.c_void_p(self.handle.multicast_ptr), shape, torch.uint32,
                                                   self.local_slots.device)
        self.peer_slots = tuple(self.handle.get_buffer(p, shape, torch.uint32) for p in range(world_size))
        self.peer_pointer_tables = torch.tensor([[self.peer_slots[p][0, 0].data_ptr() for p in range(world_size)]],
                                                dtype=torch.uint64, device=self.local_slots.device)

        cluster = min(CLUSTER_MAX, packs_per_token)
        self.cluster_scratch = torch.zeros((max_token_num, cluster + 1), dtype=torch.float32,
                                           device=self.local_slots.device)

        self.local_slots.fill_(POISON_U32)
        torch.cuda.synchronize()
        dist.barrier(group=group)


def _sm_count():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def pick_cluster(num_tokens, packs_per_token):
    cluster = min(CLUSTER_MAX, packs_per_token)
    sm = _sm_count()
    while cluster > 1 and num_tokens * cluster > sm:
        cluster //= 2
    return cluster


def _next_pow2(n):
    return 1 << max(0, (n - 1).bit_length())


# ---------------------------------------------------------------------------
# the fused operator
# ---------------------------------------------------------------------------


def fused_allreduce_residual_rmsnorm(x, residual, gamma, eps, workspace, rank, world_size, norm_out=None,
                                     residual_out=None):
    # Run kernel 1 + kernel 2.  Returns ``(norm_out, residual_out)``.
    num_tokens, hidden = x.shape
    dtype = x.dtype
    packs_per_token = hidden * dtype.itemsize // PACK_BYTES

    if norm_out is None:
        norm_out = torch.empty_like(x)
    if residual_out is None:
        residual_out = torch.empty_like(x)

    cluster = pick_cluster(num_tokens, packs_per_token)
    chunk_packs = (packs_per_token + cluster - 1) // cluster
    block_packs = _next_pow2(chunk_packs)
    triton_warps = max(4, min(16, block_packs // 32))

    fused_ar_rmsnorm_allreduce_kernel[(num_tokens, cluster)](
        x,
        workspace.peer_pointer_tables[0],
        workspace.multicast_slots[0, 1],
        num_tokens,
        packs_per_token,
        rank=rank,
        world_size=world_size,
        CLUSTER_SIZE=cluster,
        num_warps=ALLREDUCE_WARPS,
        launch_pdl=PDL_CHAIN,
    )

    fused_ar_rmsnorm_norm_bf16[(num_tokens, )](
        norm_out,
        residual_out,
        residual,
        gamma,
        workspace.local_slots[0, 1],
        workspace.cluster_scratch,
        num_tokens,
        packs_per_token,
        hidden,
        eps,
        CLUSTER_SIZE=cluster,
        BLOCK_PACKS=block_packs,
        PREFETCH=1,
        PDL_CHAIN=PDL_CHAIN,
        num_warps=triton_warps,
        cluster_dims=(1, cluster, 1),
        launch_pdl=True,
    )
    return norm_out, residual_out


# ---------------------------------------------------------------------------
# torch reference
# ---------------------------------------------------------------------------


def torch_reference(x, residual, gamma, eps, group):
    full = x.to(torch.float32)
    dist.all_reduce(full, group=group)
    broadcast = full.to(torch.bfloat16)
    y = broadcast.to(torch.float32) + residual.to(torch.float32)
    prenorm = y.to(torch.bfloat16)
    rcp_rms = torch.rsqrt(y.pow(2).sum(-1, keepdim=True) / y.shape[-1] + eps)
    norm = (prenorm.to(torch.float32) * gamma.to(torch.float32) * rcp_rms).to(torch.bfloat16)
    return prenorm, norm


def check_correctness(x, residual, gamma, eps, workspace, rank, world_size, group):
    norm_out, residual_out = fused_allreduce_residual_rmsnorm(x, residual, gamma, eps, workspace, rank, world_size)
    torch.cuda.synchronize()
    ref_prenorm, ref_norm = torch_reference(x, residual, gamma, eps, group)

    prenorm_bit_exact = torch.equal(residual_out, ref_prenorm)
    prenorm_err = (residual_out.float() - ref_prenorm.float()).abs().max().item()
    norm_err = (norm_out.float() - ref_norm.float()).abs().max().item()
    num_bit_diff = (norm_out.view(torch.int16) != ref_norm.view(torch.int16)).sum().item()

    # one bf16 ulp at the observed magnitude
    scale = max(ref_norm.float().abs().max().item(), 1.0)
    ulp = 2.0**(math.floor(math.log2(scale)) - 7)
    ok = prenorm_bit_exact and norm_err <= ulp

    if rank == 0:
        print(f"  prenorm bit-exact : {prenorm_bit_exact} (max |err| = {prenorm_err:.3e})", flush=True)
        print(
            f"  norm    max |err| : {norm_err:.6f}  ({norm_err / ulp:.2f} ulp)"
            f"  bit-different: {num_bit_diff}/{norm_out.numel()}", flush=True)
        print(f"  => {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def _time_launches(launch, iters, warmup, x, group):
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        launch()
        ends[i].record()
    torch.cuda.synchronize()

    samples = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    local = samples[len(samples) // 2]  # p50 on this rank
    # the collective finishes when the slowest rank finishes
    max_over_ranks_us = torch.tensor([local], device=x.device, dtype=torch.float64)
    dist.all_reduce(max_over_ranks_us, op=dist.ReduceOp.MAX, group=group)
    return max_over_ranks_us.item()


def benchmark(x, residual, gamma, eps, workspace, rank, world_size, warmup, iters, group, use_cuda_graph=True):
    norm_out = torch.empty_like(x)
    residual_out = torch.empty_like(x)

    def launch():
        fused_allreduce_residual_rmsnorm(x, residual, gamma, eps, workspace, rank, world_size, norm_out=norm_out,
                                         residual_out=residual_out)

    eager = _time_launches(launch, iters, warmup, x, group)

    graph_us = None
    if use_cuda_graph:
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                launch()
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()
        dist.barrier(group=group)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        graph_us = _time_launches(graph.replay, iters, warmup, x, group)

    return eager, graph_us


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="two-kernel fused AllReduce + Residual + RMSNorm")
    parser.add_argument("--hidden", type=int, default=8192, help="hidden size (must be divisible by 8)")
    parser.add_argument("--tokens", type=int, default=32, help="decode batch size num_tokens")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--skip-check", action="store_true", help="skip the torch reference comparison")
    parser.add_argument("--no-cuda-graph", action="store_true", help="only time the eager launch loop")
    return parser.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

    assert world_size >= 2, "at least two ranks are required"
    assert world_size == local_world_size, "this example is single-node: WORLD_SIZE must equal LOCAL_WORLD_SIZE"
    assert args.hidden % 8 == 0, "hidden size must be divisible by 8"

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    group = init_torch_distributed()

    num_tokens, hidden = args.tokens, args.hidden
    dtype = torch.bfloat16

    # Each rank holds a different partial sum of the row-parallel GEMM; the
    # residual and gamma are replicated, exactly as in a real layer.
    torch.manual_seed(1234 + rank)
    x = (torch.randn((num_tokens, hidden), dtype=dtype, device=device) * 0.1)
    torch.manual_seed(1234)
    residual = (torch.randn((num_tokens, hidden), dtype=dtype, device=device) * 0.1)
    gamma = torch.randn((hidden, ), dtype=dtype, device=device)

    workspace = FusedArRmsNormWorkspace(world_size, rank, num_tokens, hidden, dtype, group)
    cluster = pick_cluster(num_tokens, hidden * dtype.itemsize // PACK_BYTES)
    if rank == 0:
        print("=== fused AllReduce + Residual + RMSNorm ===", flush=True)
        print(
            f"  world_size={world_size} num_tokens={num_tokens} hidden={hidden} "
            f"dtype={dtype} pdl_chain={PDL_CHAIN}", flush=True)
        print(f"  cluster={cluster} "
              f"(SM count {_sm_count()}, numTokens*cluster="
              f"{num_tokens * cluster})", flush=True)
        print(
            f"  workspace: scatter+broadcast slots "
            f"{tuple(workspace.local_slots.shape)}, "
            f"multicast={'yes' if workspace.handle.multicast_ptr else 'no'}", flush=True)

    ok = True
    if not args.skip_check:
        if rank == 0:
            print("--- correctness vs torch reference ---", flush=True)
        ok = check_correctness(x, residual, gamma, args.eps, workspace, rank, world_size, group)

    if rank == 0:
        print(f"--- benchmark (warmup={args.warmup}, iters={args.iters}) ---", flush=True)
    eager_us, graph_us = benchmark(x, residual, gamma, args.eps, workspace, rank, world_size, args.warmup, args.iters,
                                   group, use_cuda_graph=not args.no_cuda_graph)
    payload = hidden * num_tokens * dtype.itemsize  # one rank's all-reduce shard
    if rank == 0:
        print(f"  eager launch loop (p50, max over ranks): {eager_us:8.2f} us", flush=True)
        if graph_us is not None:
            print(
                f"  CUDA graph replay (p50, max over ranks): {graph_us:8.2f} us"
                f"   <- kernel time, as vLLM runs it", flush=True)
            print(f"  all-reduce payload {payload / 1024:.0f} KiB -> "
                  f"{payload / graph_us / 1e3:.1f} GB/s", flush=True)

    dist.barrier(group=group)
    dist.destroy_process_group()
    if not ok:
        raise SystemExit("correctness check FAILED")


if __name__ == "__main__":
    main()
