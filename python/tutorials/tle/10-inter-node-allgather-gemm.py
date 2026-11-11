"""Overlap TLE push-2d AllGather with persistent GEMM.

C_rank = cat(A_0, ..., A_(world_size-1), dim=0) @ B_rank.
A_rank is [M/world_size, K]; B_rank is [K, N/world_size]. M and N
are global dimensions, while K is replicated. B may be contiguous or a
transpose of contiguous [N/world_size, K] storage.

The producer is shared with tutorial 06. All communication uses TLE remote,
signal, signal_wait, and distributed_barrier, with no raw NVSHMEM calls.
The consumer follows Triton-distributed's allgather_gemm.py; context and
stream organization follow tutorials 08/09. Requires NVIDIA SM90 or newer.
Inter-node PUTs transfer a complete source segment up to 2**25 elements;
only larger segments are split. This limit counts elements, not bytes.

Run on every node (set NODE_RANK separately):

    NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 NPROC_PER_NODE=4 \
        bash python/tutorials/tle/10-inter-node-allgather-gemm.sh \
        --M 2048 --N 4096 --K 4096 --dtype bf16

One context owns signal slots [0, world_size) and barrier channel 0 until
finalize(). The current TLE runtime allocates eight signal slots, so this
tutorial supports at most eight ranks with uniform local_world_size.
Calls must occur collectively in the same order on the same CUDA stream.
The context cannot share these synchronization slots with another operator.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
import statistics
from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

_ag = importlib.import_module("06-inter-node-allgather")
_rs = importlib.import_module("08-inter-node-reduce-scatter")
_active_context = None


@dataclasses.dataclass(frozen=True)
class TleAllGatherGemmConfig:
    block_m: int = 64
    block_n: int = 128
    block_k: int = 64
    group_m: int = 8
    num_warps: int = 4
    num_stages: int = 3
    ag_block: int = 4096
    ag_num_warps: int = 4

    def validate(self):
        for name in ("block_m", "block_n", "block_k", "ag_block"):
            value = getattr(self, name)
            if value < 16 or value & (value - 1):
                raise ValueError(f"{name} must be a power of two >= 16")
        if self.group_m < 1 or self.num_stages < 1:
            raise ValueError("group_m and num_stages must be positive")
        if self.num_warps not in (4, 8) or self.ag_num_warps not in (4, 8, 16, 32):
            raise ValueError("num_warps must be 4/8; ag_num_warps must be 4/8/16/32")


def _validate_shape(M, N, K, world_size, local_world_size, config):
    config.validate()
    if min(M, N, K, world_size, local_world_size) < 1:
        raise ValueError("dimensions and rank counts must be positive")
    if world_size > 8:
        raise ValueError("the current TLE communicator provides only eight signal slots")
    if world_size % local_world_size:
        raise ValueError("world_size must be divisible by local_world_size")
    if M % world_size or N % world_size:
        raise ValueError("global M and N must be divisible by world_size")
    if (M // world_size) % config.block_m:
        raise ValueError("M/world_size must be divisible by block_m")
    if K % 16 or (N // world_size) % 16:
        raise ValueError("K and N/world_size must be multiples of 16 for TMA")


@dataclasses.dataclass
class TleAllGatherGemmContext:
    max_M: int
    N: int
    K: int
    dtype: torch.dtype
    rank: int
    world_size: int
    local_world_size: int
    workspace: torch.Tensor
    dist_ctx: object
    ag_stream: torch.cuda.Stream
    compute_stream: torch.cuda.Stream
    num_gemm_sms: int
    config: TleAllGatherGemmConfig
    epoch: int = 0
    closed: bool = False
    world_mesh: tle.device_mesh = dataclasses.field(init=False)

    def __post_init__(self):
        self.world_mesh = tle.device_mesh(tle.MeshConfig(
            node=self.world_size // self.local_world_size, device=self.local_world_size,
        ))

    def finalize(self):
        """Collectively release the tutorial communicator and process group."""
        global _active_context
        if not self.closed:
            tle.cleanup_communicator()
            self.closed = True
            _active_context = None


def create_ag_gemm_context(max_M: int, N: int, K: int, dtype: torch.dtype, rank: int,
                           world_size: int, local_world_size: int,
                           config: Optional[TleAllGatherGemmConfig] = None,
                           num_gemm_sms: Optional[int] = None) -> TleAllGatherGemmContext:
    """Collectively allocate one reusable, symmetric input workspace."""
    global _active_context
    if _active_context is not None:
        raise RuntimeError("reuse the existing AG GEMM context; its signal slots are already reserved")
    config = config or TleAllGatherGemmConfig()
    _validate_shape(max_M, N, K, world_size, local_world_size, config)
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("AG GEMM supports float16 and bfloat16")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("persistent TMA GEMM requires SM90 or newer")
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise ValueError("context topology must match the default process group")
    if int(os.environ.get("LOCAL_RANK", rank % local_world_size)) != rank % local_world_size:
        raise ValueError("ranks must be contiguous within each node")
    total_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    # Bound persistent CTA counts to leave room for every producer CTA, including
    # relay CTAs that can wait on inter-node producers. This is not SM pinning.
    ag_sms = world_size if world_size > 1 else 0
    available = total_sms - ag_sms
    if num_gemm_sms is None:
        num_gemm_sms = available
    if not 1 <= num_gemm_sms <= available:
        raise ValueError(f"num_gemm_sms must be in [1, {available}] to leave room for AG")
    mem_pool = tle.get_mem_pool()
    if mem_pool is None:
        raise RuntimeError("TLE requires an available FlagCX symmetric memory pool")
    with torch.cuda.use_mem_pool(mem_pool):
        workspace = torch.empty((max_M, K), dtype=dtype, device="cuda")
    dist_ctx = tle.create_dist_tensor(workspace)
    dist_ctx.register_buffer(workspace)
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device=workspace.device, dtype=torch.int8))
    ctx = TleAllGatherGemmContext(
        max_M, N, K, dtype, rank, world_size, local_world_size, workspace, dist_ctx,
        torch.cuda.Stream(priority=-1), torch.cuda.current_stream(), num_gemm_sms, config,
    )
    _active_context = ctx
    return ctx


@triton.jit
def _prepare_all_gather_kernel(dist_ctx: tl.constexpr, mesh: tl.constexpr):
    # The previous call joined AG and GEMM on the compute stream. This barrier
    # prevents a fast rank from overwriting A while a slow rank still uses it.
    tle.distributed_barrier(
        mesh=mesh, device_dptr=dist_ctx, space="world", group_kind="block",
        barrier_kind="sync", order="acqrel", index=0, context_id=0,
    )


@triton.jit(do_not_specialize=["epoch"])
def kernel_ag_gemm_consumer_persistent(
    a_ptr, b_ptr, c_ptr, M, N, K, epoch,
    dist_ctx: tl.constexpr,
    RANK: tl.constexpr, WORLD_SIZE: tl.constexpr,
    B_STRIDE_K: tl.constexpr, B_STRIDE_N: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr,
):
    """Consume one ready source-rank segment at a time, starting locally."""
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    rows_per_rank = M // WORLD_SIZE
    tiles_m_per_rank = rows_per_rank // BLOCK_M
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BLOCK_M, BLOCK_K])
    if TRANS_B:
        b_desc = tl.make_tensor_descriptor(b_ptr, [N, K], [B_STRIDE_N, B_STRIDE_K], [BLOCK_N, BLOCK_K])
    else:
        b_desc = tl.make_tensor_descriptor(b_ptr, [K, N], [B_STRIDE_K, B_STRIDE_N], [BLOCK_K, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BLOCK_M, BLOCK_N])

    for tile_id in range(tl.program_id(0), num_tiles, NUM_SMS):
        tiles_per_group = GROUP_M * num_pid_n
        group_id = tile_id // tiles_per_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + tile_id % group_size_m
        pid_n = (tile_id % tiles_per_group) // group_size_m
        # A rank-aligned cyclic swizzle lets local A start computing immediately.
        pid_m = (pid_m + RANK * tiles_m_per_rank) % num_pid_m
        source_rank = pid_m // tiles_m_per_rank
        if source_rank != RANK:
            tle.signal_wait(
                dist_ctx, slot_id=source_rank, wait_kind="signal", target=epoch,
                group_kind="block", context_id=0, order="acquire",
            )
        # The producer writes through the generic/network path, whereas TMA
        # reads through the async proxy. Order that proxy after the acquire.
        tl.inline_asm_elementwise(
            "fence.proxy.async.global; mov.u32 $0, 0;", constraints="=r",
            args=[], dtype=tl.int32, is_pure=False, pack=1,
        )
        offs_m = pid_m * BLOCK_M
        offs_n = pid_n * BLOCK_N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(tl.cdiv(K, BLOCK_K)):
            a = a_desc.load([offs_m, ki * BLOCK_K])
            if TRANS_B:
                b = b_desc.load([offs_n, ki * BLOCK_K])
                accumulator = tl.dot(a, b.T, accumulator)
            else:
                b = b_desc.load([ki * BLOCK_K, offs_n])
                accumulator = tl.dot(a, b, accumulator)
        c_desc.store([offs_m, offs_n], accumulator.to(c_ptr.dtype.element_ty))


def ag_gemm(A: torch.Tensor, B: torch.Tensor, ctx: TleAllGatherGemmContext,
            *, overlap: bool = True) -> torch.Tensor:
    """Collective AllGather(A) @ B; return on the context's compute stream.

    overlap=False waits for AG before launching GEMM as a serial TLE baseline.
    No host synchronization or PyTorch collective is used in this operator.
    """
    if ctx.closed:
        raise RuntimeError("AG GEMM context has been finalized")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be matrices")
    M = A.shape[0] * ctx.world_size
    _validate_shape(M, ctx.N, ctx.K, ctx.world_size, ctx.local_world_size, ctx.config)
    if M > ctx.max_M or A.shape[1] != ctx.K or B.shape != (ctx.K, ctx.N // ctx.world_size):
        raise ValueError("input dimensions do not match the context")
    if A.dtype != ctx.dtype or B.dtype != ctx.dtype or A.device != ctx.workspace.device or B.device != A.device:
        raise ValueError("A and B must match the context's device and dtype")
    if not A.is_contiguous():
        raise ValueError("A must be contiguous")
    trans_b = B.stride(0) == 1 and B.stride(1) == ctx.K
    if not trans_b and not B.is_contiguous():
        raise ValueError("B must be contiguous or a transpose of contiguous [N/world_size, K]")
    if any(t.untyped_storage().data_ptr() == ctx.workspace.untyped_storage().data_ptr() for t in (A, B)):
        raise ValueError("A and B must use separate storage from the shared workspace")
    if B.data_ptr() % 16:
        raise ValueError("B must have a 16-byte-aligned base address for TMA")
    current_stream = torch.cuda.current_stream(A.device)
    if current_stream != ctx.compute_stream:
        raise ValueError("reuse the context on its original compute stream")
    if ctx.epoch >= (1 << 63) - 1:
        raise OverflowError("signal epoch exhausted")

    C = torch.empty((M, ctx.N // ctx.world_size), dtype=ctx.dtype, device=A.device)
    if ctx.world_size > 1:
        _prepare_all_gather_kernel[(1, )](ctx.dist_ctx, ctx.world_mesh, num_warps=4)
    row_begin = ctx.rank * A.shape[0]
    ctx.workspace[row_begin:row_begin + A.shape[0]].copy_(A)
    ctx.epoch += 1

    cfg = ctx.config
    if ctx.world_size > 1:
        node_block, num_node_chunks = _ag.get_node_transfer_config(A.numel())
        ctx.ag_stream.wait_stream(current_stream)
        with torch.cuda.stream(ctx.ag_stream):
            _ag.push_2d_all_gather_kernel[(ctx.world_size, )](
                ctx.workspace, ctx.dist_ctx, ctx.rank % ctx.local_world_size,
                ctx.rank // ctx.local_world_size, ctx.epoch,
                LOCAL_WORLD_SIZE=ctx.local_world_size, WORLD_SIZE=ctx.world_size,
                ELEM_PER_RANK=A.numel(),
                NUM_NODE_CHUNKS=num_node_chunks,
                NUM_LOCAL_CHUNKS=triton.cdiv(A.numel(), cfg.ag_block),
                NODE_BLOCK=node_block, BLOCK=cfg.ag_block,
                # The consumer supplies completion in the overlapped path.
                # The serial baseline needs AG's result-ready completion CTA.
                WAIT_FOR_ALL=not overlap, num_warps=cfg.ag_num_warps,
            )
        if not overlap:
            current_stream.wait_stream(ctx.ag_stream)

    num_tiles = triton.cdiv(M, cfg.block_m) * triton.cdiv(C.shape[1], cfg.block_n)
    kernel_ag_gemm_consumer_persistent[(min(ctx.num_gemm_sms, num_tiles), )](
        ctx.workspace, B, C, M, C.shape[1], ctx.K, ctx.epoch, ctx.dist_ctx,
        RANK=ctx.rank, WORLD_SIZE=ctx.world_size,
        B_STRIDE_K=B.stride(0), B_STRIDE_N=B.stride(1), TRANS_B=trans_b,
        BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n, BLOCK_K=cfg.block_k,
        GROUP_M=cfg.group_m, NUM_SMS=ctx.num_gemm_sms,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )
    if ctx.world_size > 1:
        current_stream.wait_stream(ctx.ag_stream)
    return C


def torch_ag_gemm(A, B, process_group=None):
    group = process_group or dist.group.WORLD
    full = torch.empty((A.shape[0] * dist.get_world_size(group), A.shape[1]), dtype=A.dtype, device=A.device)
    dist.all_gather_into_tensor(full, A, group=group)
    return full @ B


def _time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(iters):
        # Rank synchronization is outside the timed interval.
        dist.barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        elapsed = torch.tensor(start.elapsed_time(end), device="cuda", dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(elapsed.item())
    return statistics.median(samples)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, default in (("M", 2048), ("N", 4096), ("K", 4096)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default=os.environ.get("DTYPE", "bf16"))
    parser.add_argument("--check-iters", type=int, default=5, help="Repeated checks with changing inputs.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations; zero disables timing.")
    parser.add_argument("--trans-b", action="store_true", help="Back B with contiguous [N/world_size, K].")
    parser.add_argument("--gemm-sms", type=int)
    for field in dataclasses.fields(TleAllGatherGemmConfig):
        parser.add_argument("--" + field.name.replace("_", "-"), type=int, default=field.default)
    args = parser.parse_args(argv)
    if min(args.M, args.N, args.K, args.check_iters) < 1 or min(args.warmup, args.iters) < 0:
        parser.error("dimensions and check-iters must be positive; warmup and iters must be nonnegative")
    return args


def main():
    args = _parse_args()
    ctx = None
    try:
        tle.get_mem_pool()
        rank, world_size = dist.get_rank(), dist.get_world_size()
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        config = TleAllGatherGemmConfig(**{field.name: getattr(args, field.name)
                                         for field in dataclasses.fields(TleAllGatherGemmConfig)})
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
        ctx = create_ag_gemm_context(args.M, args.N, args.K, dtype, rank, world_size,
                                     local_world_size, config, args.gemm_sms)
        torch.manual_seed(1234 + rank)
        A = torch.empty((args.M // world_size, args.K), dtype=dtype, device="cuda")
        B_storage = torch.empty((args.N // world_size, args.K) if args.trans_b
                                else (args.K, args.N // world_size), dtype=dtype, device="cuda")
        B = B_storage.T if args.trans_b else B_storage
        for iteration in range(args.check_iters):
            # Rank-specific, changing values detect stale signals and buffer reuse.
            A.normal_(mean=rank * 0.02 + iteration * 0.01, std=0.1)
            B_storage.normal_(std=0.1)
            reference = torch_ag_gemm(A, B)
            for overlap in (True, False):
                output = ag_gemm(A, B, ctx, overlap=overlap)
                _rs._assert_close_on_all_ranks(
                    output, reference, shape=(args.M, args.N, args.K),
                    stage=f"AG GEMM overlap={overlap} iteration={iteration}",
                    atol=2e-2 if dtype == torch.bfloat16 else 2e-3,
                    rtol=2e-2 if dtype == torch.bfloat16 else 2e-3,
                )
        if rank == 0:
            print(f"PASS: {args.check_iters} changing-input checks, both overlapped and serial TLE; "
                  f"world={world_size}, local_world={local_world_size}, "
                  f"gemm_ctas={ctx.num_gemm_sms}, config={dataclasses.asdict(config)}", flush=True)
        if args.iters:
            timings = {}
            for name, fn in (
                ("torch AG + GEMM", lambda: torch_ag_gemm(A, B)),
                ("serial TLE AG + GEMM", lambda: ag_gemm(A, B, ctx, overlap=False)),
                ("overlapped TLE AG + GEMM", lambda: ag_gemm(A, B, ctx)),
            ):
                timings[name] = _time_ms(fn, args.warmup, args.iters)
            if rank == 0:
                for name, duration in timings.items():
                    print(f"{name}: median(max-rank)={duration:.3f} ms", flush=True)
    finally:
        if ctx is not None:
            ctx.finalize()
        else:
            tle.cleanup_communicator()


if __name__ == "__main__":
    main()
