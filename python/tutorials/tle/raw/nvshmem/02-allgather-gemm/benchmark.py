import argparse
import importlib.util
import os
from pathlib import Path

import torch
import torch.distributed

LAYER_CONFIGS = {
    "LLaMA-7B": {"N": 11008, "K": 4096},
    "LLaMA-3.1-8B": {"N": 14336, "K": 4096},
    "LLaMA-3.1-70B": {"N": 28672, "K": 8192},
    "LLaMA-3.1-405B": {"N": 53248, "K": 16384},
    "Mistral-7B": {"N": 14336, "K": 4096},
    "Qwen2-72B": {"N": 29568, "K": 8192},
    "GPT-3-175B": {"N": 49152, "K": 12288},

    # "Custom-2-1": {"N": 1024 * 4, "K": 8192},
    # "Custom-2-2": {"N": 8192 * 4, "K": 8192},
    # "Custom-2-3": {"N": 28672 * 4, "K": 8192},
    # "Custom-2-4": {"N": 8192 * 4, "K": 28672},
}


def load_ag_gemm_module():
    source = Path(__file__).with_name("ag-gemm.py")
    spec = importlib.util.spec_from_file_location("flagtree_tle_raw_nvshmem_ag_gemm", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load all-gather GEMM implementation from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def dist_print(
    *print_args,
    need_sync=False,
    allowed_ranks=None,
    group=None,
    **kwargs,
):
    rank = group.rank() if group is not None else int(os.environ.get("RANK", 0))
    world_size = group.size() if group is not None else int(os.environ.get("WORLD_SIZE", 1))
    if allowed_ranks is None:
        allowed_ranks = [0]
    elif allowed_ranks == "all":
        allowed_ranks = list(range(world_size))

    for output_rank in allowed_ranks:
        if need_sync:
            torch.distributed.barrier(group=group)
        if rank == output_rank:
            print(*print_args, **kwargs)


def rand_tensor(shape, dtype: torch.dtype, device):
    return torch.rand(shape, dtype=dtype, device=device) * 2 - 1


def assert_allclose(x: torch.Tensor, y: torch.Tensor, rtol, atol):
    if torch.any(x.isnan()):
        raise RuntimeError(f"x has nan: {x}")
    if torch.any(y.isnan()):
        raise RuntimeError(f"y has nan: {y}")
    if torch.any(x.isinf()):
        raise RuntimeError(f"x has inf: {x}")
    if torch.any(y.isinf()):
        raise RuntimeError(f"y has inf: {y}")
    torch.testing.assert_close(x, y, rtol=rtol, atol=atol)
    print("✅ all close!", flush=True)


def torch_ag_gemm(
    pg: torch.distributed.ProcessGroup,
    A: torch.Tensor,
    B: torch.Tensor,
):
    M_per_rank, K = A.shape
    A_full = torch.empty([M_per_rank * pg.size(), K], dtype=A.dtype, device=A.device)
    torch.distributed.all_gather_into_tensor(A_full, A, pg)
    return torch.matmul(A_full, B)


def make_data(M, N, K, dtype: torch.dtype, trans_b, tp_group: torch.distributed.ProcessGroup):
    rank = tp_group.rank()
    num_ranks = tp_group.size()
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks
    scale = (rank + 1) * 0.01

    current_device = torch.cuda.current_device()
    A = rand_tensor([M_per_rank, K], dtype=dtype, device=current_device) * scale
    if trans_b:
        B = rand_tensor([N_per_rank, K], dtype=dtype, device=current_device).T * scale
    else:
        B = rand_tensor([K, N_per_rank], dtype=dtype, device=current_device) * scale

    return A, B


def perf_test(M, config, pg: torch.distributed.ProcessGroup):
    N = config["N"]
    K = config["K"]
    rank = pg.rank()
    world_size = pg.size()

    if rank == 0:
        print(f"test shape: M {M}, N {N}, K {K}")

    assert M % world_size == 0
    assert N % world_size == 0

    A, B = make_data(M, N, K, dtype, args.trans_b, pg)
    A_gathered = torch.empty((M, K), dtype=A.dtype, device=A.device)
    torch.distributed.all_gather_into_tensor(A_gathered, A, pg)

    def _torch_func():
        return torch_ag_gemm(pg, A, B)

    workspace_ptr, ready_ptr, workspace, ready, mype, local_pe, local_npes = ag_gemm.create_workspace(
        HOST,
        world_size,
        M // world_size,
        K,
        dtype,
        A.device,
    )
    if mype.value != rank or local_pe.value != int(os.environ["LOCAL_RANK"]):
        raise RuntimeError("NVSHMEM rank mapping does not match torch.distributed")
    if local_npes.value != LOCAL_WORLD_SIZE:
        raise RuntimeError("NVSHMEM local team size does not match LOCAL_WORLD_SIZE")

    ag_intranode_stream = torch.cuda.Stream(priority=-1)
    is_multinode = world_size > LOCAL_WORLD_SIZE
    ag_internode_stream = torch.cuda.Stream(priority=-1) if is_multinode else None
    fake_ready = torch.ones_like(ready)

    # AG-only excludes the local copy included in fused AG-GEMM.
    def _triton_ag_func():
        current_stream = torch.cuda.current_stream()
        HOST.ag_gemm_barrier_all_on_stream(current_stream.cuda_stream)

        if is_multinode:
            ag_internode_stream.wait_stream(current_stream)
        ag_intranode_stream.wait_stream(current_stream)

        if not is_multinode:
            ag_gemm.cp_engine_producer_all_gather_intra_node(
                host=HOST,
                local_tensor=A,
                ag_buffer=workspace,
                signal_buffer=ready,
                rank=rank,
                local_world_size=LOCAL_WORLD_SIZE,
                ag_intranode_stream=ag_intranode_stream,
            )
        else:
            ag_gemm.cp_engine_producer_all_gather_inter_node(
                host=HOST,
                local_tensor=A,
                ag_buffer=workspace,
                signal_buffer=ready,
                rank=rank,
                local_world_size=LOCAL_WORLD_SIZE,
                world_size=world_size,
                ag_intranode_stream=ag_intranode_stream,
                ag_internode_stream=ag_internode_stream,
            )

        if is_multinode:
            current_stream.wait_stream(ag_internode_stream)
        current_stream.wait_stream(ag_intranode_stream)

    def _triton_gemm_func():
        return ag_gemm.gemm_persistent(
            A_gathered,
            B,
            rank,
            world_size,
            fake_ready,
            LOCAL_WORLD_SIZE,
            trans_b=args.trans_b,
        )

    def _triton_func():
        C = torch.empty((M, N // world_size), dtype=dtype, device=A.device)
        ag_gemm.local_copy(A, workspace, ready, rank)
        ag_gemm.ag_gemm_op(
            A,
            B,
            C,
            rank,
            world_size,
            workspace,
            ready,
            ag_intranode_stream,
            ag_internode_stream,
            HOST,
            LOCAL_WORLD_SIZE,
            trans_b=args.trans_b,
        )
        return C

    try:
        for _ in range(5):
            A, B = make_data(M, N, K, dtype, args.trans_b, pg)
            HOST.ag_gemm_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
            C = _triton_func()

        C_golden = _torch_func()

        for check_rank in range(world_size):
            torch.distributed.barrier(pg)
            if rank == check_rank:
                assert_allclose(C_golden, C, atol=1e-3, rtol=1e-3)

        _, triton_duration_ms = ag_gemm.perf_func(
            _triton_func,
            iters=args.iters,
            warmup_iters=args.warmup_iters,
        )
        _, triton_ag_duration_ms = ag_gemm.perf_func(
            _triton_ag_func,
            iters=args.iters,
            warmup_iters=args.warmup_iters,
        )
        _, triton_gemm_duration_ms = ag_gemm.perf_func(
            _triton_gemm_func,
            iters=args.iters,
            warmup_iters=args.warmup_iters,
        )
        _, torch_ag_duration_ms = ag_gemm.perf_func(
            lambda: torch.distributed.all_gather_into_tensor(A_gathered, A, pg),
            iters=args.iters,
            warmup_iters=args.warmup_iters,
        )
        _, torch_gemm_duration_ms = ag_gemm.perf_func(
            lambda: torch.matmul(A_gathered, B),
            iters=args.iters,
            warmup_iters=args.warmup_iters,
        )
        _, torch_duration_ms = ag_gemm.perf_func(
            _torch_func,
            iters=args.iters,
            warmup_iters=args.warmup_iters,
        )

        dist_print(
            f"Rank {rank} latency (ms): "
            f"triton total={triton_duration_ms:.2f}, ag_only={triton_ag_duration_ms:.2f}, "
            f"triton_gemm_only={triton_gemm_duration_ms:.2f}, "
            f"torch total={torch_duration_ms:.2f}, ag_only={torch_ag_duration_ms:0.2f}, "
            f"torch_gemm_only={torch_gemm_duration_ms:0.2f} "
            f"speedup {torch_duration_ms / triton_duration_ms:.2f}",
            need_sync=True,
            allowed_ranks="all",
            group=pg,
        )

        return triton_duration_ms, torch_duration_ms
    finally:
        result = HOST.ag_gemm_workspace_destroy(workspace_ptr, ready_ptr)
        if result != 0:
            raise RuntimeError(f"workspace destruction failed: {result}")


def dump_csv(results, M, world_size):
    csv_dir = Path("csv")
    csv_dir.mkdir(exist_ok=True)
    csv_file = csv_dir / f"perf_ag_gemm_{world_size}_ranks.csv"
    header = [
        "Model",
        "M",
        "N",
        "K",
        "FlagTree ag gemm latency (ms)",
        "torch ag gemm latency (ms)",
        "speed up",
    ]

    with csv_file.open("w") as fout:
        print(",".join(header), file=fout)
        for (model, config), (triton_perf, torch_perf) in zip(LAYER_CONFIGS.items(), results):
            row = [
                model,
                f"{M:d}",
                f"{config['N']:d}",
                f"{config['K']:d}",
                f"{triton_perf:02f}",
                f"{torch_perf:02f}",
                f"{torch_perf / triton_perf:02f}",
            ]
            print(",".join(row), file=fout, flush=True)

    print(f"csv file is dumped into {csv_file}")


if __name__ == "__main__":
    args = parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 8))

    if world_size < 2:
        raise RuntimeError("all-gather GEMM requires at least two GPUs")
    if world_size % LOCAL_WORLD_SIZE != 0:
        raise RuntimeError("WORLD_SIZE must be divisible by LOCAL_WORLD_SIZE")

    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("Allgather-GEMM requires an sm90 or newer GPU")

    ag_gemm = load_ag_gemm_module()
    TP_GROUP = ag_gemm.init_torch_distributed()
    HOST = ag_gemm.load_host(Path(__file__).with_name("ag-gemm-host.cu"))
    COMMON = ag_gemm.load_common_host()
    ag_gemm.configure_host_library(HOST)
    ag_gemm.init_nvshmem_by_torch_pg(COMMON, TP_GROUP)

    results = []
    try:
        for config in LAYER_CONFIGS.values():
            triton_perf, torch_perf = perf_test(args.M, config, TP_GROUP)
            results.append((triton_perf, torch_perf))

        if args.dump_csv and TP_GROUP.rank() == 0:
            dump_csv(results, args.M, TP_GROUP.size())
    finally:
        result = COMMON.nvshmem_finalize_from_torch_distributed()
        if result != 0:
            raise RuntimeError(f"NVSHMEM finalization failed: {result}")
        torch.cuda.synchronize()
        torch.distributed.barrier(group=TP_GROUP)
        torch.distributed.destroy_process_group()
