import argparse
import ctypes
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect

from triton.experimental.tle.raw.nvshmem.utils import (
    init_nvshmem_by_torch_pg,
    init_torch_distributed,
    load_common_host,
    load_host,
    tensor_from_pointer,
)

PACK_BYTES = 16
PACK_U32_WORDS = PACK_BYTES // 4
POISON_U32 = 0x80000000


@dialect(
    name="cuda",
    compiler="clang",
    file=Path(__file__).with_name("allreduce-device.cu"),
    extern_func_name="allreduce_one_shot_push_reduce_tp8",
)
def one_shot_push_reduce_tp8(*args, **kwargs):
    ...


@triton.jit
def one_shot_kernel(
    input_ptr,
    output_ptr,
    peer_scratch_ptrs,
    local_scratch,
    packs_per_rank,
    rank: tl.constexpr,
):
    rank_i32 = tl.full((), rank, tl.int32)
    tle_raw.call(
        one_shot_push_reduce_tp8,
        [
            input_ptr,
            output_ptr,
            peer_scratch_ptrs,
            local_scratch,
            packs_per_rank,
            rank_i32,
        ],
    )


def configure_host_library(host):
    host.allreduce_workspace_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    host.allreduce_workspace_create.restype = ctypes.c_int

    host.allreduce_peer_scratch_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    host.allreduce_peer_scratch_ptr.restype = ctypes.c_void_p

    host.allreduce_workspace_destroy.argtypes = [ctypes.c_void_p]
    host.allreduce_workspace_destroy.restype = ctypes.c_int

    host.allreduce_node_team_size.argtypes = []
    host.allreduce_node_team_size.restype = ctypes.c_int


def create_peer_pointer_table(host, scratch_ptr, world_size, device):
    peer_addresses = []
    for peer in range(world_size):
        address = host.allreduce_peer_scratch_ptr(scratch_ptr, peer)
        if not address:
            raise RuntimeError(f"rank cannot directly access peer {peer} scratch")
        peer_addresses.append(address)
    return torch.tensor(peer_addresses, dtype=torch.uint64, device=device)


def parse_args():
    parser = argparse.ArgumentParser(description="TLE-raw one-shot AllReduce")
    parser.add_argument("--case", choices=("allocation", "check"), default="check")
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--N", type=int, default=8192)
    return parser.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    if world_size != 8:
        raise ValueError(f"the one-shot kernel is specialized for TP8, got {world_size=}")
    if world_size != local_world_size:
        raise ValueError("the first version supports a single node only")
    if args.M <= 0 or args.N <= 0:
        raise ValueError("M and N must be positive")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    group = init_torch_distributed()

    host_source = Path(__file__).with_name("allreduce-host.cu")
    host = load_host(host_source)
    common = load_common_host()
    init_nvshmem_by_torch_pg(common, group)
    configure_host_library(host)

    scratch_ptr = ctypes.c_void_p()

    elements = args.M * args.N
    element_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
    message_bytes = elements * element_bytes
    if message_bytes % PACK_BYTES != 0:
        raise ValueError("the first version requires the BF16 message size to be divisible by 16 bytes")
    packs_per_rank = (message_bytes + PACK_BYTES - 1) // PACK_BYTES

    result = host.allreduce_workspace_create(
        world_size,
        packs_per_rank,
        ctypes.byref(scratch_ptr),
    )
    if result != 0:
        raise RuntimeError(f"allreduce_workspace_create failed with code {result}")
    if host.allreduce_node_team_size() != world_size:
        raise RuntimeError("not all ranks are in the same NVSHMEM node team")

    # uint32 gives us an exact view of the sentinel bit pattern.  The
    # logical layout is scratch[source_rank][pack_index][word_in_pack].
    scratch = tensor_from_pointer(
        scratch_ptr,
        (world_size, packs_per_rank, PACK_U32_WORDS),
        torch.uint32,
        device,
    )
    scratch.fill_(POISON_U32)
    peer_ptrs = create_peer_pointer_table(host, scratch_ptr, world_size, device)

    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)

    if not torch.all(scratch == POISON_U32).item():
        raise AssertionError("scratch sentinel initialization failed")
    if peer_ptrs.numel() != world_size or torch.any(peer_ptrs == 0).item():
        raise AssertionError("peer pointer table is incomplete")

    if args.case == "allocation" and rank == 0:
        print(
            "[allocation] Pass: "
            f"shape=({args.M}, {args.N}), bytes={message_bytes}, "
            f"packs_per_rank={packs_per_rank}, scratch_bytes={scratch.numel() * 4}",
            flush=True,
        )
    elif args.case == "check":
        input_tensor = torch.full(
            (args.M, args.N),
            rank + 1,
            dtype=torch.bfloat16,
            device=device,
        )
        output_tensor = torch.empty_like(input_tensor)

        threads_per_cta = 128
        grid = (triton.cdiv(packs_per_rank, threads_per_cta), )
        one_shot_kernel[grid](
            input_tensor,
            output_tensor,
            peer_ptrs,
            scratch,
            packs_per_rank,
            rank=rank,
            num_warps=threads_per_cta // 32,
        )
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)

        expected = float(world_size * (world_size + 1) // 2)
        max_error = (output_tensor.float() - expected).abs().max().item()
        if max_error != 0.0:
            raise AssertionError(f"rank {rank}: max error is {max_error}")
        if rank == 0:
            print(
                "[check] Pass: "
                f"shape=({args.M}, {args.N}), bytes={message_bytes}, "
                f"packs_per_rank={packs_per_rank}, expected={expected}",
                flush=True,
            )

    if scratch_ptr.value:
        result = host.allreduce_workspace_destroy(scratch_ptr)
        if result != 0:
            raise RuntimeError(f"allreduce_workspace_destroy failed with code {result}")
    common.nvshmem_finalize_from_torch_distributed()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
