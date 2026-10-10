# source /opt/dtk/env.sh
# export LD_LIBRARY_PATH="/opt/dtk/lib/dushmem:${LD_LIBRARY_PATH:-}"
# unset DUSHMEM_BOOTSTRAP
# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
#   --master_addr=127.0.0.1 --master_port=29502 gemm-ar.py

"""K-split SGEMM, then a sum allreduce on a DUSHMEM stream."""

import ctypes
import os
from pathlib import Path

import torch
import triton
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.hcu.utils import (
    compile_host_library,
    init_dushmem_by_torch_pg,
    init_torch_distributed,
    load_common_host,
    tensor_from_pointer,
)

HERE = Path(__file__).parent
TILE = 16
M, K, N = 64, 32, 64


# deferred=True is the fast path: clang bitcode is linked with its target attributes.
@dialect(name="hcu", file=HERE / "gemm-ar-device.hip", extern_func_name="gemm_partial", deferred=True)
def gemm_partial(*args, **kwargs):
    ...


@triton.jit
def gemm_partial_kernel(c_ptr, a_ptr, b_ptr, m, n, k):
    tle_raw.call(gemm_partial, [c_ptr, a_ptr, b_ptr, m, n, k], output_indices=[])


def _load_host():
    host = compile_host_library(HERE / "gemm-ar-host.hip", Path("/tmp/dushmem-gemm-ar-host.so"))
    host.gemm_ar_prepare.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ]
    host.gemm_ar_prepare.restype = ctypes.c_int
    host.gemm_ar_copy_on_stream.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
    ]
    host.gemm_ar_copy_on_stream.restype = ctypes.c_int
    host.gemm_ar_sum_reduce_on_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    host.gemm_ar_sum_reduce_on_stream.restype = ctypes.c_int
    host.gemm_ar_finish.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    host.gemm_ar_finish.restype = ctypes.c_int
    return host


def main() -> None:
    group = init_torch_distributed()
    init_dushmem_by_torch_pg(load_common_host(), group)
    host = _load_host()

    mype = ctypes.c_int()
    npes = ctypes.c_int()
    mype_node = ctypes.c_int()
    stream_ptr = ctypes.c_void_p()
    a_ptr = ctypes.c_void_p()
    b_ptr = ctypes.c_void_p()
    partial = ctypes.c_void_p()
    reduced = ctypes.c_void_p()
    world = int(os.environ["WORLD_SIZE"])
    if K % world != 0:
        raise SystemExit(f"K={K} is not divisible by npes={world}")
    k_local = K // world
    status = host.gemm_ar_prepare(
        M, k_local, N,
        ctypes.byref(mype), ctypes.byref(npes), ctypes.byref(mype_node),
        ctypes.byref(stream_ptr),
        ctypes.byref(a_ptr), ctypes.byref(b_ptr), ctypes.byref(partial), ctypes.byref(reduced),
    )
    if status != 0:
        raise SystemExit(f"gemm_ar_prepare failed: {status}")
    if npes.value != world or K % npes.value != 0:
        raise SystemExit(f"npes mismatch: mpi={world} dushmem={npes.value}")

    device = torch.device("cuda", mype_node.value)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    a_ref = torch.randn((M, K), device=device)
    b_ref = torch.randn((K, N), device=device)
    k0 = mype.value * k_local
    a_shard = a_ref[:, k0:k0 + k_local].contiguous()
    b_shard = b_ref[k0:k0 + k_local, :].contiguous()
    torch.cuda.synchronize()

    rc = host.gemm_ar_copy_on_stream(
        stream_ptr, a_ptr, b_ptr,
        ctypes.c_void_p(a_shard.data_ptr()), ctypes.c_void_p(b_shard.data_ptr()),
        M * k_local, k_local * N,
    )
    if rc != 0:
        raise SystemExit(f"copy failed: {rc}")

    partial_view = tensor_from_pointer(partial, (M, N), torch.float32, device)
    stream = torch.cuda.ExternalStream(stream_ptr.value)
    grid = (triton.cdiv(N, TILE), triton.cdiv(M, TILE))
    with torch.cuda.stream(stream):
        gemm_partial_kernel[grid](
            partial_view,
            tensor_from_pointer(a_ptr, (M, k_local), torch.float32, device),
            tensor_from_pointer(b_ptr, (k_local, N), torch.float32, device),
            M, N, k_local,
            num_warps=4,
        )
    rc = host.gemm_ar_sum_reduce_on_stream(stream_ptr, reduced, partial, M * N)
    if rc != 0:
        raise SystemExit(f"sum reduce failed: {rc}")
    stream.synchronize()

    got = tensor_from_pointer(reduced, (M, N), torch.float32, device)
    ref = a_ref @ b_ref
    err = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, rtol=1e-3, atol=1e-3)
    print(f"gemm-ar pe={mype.value} allclose={bool(ok)} max_abs_err={err}")
    host.gemm_ar_finish(stream_ptr, a_ptr, b_ptr, partial, reduced)
    torch.distributed.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
