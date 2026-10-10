# source /opt/dtk/env.sh
# export LD_LIBRARY_PATH="/opt/dtk/lib/dushmem:${LD_LIBRARY_PATH:-}"
# unset DUSHMEM_BOOTSTRAP
# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
#   --master_addr=127.0.0.1 --master_port=29501 ag-gemm.py

"""Allgather each rank's A shard on a DUSHMEM stream, then SGEMM."""

import ctypes
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
M_LOCAL, K, N = 32, 32, 64


# deferred=True is the fast path: clang bitcode is linked with its target attributes.
@dialect(name="hcu", file=HERE / "ag-gemm-device.hip", extern_func_name="ag_gemm", deferred=True)
def ag_gemm(*args, **kwargs):
    ...


@triton.jit
def ag_gemm_kernel(c_ptr, a_ptr, b_ptr, m, n, k):
    tle_raw.call(ag_gemm, [c_ptr, a_ptr, b_ptr, m, n, k], output_indices=[])


def _load_host():
    host = compile_host_library(HERE / "ag-gemm-host.hip", Path("/tmp/dushmem-ag-gemm-host.so"))
    host.ag_gemm_prepare.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ]
    host.ag_gemm_prepare.restype = ctypes.c_int
    host.ag_gemm_allgather_on_stream.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ]
    host.ag_gemm_allgather_on_stream.restype = ctypes.c_int
    host.ag_gemm_copy_b_on_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    host.ag_gemm_copy_b_on_stream.restype = ctypes.c_int
    host.ag_gemm_finish.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    host.ag_gemm_finish.restype = ctypes.c_int
    return host


def main() -> None:
    group = init_torch_distributed()
    init_dushmem_by_torch_pg(load_common_host(), group)
    host = _load_host()

    mype = ctypes.c_int()
    npes = ctypes.c_int()
    mype_node = ctypes.c_int()
    stream_ptr = ctypes.c_void_p()
    a_local = ctypes.c_void_p()
    a_full = ctypes.c_void_p()
    b_ptr = ctypes.c_void_p()
    c_ptr = ctypes.c_void_p()
    status = host.ag_gemm_prepare(
        M_LOCAL, K, N,
        ctypes.byref(mype), ctypes.byref(npes), ctypes.byref(mype_node),
        ctypes.byref(stream_ptr),
        ctypes.byref(a_local), ctypes.byref(a_full), ctypes.byref(b_ptr), ctypes.byref(c_ptr),
    )
    if status != 0:
        raise SystemExit(f"ag_gemm_prepare failed: {status}")

    device = torch.device("cuda", mype_node.value)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    m = M_LOCAL * npes.value
    a_ref = torch.randn((m, K), device=device)
    b_ref = torch.randn((K, N), device=device)
    shard = a_ref[mype.value * M_LOCAL:(mype.value + 1) * M_LOCAL].contiguous()
    torch.cuda.synchronize()

    rc = host.ag_gemm_allgather_on_stream(stream_ptr, a_full, a_local, ctypes.c_void_p(shard.data_ptr()), M_LOCAL * K)
    if rc != 0:
        raise SystemExit(f"fcollect failed: {rc}")
    rc = host.ag_gemm_copy_b_on_stream(stream_ptr, b_ptr, ctypes.c_void_p(b_ref.data_ptr()), K * N)
    if rc != 0:
        raise SystemExit(f"copy B failed: {rc}")

    c = tensor_from_pointer(c_ptr, (m, N), torch.float32, device)
    stream = torch.cuda.ExternalStream(stream_ptr.value)
    grid = (triton.cdiv(N, TILE), triton.cdiv(m, TILE))
    with torch.cuda.stream(stream):
        ag_gemm_kernel[grid](
            c,
            tensor_from_pointer(a_full, (m, K), torch.float32, device),
            tensor_from_pointer(b_ptr, (K, N), torch.float32, device),
            m, N, K,
            num_warps=4,
        )
    stream.synchronize()

    ref = a_ref @ b_ref
    err = (c - ref).abs().max().item()
    ok = torch.allclose(c, ref, rtol=1e-3, atol=1e-3)
    print(f"ag-gemm pe={mype.value} allclose={bool(ok)} max_abs_err={err}")
    host.ag_gemm_finish(stream_ptr, a_local, a_full, b_ptr, c_ptr)
    torch.distributed.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
