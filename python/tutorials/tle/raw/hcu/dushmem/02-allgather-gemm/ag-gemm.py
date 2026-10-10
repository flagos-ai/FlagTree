"""Allgather each rank's A shard on a DUSHMEM stream, then SGEMM."""

import ctypes
import os
from pathlib import Path

import torch
import triton
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect

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
    library = Path("/tmp/dushmem-ag-gemm-host.so")
    command = [
        "/opt/dtk/bin/hipcc",
        "-shared",
        "-fPIC",
        "-fgpu-rdc",
        "--offload-arch=gfx936",
        "-O3",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
        "-mcode-object-version=4",
        "-I/opt/dtk/include",
        "-I/opt/dtk/include/dushmem",
        "-I/opt/mpi/include",
        "-L/opt/dtk/lib/dushmem",
        "-L/opt/mpi/lib",
        str(HERE / "ag-gemm-host.hip"),
        "-o",
        str(library),
        "-ldushmem_host",
        "-ldushmem_device",
        "-lmpi",
    ]
    import subprocess
    build = subprocess.run(command, capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError(f"host library build failed:\n{build.stderr}")
    host = ctypes.CDLL(str(library))
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


def _view(ptr, shape, device):
    nbytes = 1
    for dim in shape:
        nbytes *= dim
    nbytes *= 4
    storage = torch._C._construct_storage_from_data_pointer(ptr, device, nbytes)
    return torch.empty(0, dtype=torch.float32, device=device).set_(storage).view(*shape)


def main() -> None:
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
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

    c = _view(c_ptr.value, (m, N), device)
    stream = torch.cuda.ExternalStream(stream_ptr.value)
    grid = (triton.cdiv(N, TILE), triton.cdiv(m, TILE))
    with torch.cuda.stream(stream):
        ag_gemm_kernel[grid](c, _view(a_full.value, (m, K), device), _view(b_ptr.value, (K, N), device), m, N, K,
                             num_warps=4)
    stream.synchronize()

    ref = a_ref @ b_ref
    err = (c - ref).abs().max().item()
    ok = torch.allclose(c, ref, rtol=1e-3, atol=1e-3)
    print(f"ag-gemm pe={mype.value} allclose={bool(ok)} max_abs_err={err}")
    host.ag_gemm_finish(stream_ptr, a_local, a_full, b_ptr, c_ptr)
    host.ag_gemm_finalize()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
