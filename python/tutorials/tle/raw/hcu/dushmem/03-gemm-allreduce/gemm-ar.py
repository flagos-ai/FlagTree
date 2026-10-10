"""K-split SGEMM, then a sum allreduce on a DUSHMEM stream."""

import ctypes
import os
from pathlib import Path

import torch
import triton
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect

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
    library = Path("/tmp/dushmem-gemm-ar-host.so")
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
        str(HERE / "gemm-ar-host.hip"),
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


def _view(ptr, shape, device):
    nbytes = 4
    for dim in shape:
        nbytes *= dim
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
    a_ptr = ctypes.c_void_p()
    b_ptr = ctypes.c_void_p()
    partial = ctypes.c_void_p()
    reduced = ctypes.c_void_p()
    # k_local is filled after we know npes. Prepare needs it, so init with K // world from MPI.
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
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

    partial_view = _view(partial.value, (M, N), device)
    stream = torch.cuda.ExternalStream(stream_ptr.value)
    grid = (triton.cdiv(N, TILE), triton.cdiv(M, TILE))
    with torch.cuda.stream(stream):
        gemm_partial_kernel[grid](
            partial_view,
            _view(a_ptr.value, (M, k_local), device),
            _view(b_ptr.value, (k_local, N), device),
            M, N, k_local,
            num_warps=4,
        )
    rc = host.gemm_ar_sum_reduce_on_stream(stream_ptr, reduced, partial, M * N)
    if rc != 0:
        raise SystemExit(f"sum reduce failed: {rc}")
    stream.synchronize()

    got = _view(reduced.value, (M, N), device)
    ref = a_ref @ b_ref
    err = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, rtol=1e-3, atol=1e-3)
    print(f"gemm-ar pe={mype.value} allclose={bool(ok)} max_abs_err={err}")
    host.gemm_ar_finish(stream_ptr, a_ptr, b_ptr, partial, reduced)
    host.gemm_ar_finalize()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
