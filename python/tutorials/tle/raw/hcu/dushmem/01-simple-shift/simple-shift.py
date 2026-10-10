import ctypes
import os
from pathlib import Path

import torch
import triton
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect

HERE = Path(__file__).parent


# deferred=True is the fast path: clang bitcode is linked with its target attributes.
@dialect(
    name="hcu",
    library="dushmem",
    file=HERE / "simple-shift-device.hip",
    extern_func_name="simple_shift",
    deferred=True,
)
def simple_shift(*args, **kwargs):
    ...


@triton.jit
def simple_shift_kernel(destination_ptr):
    tle_raw.call(simple_shift, [destination_ptr], output_indices=[])


def _load_host():
    library = Path("/tmp/dushmem-simple-shift-host.so")
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
        str(HERE / "simple-shift-host.hip"),
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
    return ctypes.CDLL(str(library))


def run_once(host, iters: int = 1):
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    mype = ctypes.c_int()
    npes = ctypes.c_int()
    mype_node = ctypes.c_int()
    stream_ptr = ctypes.c_void_p()
    destination_ptr = ctypes.c_void_p()
    status = host.simple_shift_before_launch(
        ctypes.byref(mype),
        ctypes.byref(npes),
        ctypes.byref(mype_node),
        ctypes.byref(stream_ptr),
        ctypes.byref(destination_ptr),
    )
    if status != 0:
        raise RuntimeError(f"simple_shift_before_launch failed: {status}")
    torch.cuda.set_device(mype_node.value)

    address = destination_ptr.value
    storage = torch._C._construct_storage_from_data_pointer(address, torch.device("cuda", mype_node.value), 4)
    destination = torch.empty(0, dtype=torch.int32, device="cuda").set_(storage).view(1)
    stream = torch.cuda.ExternalStream(stream_ptr.value)

    with torch.cuda.stream(stream):
        for _ in range(iters):
            simple_shift_kernel[(1, )](destination, num_warps=1)
    return host, stream, stream_ptr, destination, destination_ptr, mype, npes


def main() -> None:
    host = _load_host()
    host, stream, stream_ptr, destination, destination_ptr, mype, npes = run_once(host)
    result = host.simple_shift_after_launch(stream_ptr, destination_ptr, mype, npes)
    if result != 0:
        raise SystemExit(f"PE {mype.value}: shift mismatch")
    host.simple_shift_finalize()


if __name__ == "__main__":
    main()
