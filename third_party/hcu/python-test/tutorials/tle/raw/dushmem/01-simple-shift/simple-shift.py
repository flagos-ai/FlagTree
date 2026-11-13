# source /opt/dtk/env.sh
# export LD_LIBRARY_PATH="/opt/dtk/lib/dushmem:${LD_LIBRARY_PATH:-}"
# unset DUSHMEM_BOOTSTRAP
# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
#   --master_addr=127.0.0.1 --master_port=29500 simple-shift.py

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
    return compile_host_library(HERE / "simple-shift-host.hip", Path("/tmp/dushmem-simple-shift-host.so"))


def run_once(host, iters: int = 1):
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

    device = torch.device("cuda", mype_node.value)
    destination = tensor_from_pointer(destination_ptr, (1, ), torch.int32, device)
    stream = torch.cuda.ExternalStream(stream_ptr.value)

    with torch.cuda.stream(stream):
        for _ in range(iters):
            simple_shift_kernel[(1, )](destination, num_warps=1)
    return host, stream, stream_ptr, destination, destination_ptr, mype, npes


def main() -> None:
    group = init_torch_distributed()
    init_dushmem_by_torch_pg(load_common_host(), group)
    host = _load_host()
    host, stream, stream_ptr, destination, destination_ptr, mype, npes = run_once(host)
    result = host.simple_shift_after_launch(stream_ptr, destination_ptr, mype, npes)
    if result != 0:
        raise SystemExit(f"PE {mype.value}: shift mismatch")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
