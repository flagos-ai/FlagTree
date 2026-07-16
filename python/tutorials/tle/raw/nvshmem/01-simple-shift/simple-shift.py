import os
import ctypes
import torch
import triton
import triton.experimental.tle.language.raw as tle_raw

from pathlib import Path
from triton.experimental.tle.raw import dialect

from triton.experimental.tle.raw.nvshmem.utils import (
    load_common_host,
    load_host,
    init_torch_distributed,
    init_nvshmem_by_torch_pg,
    tensor_from_pointer,
)


@dialect(
    name="cuda",
    compiler="clang",
    file=(Path(__file__).parent / "simple-shift-device.cu"),
    extern_func_name="simple_shift",
)
def simple_shift(*args, **kwargs):
    ...


@triton.jit
def simple_shift_kernel(destination_ptr, ):
    tle_raw.call(simple_shift, [destination_ptr])


def simpe_shift():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)

    group = init_torch_distributed()
    host_source = Path(__file__).with_name("simple-shift-host.cu")
    host_lib = load_host(source=host_source)
    common = load_common_host()
    init_nvshmem_by_torch_pg(common, group)

    mype = ctypes.c_int()
    npes = ctypes.c_int()
    mype_in_node = ctypes.c_int()
    npes_in_node = ctypes.c_int()
    stream_ptr = ctypes.c_void_p()
    destination_ptr = ctypes.c_void_p()
    host_data_ptr = ctypes.c_void_p()
    host_lib.simple_shift_before_launch(
        ctypes.byref(mype),
        ctypes.byref(npes),
        ctypes.byref(mype_in_node),
        ctypes.byref(npes_in_node),
        ctypes.byref(stream_ptr),
        ctypes.byref(destination_ptr),
        ctypes.byref(host_data_ptr),
    )

    destination = tensor_from_pointer(
        destination_ptr,
        (1, ),
        torch.int32,
        device,
    )
    stream = torch.cuda.ExternalStream(stream_ptr.value, device=device)

    with torch.cuda.stream(stream):
        simple_shift_kernel[(1, )](destination)

    host_lib.simple_shift_after_launch(
        stream_ptr,
        destination_ptr,
        host_data_ptr,
        mype_in_node.value,
        npes_in_node.value,
    )

    common.nvshmem_finalize_from_torch_distributed()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    simpe_shift()
