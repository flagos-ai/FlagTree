import ctypes
import torch
import triton
import triton.experimental.tle.language.raw as tle_raw

from pathlib import Path
from triton.experimental.tle.raw import dialect

from triton.experimental.tle.raw.nvshmem.utils import (
    load_host,
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
    host_source = Path(__file__).with_name("simple-shift-host.cu")
    host_lib = load_host(source=host_source)

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

    device = triton.runtime.driver.active.get_active_torch_device()
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


if __name__ == "__main__":
    simpe_shift()
