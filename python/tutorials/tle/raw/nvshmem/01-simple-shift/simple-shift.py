import ctypes
import torch
import triton
import triton.experimental.tle.language.raw as tle_raw

from pathlib import Path
from triton.experimental.tle.raw import dialect

from common.utils import (
    install_cumodule_hook,
    load_library,
    prepare_clang_bitcode,
)


@dialect(
    name="cuda",
    compiler="clang",
    target="bc",
    file=(Path(__file__).parent / "simple-shift-device.cu"),
    extern_file=(Path(__file__).parent / "simple-shift-device-extern-call.py"),
    extern_func_name="simple_shift",
)
def simple_shift(*args, **kwargs):
    ...


@triton.jit
def simple_shift_kernel(destination_ptr, ):
    tle_raw.call(simple_shift, [destination_ptr])


def tensor_from_pointer(pointer, shape, dtype, device):
    num_elements = 1
    for extent in shape:
        num_elements *= extent
    storage = torch._C._construct_storage_from_data_pointer(
        pointer.value,
        device,
        num_elements * dtype.itemsize,
    )
    return torch.empty(0, dtype=dtype, device=device).set_(storage).view(shape)


def simpe_shift():
    common_path = Path(__file__).parents[1] / "common" / "common-host.so"
    host_path = Path(__file__).with_name("simple-shift-host.so")
    common_lib = load_library(common_path)
    host_lib = load_library(host_path)

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
    install_cumodule_hook(common_lib)

    extern_libs = prepare_clang_bitcode(
        common_lib,
        mype_in_node.value,
        Path(__file__).with_name("simple-shift-device.bc"),
        simple_shift,
    )

    with torch.cuda.stream(stream):
        simple_shift_kernel[(1, )](destination, extern_libs=extern_libs)

    host_lib.simple_shift_after_launch(
        stream_ptr,
        destination_ptr,
        host_data_ptr,
        mype_in_node.value,
        npes_in_node.value,
    )


if __name__ == "__main__":
    simpe_shift()
