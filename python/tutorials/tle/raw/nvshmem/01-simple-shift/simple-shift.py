import os
import subprocess
import ctypes
import torch
import triton
import triton.knobs as knobs
import triton.experimental.tle.language.raw as tle_raw

from pathlib import Path
from triton.experimental.tle.raw import dialect

NVSHMEM_HOME = "/data/zyuli/miniconda3/envs/flagtree_triton_v3.6.x/lib/python3.12/site-packages/nvidia/nvshmem"

@dialect(
    name="cuda",
    compiler="nvcc",
    file=(Path(__file__).parent / "simple-shift-device.cu"),
    extern=(Path(__file__).parent / "simple-shift-device-extern-call.py"),
    extern_func_name="simple_shift",
    libs={"nvshmem": NVSHMEM_HOME},
    links=["nvshmem_device"]
)
def edsl(*args, **kwargs):
    ...


@triton.jit
def simple_shift_kernel(destination_ptr, ):
    tle_raw.call(edsl, [destination_ptr])


def simpe_shift():
    cu_file = (Path(__file__).parent / "simple-shift-host.cu").resolve()
    lib_file = Path(cu_file).with_suffix('.so')
    
    rank = int(os.getenv("PMI_RANK", "0"))

    lib = ctypes.CDLL(lib_file)
    lib.nvshmem_init_wrapper.argtypes = []
    lib.nvshmem_init_wrapper.restype = None
    lib.nvshmemx_cumodule_init_wrapper.argtypes = [ctypes.c_void_p]
    lib.nvshmemx_cumodule_init_wrapper.restype = ctypes.c_int
    lib.nvshmem_team_mype_wrapper.argtypes = []
    lib.nvshmem_team_mype_wrapper.restype = ctypes.c_int
    lib.nvshmem_alloc_wrapper.argtypes = [ctypes.c_int]
    lib.nvshmem_alloc_wrapper.restype = ctypes.POINTER(ctypes.c_int)
    lib.nvshmemx_barrier_warpper.argtypes = [ctypes.c_void_p]
    lib.nvshmemx_barrier_warpper.restype = None
    lib.nvshmem_finalize_wrapper.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.nvshmem_finalize_wrapper.restype = None

    lib.nvshmem_init_wrapper()
    mype_node = lib.nvshmem_team_mype_wrapper()
    torch.cuda.set_device(mype_node)
    device = triton.runtime.driver.active.get_active_torch_device()
    stream = torch.cuda.Stream()

    dest = lib.nvshmem_alloc_wrapper(1)
    dest_addr = ctypes.cast(dest, ctypes.c_void_p).value
    storage = torch._C._construct_storage_from_data_pointer(dest_addr, device, 4)
    dest_tensor = torch.empty(0, dtype=torch.int32, device=device).set_(storage).view(1)
    msg = torch.empty((1, ), dtype=torch.int32, pin_memory=True)

    def cumodule_init_hook(*args, **kwargs):
        key = kwargs["key"]
        jit_function = kwargs["fn"].jit_function
        device = kwargs["compile"]["device"]
        kernel_cache = jit_function.device_caches[device][0]
        kernel = kernel_cache.get(key, None)
        assert kernel is not None
        kernel._init_handles()
        ret = lib.nvshmemx_cumodule_init_wrapper(ctypes.c_void_p(kernel.module))
        assert ret == 0, f"nvshmemx_cumodule_init_wrapper failed: {ret}"
    knobs.runtime.jit_post_compile_hook = cumodule_init_hook

    simple_shift_kernel[(1, )](dest_tensor)

    stream_ptr = stream.cuda_stream
    lib.nvshmemx_barrier_warpper(ctypes.c_void_p(stream_ptr))
    with torch.cuda.stream(stream):
        msg.copy_(dest_tensor, non_blocking=True)
    stream.synchronize()

    lib.nvshmem_finalize_wrapper(dest)
    print(f"Rank {mype_node}: {msg}")


if __name__ == "__main__":
    simpe_shift()
