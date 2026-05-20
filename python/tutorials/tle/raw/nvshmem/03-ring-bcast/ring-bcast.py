import os
import subprocess
import ctypes
import torch
import triton
import triton.knobs as knobs
import triton.experimental.tle.language.raw as tle_raw

from pathlib import Path
from triton.experimental.tle.raw import dialect
from triton.language.extra.cuda import libnvshmem_device


@dialect(
    name="cuda",
    file=(Path(__file__).parent / "ring-bcast-device.cu").resolve(),
    library={"nvshmem": "/home/zyuli/miniconda3/envs/flagtree/lib/python3.12/site-packages/nvidia/nvshmem"},
)
def edsl(*args, **kwargs):
    ...


@triton.jit
def ring_bcast_kernel(
    data,
    nelem,
    root,
    psync,
):
    tle_raw.call(edsl, [])
    libnvshmem_device.ring_bcast(data, nelem, root, psync)


def cuda_host_compile(cuda_host_path, cuda_host_lib):
    NVCC = os.getenv("NVCC", "nvcc")
    NVSHMEM_HOME = "/home/zyuli/miniconda3/envs/flagtree/lib/python3.12/site-packages/nvidia/nvshmem"
    include_path = f"-I{os.path.join(NVSHMEM_HOME, 'include')}"
    lib_path = f"-L{os.path.join(NVSHMEM_HOME, 'lib')}"

    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = f"-arch=sm_{prop.major}{prop.minor}"
    tmp_file = Path(cuda_host_lib).with_suffix('.so.tmp')
    build = [
        NVCC, "-shared", "-Xcompiler", "-fPIC", "-rdc=true", arch, include_path, lib_path, "-lnvshmem_host",
        "-lnvshmem_device", "-o", tmp_file, cuda_host_path
    ]
    build = subprocess.run(build, capture_output=True)
    assert build.returncode == 0, (f"NVCC host failed\nstderr:\n{build.stderr.decode()}")
    tmp_file.rename(cuda_host_lib)


def ring_bcast():
    cu_file = (Path(__file__).parent / "ring-bcast-host.cu").resolve()
    lib_file = Path(cu_file).with_suffix('.so')

    rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    if rank == 0:
        cuda_host_compile(cu_file, lib_file)

    import time
    timeout = 60
    start = time.time()
    while True:
        if lib_file.exists():
            try:
                ctypes.CDLL(str(lib_file))
                break
            except OSError:
                pass
        if time.time() - start > timeout:
            raise RuntimeError(f"Timeout waiting for {lib_file}")
        time.sleep(0.1)

    lib = ctypes.CDLL(lib_file)
    mype = ctypes.c_int()
    npes = ctypes.c_int()
    mype_in_node = ctypes.c_int()
    npes_in_node = ctypes.c_int()
    
    stream = ctypes.c_void_p()
    
    data = ctypes.c_void_p()
    data_h = ctypes.c_void_p()
    psync = ctypes.c_void_p()
    
    data_len = 32
    root = 0
    lib.ring_bcast_before_launch(
        ctypes.byref(mype), ctypes.byref(npes), ctypes.byref(mype_in_node), ctypes.byref(npes_in_node),
        ctypes.byref(stream),
        ctypes.byref(data), ctypes.byref(data_h), ctypes.byref(psync),
        data_len
    )
    
    # print("PE:", mype_in_node.value)
    dtype = torch.int32
    num_blocks = 1
    num_warps = 1
    device = triton.runtime.driver.active.get_active_torch_device()
    
    data_storage = torch._C._construct_storage_from_data_pointer(data.value, device, dtype.itemsize * data_len)
    data_tensor = torch.empty(0, dtype=dtype, device=device).set_(data_storage).view(data_len, )
    
    psync_storage = torch._C._construct_storage_from_data_pointer(psync.value, device, 8 * 1)
    psync_tensor = torch.empty(0, dtype=torch.uint64, device=device).set_(psync_storage).view(1, )

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

    curr_stream = torch.cuda.ExternalStream(stream.value, device=device)
    with torch.cuda.stream(curr_stream):
        ring_bcast_kernel[(num_blocks, )](
            data_tensor, 
            data_len, 
            root, 
            psync_tensor
        )
    
    lib.ring_bcast_after_launch(
        stream, 
        data, data_h, psync, 
        mype.value, npes.value,
        data_len
    )


if __name__ == "__main__":
    ring_bcast()
