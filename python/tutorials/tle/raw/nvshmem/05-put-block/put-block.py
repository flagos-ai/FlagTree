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
    file=(Path(__file__).parent / "put-block-device.cu").resolve(),
    library={"nvshmem": "/home/zyuli/miniconda3/envs/flagtree/lib/python3.12/site-packages/nvidia/nvshmem"},
)
def edsl(*args, **kwargs):
    ...



@triton.jit
def set_and_shift_kernel(
    send_data, recv_data, num_elems, mype, npes
):
    tle_raw.call(edsl, [])
    libnvshmem_device.set_and_shift(send_data, recv_data, num_elems, mype, npes)


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


def put_block():
    cu_file = (Path(__file__).parent / "put-block-host.cu").resolve()
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
    
    send_data = ctypes.c_void_p()
    recv_data = ctypes.c_void_p()
    
    num_elems = 8192
    lib.put_block_before_launch(
        ctypes.byref(mype), ctypes.byref(npes), ctypes.byref(mype_in_node), ctypes.byref(npes_in_node),
        ctypes.byref(send_data), ctypes.byref(recv_data), 
        num_elems
    )
    
    dtype = torch.float32
    THREADS_PER_BLOCK = 1024
    num_blocks = num_elems // THREADS_PER_BLOCK
    num_warps = (THREADS_PER_BLOCK + 31) // 32
    device = triton.runtime.driver.active.get_active_torch_device()
    
    send_data_storage = torch._C._construct_storage_from_data_pointer(send_data.value, device, dtype.itemsize * num_elems)
    send_data_tensor = torch.empty(0, dtype=dtype, device=device).set_(send_data_storage).view(num_elems, )
    recv_data_storage = torch._C._construct_storage_from_data_pointer(recv_data.value, device, dtype.itemsize * num_elems)
    recv_data_tensor = torch.empty(0, dtype=dtype, device=device).set_(recv_data_storage).view(num_elems, )

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
    
    set_and_shift_kernel[(num_blocks, )](
        send_data_tensor,
        recv_data_tensor,
        num_elems,
        mype.value,
        npes.value,
        num_warps=num_warps
    )
    
    lib.put_block_after_launch(
        mype.value, npes.value,
        send_data, recv_data,
        num_elems
    )


if __name__ == "__main__":
    put_block()