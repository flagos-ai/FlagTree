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
    file=(Path(__file__).parent / "simple-gemm.cu").resolve(),
    library={"nvshmem": "/home/zyuli/miniconda3/envs/flagtree/lib/python3.12/site-packages/nvidia/nvshmem"},
)
def edsl(*args, **kwargs):
    ...


@triton.jit
def gemm_allreduce_kernel(
    C, A, B,
    m, n, k
):
    tle_raw.call(edsl, [])
    libnvshmem_device.tiled_gemm(C, A, B, m, n, k)


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


def gemm_allreduce():
    cu_file = (Path(__file__).parent / "gemm-allreduce-host.cu").resolve()
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
    # stream = ctypes.c_void_p()
    
    d_A = ctypes.c_void_p()
    d_B = ctypes.c_void_p()
    d_C = ctypes.c_void_p()
    h_A = ctypes.c_void_p()
    h_B = ctypes.c_void_p()
    
    M, N, K = 64, 64, 64
    lib.gemm_allreduce_before_launch(
        ctypes.byref(mype), ctypes.byref(npes), ctypes.byref(mype_in_node), ctypes.byref(npes_in_node),
        ctypes.byref(d_A), ctypes.byref(d_B), ctypes.byref(d_C), 
        ctypes.byref(h_A), ctypes.byref(h_B),
        M, N, K
    )
    
    # define tile_size, dtype, len(dtype), device
    dtype = torch.float32
    TILE_SIZE = 16
    grid = (((M + TILE_SIZE - 1) // TILE_SIZE), ((N + TILE_SIZE - 1) // TILE_SIZE))
    num_warps = TILE_SIZE * TILE_SIZE // 32
    device = triton.runtime.driver.active.get_active_torch_device()

    d_A_storage = torch._C._construct_storage_from_data_pointer(d_A.value, device, dtype.itemsize * M * K)
    d_A_tensor = torch.empty(0, dtype=dtype, device=device).set_(d_A_storage).view(M, K)
    
    d_B_storage = torch._C._construct_storage_from_data_pointer(d_B.value, device, dtype.itemsize * K * N)
    d_B_tensor = torch.empty(0, dtype=dtype, device=device).set_(d_B_storage).view(K, N)

    d_C_storage = torch._C._construct_storage_from_data_pointer(d_C.value, device, dtype.itemsize * M * N)
    d_C_tensor = torch.empty(0, dtype=dtype, device=device).set_(d_C_storage).view(M, N)

    # def cumodule_init_hook(*args, **kwargs):
    #     key = kwargs["key"]
    #     jit_function = kwargs["fn"].jit_function
    #     device = kwargs["compile"]["device"]
    #     kernel_cache = jit_function.device_caches[device][0]
    #     kernel = kernel_cache.get(key, None)
    #     assert kernel is not None
    #     kernel._init_handles()
    #     ret = lib.nvshmemx_cumodule_init_wrapper(ctypes.c_void_p(kernel.module))
    #     assert ret == 0, f"nvshmemx_cumodule_init_wrapper failed: {ret}"
    # knobs.runtime.jit_post_compile_hook = cumodule_init_hook

    gemm_allreduce_kernel[grid](
        d_C_tensor, 
        d_A_tensor, 
        d_B_tensor,
        M, N, K,
        num_warps = num_warps
    )
    
    lib.gemm_allreduce_after_launch(
        mype.value, npes.value,
        d_A, d_B, d_C,
        h_A, h_B,
        M, N, K
    )


if __name__ == "__main__":
    gemm_allreduce()
