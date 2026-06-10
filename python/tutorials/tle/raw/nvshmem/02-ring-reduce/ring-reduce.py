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

NVSHMEM_HOME = "/data/zyuli/miniconda3/envs/flagtree_triton_v3.6.x/lib/python3.12/site-packages/nvidia/nvshmem"
@dialect(
    name="cuda",
    compiler="nvcc",
    file=(Path(__file__).parent / "ring-reduce-device.cu"),
    extern=(Path(__file__).parent / "ring-reduce-device-extern-call.py"),
    extern_func_name="ring_reduce",
    libs={"nvshmem": NVSHMEM_HOME},
    links=["nvshmem_device"]
)
def edsl(*args, **kwargs):
    ...


@triton.jit
def ring_reduce_kernel(
    dst,
    src,
    nreduce,
    signal,
    chunk_size,
):
    tle_raw.call(edsl, [dst, src, nreduce, signal, chunk_size])


def cuda_host_compile(cuda_host_path, cuda_host_lib):
    NVCC = os.getenv("NVCC", "nvcc")
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


def ring_reduce():
    cu_file = (Path(__file__).parent / "ring-reduce-host.cu").resolve()
    lib_file = Path(cu_file).with_suffix('.so')

    # rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    rank = int(os.environ["PMI_RANK"])
    # if rank == 0:
    #     cuda_host_compile(cu_file, lib_file)

    # import time
    # timeout = 60
    # start = time.time()
    # while True:
    #     if lib_file.exists():
    #         try:
    #             ctypes.CDLL(str(lib_file))
    #             break
    #         except OSError:
    #             pass
    #     if time.time() - start > timeout:
    #         raise RuntimeError(f"Timeout waiting for {lib_file}")
    #     time.sleep(0.1)

    M, N = 64, 8
    lib = ctypes.CDLL(lib_file)
    mype = ctypes.c_int()
    npes = ctypes.c_int()
    mype_in_node = ctypes.c_int()
    npes_in_node = ctypes.c_int()
    
    stream = ctypes.c_void_p()
    
    src = ctypes.c_void_p()
    dst = ctypes.c_void_p()
    data_h = ctypes.c_void_p()
    signal = ctypes.c_void_p()
    
    size = M * N
    lib.ring_reduce_before_launch(
        ctypes.byref(mype), ctypes.byref(npes), ctypes.byref(mype_in_node), ctypes.byref(npes_in_node),
        ctypes.byref(stream),
        ctypes.byref(src), ctypes.byref(dst), ctypes.byref(data_h), ctypes.byref(signal),
        size
    )
    
    dtype = torch.int32
    num_blocks = npes_in_node.value
    device = triton.runtime.driver.active.get_active_torch_device()
    src_storage = torch._C._construct_storage_from_data_pointer(src.value, device, 4 * M * N)
    src_tensor = torch.empty(0, dtype=torch.int32, device=device).set_(src_storage).view(M, N)
    
    dst_storage = torch._C._construct_storage_from_data_pointer(dst.value, device, 4 * M * N)
    dst_tensor = torch.empty(0, dtype=torch.int32, device=device).set_(dst_storage).view(M, N)
    
    signal_storage = torch._C._construct_storage_from_data_pointer(signal.value, device, 8 * num_blocks)
    signal_tensor = torch.empty(0, dtype=torch.uint64, device=device).set_(signal_storage).view(num_blocks, )

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
        chunk_size = int(M // num_blocks * dtype.itemsize)
        ring_reduce_kernel[(num_blocks, )](
            dst_tensor, 
            src_tensor, 
            M * N, 
            signal_tensor, 
            chunk_size
        )
    
    print(f"PE {mype.value}: {npes.value}")
    lib.ring_reduce_after_launch(
        stream, 
        src, dst, data_h, signal, 
        mype_in_node.value, npes_in_node.value,
        size
    )


if __name__ == "__main__":
    ring_reduce()
