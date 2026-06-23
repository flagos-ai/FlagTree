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


# MAT_M = 16   
# MAT_N = 32  
# MAT_K = 16  

MAT_M = 128  
MAT_N = 256  
MAT_K = 128  

# ------------------------------------------------------------------
# 通信数据规模
# SIZE 为每个 PE 的 int32 元素总数
# NUM_CHUNKS 为每个 block 内的 chunk 数量
# ------------------------------------------------------------------
SIZE       = 1024 * 1024 * 32   
NUM_CHUNKS = 1 


def _make_dialect(extern_func_name):
    return dialect(
        name="cuda",
        compiler="nvcc",
        file=(Path(__file__).parent / "ring-reduce-put.cu"),
        extern=(Path(__file__).parent / "ring-reduce-put-extern-call.py"),
        extern_func_name=extern_func_name,
        libs={"nvshmem": NVSHMEM_HOME},
        links=["nvshmem_device"]
    )

@_make_dialect("ring_reduce_put_one_chunk_timed")
def edsl_put_one_chunk_timed(*args, **kwargs): ...

@_make_dialect("ring_reduce_wait_timed")
def edsl_wait_timed(*args, **kwargs): ...

@_make_dialect("local_matmul")
def edsl_local_matmul(*args, **kwargs): ...


@triton.jit
def overlap_kernel(
    dst,
    src,
    nreduce: triton.language.constexpr,
    signal,
    put_cycles,
    wait_cycles,
    chunk_size: triton.language.constexpr,
    num_chunks: triton.language.constexpr,
    expected,
    mat_A,
    mat_B,
    mat_C,
    mat_M: triton.language.constexpr,
    mat_N: triton.language.constexpr,
    mat_K: triton.language.constexpr,
):
    for chunk in range(num_chunks):
        tle_raw.call(edsl_put_one_chunk_timed,
                     [dst, src, nreduce, signal, chunk_size, chunk,
                      put_cycles, num_chunks])
        tle_raw.call(edsl_local_matmul,
                     [mat_A, mat_B, mat_C, mat_M, mat_N, mat_K])
    
    tle_raw.call(edsl_wait_timed, [signal, expected, wait_cycles, num_chunks, 0])


def setup_nvshmem(lib, size):
    mype         = ctypes.c_int()
    npes         = ctypes.c_int()
    mype_in_node = ctypes.c_int()
    npes_in_node = ctypes.c_int()
    stream       = ctypes.c_void_p()
    src          = ctypes.c_void_p()
    dst          = ctypes.c_void_p()
    data_h       = ctypes.c_void_p()
    signal       = ctypes.c_void_p()

    lib.ring_reduce_before_launch(
        ctypes.byref(mype), ctypes.byref(npes),
        ctypes.byref(mype_in_node), ctypes.byref(npes_in_node),
        ctypes.byref(stream),
        ctypes.byref(src), ctypes.byref(dst), ctypes.byref(data_h), ctypes.byref(signal),
        size
    )
    return mype, npes, mype_in_node, npes_in_node, stream, src, dst, data_h, signal


def make_tensors(src, dst, signal, num_blocks, size, device):
    src_storage    = torch._C._construct_storage_from_data_pointer(
        src.value, device, 4 * size)
    src_tensor     = torch.empty(0, dtype=torch.int32, device=device).set_(
        src_storage).view(size)

    dst_storage    = torch._C._construct_storage_from_data_pointer(
        dst.value, device, 4 * size)
    dst_tensor     = torch.empty(0, dtype=torch.int32, device=device).set_(
        dst_storage).view(size)

    signal_storage = torch._C._construct_storage_from_data_pointer(
        signal.value, device, 8 * num_blocks)
    signal_tensor  = torch.empty(0, dtype=torch.uint64, device=device).set_(
        signal_storage).view(num_blocks)

    return src_tensor, dst_tensor, signal_tensor


def install_cumodule_hook(lib):
    def cumodule_init_hook(*args, **kwargs):
        key          = kwargs["key"]
        jit_function = kwargs["fn"].jit_function
        dev          = kwargs["compile"]["device"]
        kernel_cache = jit_function.device_caches[dev][0]
        kernel       = kernel_cache.get(key, None)
        assert kernel is not None
        kernel._init_handles()
        ret = lib.nvshmemx_cumodule_init_wrapper(ctypes.c_void_p(kernel.module))
        assert ret == 0, f"nvshmemx_cumodule_init_wrapper failed: {ret}"
    knobs.runtime.jit_post_compile_hook = cumodule_init_hook


def reset_signal_on_stream(signal_tensor, curr_stream):
    with torch.cuda.stream(curr_stream):
        signal_tensor.zero_()
    curr_stream.synchronize()


def reset_signal_collective(lib, signal_tensor, curr_stream):
    lib.ring_reduce_barrier_all()
    reset_signal_on_stream(signal_tensor, curr_stream)
    lib.ring_reduce_barrier_all()


def reset_tensors_on_stream(curr_stream, *tensors):
    with torch.cuda.stream(curr_stream):
        for tensor in tensors:
            tensor.zero_()
    curr_stream.synchronize()


def format_cycle_stats(tensor, clock_rate_khz):
    tensor = tensor.to(torch.int64)
    values = tensor[tensor > 0].detach().cpu().to(torch.float64)
    if values.numel() == 0:
        return "n/a"
    us = values / clock_rate_khz * 1000.0
    return (
        f"avg {us.mean().item():.3f} us, "
        f"min {us.min().item():.3f} us, "
        f"max {us.max().item():.3f} us "
        f"({int(values.mean().item())} cycles avg)"
    )


def run_experiment():
    cu_file  = (Path(__file__).parent / "ring-reduce-host.cu").resolve()
    lib_file = cu_file.with_suffix(".so")

    rank = int(os.getenv("PMI_RANK", "0"))
    lib = ctypes.CDLL(str(lib_file))
    lib.ring_reduce_barrier_all.argtypes = []
    lib.ring_reduce_barrier_all.restype = None

    # ---- nvshmem 初始化 ----
    size = SIZE
    (mype, npes, mype_in_node, npes_in_node,
     stream, src, dst, data_h, signal) = setup_nvshmem(lib, size)

    num_blocks = npes_in_node.value
    device     = triton.runtime.driver.active.get_active_torch_device()

    src_tensor, dst_tensor, signal_tensor = make_tensors(
        src, dst, signal, num_blocks, size, device)

    # ---- 矩阵乘法 buffer（每个 PE 分配，所有 block 共享同一组指针）
    # 为了让不同 block 的 matmul 真正，按 block 偏移寻址
    # mat_A, mat_B, mat_C 各 num_blocks 份，每份 MAT_M*MAT_N / MAT_N*MAT_K 大小
    mat_A = torch.randn(num_blocks, MAT_M, MAT_N, dtype=torch.float32, device=device)
    mat_B = torch.randn(num_blocks, MAT_N, MAT_K, dtype=torch.float32, device=device)
    mat_C = torch.zeros(num_blocks, MAT_M, MAT_K, dtype=torch.float32, device=device)
    overlap_put_cycles = torch.zeros(num_blocks, NUM_CHUNKS, dtype=torch.uint64, device=device)
    overlap_wait_cycles = torch.zeros(num_blocks, NUM_CHUNKS, dtype=torch.uint64, device=device)

    # chunk 参数
    elems_per_block = size // num_blocks
    chunk_elems     = elems_per_block // NUM_CHUNKS
    chunk_size      = chunk_elems * 4   # bytes（int32）

    assert chunk_elems > 0, "SIZE 太小或 NUM_CHUNKS 太大，chunk_elems = 0"
    assert elems_per_block % NUM_CHUNKS == 0, \
        f"elems_per_block={elems_per_block} 不能被 NUM_CHUNKS={NUM_CHUNKS} 整除"

    # ---- install cumodule hook ----
    install_cumodule_hook(lib)

    # ---- CUDA event 计时 ----
    curr_stream = torch.cuda.ExternalStream(stream.value, device=device)
    num_warps = 4   

    # ---- 预热（warmup） ----
    with torch.cuda.stream(curr_stream):
        overlap_kernel[(num_blocks,)](
            dst_tensor, src_tensor, size, signal_tensor,
            overlap_put_cycles, overlap_wait_cycles,
            chunk_size, NUM_CHUNKS, 
            NUM_CHUNKS, 
            mat_A, mat_B, mat_C, MAT_M, MAT_N, MAT_K,
            num_warps=num_warps
        )
    curr_stream.synchronize()
    reset_signal_collective(lib, signal_tensor, curr_stream)
    reset_tensors_on_stream(curr_stream, overlap_put_cycles, overlap_wait_cycles)
    
    evt_start_ov = torch.cuda.Event(enable_timing=True)
    evt_end_ov   = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(curr_stream):
        evt_start_ov.record(curr_stream)
        overlap_kernel[(num_blocks,)](
            dst_tensor, src_tensor, size, signal_tensor,
            overlap_put_cycles, overlap_wait_cycles,
            chunk_size, NUM_CHUNKS,
            NUM_CHUNKS,
            mat_A, mat_B, mat_C, MAT_M, MAT_N, MAT_K,
            num_warps=num_warps
        )
        evt_end_ov.record(curr_stream)

    curr_stream.synchronize()
    time_overlap = evt_start_ov.elapsed_time(evt_end_ov)  # ms
    lib.ring_reduce_barrier_all()

    clock_rate_khz = torch.cuda.get_device_properties(device).clock_rate
    overlap_put_stats = format_cycle_stats(overlap_put_cycles, clock_rate_khz)
    overlap_wait_stats = format_cycle_stats(overlap_wait_cycles, clock_rate_khz)

    pe = mype_in_node.value
    if pe == 0 or pe == 1:
        print("\n" + "=" * 30 + f" PE {pe}: " + "=" * 30)
        print("  通信-计算重叠 vs 无重叠  耗时对比")
        print("=" * 60)
        print(f"  配置:")
        print(f"    PEs (GPUs)    : {npes_in_node.value}")
        print(f"    数据规模      : {size} int32 elements ({size * 4 / 1024:.1f} KB per PE)")
        print(f"    chunk 数      : {NUM_CHUNKS} per block")
        print(f"    chunk 大小    : {chunk_size} bytes ({chunk_elems} int32 each)")
        print(f"    matmul 规模   : {MAT_M}×{MAT_N} × {MAT_N}×{MAT_K} (float32)")
        print(f"    计时 launch   : 1 per scheme")
        print("-" * 60)
        print(f"  方案一 overlap   : {time_overlap:.3f} ms")
        print("-" * 60)
        print("  device clock64 统计:")
        print(f"    overlap put_nbi 调用        : {overlap_put_stats}")
        print(f"    overlap quiet + wait 收口     : {overlap_wait_stats}")
        print()

    lib.ring_reduce_after_launch(
        stream,
        src, dst, data_h, signal,
        mype.value, npes.value,
        size
    )


if __name__ == "__main__":
    run_experiment()

