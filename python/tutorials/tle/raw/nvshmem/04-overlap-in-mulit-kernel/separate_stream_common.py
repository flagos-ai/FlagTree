import ctypes
import os
import subprocess
from pathlib import Path

import torch
import triton
import triton.experimental.tle.language.raw as tle_raw
import triton.knobs as knobs
from triton.experimental.tle.raw import dialect


NVSHMEM_HOME = "/data/zyuli/miniconda3/envs/flagtree_triton_v3.6.x/lib/python3.12/site-packages/nvidia/nvshmem"

SIZE = 1024 * 1024 * 32
MAT_M = 128
MAT_N = 256
MAT_K = 128


def _make_dialect(extern_func_name):
    return dialect(
        name="cuda",
        compiler="nvcc",
        file=(Path(__file__).parent / "ring-reduce-put.cu"),
        extern=(Path(__file__).parent / "ring-reduce-put-extern-call.py"),
        extern_func_name=extern_func_name,
        libs={"nvshmem": NVSHMEM_HOME},
        links=["nvshmem_device"],
    )


@_make_dialect("ring_reduce_put_timed")
def edsl_put_timed(*args, **kwargs): ...


@_make_dialect("ring_reduce_wait_timed")
def edsl_wait_timed(*args, **kwargs): ...


@_make_dialect("local_matmul")
def edsl_local_matmul(*args, **kwargs): ...


@triton.jit
def communication_kernel(
    dst,
    src,
    nreduce: triton.language.constexpr,
    signal,
    put_cycles,
    wait_cycles,
):
    tle_raw.call(
        edsl_put_timed,
        [dst, src, nreduce, signal, put_cycles],
    )
    tle_raw.call(
        edsl_wait_timed,
        [signal, wait_cycles],
    )


@triton.jit
def matmul_kernel(
    mat_A,
    mat_B,
    mat_C,
    mat_M: triton.language.constexpr,
    mat_N: triton.language.constexpr,
    mat_K: triton.language.constexpr,
):
    tle_raw.call(
        edsl_local_matmul,
        [mat_A, mat_B, mat_C, mat_M, mat_N, mat_K],
    )


def cuda_host_compile(cuda_host_path, cuda_host_lib):
    nvcc = os.getenv("NVCC", "nvcc")
    include_path = f"-I{os.path.join(NVSHMEM_HOME, 'include')}"
    lib_path = f"-L{os.path.join(NVSHMEM_HOME, 'lib')}"

    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = f"-arch=sm_{prop.major}{prop.minor}"
    tmp_file = Path(cuda_host_lib).with_suffix(".so.tmp")
    build = [
        nvcc,
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-rdc=true",
        arch,
        include_path,
        lib_path,
        "-lnvshmem_host",
        "-lnvshmem_device",
        "-o",
        tmp_file,
        cuda_host_path,
    ]
    result = subprocess.run(build, capture_output=True)
    assert result.returncode == 0, (
        f"NVCC host failed\nstderr:\n{result.stderr.decode()}"
    )
    tmp_file.rename(cuda_host_lib)


def wait_for_lib(lib_file, timeout=60):
    import time

    start = time.time()
    while True:
        if lib_file.exists():
            try:
                ctypes.CDLL(str(lib_file))
                return
            except OSError:
                pass
        if time.time() - start > timeout:
            raise RuntimeError(f"Timeout waiting for {lib_file}")
        time.sleep(0.1)


def setup_nvshmem(lib, size):
    mype = ctypes.c_int()
    npes = ctypes.c_int()
    mype_in_node = ctypes.c_int()
    npes_in_node = ctypes.c_int()
    stream = ctypes.c_void_p()
    src = ctypes.c_void_p()
    dst = ctypes.c_void_p()
    data_h = ctypes.c_void_p()
    signal = ctypes.c_void_p()

    lib.ring_reduce_before_launch(
        ctypes.byref(mype),
        ctypes.byref(npes),
        ctypes.byref(mype_in_node),
        ctypes.byref(npes_in_node),
        ctypes.byref(stream),
        ctypes.byref(src),
        ctypes.byref(dst),
        ctypes.byref(data_h),
        ctypes.byref(signal),
        size,
    )
    return (
        mype,
        npes,
        mype_in_node,
        npes_in_node,
        stream,
        src,
        dst,
        data_h,
        signal,
    )


def make_tensors(src, dst, signal, size, device):
    src_storage = torch._C._construct_storage_from_data_pointer(
        src.value, device, 4 * size
    )
    src_tensor = torch.empty(0, dtype=torch.int32, device=device).set_(
        src_storage
    ).view(size)

    dst_storage = torch._C._construct_storage_from_data_pointer(
        dst.value, device, 4 * size
    )
    dst_tensor = torch.empty(0, dtype=torch.int32, device=device).set_(
        dst_storage
    ).view(size)

    signal_storage = torch._C._construct_storage_from_data_pointer(
        signal.value, device, 8
    )
    signal_tensor = torch.empty(0, dtype=torch.uint64, device=device).set_(
        signal_storage
    ).view(1)

    return src_tensor, dst_tensor, signal_tensor


def install_cumodule_hook(lib):
    def cumodule_init_hook(*args, **kwargs):
        key = kwargs["key"]
        jit_function = kwargs["fn"].jit_function
        dev = kwargs["compile"]["device"]
        kernel_cache = jit_function.device_caches[dev][0]
        kernel = kernel_cache.get(key)
        assert kernel is not None
        kernel._init_handles()
        ret = lib.nvshmemx_cumodule_init_wrapper(ctypes.c_void_p(kernel.module))
        assert ret == 0, f"nvshmemx_cumodule_init_wrapper failed: {ret}"

    knobs.runtime.jit_post_compile_hook = cumodule_init_hook


def reset_signal_collective(lib, signal_tensor, comm_stream):
    lib.ring_reduce_barrier_all()
    with torch.cuda.stream(comm_stream):
        signal_tensor.zero_()
    comm_stream.synchronize()
    lib.ring_reduce_barrier_all()


def reset_tensors_on_stream(stream, *tensors):
    with torch.cuda.stream(stream):
        for tensor in tensors:
            tensor.zero_()
    stream.synchronize()


def format_cycle_stats(tensor, clock_rate_khz):
    tensor = tensor.to(torch.int64)
    values = tensor[tensor > 0].detach().cpu().to(torch.float64)
    if values.numel() == 0:
        return "n/a"
    us = values / clock_rate_khz * 1000.0
    return (
        f"{us.mean().item():.3f} us "
        f"({int(values.mean().item())} cycles)"
    )


def launch_communication(
    stream,
    args,
    wait_event=None,
    done_event=None,
):
    with torch.cuda.stream(stream):
        if wait_event is not None:
            stream.wait_event(wait_event)
        communication_kernel[(1,)](
            *args,
            num_warps=4,
        )
        if done_event is not None:
            done_event.record(stream)


def launch_matmul(
    stream,
    grid,
    args,
    wait_event=None,
    done_event=None,
):
    with torch.cuda.stream(stream):
        if wait_event is not None:
            stream.wait_event(wait_event)
        matmul_kernel[grid](
            *args,
            num_warps=4,
        )
        if done_event is not None:
            done_event.record(stream)


def run_overlap(
    timing_stream,
    comm_stream,
    compute_stream,
    comm_args,
    matmul_grid,
    matmul_args,
):
    gate = torch.cuda.Event(enable_timing=True)
    comm_done = torch.cuda.Event()
    compute_done = torch.cuda.Event()
    all_done = torch.cuda.Event(enable_timing=True)

    launch_communication(
        comm_stream,
        comm_args,
        wait_event=gate,
        done_event=comm_done,
    )
    launch_matmul(
        compute_stream,
        matmul_grid,
        matmul_args,
        wait_event=gate,
        done_event=compute_done,
    )

    with torch.cuda.stream(timing_stream):
        gate.record(timing_stream)
        timing_stream.wait_event(comm_done)
        timing_stream.wait_event(compute_done)
        all_done.record(timing_stream)

    all_done.synchronize()
    return gate.elapsed_time(all_done)


def run_no_overlap(
    timing_stream,
    comm_stream,
    compute_stream,
    comm_args,
    matmul_grid,
    matmul_args,
):
    gate = torch.cuda.Event(enable_timing=True)
    comm_done = torch.cuda.Event()
    compute_done = torch.cuda.Event()
    all_done = torch.cuda.Event(enable_timing=True)

    launch_communication(
        comm_stream,
        comm_args,
        wait_event=gate,
        done_event=comm_done,
    )
    launch_matmul(
        compute_stream,
        matmul_grid,
        matmul_args,
        wait_event=comm_done,
        done_event=compute_done,
    )

    with torch.cuda.stream(timing_stream):
        gate.record(timing_stream)
        timing_stream.wait_event(compute_done)
        all_done.record(timing_stream)

    all_done.synchronize()
    return gate.elapsed_time(all_done)


def run_experiment(mode):
    if mode not in {"overlap", "no-overlap"}:
        raise ValueError(f"Unsupported mode: {mode}")
    cu_file = (Path(__file__).parent / "ring-reduce-host.cu").resolve()
    lib_file = cu_file.with_suffix(".so")

    rank = int(os.getenv("MPI_RANK", "0"))
    lib = ctypes.CDLL(str(lib_file))
    lib.ring_reduce_barrier_all.argtypes = []
    lib.ring_reduce_barrier_all.restype = None

    (
        mype,
        npes,
        mype_in_node,
        npes_in_node,
        stream,
        src,
        dst,
        data_h,
        signal,
    ) = setup_nvshmem(lib, SIZE)

    device = triton.runtime.driver.active.get_active_torch_device()
    props = torch.cuda.get_device_properties(device)
    compute_blocks = max(1, props.multi_processor_count - 1)

    src_tensor, dst_tensor, signal_tensor = make_tensors(
        src, dst, signal, SIZE, device
    )

    mat_A = torch.randn(
        compute_blocks, MAT_M, MAT_N, dtype=torch.float32, device=device
    )
    mat_B = torch.randn(
        compute_blocks, MAT_N, MAT_K, dtype=torch.float32, device=device
    )
    mat_C = torch.zeros(
        compute_blocks, MAT_M, MAT_K, dtype=torch.float32, device=device
    )

    overlap_put_cycles = torch.zeros(1, dtype=torch.uint64, device=device)
    overlap_wait_cycles = torch.zeros(1, dtype=torch.uint64, device=device)
    no_overlap_put_cycles = torch.zeros(1, dtype=torch.uint64, device=device)
    no_overlap_wait_cycles = torch.zeros(1, dtype=torch.uint64, device=device)

    install_cumodule_hook(lib)

    comm_stream = torch.cuda.ExternalStream(stream.value, device=device)
    compute_stream = torch.cuda.Stream(device=device, priority=0)
    timing_stream = torch.cuda.Stream(device=device, priority=0)
    matmul_grid = (compute_blocks,)

    overlap_comm_args = (
        dst_tensor,
        src_tensor,
        SIZE,
        signal_tensor,
        overlap_put_cycles,
        overlap_wait_cycles,
    )
    no_overlap_comm_args = (
        dst_tensor,
        src_tensor,
        SIZE,
        signal_tensor,
        no_overlap_put_cycles,
        no_overlap_wait_cycles,
    )
    matmul_args = (
        mat_A,
        mat_B,
        mat_C,
        MAT_M,
        MAT_N,
        MAT_K,
    )

    torch.cuda.synchronize(device)

    selected_comm_args = (
        overlap_comm_args if mode == "overlap" else no_overlap_comm_args
    )
    if mode == "overlap":
        run_overlap(
            timing_stream,
            comm_stream,
            compute_stream,
            selected_comm_args,
            matmul_grid,
            matmul_args,
        )
    else:
        run_no_overlap(
            timing_stream,
            comm_stream,
            compute_stream,
            selected_comm_args,
            matmul_grid,
            matmul_args,
        )
    reset_signal_collective(lib, signal_tensor, comm_stream)

    selected_put_cycles = (
        overlap_put_cycles if mode == "overlap" else no_overlap_put_cycles
    )
    selected_wait_cycles = (
        overlap_wait_cycles if mode == "overlap" else no_overlap_wait_cycles
    )
    reset_tensors_on_stream(
        comm_stream, selected_put_cycles, selected_wait_cycles
    )
    lib.ring_reduce_barrier_all()
    
    if mode == "overlap":
        elapsed_ms = run_overlap(
            timing_stream,
            comm_stream,
            compute_stream,
            selected_comm_args,
            matmul_grid,
            matmul_args,
        )
    else:
        elapsed_ms = run_no_overlap(
            timing_stream,
            comm_stream,
            compute_stream,
            selected_comm_args,
            matmul_grid,
            matmul_args,
        )

    lib.ring_reduce_barrier_all()

    clock_rate_khz = props.clock_rate
    put_stats = format_cycle_stats(selected_put_cycles, clock_rate_khz)
    wait_stats = format_cycle_stats(selected_wait_cycles, clock_rate_khz)

    if mype.value == 0 or mype.value == 1:
        print("\n" + "=" * 64)
        print(f"  Separate-stream {mode}")
        print("=" * 64)
        print(f"  PEs (GPUs)       : {npes.value}")
        print(f"  GPU SMs          : {props.multi_processor_count}")
        print("  Communication    : 1 block on communication stream")
        print(f"  Matmul           : {compute_blocks} blocks on compute stream")
        print(
            f"  Message size     : {SIZE * 4 / 1024 / 1024:.1f} MB per PE"
        )
        print(
            f"  Per-block matmul : {MAT_M}x{MAT_N} x {MAT_N}x{MAT_K}"
        )
        print("-" * 64)
        print(f"  elapsed          : {elapsed_ms:.3f} ms")
        print("-" * 64)
        print(f"  put_nbi          : {put_stats}")
        print(f"  quiet/wait       : {wait_stats}")
        print("=" * 64)

    compute_stream.synchronize()
    timing_stream.synchronize()
    lib.ring_reduce_after_launch(
        stream,
        src,
        dst,
        data_h,
        signal,
        mype.value,
        npes.value,
        SIZE,
    )
