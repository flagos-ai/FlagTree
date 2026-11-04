# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

try:
    from triton._flagtree_backend import FLAGTREE_BACKEND
except ImportError:
    FLAGTREE_BACKEND = "nvidia"
try:
    import importlib
    flagcx_rt_conf = importlib.import_module(
        f"triton.backends.{FLAGTREE_BACKEND or 'nvidia'}.distributed").flagcx_rt_conf
    enabled = flagcx_rt_conf.is_available
except Exception:
    enabled = False

if enabled:
    import os
    import tempfile
    from pathlib import Path

    import torch
    import torch.distributed as dist
    from torch.cuda.memory import CUDAPluggableAllocator
    from torch.utils.cpp_extension import load_inline

    from .flagcx_wrapper import (
        FLAGCX_WIN_COLL_SYMMETRIC,
        FLAGCXLibrary,
        flagcxDevCommRequirements,
        flagcxUniqueId,  # noqa: F401
    )

    FLAGCX_LIB_PATH = str(flagcx_rt_conf.shared_lib_path.parent)
    FLAGCX_INCLUDE_PATH = str(flagcx_rt_conf.include_path)

    def _libflagcx_path():
        return Path(FLAGCX_LIB_PATH) / "libflagcx.so"

    _allocator = None
    _allocator_wrapper = None
    _mem_pool = None
    _flagcx_allocator_failed_to_compile = False
    _init_communicator_ = False
    # ``init_communicator`` may be called by a program that already owns the
    # torch process group (for example a torchrun model driver).  Keep that
    # ownership explicit so TLE can reuse the group without initializing it a
    # second time or destroying it during cleanup.
    _owns_process_group = False
    _sync_group = None
    _FLAGCX_WINDOW_ALIGNMENT = 4096

    def _distributed_barrier():
        """Synchronize without touching NCCL after FlagCX owns CUDA state.

        FlagCX's symmetric-window registration installs its own CUDA-side
        communicator state.  A subsequent NCCL barrier on the same process
        group can report ``cudaErrorInvalidValue`` even though NCCL itself is
        healthy.  A CPU Gloo subgroup provides the host rendezvous needed for
        window publication while leaving the NCCL communicator available to
        the caller and to generated kernels.
        """
        if _sync_group is not None:
            dist.barrier(group=_sync_group)
            return
        local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
        dist.barrier(device_ids=[local_rank])

    def _cleanup_flagcx_mem_pool():
        global _mem_pool
        _mem_pool = None

    def _cleanup_flagcx_allocator_wrapper():
        global _allocator_wrapper
        _allocator_wrapper = None

    import atexit
    atexit.register(_cleanup_flagcx_mem_pool)
    atexit.register(_cleanup_flagcx_allocator_wrapper)
    flagcx = FLAGCXLibrary(so_file=_libflagcx_path())
    global comm, rank, dev_mem, dev_comm, win
    comm = None
    dev_mem = None
    dev_comm = None
    win = None

    flagcx_allocator_source = """
    #include <flagcx.h>
    extern "C" {

    void* flagcx_alloc_plug(size_t size, int device, void* stream) {
    void* ptr = nullptr;
    flagcxResult_t err = flagcxMemAlloc(&ptr, size);
    if (err != flagcxSuccess) {
        return nullptr;
    }
    return ptr;
    }

    void flagcx_free_plug(void* ptr, size_t size, int device, void* stream) {
    if (ptr != nullptr) {
        flagcxMemFree(ptr);
    }
    }

    }
    """


def _flagcx_allocator_build_identity():
    """Bind the cached native wrapper to its SDK, source and Torch/CUDA ABI."""
    import hashlib
    import json
    import sys
    from pathlib import Path

    library_dir = Path(FLAGCX_LIB_PATH).resolve()
    include_dir = Path(FLAGCX_INCLUDE_PATH).resolve()

    def digest(path):
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(chunk)
        return result.hexdigest()

    identity = {
        "source": flagcx_allocator_source,
        "host_library": digest(library_dir / "libflagcx.so"),
        "header": digest(include_dir / "flagcx.h"),
        "library_dir": str(library_dir),
        "include_dir": str(include_dir),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "python_cache_tag": sys.implementation.cache_tag,
        "cuda_home": os.environ.get("CUDA_HOME"),
        "cxx": os.environ.get("CXX"),
        "with_cuda": True,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return key, library_dir, include_dir


def compile_flagcx_allocator():
    global _allocator, _allocator_wrapper, _flagcx_allocator_failed_to_compile

    try:
        identity, library_dir, include_dir = _flagcx_allocator_build_identity()
        out_dir = os.path.join(tempfile.gettempdir(), "tle_flagcx_allocator", identity)
        # A unique module name also prevents Torch's in-process extension
        # versioner from renaming a different SDK build behind our loader.
        lib_name = f"flagcx_allocator_{identity}"
        lib_path = os.path.join(out_dir, f"{lib_name}.so")

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0 and not os.path.isfile(lib_path):
            os.makedirs(out_dir, exist_ok=True)
            print(
                f"[INFO] FlagCX allocator not found, compiling: {lib_path}",
                flush=True,
            )

            load_inline(
                name=lib_name,
                cpp_sources=flagcx_allocator_source,
                with_cuda=True,
                extra_ldflags=[
                    f"-L{library_dir}",
                    "-lflagcx",
                    f"-Wl,-rpath,{library_dir}",
                ],
                verbose=True,
                is_python_module=False,
                build_directory=out_dir,
                extra_include_paths=[str(include_dir)],
            )
        else:
            print(
                f"[INFO] Using cached FlagCX allocator: {lib_path}",
                flush=True,
            )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            _distributed_barrier()

        if not os.path.isfile(lib_path):
            raise FileNotFoundError(f"FlagCX allocator library not found after compilation: {lib_path}")

        _allocator_wrapper = CUDAPluggableAllocator(
            lib_path,
            "flagcx_alloc_plug",
            "flagcx_free_plug",
        )

        _allocator = _allocator_wrapper.allocator()

    except Exception as e:
        _flagcx_allocator_failed_to_compile = True
        print(
            f"[WARNING] Failed to load FlagCX memory allocator: {e}",
            flush=True,
        )


def get_mem_pool():
    """Return a cached PyTorch MemPool backed by flagcxMemAlloc."""
    init_communicator()
    global _mem_pool, _flagcx_allocator_failed_to_compile
    if _mem_pool is None and not _flagcx_allocator_failed_to_compile:
        compile_flagcx_allocator()
        if _allocator is not None:
            _mem_pool = torch.cuda.MemPool(_allocator)
    return _mem_pool


def allocate_dist_tensor(shape, *, dtype, device):
    """Allocate a CUDA-allocation-root tensor for a FlagCX symmetric window.

    IPC imports the underlying CUDA allocation, including when the exported
    pointer was a caching-allocator suballocation. An aligned view within a
    cached block therefore gives remote peers a different byte origin from
    the local tensor. A fresh private pool's first allocation starts at its
    native CUDA allocation root; no other tensor can consume a prefix.
    """
    if not enabled:
        raise RuntimeError("FlagCX/TLE is unavailable")
    get_mem_pool()  # initialize and validate the communicator before allocation
    shape = (shape,) if isinstance(shape, int) else tuple(int(value) for value in shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"symmetric tensor shape must be positive, got {shape}")
    pool = torch.cuda.MemPool()
    with torch.cuda.device(device), torch.cuda.use_mem_pool(pool):
        tensor = torch.empty(shape, dtype=dtype, device=device)
    # Retain the allocation domain as long as the registered tensor lives.
    tensor._tle_window_pool = pool
    return tensor


def _cleanup_flagcx_mem_pool():
    global _mem_pool
    _mem_pool = None


def _cleanup_flagcx_allocator_wrapper():
    global _allocator_wrapper
    _allocator_wrapper = None


def cleanup_communicator():
    global comm, rank, dev_mem, dev_comm, win, _init_communicator_, _owns_process_group, _sync_group
    if not _init_communicator_:
        return
    _distributed_barrier()
    if dev_mem is not None:
        flagcx.flagcxDevMemFreeDevicePtr(dev_mem)
    if dev_comm is not None:
        flagcx.flagcxDevCommFreeDevicePtr(dev_comm)
    if dev_mem is not None:
        flagcx.flagcxDevMemDestroy(comm, dev_mem)
    if dev_comm is not None:
        flagcx.flagcxDevCommDestroy(comm, dev_comm)
    if win is not None:
        flagcx.flagcxCommWindowDeregister(comm, win)
    # buf_tensor memory is managed by torch, no explicit free needed
    flagcx.flagcxCommDestroy(comm)
    comm = dev_mem = dev_comm = win = None

    _init_communicator_ = False
    from triton.runtime import DistributedRtContext
    DistributedRtContext._reset()
    if _sync_group is not None and dist.is_initialized():
        dist.destroy_process_group(_sync_group)
    _sync_group = None
    if _owns_process_group and dist.is_initialized():
        dist.destroy_process_group()
    _owns_process_group = False
    print(f"[Rank {rank}] Done")


def init_communicator():
    global comm, rank, _init_communicator_, _owns_process_group, _sync_group
    if enabled and _init_communicator_:
        return
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        _owns_process_group = True
    else:
        _owns_process_group = False
    # Create the host-only rendezvous group before FlagCX registers any
    # symmetric window.  Every rank enters this call in the same order.
    if _sync_group is None:
        _sync_group = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    print(f"[Rank {rank}] Starting (world_size={world_size})")

    # Create unique ID on rank 0 and broadcast
    if rank == 0:
        unique_id = flagcx.flagcxGetUniqueId()
        id_bytes = bytes(unique_id.internal)
    else:
        id_bytes = b"\x00" * 256

    # Bootstrap bytes are host data. Using the CUDA default group here would
    # initialize an otherwise unused NCCL communicator and retain its device
    # buffers for the lifetime of this FlagCX-only execution.
    id_tensor = torch.frombuffer(bytearray(id_bytes), dtype=torch.uint8)
    dist.broadcast(id_tensor, src=0, group=_sync_group)
    id_bytes = id_tensor.numpy().tobytes()

    if rank != 0:
        unique_id = flagcx.unique_id_from_bytes(id_bytes)

    # Init FlagCX communicator
    comm = flagcx.flagcxCommInitRank(world_size, unique_id, rank)
    print(f"[Rank {rank}] FlagCX comm initialized")
    _init_communicator_ = True


def create_dist_tensor(buf_tensor):
    global comm, rank, dev_mem, dev_comm, win
    buf_ptr = buf_tensor.data_ptr()
    buf_size = buf_tensor.numel() * buf_tensor.element_size()

    if dev_comm is not None or dev_mem is not None or win is not None:
        raise RuntimeError(
            "An active FlagCX window already owns its communication resources; "
            "call cleanup_communicator() before creating another window."
        )
    torch.cuda.synchronize(buf_tensor.device)
    if dist.is_initialized():
        _distributed_barrier()

    # Register buffer with symmetric window for LSA
    win = flagcx.flagcxCommWindowRegister(comm, buf_ptr, buf_size, flags=FLAGCX_WIN_COLL_SYMMETRIC)
    print(f"[Rank {rank}] Window registered (symmetric)")

    # Create DevComm with 1 intra barrier
    reqs = flagcxDevCommRequirements()
    reqs.intraMulticast = False
    reqs.barrierCount = 8
    reqs.intraBarrierCount = 8
    reqs.interBarrierCount = 8
    reqs.intraLLA2ABlockCount = 0
    reqs.intraLLA2ASlotCount = 0
    reqs.interForceEnable = False
    reqs.interContextCount = 4
    reqs.interSignalCount = 8
    reqs.interCounterCount = 8

    dev_comm = flagcx.flagcxDevCommCreate(comm, reqs)
    print(f"[Rank {rank}] DevComm created")

    # Create DevMem (with window)
    dev_mem = flagcx.flagcxDevMemCreate(comm, buf_ptr, buf_size, win)
    print(f"[Rank {rank}] DevMem created")

    # Get device pointers for Triton
    dev_comm_dptr = flagcx.flagcxDevCommGetDevicePtr(dev_comm)
    dev_mem_dptr = flagcx.flagcxDevMemGetDevicePtr(dev_mem)
    print(f"[Rank {rank}] Device pointers: comm={dev_comm_dptr.value:#x}, "
          f"mem={dev_mem_dptr.value:#x}")

    # Synchronize all ranks before kernel launch
    _distributed_barrier()

    from triton.runtime import DistributedRtContext
    ctx = DistributedRtContext(dev_comm_dptr.value, dev_mem_dptr.value)
    return ctx
