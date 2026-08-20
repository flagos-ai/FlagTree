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
    from triton.backends.nvidia.distributed import flagcx_rt_conf
    enabled = flagcx_rt_conf.is_available
except Exception:
    enabled = False

if enabled:
    from .flagcx_wrapper import (
        FLAGCXLibrary,
        flagcxDevCommRequirements,
        flagcxUniqueId,  # noqa: F401
        FLAGCX_WIN_COLL_SYMMETRIC,
    )
    import os
    import tempfile
    import torch
    import torch.distributed as dist
    from torch.cuda.memory import CUDAPluggableAllocator
    from torch.utils.cpp_extension import load_inline
    from pathlib import Path

    FLAGCX_LIB_PATH = str(flagcx_rt_conf.shared_lib_path.parent)
    FLAGCX_INCLUDE_PATH = str(flagcx_rt_conf.include_path)

    def _libflagcx_path():
        return Path(FLAGCX_LIB_PATH) / "libflagcx.so"

    _allocator = None
    _allocator_wrapper = None
    _mem_pool = None
    _flagcx_allocator_failed_to_compile = False
    _init_communicator_ = False

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


def compile_flagcx_allocator():
    global _allocator, _allocator_wrapper, _flagcx_allocator_failed_to_compile

    try:
        out_dir = tempfile.gettempdir()
        lib_name = "flagcx_allocator"
        lib_path = os.path.join(out_dir, f"{lib_name}.so")

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if local_rank == 0 and not os.path.isfile(lib_path):
            print(
                f"[INFO] FlagCX allocator not found, compiling: {lib_path}",
                flush=True,
            )

            load_inline(
                name=lib_name,
                cpp_sources=flagcx_allocator_source,
                with_cuda=True,
                extra_ldflags=[
                    f"-L{FLAGCX_LIB_PATH}",
                    "-lflagcx",
                    f"-Wl,-rpath,{FLAGCX_LIB_PATH}",
                ],
                verbose=True,
                is_python_module=False,
                build_directory=out_dir,
                extra_include_paths=[FLAGCX_INCLUDE_PATH],
            )
        else:
            print(
                f"[INFO] Using cached FlagCX allocator: {lib_path}",
                flush=True,
            )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

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
    init_communicator()
    """Return a cached PyTorch MemPool backed by flagcxMemAlloc."""
    global _mem_pool, _flagcx_allocator_failed_to_compile
    if _mem_pool is None and not _flagcx_allocator_failed_to_compile:
        compile_flagcx_allocator()
        if _allocator is not None:
            _mem_pool = torch.cuda.MemPool(_allocator)
    return _mem_pool


def _cleanup_flagcx_mem_pool():
    global _mem_pool
    _mem_pool = None


def _cleanup_flagcx_allocator_wrapper():
    global _allocator_wrapper
    _allocator_wrapper = None


def cleanup_communicator():
    global comm, rank, dev_mem, dev_comm, win
    dist.barrier()
    flagcx.flagcxDevMemFreeDevicePtr(dev_mem)
    flagcx.flagcxDevCommFreeDevicePtr(dev_comm)
    flagcx.flagcxDevMemDestroy(comm, dev_mem)
    flagcx.flagcxDevCommDestroy(comm, dev_comm)
    flagcx.flagcxCommWindowDeregister(comm, win)
    # buf_tensor memory is managed by torch, no explicit free needed
    flagcx.flagcxCommDestroy(comm)

    dist.destroy_process_group()
    print(f"[Rank {rank}] Done")


def init_communicator():
    global comm, rank, _init_communicator_
    if enabled and _init_communicator_:
        return
    dist.init_process_group(backend="nccl")
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

    # Broadcast unique_id bytes via torch distributed
    id_tensor = torch.frombuffer(bytearray(id_bytes), dtype=torch.uint8).cuda()
    dist.broadcast(id_tensor, src=0)
    id_bytes = id_tensor.cpu().numpy().tobytes()

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

    # Register buffer with symmetric window for LSA
    win = flagcx.flagcxCommWindowRegister(comm, buf_ptr, buf_size, flags=FLAGCX_WIN_COLL_SYMMETRIC)
    print(f"[Rank {rank}] Window registered (symmetric)")

    # Create DevComm with 1 intra barrier
    reqs = flagcxDevCommRequirements()
    reqs.intraMulticast = False
    reqs.barrierCount = 0
    reqs.intraBarrierCount = 1
    reqs.interBarrierCount = 0
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
    dist.barrier()

    from triton.runtime import DistributedRtContext
    ctx = DistributedRtContext(dev_mem_dptr.value, dev_comm_dptr.value)
    return ctx
