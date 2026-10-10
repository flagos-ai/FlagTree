"""Torch launch helpers for HCU DUSHMEM.

``tensor_from_pointer`` is the same PyTorch storage view NVSHMEM uses. The
unique-id init is DUSHMEM-specific and lives in ``common-host.hip``.
"""

from __future__ import annotations

import ctypes
import datetime
import fcntl
import os
import subprocess
from pathlib import Path

import torch
from triton.experimental.tle.raw.nvshmem.utils import tensor_from_pointer

__all__ = [
    "compile_host_library",
    "init_dushmem_by_torch_pg",
    "init_torch_distributed",
    "load_common_host",
    "tensor_from_pointer",
]

_UID_BYTES = 1024


def _compile_shared(source: Path, library: Path) -> None:
    library.parent.mkdir(parents=True, exist_ok=True)
    lock_path = library.with_suffix(library.suffix + ".lock")
    command = [
        "/opt/dtk/bin/hipcc",
        "-shared",
        "-fPIC",
        "-fgpu-rdc",
        "--offload-arch=gfx936",
        "-O3",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
        "-mcode-object-version=4",
        "-I/opt/dtk/include",
        "-I/opt/dtk/include/dushmem",
        "-L/opt/dtk/lib/dushmem",
        str(source),
        "-o",
        str(library),
        "-ldushmem_host",
        "-ldushmem_device",
    ]
    with open(lock_path, "a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            stale = not library.is_file() or source.stat().st_mtime_ns > library.stat().st_mtime_ns
            if stale:
                build = subprocess.run(command, capture_output=True, text=True)
                if build.returncode != 0:
                    raise RuntimeError(f"host library build failed:\n{build.stderr}")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def compile_host_library(source: Path, library: Path) -> ctypes.CDLL:
    _compile_shared(source, library)
    return ctypes.CDLL(str(library))


def load_common_host() -> ctypes.CDLL:
    source = Path(__file__).with_name("common-host.hip")
    library = Path("/tmp/dushmem-common-host.so")
    _compile_shared(source, library)
    host = ctypes.CDLL(str(library))
    host.dushmem_get_unique_id_bytes.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    host.dushmem_get_unique_id_bytes.restype = ctypes.c_int
    host.dushmem_init_from_torch_distributed.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    host.dushmem_init_from_torch_distributed.restype = ctypes.c_int
    return host


def init_torch_distributed():
    """Create the process group torchrun already launched.

    gloo carries the unique-id broadcast. A later model can pass its own group
    into ``init_dushmem_by_torch_pg`` instead of calling this helper.
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=1800),
        )
    torch.distributed.barrier()
    return torch.distributed.group.WORLD


def init_dushmem_by_torch_pg(common, group) -> None:
    rank = group.rank()
    world_size = group.size()
    if rank == 0:
        temp_buffer = ctypes.create_string_buffer(_UID_BYTES)
        result = common.dushmem_get_unique_id_bytes(temp_buffer, _UID_BYTES)
        if result != 0:
            raise RuntimeError(f"dushmemx_get_uniqueid failed: {result}")
        uid = bytes(temp_buffer.raw)
    else:
        uid = bytes(_UID_BYTES)

    objects = [uid]
    torch.distributed.broadcast_object_list(objects, src=0, group=group)
    uid_buffer = ctypes.create_string_buffer(objects[0], _UID_BYTES)
    result = common.dushmem_init_from_torch_distributed(
        rank,
        world_size,
        int(os.environ["LOCAL_RANK"]),
        uid_buffer,
        _UID_BYTES,
    )
    if result != 0:
        raise RuntimeError(f"DUSHMEM init failed: {result}")
    torch.distributed.barrier(group=group)
