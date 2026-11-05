# Copyright 2026-     FlagOS Contributors
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
"""Host helpers for library=\"hccl\" (AscendC device HCCL).

Mirrors nvshmem: compile a small host .cpp → .so (ctypes.CDLL), no Python op
surface. ``library=\"hccl\"`` only adds AscendC HCCL include paths for JIT.
"""

from __future__ import annotations

import ctypes
import datetime
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
from triton.runtime.cache import get_cache_manager
from triton.experimental.tle.raw.cache_key import compute_tle_raw_host_cache_key

# library="hccl": open AscendC HCCL -I paths during cann JIT
_hccl_enabled: bool = False


def enable_hccl(enabled: bool = True) -> None:
    global _hccl_enabled
    _hccl_enabled = bool(enabled)


def is_hccl_enabled() -> bool:
    return _hccl_enabled


def _ascend_root() -> Path:
    for env in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME", "ASCEND_AICPU_PATH"):
        value = os.environ.get(env)
        if value:
            p = Path(value)
            if p.is_dir():
                return p.resolve()
    for candidate in (
            Path("/usr/local/Ascend/ascend-toolkit/latest"),
            Path("/usr/local/Ascend/cann-9.1.0"),
    ):
        if candidate.is_dir():
            return candidate.resolve()
    raise RuntimeError("cannot resolve Ascend toolkit root for HCCL includes")


def get_hccl_include_dirs() -> List[str]:
    """Include dirs for library=\"hccl\" (AscendC Hccl device API)."""
    if not is_hccl_enabled():
        return []
    root = _ascend_root()
    # Match cannshmem: tikcfw + impl/interface so kernel_operator.h finds
    # kernel_tpipe.h; lib/hccl/hccl.h lives under tikcfw/.
    candidates = [
        root / "compiler" / "tikcpp" / "tikcfw",
        root / "compiler" / "tikcpp" / "tikcfw" / "impl",
        root / "compiler" / "tikcpp" / "tikcfw" / "interface",
        root / "tools" / "tikcpp" / "tikcfw",
        root / "tools" / "tikcpp" / "tikcfw" / "impl",
        root / "tools" / "tikcpp" / "tikcfw" / "interface",
        root / "aarch64-linux" / "tikcpp" / "tikcfw",
        root / "aarch64-linux" / "tikcpp" / "tikcfw" / "impl",
        root / "aarch64-linux" / "tikcpp" / "tikcfw" / "interface",
        root / "aarch64-linux" / "asc" / "include" / "interface",
        root / "aarch64-linux" / "asc" / "include" / "basic_api",
        root / "aarch64-linux" / "asc" / "include" / "adv_api",
        root / "aarch64-linux" / "ascendc" / "include" / "basic_api" / "impl",
        root / "include" / "ascendc" / "include",
        root / "include" / "ascendc" / "highlevel_api",
        root / "include" / "ascendc" / "include" / "adv_api",
    ]
    out: List[str] = []
    for d in candidates:
        if d.is_dir() and str(d.resolve()) not in out:
            out.append(str(d.resolve()))
    if not out:
        raise RuntimeError(f"no HCCL/AscendC include dirs under {root}")
    return out


def init_torch_distributed():
    """NPU process group (same role as nvshmem init_torch_distributed)."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("npu", local_rank),
        timeout=datetime.timedelta(seconds=1800),
    )
    torch.distributed.barrier()
    return dist.group.WORLD


def get_hccl_comm(rank: Optional[int] = None):
    """Raw HcclComm handle from torch ProcessGroup (intptr / capsule)."""
    if rank is None:
        rank = dist.get_rank()
    backend = dist.group.WORLD._get_backend(torch.device("npu"))
    if hasattr(backend, "get_hccl_comm"):
        return backend.get_hccl_comm(rank)
    raise RuntimeError("ProcessGroupHCCL has no get_hccl_comm(rank)")


class HostLibrary:

    def __init__(self, library_path: Path):
        self.path = Path(library_path).expanduser().resolve()
        self.library = ctypes.CDLL(str(self.path))

    def __getattr__(self, name):
        return getattr(self.library, name)


def _compile_host_to_cache(source: Path, force: bool = False) -> Path:
    source_path = Path(source).expanduser().resolve()
    root = _ascend_root()
    host_cache_key = compute_tle_raw_host_cache_key(source_path, "npu-hccl")
    output_name = source_path.with_suffix(".so").name
    cache = get_cache_manager(host_cache_key)
    cached = None if force else cache.get_file(output_name)
    if cached is not None:
        return Path(cached)

    include_flags: List[str] = []
    for d in (
            root / "include",
            root / "include" / "hccl",
            root / "aarch64-linux" / "include",
            root / "aarch64-linux" / "pkg_inc",
            root / "aarch64-linux" / "pkg_inc" / "runtime",
            root / "aarch64-linux" / "pkg_inc" / "aicpu",
    ):
        if d.is_dir():
            include_flags += ["-I", str(d)]

    lib_dirs = []
    for d in (root / "lib64", root / "aarch64-linux" / "lib64"):
        if d.is_dir():
            lib_dirs.append(d)

    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{output_name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()

    command = [
        "g++",
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c++17",
        *include_flags,
        "-o",
        str(temporary_path),
        str(source_path),
    ]
    for d in lib_dirs:
        command += [f"-L{d}", f"-Wl,-rpath,{d}"]
    command += ["-lhccl", "-lascendcl"]

    try:
        build = subprocess.run(command, capture_output=True)
        if build.returncode != 0:
            raise RuntimeError("g++ failed while compiling HCCL host library\n"
                               f"command: {shlex.join(command)}\n"
                               f"stderr:\n{build.stderr.decode()}")
        return Path(cache.put(temporary_path.read_bytes(), output_name, binary=True))
    finally:
        temporary_path.unlink(missing_ok=True)


def load_host(source, force: bool = False) -> HostLibrary:
    """Compile host .cpp → .so and load (nvshmem load_host analogue)."""
    return HostLibrary(_compile_host_to_cache(Path(source), force=force))


# Verified non-GE KFC host (cann-hccl AicpuUnfoldKernelLaunch contract).
# Do not change packing / kernel name without re-validating against CANN.
_KFC_HOST_SRC = Path(__file__).resolve().parent / "kfc_host.cpp"


def load_kfc_host(force: bool = False) -> HostLibrary:
    """Load the shared KFC host .so used by raw device-HCCL tutorials."""
    return load_host(_KFC_HOST_SRC, force=force)


def create_op_res_ctx(host: HostLibrary, comm, count: int = 1) -> int:
    """HcclCreateOpResCtx for AICPU MC2 client (CombinOpParam* as uint64)."""
    out = ctypes.c_uint64(0)
    fn = host.hccl_kfc_probe_create_ctx
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    fn.restype = ctypes.c_int
    rc = fn(ctypes.c_void_p(int(comm)), ctypes.c_uint64(count), ctypes.byref(out))
    if rc != 0 or out.value == 0:
        raise RuntimeError(f"HcclCreateOpResCtx failed: rc={rc}, context=0x{out.value:x}")
    return int(out.value)


def start_kfc_server(
    host: HostLibrary,
    context: int,
    scratch: "torch.Tensor",
    main_stream: Optional[int] = None,
) -> tuple[int, int]:
    """Enqueue RunAicpuRpcSrvLaunch on HCCL AICPU attach stream.

    Ordering matches the validated probe: ACL record(main) → wait(attach) →
    rtAicpuKernelLaunchExWithArgs(ApiParamDef, KERNEL_TYPE_AICPU_KFC).
    Call before the AICore raw kernel that uses Hccl::Init / Finalize.
    """
    if main_stream is None:
        main_stream = int(torch.npu.current_stream().npu_stream)
    handle = ctypes.c_uint64(0)
    launch_rc = ctypes.c_int(-1)
    fn = host.hccl_kfc_probe_start
    fn.argtypes = [
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int),
    ]
    fn.restype = ctypes.c_int
    rc = fn(
        ctypes.c_uint64(context),
        ctypes.c_void_p(main_stream),
        ctypes.c_uint64(scratch.data_ptr()),
        ctypes.c_uint64(scratch.numel()),
        ctypes.byref(handle),
        ctypes.byref(launch_rc),
    )
    if rc != 0:
        raise RuntimeError(f"KFC enqueue failed: wrapper_rc={rc}, rt_launch_rc={launch_rc.value}")
    return int(handle.value), int(launch_rc.value)


def finish_kfc_server(
    host: HostLibrary,
    handle: int,
    timeout_ms: int = 10000,
) -> int:
    """Synchronize attach stream and free probe state. 0 = success."""
    fn = host.hccl_kfc_probe_finish
    fn.argtypes = [ctypes.c_uint64, ctypes.c_int32]
    fn.restype = ctypes.c_int
    return int(fn(ctypes.c_uint64(handle), ctypes.c_int32(timeout_ms)))


class HcclKfcSession:
    """Host AICPU KFC server for one HcclComm (nvshmem-style explicit init).

    Device AscendC ``Hccl::*`` needs this server on an attach stream before
    Commit/Wait. Usage (all ranks)::

      session = HcclKfcSession.open(comm, count=n)
      dist.barrier()
      session.ensure_server(scratch=buf)  # buf is also send/recv for packing
      # launch @dialect(library=\"hccl\") AICore client
      session.finish()
    """

    def __init__(self, host: HostLibrary, context: int):
        self.host = host
        self.context = context
        self._handle: Optional[int] = None
        self._scratch: Optional[torch.Tensor] = None

    @classmethod
    def open(cls, comm, count: int = 1, force_rebuild: bool = False) -> "HcclKfcSession":
        host = load_kfc_host(force=force_rebuild)
        context = create_op_res_ctx(host, comm, count=count)
        return cls(host, context)

    def ensure_server(self, scratch: Optional[torch.Tensor] = None) -> int:
        if self._handle is not None:
            return self._handle
        if scratch is None:
            scratch = torch.zeros((16, ), dtype=torch.float16, device="npu")
        self._scratch = scratch
        handle, _launch_rc = start_kfc_server(self.host, self.context, scratch)
        self._handle = handle
        return handle

    def finish(self, timeout_ms: Optional[int] = None) -> None:
        if self._handle is None:
            return
        if timeout_ms is None:
            timeout_ms = int(os.environ.get("HCCL_KFC_PROBE_TIMEOUT_MS", "10000"))
        rc = finish_kfc_server(self.host, self._handle, timeout_ms)
        self._handle = None
        if rc != 0:
            # Do not tear down ACL while a live KFC task may still own the stream.
            raise RuntimeError(f"KFC attach stream did not finish within {timeout_ms} ms, acl_rc={rc}")
