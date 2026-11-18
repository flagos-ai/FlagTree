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
"""Host helpers for library=\"hccl\" (AscendC device HCCL + hcomm KFC).

Compile a small host .cpp → .so (ctypes.CDLL). ``library=\"hccl\"`` adds
AscendC HCCL includes.

Lifecycle (aligned with hcomm CommonKFCMc2GenTask / ops MatmulAllReduce)::

  session = HcclKfcSession.open(comm, count=n)   # HcclCreateOpResCtx
  session.configure(...)                         # DoAllReduceTiling fields
  session.start()                                # AICPU TASK_DELIVER on attach
  # AICore: Hccl::Init → AllReduce(register) → Commit* → Wait → Finalize
  session.finish()                               # sync attach after Finalize

KFC start/finish are **host-explicit**. Triton launch does **not** auto
start/finish AICPU (legacy ``auto_kfc=True`` / ``HCCL_KFC_AUTO_LAUNCH=1``
restores the old per-launch hook — slow and wrong for multi-turn Commit).

Deadlock / timeout rules after ``start()`` (AICPU on attach waits for device
HCCL msgs; default server timeout ~30s):

  * Do **not** ``torch.npu.synchronize()`` (device-wide) before AICore launch
    — waits on attach → host↔AICPU deadlock (507017).
  * Do **not** start AICPU before a cold Triton JIT of a heavy mix-CV kernel
    — compile can exceed the AICPU wait → ``msgPos 0`` / 507017. Warm up
    (``kernel.warmup``) first, then ``start()`` once per op, then launch.
    (Legacy ``auto_kfc`` hid this because start lived in pre_launch *after*
    compile; do not bring back per-launch start/finish for perf.)
  * After launch, sync the **main** stream only; ``finish()`` syncs attach.
"""

from __future__ import annotations

import ctypes
import datetime
import os
import shlex
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
from triton.runtime.cache import get_cache_manager
from triton.experimental.tle.raw.cache_key import compute_tle_raw_host_cache_key

# library="hccl": open AscendC HCCL -I paths during cann JIT
_hccl_enabled: bool = False
_tls = threading.local()


def enable_hccl(enabled: bool = True) -> None:
    global _hccl_enabled
    _hccl_enabled = bool(enabled)


def is_hccl_enabled() -> bool:
    return _hccl_enabled


def _ascend_root() -> Path:
    for env in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "ASCEND_HOME", "ASCEND_AICPU_PATH"):
        value = os.environ.get(env)
        if value:
            p = Path(value)
            if p.is_dir():
                return p.resolve()
    # Generic install layouts only (no versioned / machine-local paths).
    for candidate in (
            Path("/usr/local/Ascend/ascend-toolkit/latest"),
            Path("/usr/local/Ascend/cann"),
    ):
        if candidate.is_dir():
            return candidate.resolve()
    raise RuntimeError(
        "cannot resolve Ascend toolkit root for HCCL includes; "
        "source CANN setenv.bash first, or set ASCEND_TOOLKIT_HOME / ASCEND_HOME_PATH")


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
    """Initialize the NPU process group (backend=hccl)."""
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
    """Compile host .cpp → .so and load via ctypes."""
    return HostLibrary(_compile_host_to_cache(Path(source), force=force))


# Official hcomm path: AicpuUnfoldKernelLaunch (5×u64 ApiParam) +
# AicpuAclKernelLaunch (ccl_kernel.json binHandle, RunAicpuRpcSrvLaunch) +
# DoAllReduceTiling DELIVER. Do not change without re-checking hcomm.
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


# MC2_BUFFER_TYPE — match ops-transformer / aicpu_operator_pub.
MC2_BUFFER_TYPE_DEFAULT = 0
MC2_BUFFER_TYPE_OUTPUT = 1
MC2_BUFFER_TYPE_WINDOW_IN = 2


def start_kfc_server(
    host: HostLibrary,
    context: int,
    send_ptr: int,
    recv_ptr: int,
    send_cnt: int,
    turn_num: int = 1,
    use_buffer_type: int = MC2_BUFFER_TYPE_OUTPUT,
    main_stream: Optional[int] = None,
) -> tuple[int, int]:
    """Enqueue RunAicpuRpcSrvLaunch on HCCL AICPU attach stream.

    Tiling matches ops-transformer ``DoAllReduceTiling`` (DELIVER, exitPolicy=0,
    preparePosition=kernel). ``send_cnt`` is per-tile elements; ``turn_num`` is
    tileCnt. In-place AllReduce → ``use_buffer_type=OUTPUT`` and send==recv.

    Stream order approximates ``CommonKFCMc2GenTask``: Record(main) →
    Wait(attach) → AICPU; caller then launches AICore on main.
    """
    if main_stream is None:
        main_stream = int(torch.npu.current_stream().npu_stream)
    send_cnt = int(send_cnt)
    turn_num = max(int(turn_num), 1)
    use_buffer_type = int(use_buffer_type)
    if send_cnt <= 0 or send_ptr == 0 or recv_ptr == 0:
        raise ValueError(
            f"KFC needs real buffers and send_cnt>0, got send_ptr={send_ptr:#x} "
            f"recv_ptr={recv_ptr:#x} send_cnt={send_cnt}")
    handle = ctypes.c_uint64(0)
    launch_rc = ctypes.c_int(-1)
    fn = host.hccl_kfc_probe_start
    # Must match kfc_host.cpp::hccl_kfc_probe_start signature.
    fn.argtypes = [
        ctypes.c_uint64,  # context
        ctypes.c_void_p,  # mainStream
        ctypes.c_uint64,  # sendPtr
        ctypes.c_uint64,  # recvPtr
        ctypes.c_uint64,  # sendCnt
        ctypes.c_uint64,  # turnNum
        ctypes.c_uint32,  # useBufferType
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int),
    ]
    fn.restype = ctypes.c_int
    rc = fn(
        ctypes.c_uint64(context),
        ctypes.c_void_p(main_stream),
        ctypes.c_uint64(int(send_ptr)),
        ctypes.c_uint64(int(recv_ptr)),
        ctypes.c_uint64(send_cnt),
        ctypes.c_uint64(turn_num),
        ctypes.c_uint32(use_buffer_type),
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


def get_active_hccl_kfc_session() -> Optional["HcclKfcSession"]:
    return getattr(_tls, "session", None)


def set_active_hccl_kfc_session(session: Optional["HcclKfcSession"]) -> None:
    _tls.session = session


def hccl_kfc_pre_launch() -> None:
    """Legacy launch-enter hook (only when ``session.auto_kfc`` is True)."""
    session = get_active_hccl_kfc_session()
    if session is None:
        raise RuntimeError(
            "library='hccl' kernel launched without an active HcclKfcSession; "
            "call HcclKfcSession.open() or from_context() first")
    # Default path: host already called start(); launch only runs AICore.
    if not getattr(session, "auto_kfc", False):
        return
    session.start()


def hccl_kfc_post_launch() -> None:
    """Legacy launch-exit hook (only when ``session.auto_kfc`` is True)."""
    session = get_active_hccl_kfc_session()
    if session is None:
        return
    if not getattr(session, "auto_kfc", False):
        return
    # Host-MM ∥ device-Commit: defer finish so another stream can feed ready[].
    if getattr(session, "defer_finish", False):
        return
    session.finish()


class HcclKfcSession:
    """Host-side AICPU KFC session (hcomm TASK_DELIVER) for AscendC HCCL.

    ``HcclComm`` is owned by torch. Prefer explicit lifecycle::

      session = HcclKfcSession.open(comm, count=n)
      session.configure(c_ptr, c_ptr, total_cnt=n, turn_num=tile_cnt)
      session.start()                 # AICPU on attach (before AICore)
      kernel[(grid,)](...)            # device: Init / Commit / Finalize
      session.finish()                # after device Finalize

    ``auto_kfc=True`` restores the old per-Triton-launch start/finish hook
    (not recommended; conflicts with multi-turn Commit inside one KFC server).
    """

    def __init__(self, host: HostLibrary, context: int, count: int = 1,
                 auto_kfc: bool = False):
        self.host = host
        self.context = int(context)
        # OpRes sizing hint from HcclCreateOpResCtx (not KFC tiling count).
        self.count = max(int(count), 1)
        self._handle: Optional[int] = None
        # Legacy: when True, hccl_kfc_* launch hooks call start/finish.
        self.auto_kfc: bool = bool(auto_kfc)
        # When True with auto_kfc, post_launch skips finish() (caller finish()).
        self.defer_finish: bool = False
        # Must configure before start().
        self._configured: bool = False
        self._send_ptr: int = 0
        self._recv_ptr: int = 0
        self._send_cnt: int = 0
        self._turn_num: int = 1
        self._use_buffer_type: int = MC2_BUFFER_TYPE_OUTPUT
        set_active_hccl_kfc_session(self)

    @property
    def started(self) -> bool:
        return self._handle is not None

    def configure(
        self,
        send_ptr: int,
        recv_ptr: int,
        *,
        total_cnt: int,
        turn_num: int = 1,
        use_buffer_type: int = MC2_BUFFER_TYPE_OUTPUT,
    ) -> "HcclKfcSession":
        """Set DoAllReduceTiling fields for the next ``start()`` (DELIVER)."""
        turn_num = max(int(turn_num), 1)
        total_cnt = max(int(total_cnt), 1)
        if total_cnt % turn_num != 0:
            raise ValueError(f"total_cnt={total_cnt} not divisible by turn_num={turn_num}")
        self._send_ptr = int(send_ptr)
        self._recv_ptr = int(recv_ptr)
        self._turn_num = turn_num
        self._send_cnt = total_cnt // turn_num
        self._use_buffer_type = int(use_buffer_type)
        self._configured = True
        return self

    # Backward-compatible alias.
    def configure_launch(
        self,
        send_ptr: int,
        recv_ptr: int,
        *,
        total_cnt: int,
        turn_num: int = 1,
        use_buffer_type: int = MC2_BUFFER_TYPE_OUTPUT,
    ) -> "HcclKfcSession":
        return self.configure(
            send_ptr,
            recv_ptr,
            total_cnt=total_cnt,
            turn_num=turn_num,
            use_buffer_type=use_buffer_type,
        )

    @classmethod
    def from_context(cls, context: int, count: int = 1, force_rebuild: bool = False,
                     auto_kfc: bool = False) -> "HcclKfcSession":
        """Wrap an existing OpRes context (framework / caller owned)."""
        if not context:
            raise ValueError("context must be a non-zero HcclCreateOpResCtx pointer")
        return cls(load_kfc_host(force=force_rebuild), int(context), count=count,
                   auto_kfc=auto_kfc)

    @classmethod
    def open(cls, comm, count: int = 1, force_rebuild: bool = False,
             auto_kfc: bool = False) -> "HcclKfcSession":
        """CreateOpResCtx from a torch HcclComm, then wrap it."""
        host = load_kfc_host(force=force_rebuild)
        count = max(int(count), 1)
        return cls(host, create_op_res_ctx(host, comm, count=count), count=count,
                   auto_kfc=auto_kfc)

    def activate(self) -> "HcclKfcSession":
        """Make this session the active one for this thread."""
        set_active_hccl_kfc_session(self)
        return self

    def start(self, scratch: Optional[torch.Tensor] = None) -> int:
        """Enqueue AICPU RunAicpuRpcSrvLaunch (TASK_DELIVER) on attach stream.

        Call **before** AICore launch (hcomm CommonKFCMc2GenTask order).
        Keep the server across multi-turn Commits until device Finalize, then
        ``finish()``. Re-entrant when already started (idempotent).

        After this returns, do **not** call ``torch.npu.synchronize()`` until
        the AICore kernel that talks to HCCL has been launched — that is a
        device-wide wait on the attach AICPU and deadlocks (507017). Prefer
        ``torch.npu.current_stream().synchronize()`` after the launch instead.
        """
        if self._handle is not None:
            return self._handle
        if scratch is not None:
            self.configure(
                int(scratch.data_ptr()),
                int(scratch.data_ptr()),
                total_cnt=int(scratch.numel()),
                turn_num=1,
                use_buffer_type=MC2_BUFFER_TYPE_OUTPUT,
            )
        if not self._configured:
            raise RuntimeError(
                "HcclKfcSession: call configure(send, recv, total_cnt=..., "
                "turn_num=...) before start() "
                "(DoAllReduceTiling needs real buffers / tile sizes)")
        handle, _launch_rc = start_kfc_server(
            self.host,
            self.context,
            self._send_ptr,
            self._recv_ptr,
            self._send_cnt,
            turn_num=self._turn_num,
            use_buffer_type=self._use_buffer_type,
        )
        self._handle = handle
        return handle

    # Backward-compatible name; same as start().
    def ensure_server(self, scratch: Optional[torch.Tensor] = None) -> int:
        return self.start(scratch)

    def finish(self, timeout_ms: Optional[int] = None, sync_main: bool = True) -> None:
        """Sync after device ``Hccl::Finalize`` and free AICPU attach state.

        Default ``sync_main=True`` waits for the current NPU stream first so
        device Finalize has completed before attach-stream sync (otherwise
        ``aclrtSynchronizeStreamWithTimeout`` can return 507046).
        """
        if self._handle is None:
            return
        if sync_main:
            torch.npu.current_stream().synchronize()
        if timeout_ms is None:
            timeout_ms = int(os.environ.get("HCCL_KFC_PROBE_TIMEOUT_MS", "10000"))
        handle = self._handle
        self._handle = None
        rc = finish_kfc_server(self.host, handle, timeout_ms)
        if rc != 0:
            raise RuntimeError(
                f"KFC attach stream did not finish within {timeout_ms} ms, acl_rc={rc}")
