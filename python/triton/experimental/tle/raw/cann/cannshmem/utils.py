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
"""Host helpers + native device includes for library=\"cannshmem\".

``library=\"cannshmem\"`` adds official cann/shmem header paths so AscendC can
call ``aclshmem_*`` directly (no AscendNPU-IR mlir template late-link).
Host setup uses ``import shmem as ash``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist

# library="cannshmem": JIT includes official shmem headers
_cannshmem_enabled: bool = False

_g_pe: Optional[int] = None
_g_world_size: Optional[int] = None
_g_mem_count: int = 0


def enable_cannshmem(enabled: bool = True) -> None:
    global _cannshmem_enabled
    _cannshmem_enabled = bool(enabled)


def is_cannshmem_enabled() -> bool:
    return _cannshmem_enabled


def add_cannshmem_extra_link_bitcode(path: str) -> None:
    """Backward-compat alias; prefer cann.runtime.add_op_extra_link_bitcode."""
    from triton.experimental.tle.raw.cann.runtime import add_op_extra_link_bitcode
    add_op_extra_link_bitcode(path)


def get_cannshmem_extra_link_bitcodes() -> List[str]:
    """Backward-compat alias; prefer cann.runtime.get_op_extra_link_bitcodes."""
    from triton.experimental.tle.raw.cann.runtime import get_op_extra_link_bitcodes
    return get_op_extra_link_bitcodes()


def resolve_cannshmem_home() -> Optional[Path]:
    """Root of a cann/shmem source install or checkout (has include/ + src/)."""
    for env in ("FLAGTREE_CANNSHMEM_HOME", "SHMEM_HOME_PATH", "ACLSHMEM_HOME"):
        value = os.environ.get(env)
        if not value:
            continue
        path = Path(value)
        for candidate in (path, path / "shmem", path.parent if path.name == "shmem" else path):
            if (candidate / "include" / "shmem.h").is_file():
                return candidate.resolve()
    # Walk up from this file (depth-independent after cann/ nesting).
    here = Path(__file__).resolve().parent
    for root in [here, *here.parents]:
        flash = root / "third_party" / "shmem"
        for candidate in (flash / "install" / "shmem", flash):
            if (candidate / "include" / "shmem.h").is_file():
                return candidate.resolve()
        # FlagTree checkout root often has third_party next to python/
        flash = root / "FlagTree" / "third_party" / "shmem"
        for candidate in (flash / "install" / "shmem", flash):
            if (candidate / "include" / "shmem.h").is_file():
                return candidate.resolve()
    try:
        import shmem
        root = Path(shmem.__file__).resolve().parent
        if (root / "include" / "shmem.h").is_file() and (root / "src" / "host_device").is_dir():
            return root
        # Prefer source install over incomplete pip wheels
        if (root / "include" / "shmem.h").is_file() and not (root / "src" / "host_device").is_dir():
            pass
    except ImportError:
        pass
    return None


def get_cannshmem_include_dirs() -> List[str]:
    """Include dirs for library=\"cannshmem\" (official headers)."""
    if not is_cannshmem_enabled():
        return []
    home = resolve_cannshmem_home()
    if home is None:
        raise RuntimeError("library='cannshmem' enabled but cann/shmem headers not found; "
                           "build from https://gitcode.com/cann/shmem and set "
                           "FLAGTREE_CANNSHMEM_HOME to install/shmem (or the repo root)")
    if not (home / "src" / "host_device").is_dir():
        raise RuntimeError(f"cann/shmem home {home} is missing src/host_device (incomplete wheel?); "
                           "use a source build install/ and set FLAGTREE_CANNSHMEM_HOME")
    dirs: List[Path] = []
    # include/: public shmem.h; src/device: hpp bodies; src/: host_device/* relative includes
    for sub in (home / "include", home / "src" / "device", home / "src"):
        if sub.is_dir():
            dirs.append(sub)
    ascend = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if ascend:
        root = Path(ascend)
        for sub in (
                root / "aarch64-linux" / "include",
                root / "x86_64-linux" / "include",
                root / "compiler" / "tikcpp" / "tikcfw",
                root / "compiler" / "tikcpp" / "tikcfw" / "impl",
                root / "compiler" / "tikcpp" / "tikcfw" / "interface",
                root / "tools" / "mssanitizer" / "include",
        ):
            if sub.is_dir():
                dirs.append(sub)
    if len(dirs) < 2:
        raise RuntimeError(f"cann/shmem home {home} has no include/ or src/device/")
    return [str(d.resolve()) for d in dirs]


def _get_ash_ip_port() -> str:
    ash_port = os.environ.get("ASH_MASTER_PORT", "8666")
    ash_addr = os.environ.get("ASH_MASTER_ADDR", "127.0.0.1")
    return f"tcp://{ash_addr}:{ash_port}"


def init_cannshmem_by_torch_pg(group=None, ash_size=None, ip_port=None, engine_type=None):
    """Init HCCL process group + aclshmem (installed `shmem` / ash package)."""
    del group
    global _g_pe, _g_world_size, _g_mem_count
    import shmem as ash

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl", rank=local_rank)
    _g_pe = dist.get_rank()
    _g_world_size = dist.get_world_size()
    dist.barrier()

    ash.set_conf_store_tls(False, "")
    attributes = ash.InitAttr()
    attributes.my_rank = _g_pe
    attributes.n_ranks = _g_world_size
    attributes.local_mem_size = ash_size if ash_size is not None else 1024 * 1024 * 1024
    attributes.ip_port = ip_port if ip_port is not None else _get_ash_ip_port()
    if engine_type is None:
        engine_type = ash.OpEngineType.MTE
    attributes.option_attr.data_op_engine_type = engine_type
    ret = ash.aclshmem_init(attributes)
    if ret != 0:
        raise RuntimeError(f"aclshmem_init failed: {ret}")
    _g_mem_count = 0


def create_symmetric_tensor(buf_tensor):
    """Allocate a symmetric ash tensor sized like ``buf_tensor``."""
    global _g_mem_count
    import shmem as ash

    if not isinstance(buf_tensor, torch.Tensor):
        raise TypeError("buf_tensor must be a torch.Tensor")
    if _g_pe is None:
        raise RuntimeError("call init_cannshmem_by_torch_pg before create_symmetric_tensor")
    peer_mem = ash.aclshmem_create_tensor([buf_tensor.numel()], dtype=buf_tensor.dtype, device_id=_g_pe)
    _g_mem_count += 1
    return peer_mem


def cleanup_symmetric_tensor(peer_mem):
    """Free one symmetric tensor; finalize ash when the last one is freed."""
    global _g_mem_count
    import shmem as ash

    torch.npu.synchronize()
    dist.barrier()
    ash.aclshmem_free_tensor(peer_mem)
    _g_mem_count -= 1
    if _g_mem_count < 0:
        raise RuntimeError("cleanup_symmetric_tensor called more times than create_symmetric_tensor")
    if _g_mem_count == 0:
        ash.aclshmem_finalize()


def bench_us(fn, warmup: int = 3, iters: int = 10) -> float:
    """Average device time in microseconds after warmup."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    dist.barrier()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    dist.barrier()
    return start.elapsed_time(end) * 1000.0 / iters
