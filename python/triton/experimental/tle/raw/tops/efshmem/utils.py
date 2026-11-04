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
"""efshmem (Enflame GCU NVSHMEM) support for TLE-Raw TOPS kernels.

TOPS counterpart of `triton.experimental.tle.raw.nvshmem.utils`. efshmem is the
Enflame GCU port of NVSHMEM with an `efshmem`/`efshmemx` naming scheme (its
upstream NVIDIA spelling is `nvshmem`/`nvshmemx`); the install layout is the
NVSHMEM one:

    $EFSHMEM_HOME/include/                   efshmem.h, efshmemx.h ...
    $EFSHMEM_HOME/lib/libefshmem_host.so[.1]
    $EFSHMEM_HOME/lib/libefshmem_device.bc   (device bitcode for TOPS)

Two things the TOPS backend needs from here:

1. `enable_efshmem_device_bc()` — set by `@dialect(name="tops", library="efshmem")`.
   `third_party/enflame/backend/compiler.py` then attaches the device bitcode to
   the kernel's `extern_libs`, which the GCU backend forwards as
   `--gcu-attach-target ... l=<bc>` ("extra bitcode libraries to link"). That is
   what resolves the device-side `efshmem_*` symbols a `.tops` kernel calls.

2. `register_efshmem_cumodule_init_hook()` — efshmem keeps per-module state that
   has to be initialised on the loaded module (`efshmemx_cumodule_init`), same as
   CUDA NVSHMEM. The hook runs from `triton.runtime.jit` right after compile.
"""

from __future__ import annotations

import ctypes
import functools
import datetime
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from triton.experimental.tle.raw.cache_key import compute_tle_raw_host_cache_key
from triton.runtime.cache import get_cache_manager

_EFSHMEM_CUMODULE_INIT_HOOK = "efshmem_cumodule_init"
# Key the efshmem device bitcode is filed under in the kernel's `extern_libs`.
# It is only a label for the `--gcu-attach-target ... l=<path>` list (the GCU
# backend forwards the paths, and the only name it matches specially is
# `libdevice`), so it is spelled after the artifact it points at:
# libefshmem_device.bc.
EFSHMEM_EXTERN_LIB_KEY = "libefshmem_device"

# Set by @dialect(..., library="efshmem"); the GCU backend links the efshmem
# device bitcode only when True.
_efshmem_device_bc_enabled: bool = False


def enable_efshmem_device_bc(enabled: bool = True) -> None:
    global _efshmem_device_bc_enabled
    _efshmem_device_bc_enabled = bool(enabled)


def is_efshmem_device_bc_enabled() -> bool:
    return _efshmem_device_bc_enabled


def get_efshmem_home() -> Path:
    """Resolve the efshmem install prefix (the directory containing include/ and lib/)."""
    for var in ("EFSHMEM_HOME", "NVSHMEM_HOME"):
        value = os.environ.get(var)
        if value:
            home = Path(value)
            # Accept both an install prefix and the src/ subdir of a build tree.
            if (home / "include" / "efshmem.h").is_file():
                return home
            if (home / "src" / "include" / "efshmem.h").is_file():
                return home / "src"
            raise RuntimeError(f"{var}={value} does not look like an efshmem install "
                               "(no include/efshmem.h)")
    raise RuntimeError("Cannot resolve EFSHMEM_HOME: set EFSHMEM_HOME (or NVSHMEM_HOME) to "
                       "the efshmem install prefix")


def try_get_efshmem_home() -> Path | None:
    try:
        return get_efshmem_home()
    except RuntimeError:
        return None


def resolve_efshmem_host_library(efshmem_home: Path | None = None) -> Path:
    """Prefer unversioned .so; fall back to the versioned SONAME."""
    home = Path(efshmem_home) if efshmem_home is not None else get_efshmem_home()
    lib_dir = home / "lib"
    for name in ("libefshmem_host.so", "libefshmem_host.so.1", "libefshmem_host.so.1.0.0"):
        path = lib_dir / name
        if path.exists():
            return path
    raise RuntimeError(f"Cannot find libefshmem_host.so under {lib_dir}")


def resolve_efshmem_device_library(efshmem_home: Path | None = None) -> Path | None:
    """Device-side efshmem bitcode linked into the kernel.

    Bitcode only: the value ends up in `--gcu-attach-target ... l=<path>`, which
    documents itself as "extra bitcode libraries", so a static archive of device
    objects would not be a usable substitute.
    """
    home = Path(efshmem_home) if efshmem_home is not None else try_get_efshmem_home()
    if home is None:
        return None
    bitcode = home / "lib" / "libefshmem_device.bc"
    return bitcode if bitcode.is_file() else None


def get_efshmem_extern_libs(efshmem_home: Path | None = None) -> dict[str, str]:
    """Return {EFSHMEM_EXTERN_LIB_KEY: path} when enabled; else {}."""
    if not is_efshmem_device_bc_enabled():
        return {}
    device_lib = resolve_efshmem_device_library(efshmem_home)
    return {EFSHMEM_EXTERN_LIB_KEY: str(device_lib)} if device_lib is not None else {}


def get_efshmem_include_dir(efshmem_home: Path | None = None) -> Path:
    home = Path(efshmem_home) if efshmem_home is not None else get_efshmem_home()
    return home / "include"


def get_efshmem_lib_dir(efshmem_home: Path | None = None) -> Path:
    home = Path(efshmem_home) if efshmem_home is not None else get_efshmem_home()
    return home / "lib"


def get_efshmem_include_flags() -> list[str]:
    """`-I` flags for topscc; empty when efshmem is not installed."""
    home = try_get_efshmem_home()
    if home is None:
        return []
    return [f"-I{home / 'include'}"]


# ---------------------------------------------------------------------------
# Host shim: compile + load
# ---------------------------------------------------------------------------


@functools.lru_cache()
def _tops_home() -> Path:
    caps_path = os.environ.get("CAPS_PATH", "/opt/tops")
    return Path(caps_path)


def _tops_include_dir() -> str:
    from triton.experimental.tle.raw.tops.runtime import _find_tops_include_dir
    return _find_tops_include_dir()


def _get_host_compiler() -> str:
    """Host C++ compiler for the shim.

    Only libefshmem_host.so is linked: the shim goes through the host-library
    entry points (`efshmemx_hostlib_init_attr` / `efshmemx_hostlib_finalize`),
    which are real exported symbols, rather than the `efshmemx_init_attr` /
    `efshmem_finalize` static inlines that call the runtime's
    `efshmemi_init_thread` / `efshmemi_finalize`. Those live in
    libefshmem_device.a next to a device fatbin reference, which would force a
    `--tops-link` host link; going through the host-library entry points keeps
    the shim an ordinary host .so.

    Override with $EFSHMEM_HOST_CC.
    """
    override = os.environ.get("EFSHMEM_HOST_CC")
    if override:
        return override
    for name in (os.environ.get("CXX"), "g++", "c++", "clang++"):
        if name and shutil.which(name):
            return name
    raise RuntimeError("Cannot find a host C++ compiler for the efshmem host shim; "
                       "set EFSHMEM_HOST_CC")


def _build_host_command(compiler: str, source_path: Path, output_path: Path,
                        efshmem_home: Path) -> list[str]:
    lib_dir = efshmem_home / "lib"
    host_lib = resolve_efshmem_host_library(efshmem_home)
    tops_lib_dir = _tops_home() / "lib"
    # The source has to precede the -L/-l flags: with the default --as-needed a
    # library seen before any undefined symbol exists is dropped, which shows up
    # later as an unresolved symbol at dlopen time.
    return [
        compiler,
        "-shared",
        "-fPIC",
        "-std=c++17",
        f"-I{efshmem_home / 'include'}",
        f"-I{_tops_include_dir()}",
        "-o",
        str(output_path),
        str(source_path),
        f"-L{lib_dir}",
        # `-l:` so the versioned SONAME can be used verbatim, like the nvshmem
        # CUDA path does for a wheel that ships no unversioned symlink.
        f"-l:{host_lib.name}",
        f"-L{tops_lib_dir}",
        "-ltopsrt",
        "-Xlinker",
        "-rpath",
        "-Xlinker",
        str(lib_dir),
    ]


def _compile_host_to_cache(
    source,
    efshmem_home: Path | None = None,
    force: bool = False,
) -> tuple[Path, str, bool]:
    source_path = Path(source).expanduser().resolve()
    home = Path(efshmem_home) if efshmem_home is not None else get_efshmem_home()

    compiler = _get_host_compiler()
    # Unlike the CUDA shim, this one is arch-independent (nothing device-side is
    # compiled or linked here), so the cache key only has to cover the build
    # inputs: the source, the toolchain, the efshmem install, and the TOPS
    # toolkit (its -I/-L/-ltopsrt come from CAPS_PATH / TOPS_INCLUDE_DIR, so
    # changing either must invalidate the cached .so).
    host_cache_key = compute_tle_raw_host_cache_key(
        source_path,
        f"{compiler}\0{home}\0{_tops_home()}\0{_tops_include_dir()}",
    )
    output_name = source_path.with_suffix(".so").name
    cache = get_cache_manager(host_cache_key)

    cached = None if force else cache.get_file(output_name)
    if cached is not None:
        return Path(cached), host_cache_key, True

    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{output_name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()

    command = _build_host_command(compiler, source_path, temporary_path, home)
    try:
        build = subprocess.run(command, capture_output=True)
        if build.returncode != 0:
            raise RuntimeError("failed to compile the efshmem host library\n"
                               f"command: {shlex.join(command)}\n"
                               f"stderr:\n{build.stderr.decode()}")
        cached_path = cache.put(temporary_path.read_bytes(), output_name, binary=True)
        return Path(cached_path), host_cache_key, False
    finally:
        temporary_path.unlink(missing_ok=True)


class EfshmemHostLibrary:
    """ctypes handle on a compiled efshmem host shim."""

    def __init__(self, library_path):
        self.path = Path(library_path).expanduser().resolve()
        self.library = ctypes.CDLL(str(self.path))

    def __getattr__(self, name):
        return getattr(self.library, name)


def get_common_host_source() -> Path:
    return Path(__file__).resolve().parents[0] / "common-host.cc"


def load_host(source, efshmem_home=None, force: bool = False) -> EfshmemHostLibrary:
    path, _, _ = _compile_host_to_cache(source, efshmem_home, force=force)
    return EfshmemHostLibrary(path)


def load_common_host(source=None, efshmem_home=None, force: bool = False) -> EfshmemHostLibrary:
    return load_host(source or get_common_host_source(), efshmem_home, force)


@functools.lru_cache()
def efshmem_module() -> ctypes.CDLL:
    """ctypes handle on libefshmem_host.so, for direct host API calls."""
    return ctypes.CDLL(str(resolve_efshmem_host_library(get_efshmem_home())))


# ---------------------------------------------------------------------------
# Post-compile cumodule init
# ---------------------------------------------------------------------------


def _get_efshmemx_cumodule_init():
    library = efshmem_module()
    fn = library.efshmemx_cumodule_init
    fn.argtypes = [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def _initialize_efshmem_cumodule(kernel):
    kernel._init_handles()
    result = _get_efshmemx_cumodule_init()(ctypes.c_void_p(kernel.module))
    assert result == 0, f"efshmemx_cumodule_init failed: {result}"


def register_efshmem_cumodule_init_hook() -> None:
    """Register the efshmem cumodule init hook, idempotently.

    The hook registry lives in the TLE-Raw CUDA runtime module because that is
    the one `triton.runtime.jit` consults after every compile, on all backends;
    the registry itself is backend-neutral (kernel metadata drives it).

    This is invoked from `triton.experimental.tle.raw.tops.runtime` at import
    time (mirroring how `cuda.runtime` registers its own nvshmem hook), so just
    importing the TOPS dialect runtime is enough; no need to call it by hand.
    """
    from triton.experimental.tle.raw.cuda.runtime import (
        _KERNEL_INIT_HOOKS,
        register_kernel_init_hook,
    )
    if _EFSHMEM_CUMODULE_INIT_HOOK in _KERNEL_INIT_HOOKS:
        return
    register_kernel_init_hook(_EFSHMEM_CUMODULE_INIT_HOOK, _initialize_efshmem_cumodule)


# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------


def _torch_device_module():
    import torch
    return getattr(torch, "gcu", torch.cuda)


def resolve_gcu_device(local_rank: int, local_world_size: int | None = None) -> int:
    """GCU index for a local rank (a PE), matching efshmem's own PE->device map.

    efshmem picks the device itself while initialising (`src/host/init/init.tops`):

        topsGetDeviceCount(&dev_count);
        if (dev_count > 1 && mype_node >= 0) {
            npes_per_gpu = ceil(npes_node / dev_count);
            cudevice = mype_node / npes_per_gpu;      // block mapping
        } else {
            cudevice = 0;                             // single device: everyone
        }

    so this reproduces that: device 0 when there is one device (any number of PEs
    packed onto it), and otherwise `local_rank // npes_per_gpu`. With the usual
    one-PE-per-device launch it is the identity, which is also what a plain
    `torch.device("gcu", local_rank)` would give.
    """
    device_module = _torch_device_module()
    count = device_module.device_count() if hasattr(device_module, "device_count") else 0
    if count <= 1:
        return 0
    if local_world_size is None:
        env = os.environ.get("LOCAL_WORLD_SIZE")
        # Unknown PE-per-node: every local rank gets its own device (and the
        # block mapping below then degrades to the identity).
        local_world_size = int(env) if env else local_rank + 1
    npes_per_gpu = max(1, -(-local_world_size // count))
    return min(local_rank // npes_per_gpu, count - 1)


def _default_torch_dist_backend() -> str:
    """Pick the best available ProcessGroup backend for a GCU job.

    Mirrors what `triton_dist` does: ECCL (the Enflame collective comms
    library) is what gives the GCU path its performance, but it needs
    libefml.so; a torch_gcu build without it would fail inside
    init_process_group(backend="cpu:gloo,gcu:eccl"). So probe it and fall back
    to host gloo.
    """
    import torch

    if not hasattr(torch, "gcu"):
        return "cpu:gloo,cuda:nccl"
    try:
        ctypes.CDLL("libefml.so")
    except OSError:
        return "gloo"
    return "cpu:gloo,gcu:eccl"


def init_torch_distributed(
    backend: str | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
    timeout_seconds: int = 1800,
):
    """Initialise torch.distributed on GCU from the torchrun environment.

    Args:
        backend: ProcessGroup backend spec. Defaults to
            $EFSHMEM_TORCH_DIST_BACKEND, else `cpu:gloo,gcu:eccl` when ECCL is
            loadable, else `gloo` (see `_default_torch_dist_backend`).

    Only host-side collectives are needed here (the efshmem unique id is a ~1KB
    object broadcast), so the CPU part of a composite backend carries it; the
    data path is efshmem's own transport, not the ProcessGroup.

    ECCL is used only when there is at least one GCU per PE: several PEs packed
    onto one GCU (efshmem MPG) makes ECCL fail with `ecclInvalidUsage`, so such
    a job falls back to host gloo -- which is enough for the uid broadcast.
    """
    import torch

    backend = backend or os.environ.get("EFSHMEM_TORCH_DIST_BACKEND") or _default_torch_dist_backend()
    rank = int(os.environ["RANK"]) if rank is None else int(rank)
    local_rank = int(os.environ["LOCAL_RANK"]) if local_rank is None else int(local_rank)
    world_size = int(os.environ["WORLD_SIZE"]) if world_size is None else int(world_size)

    device_index = resolve_gcu_device(local_rank)
    device = torch.device("gcu", device_index)
    device_module = _torch_device_module()
    if hasattr(device_module, "set_device"):
        device_module.set_device(device_index)

    # One rank per device is a hard requirement of the device backends; more PEs
    # *per node* than local GCUs means efshmem is packing several PEs per card
    # (MPG), where ECCL rejects init_process_group with `ecclInvalidUsage`.
    # Compare LOCAL_WORLD_SIZE, not the global world_size: in a cross-node job
    # world_size > device_count even with one PE per GPU and ECCL is valid.
    device_backend = backend.split(",")[-1].split(":")[-1]
    device_count = device_module.device_count() if hasattr(device_module, "device_count") else 0
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if device_backend not in ("gloo", "mpi") and device_count and local_world_size > device_count:
        backend = "gloo"
        device_backend = "gloo"

    # `device_id` is only meaningful for a device backend (NCCL/ECCL), not gloo.
    init_kwargs = {"device_id": device} if device_backend not in ("gloo", "mpi") else {}
    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=timeout_seconds),
        **init_kwargs,
    )
    # Object collectives (broadcast_object_list) run on the CPU part of a
    # composite backend, so the world group is the right handle; a subgroup
    # narrowed to the device backend could not carry them.
    return torch.distributed.group.WORLD


def get_unique_id_bytes(common: EfshmemHostLibrary, uid_size: int = 1024) -> bytes:
    """Ask the local PE for an efshmem unique id (the multi-PE bootstrap token)."""
    temp_buffer = ctypes.create_string_buffer(uid_size)
    result = common.efshmem_get_unique_id_bytes(temp_buffer, uid_size)
    assert result == 0, f"efshmemx_get_uniqueid failed: {result}"
    return bytes(temp_buffer.raw)


def init_efshmem(
    common: EfshmemHostLibrary,
    uid: bytes,
    *,
    rank: int,
    nranks: int,
    device: int,
) -> None:
    """Bring up this PE from a unique id already shared by the launcher."""
    uid_size = len(uid)
    uid_buffer = ctypes.create_string_buffer(uid, uid_size)
    result = common.efshmem_init_from_torch_distributed(
        rank,
        nranks,
        device,
        uid_buffer,
        uid_size,
    )
    assert result == 0, f"efshmem init failed: {result}"


def init_efshmem_local(
    common: EfshmemHostLibrary,
    *,
    rank: int = 0,
    nranks: int = 1,
    device: int = 0,
) -> None:
    """Single-PE bring-up: no launcher and no torch process group involved."""
    init_efshmem(common, get_unique_id_bytes(common), rank=rank, nranks=nranks, device=device)


def init_efshmem_by_torch_pg(common: EfshmemHostLibrary, group, device: int | None = None) -> None:
    """Broadcast the efshmem unique id over `group` and bring up every PE."""
    import torch

    rank = group.rank()
    world_size = group.size()
    uid_size = 1024

    if rank == 0:
        uid = get_unique_id_bytes(common, uid_size)
    else:
        uid = bytes(uid_size)

    objects = [uid]
    torch.distributed.broadcast_object_list(
        objects,
        src=torch.distributed.get_global_rank(group, 0),
        group=group,
    )
    init_efshmem(
        common,
        objects[0],
        rank=rank,
        nranks=world_size,
        device=resolve_gcu_device(int(os.environ["LOCAL_RANK"])) if device is None else device,
    )
    torch.distributed.barrier(group=group)


def finalize_efshmem(common: EfshmemHostLibrary) -> None:
    result = common.efshmem_finalize_from_torch_distributed()
    assert result == 0, f"efshmem finalize failed: {result}"


def tensor_from_pointer(
    pointer: int | ctypes.c_void_p,
    shape: tuple[int, ...],
    dtype,
    device,
):
    """Create a non-owning Torch tensor view over an efshmem symmetric allocation."""
    import torch

    address = pointer.value if isinstance(pointer, ctypes.c_void_p) else pointer
    if address is not None and not isinstance(address, int):
        raise TypeError(f"pointer must be int or ctypes.c_void_p, got {type(pointer).__name__}")
    if not address:
        raise ValueError("pointer cannot be null; efshmem memory must be allocated")
    elements = 1
    for extent in shape:
        elements *= extent
    storage = torch._C._construct_storage_from_data_pointer(
        address,
        device,
        elements * dtype.itemsize,
    )
    return torch.empty(0, dtype=dtype, device=device).set_(storage).view(shape)


def print_perf(
    name: str,
    value: float,
    group,
    rank: int,
    world_size: int,
    unit: str = "ms",
):
    import torch

    for index in range(world_size):
        torch.distributed.barrier(group=group)
        if rank == index:
            print(f"{name} #{rank}: {value:.3f} {unit}", flush=True)


def print_perf_mean(
    name: str,
    value: float,
    group,
    rank: int,
    world_size: int,
    unit: str = "ms",
):
    import torch

    device = torch.device("gcu", int(os.environ["LOCAL_RANK"]))
    value_tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    torch.distributed.all_reduce(
        value_tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=group,
    )
    mean_value = (value_tensor / world_size).item()
    if rank == 0:
        print(f"{name} mean: {mean_value:.3f} {unit}", flush=True)
