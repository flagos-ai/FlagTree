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

import ctypes
import datetime
import os
import warnings
from pathlib import Path

import torch
import functools
import sysconfig
import subprocess
import re
import tempfile
import shlex
from triton.runtime.cache import get_cache_manager
from triton.experimental.tle.raw.cache_key import compute_tle_raw_host_cache_key

_cuda = None
_cudart = None


def _get_cuda_modules():
    global _cuda, _cudart
    if _cuda is not None and _cudart is not None:
        return _cuda, _cudart
    try:
        from cuda.bindings import driver as cuda
        from cuda.bindings import runtime as cudart
    except ImportError:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The cuda\.(cuda|cudart) module is deprecated.*",
                category=FutureWarning,
            )
            from cuda import cuda as cuda
            from cuda import cudart as cudart
    _cuda = cuda
    _cudart = cudart
    return _cuda, _cudart


def __getattr__(name):
    # Preserve `from ...utils import cuda/cudart` after lazy-loading.
    if name in ("cuda", "cudart"):
        cuda, cudart = _get_cuda_modules()
        return cuda if name == "cuda" else cudart
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def CUDA_CHECK(err):
    cuda, cudart = _get_cuda_modules()
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


@functools.lru_cache()
def get_nvshmem_home() -> Path:
    from triton import knobs
    if knobs.nvidia.nvshmem_home is not None:
        return Path(knobs.nvidia.nvshmem_home)

    try:
        import nvidia.nvshmem
        return Path(nvidia.nvshmem.__path__[0])
    except Exception:
        raise RuntimeError("Cannot resolve NVSHMEM_HOME: environment variable not set and nvidia.nvshmem not found")


def try_get_nvshmem_home() -> Path | None:
    try:
        return get_nvshmem_home()
    except RuntimeError:
        return None


def resolve_nvshmem_host_library(nvshmem_home: Path | None = None) -> Path:
    """Prefer unversioned .so; fall back to .so.N (pip wheels often omit the symlink)."""
    home = Path(nvshmem_home) if nvshmem_home is not None else get_nvshmem_home()
    lib_dir = home / "lib"
    for name in ("libnvshmem_host.so", "libnvshmem_host.so.3"):
        path = lib_dir / name
        if path.exists():
            return path
    raise RuntimeError(f"Cannot find libnvshmem_host.so[.3] under {lib_dir}")


# Set by @dialect(..., library="nvshmem"); CUDA backend links device .bc only when True.
_nvshmem_device_bc_enabled: bool = False


def enable_nvshmem_device_bc(enabled: bool = True) -> None:
    global _nvshmem_device_bc_enabled
    _nvshmem_device_bc_enabled = bool(enabled)


def is_nvshmem_device_bc_enabled() -> bool:
    return _nvshmem_device_bc_enabled


def get_nvshmem_extern_libs(arch: str | int | None = None) -> dict[str, str]:
    """Return {libnvshmem_device: path} when enabled; else {}."""
    if not is_nvshmem_device_bc_enabled():
        return {}
    bc = resolve_nvshmem_device_bitcode(arch=arch)
    return {"libnvshmem_device": str(bc)} if bc is not None else {}


def resolve_nvshmem_device_bitcode(nvshmem_home: Path | None = None, arch: str | int | None = None) -> Path | None:
    """Resolve device bitcode: unified .bc, else per-SM .bc (NVSHMEM >= ~3.7)."""
    home = Path(nvshmem_home) if nvshmem_home is not None else try_get_nvshmem_home()
    if home is None:
        return None
    lib_dir = home / "lib"
    unified = lib_dir / "libnvshmem_device.bc"
    if unified.is_file():
        return unified
    # arch: sm90 / sm_90a / 90 -> try exact then next-lower known ships
    if isinstance(arch, int):
        sm = arch
    else:
        s = str(arch or "").lower().replace("sm_", "").replace("sm", "").rstrip("a")
        sm = int(s) if s.isdigit() else None
    sms = (90, 89, 80, 75, 70)
    for n in ([sm] + [x for x in sms if sm is not None and x < sm]) if sm is not None else sms:
        path = lib_dir / f"libnvshmem_device_sm_{n}.bc"
        if path.is_file():
            return path
    return None


@functools.lru_cache()
def get_nvcc():
    return _path_to_binary("nvcc")


@functools.lru_cache()
def _path_to_binary(binary: str):
    binary += sysconfig.get_config_var("EXE")
    paths = [os.environ.get(f"TRITON_{binary.upper()}_PATH", "")]

    cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")
    paths += [f"{cuda_home}/bin/{binary}"]

    for path in paths:
        if os.path.exists(path) and os.path.isfile(path):
            result = subprocess.check_output([path, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
                if version is not None:
                    return path, version.group(1)
    raise RuntimeError(f"Cannot find {binary}")


def _compile_cuda_host_to_cache(
    source,
    nvshmem_home,
    arch: str = None,
    force: bool = False,
) -> tuple[Path, str, bool]:
    source_path = Path(source).expanduser().resolve()
    if not arch:
        from triton.experimental.tle.raw.cuda.runtime import _get_cuda_gpu_arch
        arch = _get_cuda_gpu_arch().split('=')[1]
    nvshmem_home = get_nvshmem_home()

    host_cache_key = compute_tle_raw_host_cache_key(source_path, arch)
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
    nvcc, _ = get_nvcc()
    lib_dir = nvshmem_home / "lib"
    host_lib = resolve_nvshmem_host_library(nvshmem_home)
    device_lib = lib_dir / "libnvshmem_device.a"
    if not device_lib.is_file():
        raise RuntimeError(f"Cannot find libnvshmem_device.a under {lib_dir}")
    # nvcc cannot take .so as a positional input; use -L/-l (-l: for versioned SONAME).
    command = [
        nvcc,
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-rdc=true",
        f"-arch={arch}",
        f"-I{nvshmem_home / 'include'}",
        f"-L{lib_dir}",
        f"-l:{host_lib.name}",
        "-lnvshmem_device",
        "-lcuda",
        "-Xlinker",
        "-rpath",
        "-Xlinker",
        str(lib_dir),
        "-o",
        str(temporary_path),
        str(source_path),
    ]
    try:
        build = subprocess.run(command, capture_output=True)
        if build.returncode != 0:
            raise RuntimeError("nvcc failed while compiling CUDA host library\n"
                               f"command: {shlex.join(command)}\n"
                               f"stderr:\n{build.stderr.decode()}")
        cached_path = cache.put(temporary_path.read_bytes(), output_name, binary=True)
        return Path(cached_path), host_cache_key, False
    finally:
        temporary_path.unlink(missing_ok=True)


class CudaHostLibrary:

    def __init__(self, library_path):
        self.path = Path(library_path).expanduser().resolve()
        self.library = ctypes.CDLL(str(self.path))

    def __getattr__(self, name):
        return getattr(self.library, name)


def get_common_host_source() -> Path:
    return Path(__file__).resolve().parents[0] / "common-host.cu"


def load_host(source, nvshmem_home=None, arch=None, force: bool = False) -> CudaHostLibrary:
    path, _, _ = _compile_cuda_host_to_cache(source, nvshmem_home, arch, force=force)
    return CudaHostLibrary(path)


def load_common_host(source=None, nvshmem_home=None, arch=None, force: bool = False) -> CudaHostLibrary:
    return load_host(source or get_common_host_source(), nvshmem_home, arch, force)


def init_torch_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", local_rank),
        timeout=datetime.timedelta(seconds=1800),
    )
    group = torch.distributed.new_group(ranks=list(range(world_size)), backend="nccl")
    torch.distributed.barrier(group=group)
    return group


def init_nvshmem_by_torch_pg(common, group):
    rank = group.rank()
    world_size = group.size()
    uid_size = 1024

    if rank == 0:
        temp_buffer = ctypes.create_string_buffer(uid_size)
        result = common.nvshmem_get_unique_id_bytes(temp_buffer, uid_size)
        assert result == 0, f"nvshmemx_get_uniqueid failed: {result}"
        uid = bytes(temp_buffer.raw)
    else:
        uid = bytes(uid_size)

    objects = [uid]
    torch.distributed.broadcast_object_list(
        objects,
        src=torch.distributed.get_global_rank(group, 0),
        group=group,
    )
    uid_buffer = ctypes.create_string_buffer(objects[0], uid_size)
    result = common.nvshmem_init_from_torch_distributed(
        rank,
        world_size,
        int(os.environ["LOCAL_RANK"]),
        uid_buffer,
        uid_size,
    )
    assert result == 0, f"NVSHMEM init failed: {result}"
    torch.distributed.barrier(group=group)


def tensor_from_pointer(
    pointer: int | ctypes.c_void_p,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a non-owning Torch tensor view over a CUDA allocation."""
    address = pointer.value if isinstance(pointer, ctypes.c_void_p) else pointer
    if address is not None and not isinstance(address, int):
        raise TypeError(f"pointer must be int or ctypes.c_void_p, got {type(pointer).__name__}")
    if not address:
        raise ValueError("pointer cannot be null; CUDA memory must be allocated")
    elements = 1
    for extent in shape:
        elements *= extent
    storage = torch._C._construct_storage_from_data_pointer(
        address,
        device,
        elements * dtype.itemsize,
    )
    return torch.empty(0, dtype=dtype, device=device).set_(storage).view(shape)


def set_signal_cuda_ptr(signal_ptr, signal, stream):
    cuda, _ = _get_cuda_modules()
    (err, ) = cuda.cuStreamWriteValue64(
        stream.cuda_stream,
        signal_ptr,
        signal,
        cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
    )
    CUDA_CHECK(err)


def print_perf(
    name: str,
    value: float,
    group,
    rank: int,
    world_size: int,
    unit: str = "ms",
):
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
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    value_tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    torch.distributed.all_reduce(
        value_tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=group,
    )
    mean_value = (value_tensor / world_size).item()
    if rank == 0:
        print(f"{name} mean: {mean_value:.3f} {unit}", flush=True)
