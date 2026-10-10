"""Host and device glue for tle.raw library=\"dushmem\".

NVSHMEM links a ready ``libnvshmem_device.bc``. DUSHMEM's device state lives in
an offload bundle inside ``libdushmem_device.a``; the put/get bodies are
``always_inline`` headers compiled with the user's HIP. After the hsaco loads,
``dushmemx_cumodule_init`` fills ``dushmemi_device_state_d``.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

DUSHMEM_CUMODULE_INIT_HOOK = "dushmem_cumodule_init"

_cumodule_init = None


def dushmem_lib_dir() -> Path:
    env = os.environ.get("DUSHMEM_HOME")
    if env:
        lib = Path(env) / "lib"
        if lib.is_dir():
            return lib
    return Path("/opt/dtk/lib/dushmem")


def resolve_dushmem_host_library() -> Path:
    lib_dir = dushmem_lib_dir()
    for name in ("libdushmem_host.so", "libdushmem_host.so.3"):
        path = lib_dir / name
        if path.is_file():
            return path
    raise RuntimeError(f"Cannot find libdushmem_host.so under {lib_dir}")


def resolve_dushmem_device_bitcode(arch: str) -> Path:
    """Unbundle the device archive for ``arch`` and link the pieces Triton needs.

    ``init_device`` defines ``dushmemi_device_state_d``. ``transfer_device``
    defines the non-inline ``dushmemi_transfer_rma_p`` / ``quiet`` bodies that
    ``dushmem_int_p`` calls when ``DUSHMEM_ENABLE_ALL_DEVICE_INLINING`` is off.
    """
    archive = dushmem_lib_dir() / "libdushmem_device.a"
    if not archive.is_file():
        raise RuntimeError(f"Cannot find {archive}")
    stamp = archive.stat().st_mtime_ns
    cached = Path(tempfile.gettempdir()) / f"dushmem-device-slim2-{arch}-{stamp}.bc"
    if cached.is_file():
        return cached
    clang_dir = Path(os.environ.get("TRITON_HIP_CLANG_PATH", "/opt/dtk/aillvm/bin/clang-18")).parent
    bundler = clang_dir / "clang-offload-bundler"
    llvm_ar = clang_dir / "llvm-ar"
    llvm_link = clang_dir / "llvm-link"
    members = ("init_device.cu.o", "transfer_device.cu.o")
    # simple_shift only needs the int put and the quiet helpers the header emits.
    # The rest of transfer_device is every collective; feeding that to dcc
    # produces an assembler error and a multi-minute compile.
    needed = (
        "_Z23dushmemi_transfer_rma_pIiEvPvT_i",
        "_Z23dushmemi_transfer_quietIL13threadgroup_t0EEvb",
        "_Z23dushmemi_transfer_quietIL13threadgroup_t1EEvb",
        "_Z23dushmemi_transfer_quietIL13threadgroup_t2EEvb",
    )
    with tempfile.TemporaryDirectory() as tmp:
        extract = subprocess.run([str(llvm_ar), "x", str(archive), *members], cwd=tmp, capture_output=True, text=True)
        if extract.returncode != 0:
            raise RuntimeError(f"llvm-ar failed to extract {members} from {archive}:\n{extract.stderr}")
        bitcodes: list[str] = []
        for member_name in members:
            member = Path(tmp) / member_name
            extracted = Path(tmp) / f"{member_name}.bc"
            unbundle = subprocess.run(
                [
                    str(bundler),
                    "-type=o",
                    f"-targets=hip-amdgcn-amd-amdhsa--{arch}",
                    f"-input={member}",
                    f"-output={extracted}",
                    "-unbundle",
                ],
                capture_output=True,
                text=True,
            )
            if unbundle.returncode != 0:
                raise RuntimeError(f"clang-offload-bundler failed for {member_name} ({arch}):\n{unbundle.stderr}")
            bitcodes.append(str(extracted))
        init_bc, transfer_bc = bitcodes
        slim = Path(tmp) / "transfer-slim.bc"
        slim_cmd = [str(clang_dir / "llvm-extract"), "--recursive", "-o", str(slim)]
        for func in needed:
            slim_cmd.append(f"--func={func}")
        # llvm-extract leaves string constants used by the put body as
        # declarations. Pull the one this gfx936 build references.
        slim_cmd.append("--glob=.str.3")
        slim_cmd.append(transfer_bc)
        extracted_funcs = subprocess.run(slim_cmd, capture_output=True, text=True)
        if extracted_funcs.returncode != 0:
            raise RuntimeError(f"llvm-extract failed:\n{extracted_funcs.stderr}")
        linked = Path(tmp) / "dushmem-device.bc"
        link = subprocess.run([str(llvm_link), init_bc, str(slim), "-o", str(linked)], capture_output=True, text=True)
        if link.returncode != 0:
            raise RuntimeError(f"llvm-link failed for DUSHMEM device bitcode:\n{link.stderr}")
        cached.write_bytes(linked.read_bytes())
    return cached


def _get_cumodule_init():
    global _cumodule_init
    if _cumodule_init is not None:
        return _cumodule_init
    library = ctypes.CDLL(str(resolve_dushmem_host_library()))
    fn = library.dushmemx_cumodule_init
    fn.argtypes = [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    _cumodule_init = fn
    return fn


def initialize_dushmem_cumodule(kernel) -> None:
    kernel._init_handles()
    result = _get_cumodule_init()(ctypes.c_void_p(kernel.module))
    if result != 0:
        raise RuntimeError(f"dushmemx_cumodule_init failed: {result}")
