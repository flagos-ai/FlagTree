"""Backend-side runtime for `tle.raw`: turning user source into an LLVM IR payload.

`tle.raw.call` only requires its first argument to satisfy the contract

    make_llvm(context=None, arch=None) -> str   # LLVM-dialect MLIR text or textual LLVM IR
    name -> str                                 # symbol to call in that payload

`triton_xpu.raw` accepts either form, so a backend only has to produce the text;
`xpu-clang -S -emit-llvm` output can be handed over verbatim.

FlagTree ships one stack, the XPU one (`third_party/xpu`, XPUBackend), so there is
one dialect name:

    @tle.raw.dialect("xpu3", file="k.xpu")     # XPU cluster path

The name only picks the stack; the arch itself still comes from whichever backend
compiles the kernel. (Internal Triton also has `"xpu4"`/`"xpu5"` for the mars and
jupiter stacks and `"houyi"` for the jupiter cluster path; neither backend exists
in FlagTree, so those names are rejected here rather than silently mapped to XPU.)

Payloads are deferred by default: the architecture is only known once a backend
actually compiles the kernel, so the source is registered in
`source_store` and compiled during `make_llir` with the real arch (see
`deferred.py`). Pass `deferred=False` to compile eagerly at trace time, which
pins the payload to `arch=`/`TRITON_XPU_ARCH` instead.

Toolchain lookup: `TRITON_XPU_CLANG_PATH` wins, then the clang packaged with the
backend, then `_deps/xtdk/*/bin/clang`, then `LLVM_SYSPATH`/`XPU_HOME`/`PATH`. A
candidate that cannot resolve `xpu/kernel/xtdk.h` is skipped in favour of one that
can -- the packaged clang ships without its resource dir, so a payload that
includes xtdk.h needs one of the full installs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .cache_key import bind_source_cache_key, compute_source_cache_key
from .source_store import register_source

__all__ = [
    "RawJITFunction",
    "XPUJITFunction",
    "registry",
    "dialect",
]


class RawJITFunction:
    """Compiles an XPU C++ (`.xpu`) payload to LLVM IR with the xtdk clang.

    Two requirements on the payload's entry point:

    - `extern "C"`, so its symbol matches the Python function name that
      `triton_xpu.raw` looks the callee up by (no C++ mangling).
    - a device attribute: either `__device__` (needs
      `#include "xpu/kernel/xtdk.h"`) or `__attribute__((device))`. Without it
      `--xpu-device-only` treats the function as host-only and emits a module
      with no definition at all -- clang still exits 0, so `make_llvm` checks
      for this explicitly.

    Signature seen from C++, per `triton_xpu.raw`'s calling convention on the
    cluster path (one C++ parameter per operand, outputs first then inputs; a
    `tt.ptr<T>` arrives as `T*` and a scalar keeps its type):

        extern "C" __device__ void vec_add(_global_ptr_ const float *x,
                                           _global_ptr_ float *out, int n);
    """

    # Subclasses set these.
    dialect_name: str = ""
    arch_env_vars: tuple = ("TRITON_XPU_ARCH", )
    default_arch: int = 3
    # Headers a usable toolchain has to ship, relative to its resource dir.
    device_headers: tuple = ("xpu/kernel/xtdk.h", )
    no_definition_hint: str = ('Declare it as extern "C" __device__ (or __attribute__((device))) -- '
                               "without a device attribute --xpu-device-only emits an empty module.")

    def __init__(self, fn, file=None, source=None, arch=None, flags=(), deferred=True):
        self.fn = fn
        self.name = fn.__name__
        self.flags = list(flags)
        self.deferred = bool(deferred)
        self.arch = int(arch) if arch is not None else None
        self.file = None
        self._inline_source = None
        if source is not None:
            self._inline_source = source
        else:
            if file is None:
                raise ValueError(f"{type(self).__name__} needs either `source` or `file`")
            path = Path(file)
            if not path.is_absolute():
                path = Path(fn.__code__.co_filename).parent / path
            self.file = path
        # Payload compiled per arch: eager mode fills the entry for its own arch,
        # deferred mode for whichever arch the backend is building for.
        self._llvm = {}
        bind_source_cache_key(self, self._source_cache_key)

    # -- source ------------------------------------------------------------

    @property
    def source(self) -> str:
        """Payload text, re-read on every access.

        This keeps `make_llvm`'s per-source cache keyed on what is actually on
        disk. Kernel-level cache invalidation is a separate matter -- see
        cache_key.py.
        """
        if self.file is not None:
            return self.file.read_text()
        return self._inline_source

    def _source_cache_key(self) -> str:
        return compute_source_cache_key(
            dialect=self.dialect_name,
            callee=self.name,
            source=self.source,
            arch=self.arch,
            flags=self.flags,
            file=self.file,
        )

    @property
    def cache_key(self) -> str:
        return self._source_cache_key()

    # -- toolchain ---------------------------------------------------------

    def resolve_arch(self, arch=None) -> int:
        """Arch to compile the payload for.

        The backend wins (it knows what it is building), then an explicit
        `arch=` on the decorator, then the environment, then the stack default.
        """
        if arch is not None:
            return int(arch)
        if self.arch is not None:
            return self.arch
        for env_var in self.arch_env_vars:
            value = os.environ.get(env_var)
            if value:
                return int(value)
        return self.default_arch

    def _backend_clang_dir(self, arch: int) -> Path:
        raise NotImplementedError

    @classmethod
    def _has_device_headers(cls, clang: Path) -> bool:
        """True if this clang ships the payload's device headers in its resource dir."""
        try:
            out = subprocess.run([str(clang), "-print-resource-dir"], capture_output=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return False
        resource_dir = Path(out.stdout.decode().strip())
        return all((resource_dir / "include" / header).is_file() for header in cls.device_headers)

    def _clang(self, arch: int) -> str:
        """Locate an XPU clang, preferring one that can resolve the device headers.

        `TRITON_XPU_CLANG_PATH` wins, same as the backends themselves.
        Otherwise the packaged per-arch toolchain is the default, but in a source
        checkout it only carries CUDA-compat headers, so an xtdk toolchain under
        `_deps/` (the one the device library itself is built with) is preferred
        when present.
        """
        candidates = [self._backend_clang_dir(arch) / "clang"]
        if "TRITON_XPU_CLANG_PATH" not in os.environ:
            for parent in candidates[0].resolve().parents:
                deps = parent / "_deps" / "xtdk"
                if deps.is_dir():
                    candidates += sorted(deps.glob("*/bin/clang"))
                    break
            candidates += self._extra_clang_candidates()

        existing = [c for c in candidates if c.is_file()]
        if not existing:
            raise RuntimeError(f"tle.raw: no XPU clang found for arch xpu{arch}; tried "
                               f"{[str(c) for c in candidates]}. Set TRITON_XPU_CLANG_PATH.")
        for clang in existing:
            if self._has_device_headers(clang):
                return str(clang)
        return str(existing[0])

    def _extra_clang_candidates(self) -> list:
        """Fallback toolchains to look at when the packaged one is not usable."""
        return []

    def _clang_cmd(self, clang: str, arch: int) -> list:
        """Command compiling the payload on stdin to LLVM IR on stdout."""
        return [
            clang,
            f"--xpu-arch=xpu{arch}",
            "-x",
            "xpu",
            "--xpu-device-only",
            "-S",
            "-emit-llvm",
            "-O2",
            "-std=c++11",
            "-",
            "-o",
            "-",
        ]

    # -- compilation -------------------------------------------------------

    def make_llvm(self, context=None, arch=None) -> str:
        arch = self.resolve_arch(arch)
        source = self.source
        cached = self._llvm.get((arch, source))
        if cached is not None:
            return cached
        cmd = self._clang_cmd(self._clang(arch), arch) + self.flags
        result = subprocess.run(cmd, input=source.encode(), capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"tle.raw: compiling payload '{self.name}' for xpu{arch} failed:\n"
                               f"{result.stderr.decode(errors='replace')}")
        llvm_ir = result.stdout.decode()
        if "define " not in llvm_ir or f"@{self.name}(" not in llvm_ir:
            raise RuntimeError(f"tle.raw: payload compiled but defines no '{self.name}'. "
                               f"{self.no_definition_hint}")
        self._llvm[(arch, source)] = llvm_ir
        return llvm_ir

    def register_pending_source(self) -> str:
        """Record the payload for deferred compilation; returns its source id."""
        return register_source(dialect=self.dialect_name, callee=self.name, source=self.source, handle=self)


class XPUJITFunction(RawJITFunction):
    """XPU stack (`third_party/xpu`, XPUBackend): arch 3."""

    dialect_name = "xpu"
    arch_env_vars = ("TRITON_XPU_ARCH", )
    default_arch = 3

    def _backend_clang_dir(self, arch: int) -> Path:
        # Importing the backend is safe here: it does not pull in torch.
        from triton.backends.xpu.compiler import XPUBackend

        class _Opt:
            pass

        opt = _Opt()
        opt.arch = arch
        return Path(XPUBackend.path_to_xpu_compile_tool(opt))

    def _extra_clang_candidates(self) -> list:
        """Fall back to a full XPU clang install for the device headers.

        FlagTree copies only the clang binary and `lib/linux` into
        `third_party/xpu/backend/xpu3`, not the resource dir, so the packaged
        clang cannot resolve `xpu/kernel/xtdk.h` -- `_has_device_headers` rejects
        it and the search moves on to these. `LLVM_SYSPATH` is the toolchain that
        clang was copied from (and the one the device library is built with), so
        it is tried first, then `XPU_HOME`, then `PATH`.
        """
        candidates = []
        for env_var in ("LLVM_SYSPATH", "XPU_HOME"):
            root = os.environ.get(env_var)
            if root:
                candidates.append(Path(root) / "bin" / "clang")
        found = shutil.which("clang")
        if found:
            candidates.append(Path(found))
        return candidates


registry = {
    # Chip generation -> stack. FlagTree only has the XPU one.
    "xpu3": XPUJITFunction,
}


def dialect(name="xpu3", **kwargs):
    """Decorator turning a Python stub into a `tle.raw` payload handle.

    Example:
        @tle.raw.dialect("xpu3", file="vec_add.xpu")
        def vec_add(x, y, out, n):
            ...
    """
    if name not in registry:
        raise ValueError(f"unknown tle.raw dialect '{name}', expected one of {sorted(registry)}")
    cls = registry[name]

    def decorator(fn):
        return cls(fn, **kwargs)

    return decorator
