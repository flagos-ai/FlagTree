# Copyright 2026- Xcoresigma Technology Co., Ltd
from __future__ import annotations

import functools
import hashlib
import inspect
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

# This module must stay importable on non-Ascend builds: nothing here may
# import triton.language.extra.cann at module level. CANN-specific imports are
# deferred to CANNJITFunction construction / bitcode resolution time, which
# only happens on the Ascend backend.

# ccec aicore target per core type (Ascend 910B family), mirroring
# custom_ops/build_custom_ops.sh.
_ARCH_BY_CORE = {
    "VECTOR": "dav-c220-vec",
    "CUBE": "dav-c220-cube",
}

_DEFAULT_PIPELINE: Final[Dict[str, str]] = {
    "core": "vector",
    "pipe": "PIPE_V",
    "mode": "SIMD",
}

_BITCODE_FORMATS = ("bitcode", "bc")
_SOURCE_FORMATS = (None, "", "source", "cpp")


@functools.lru_cache(maxsize=None)
def _find_ccec() -> str:
    candidates = []
    env = os.getenv("CCEC")
    if env:
        candidates.append(Path(env))
    which = shutil.which("ccec")
    if which:
        candidates.append(Path(which))
    npu_compiler = os.getenv("TRITON_NPU_COMPILER_PATH")
    if npu_compiler:
        candidates.append(Path(npu_compiler) / "ccec")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise RuntimeError(
        "ccec not found: source the CANN environment first, or set $CCEC / $TRITON_NPU_COMPILER_PATH")


def _find_llvm_link(ccec: str) -> str:
    env = os.getenv("LLVM_LINK")
    if env and Path(env).is_file():
        return str(Path(env).resolve())
    sibling = Path(ccec).parent / "llvm-link"
    if sibling.is_file():
        return str(sibling.resolve())
    which = shutil.which("llvm-link")
    if which:
        return str(Path(which).resolve())
    raise RuntimeError("llvm-link not found next to ccec; set $LLVM_LINK")


@functools.lru_cache(maxsize=None)
def _ccec_version(ccec: str) -> str:
    try:
        build = subprocess.run([ccec, "--version"], capture_output=True)
        return (build.stdout + build.stderr).decode(errors="replace")
    except OSError:
        return ""


def _template_include_dirs(extra: Sequence[Path] = ()) -> List[str]:
    dirs = []
    env = os.getenv("TLE_CANN_TEMPLATE_INCLUDE")
    if env:
        dirs.append(Path(env))
    else:
        import triton
        # triton.__file__ = <repo>/python/triton/__init__.py under an editable
        # install; the AscendC Template headers live in the source tree.
        default = (Path(triton.__file__).resolve().parent.parent.parent / "third_party" / "ascend" /
                   "AscendNPU-IR" / "bishengir" / "lib" / "Template" / "include")
        if default.is_dir():
            dirs.append(default)
    dirs.extend(Path(p) for p in extra)
    return [str(d.resolve()) for d in dirs if d.is_dir()]


def _compile_bitcode(src: Path, arch: str, includes: Sequence[str]) -> str:
    """JIT-compile an AscendC source file to bitcode with ccec.

    Mirrors custom_ops/build_custom_ops.sh: compile the source twice (plain
    and -cce-enable-mix variants) and llvm-link both into a single .bc, then
    store it in the Triton cache keyed by (source content, arch, includes,
    ccec path + version). Returns the cached .bc path.
    """
    from triton.runtime.cache import get_cache_manager

    ccec = _find_ccec()
    key_payload = "\x00".join([
        "cann-jit",
        hashlib.sha256(src.read_bytes()).hexdigest(),
        arch,
        ccec,
        _ccec_version(ccec),
        *includes,
    ])
    key = hashlib.sha256(key_payload.encode()).hexdigest()
    cache = get_cache_manager(key)
    cached = cache.get_file("custom_ops.bc")
    if cached is not None:
        return cached

    common = [
        "-O2", "-x", "cce", "--cce-auto-sync=off", "--cce-aicore-only",
        "--cce-generic-addrspace=off", "-mllvm", "-disable-llvm-optzns",
        f"--cce-aicore-arch={arch}", "--cce-enable-print", "--cce-enable-sanitizer",
        "-std=c++17",
    ]
    for include in includes:
        common += ["-I", str(include)]

    with tempfile.TemporaryDirectory(prefix="tle-cann-jit-") as tmp:
        plain_bc = Path(tmp) / "op.bc"
        mix_bc = Path(tmp) / "op.mix.bc"
        linked_bc = Path(tmp) / "custom_ops.bc"
        variants = [([], plain_bc), (["-cce-enable-mix", "-mllvm", "-enable-mix=true"], mix_bc)]
        for extra_args, out_bc in variants:
            cmd = [ccec, *common, *extra_args, str(src), "-emit-llvm", "-c", "-o", str(out_bc)]
            build = subprocess.run(cmd, capture_output=True)
            if build.returncode != 0:
                raise RuntimeError("ccec failed\n"
                                   f"cmd: {' '.join(cmd)}\n"
                                   f"stderr:  {build.stderr.decode(errors='replace')}")
        llvm_link = _find_llvm_link(ccec)
        cmd = [llvm_link, str(plain_bc), str(mix_bc), "-o", str(linked_bc)]
        build = subprocess.run(cmd, capture_output=True)
        if build.returncode != 0:
            raise RuntimeError("llvm-link failed\n"
                               f"cmd: {' '.join(cmd)}\n"
                               f"stderr:  {build.stderr.decode(errors='replace')}")
        return cache.put(linked_bc.read_bytes(), "custom_ops.bc", binary=True)


class CANNJITFunction(object):
    """CANN custom op bound by @dialect(name="cann", ...).

    The instance duck-types the custom-op attribute interface consumed by
    triton.language.extra.cann.extension.custom_op (core / pipe / mode /
    symbol / bitcode), so tle_raw.call can lower it to hivm.custom without
    the string-keyed registry of the OP extension framework:

        @dialect(name="cann", format="bitcode",
                 file=Path(__file__).parent / "sort_topk.bc",
                 extern_func_name="custom_sort_1d_topk_proposals_float",
                 pipeline={"core": "vector", "pipe": "PIPE_V"})
        def sort_topk(*args, **kwargs):
            ...

        @triton.jit
        def kernel(src, tmp, out, K: tl.constexpr):
            out = tle_raw.call(sort_topk, [src, tmp, out, K], output_indices=[2])

    When format is "bitcode" / "bc" the file is referenced as-is. When format
    is omitted the file is treated as AscendC source and JIT-compiled to
    bitcode with ccec on first use; the produced bitcode is cached under the
    Triton cache directory (~/.triton/cache).
    """

    def __init__(self, fn: Any, file, format: Optional[str] = None, extern_func_name: str = "",
                 pipeline: Optional[Dict[str, str]] = None, arch: Optional[str] = None,
                 includes: Sequence = (), extra_buffers=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[Any] = fn
        self.format: Final[Optional[str]] = format
        self.symbol: Final[str] = extern_func_name or fn.__name__
        self.name: Final[str] = fn.__name__
        self.arch: Final[Optional[str]] = arch
        self._extra_includes: Final[Tuple[Path, ...]] = tuple(Path(p) for p in includes)
        self.__triton_builtin__: Final[bool] = True
        if extra_buffers is not None:
            self.extra_buffers = extra_buffers

        if self.format not in _BITCODE_FORMATS and self.format not in _SOURCE_FORMATS:
            raise ValueError(f"unsupported @dialect(name='cann') format: {self.format!r} "
                             f"(expected 'bitcode'/'bc' or a source format)")

        merged = dict(_DEFAULT_PIPELINE)
        if pipeline:
            merged.update(pipeline)

        import triton.language.extra.cann.extension as al
        try:
            self.core = al.CORE[merged["core"].upper()]
            self.pipe = al.PIPE[merged["pipe"].upper()]
            self.mode = al.MODE[merged["mode"].upper()]
        except KeyError as e:
            raise ValueError(f"invalid pipeline metadata {pipeline!r}: unknown entry {e}") from None

        path = Path(file)
        if not path.is_absolute():
            # Relative paths resolve against the decorated function's source
            # directory first, so operators can ship source next to the call.
            try:
                base = Path(inspect.getabsfile(fn)).parent
                if (base / path).exists():
                    path = base / path
            except TypeError:
                pass
        self.file: Final[Path] = path
        self._bitcode: Optional[str] = None

    def _resolve_bitcode(self) -> str:
        if self.format in _BITCODE_FORMATS:
            resolved = str(self.file.resolve())
            assert self.file.exists(), f"@dialect(name='cann') bitcode file not found: {resolved}"
            return resolved
        if not self.file.exists():
            raise FileNotFoundError(f"@dialect(name='cann') source file not found: {self.file}")
        arch = self.arch
        if arch is None:
            arch = _ARCH_BY_CORE.get(self.core.name)
            if arch is None:
                raise ValueError(f"cannot infer ccec aicore arch for core={self.core.name!r}; "
                                 "pass arch=... explicitly")
        includes = _template_include_dirs(self._extra_includes)
        return _compile_bitcode(self.file, arch, includes)

    @property
    def bitcode(self) -> str:
        if self._bitcode is None:
            self._bitcode = self._resolve_bitcode()
        return self._bitcode

    def __deepcopy__(self, memo: Dict[int, Any]) -> "CANNJITFunction":
        return self
